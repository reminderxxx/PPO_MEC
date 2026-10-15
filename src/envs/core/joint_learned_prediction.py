"""Train-window-only residual GRU forecast; inference consumes observed frames."""
from collections import Counter, deque
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from src.envs.core.joint_causal_observation import CausalMobilityPredictor


class ResidualMobilityGRU(nn.Module):
    def __init__(self, horizon=3):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(2, 16, batch_first=True)
        self.output = nn.Linear(16, horizon * 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, history):
        latent, _ = self.gru(history)
        offsets = torch.arange(1, self.horizon + 1, device=history.device)[None, :, None]
        return history[:, -1:, :] * offsets + self.output(latent[:, -1]).reshape(-1, self.horizon, 2)


def prediction_samples(windows, history=8, horizon=3):
    xs, ys = [], []
    for window in windows:
        positions = {}
        for frame in window['frames']:
            for v in frame['vehicles']:
                positions.setdefault(v.vehicle_id, []).append([v.position_x, v.position_y])
        for track in positions.values():
            p = np.asarray(track, dtype=np.float32)
            d = np.diff(p, axis=0)
            for t in range(history, len(p)-horizon):
                xs.append(d[t-history:t])
                ys.append(p[t+1:t+1+horizon]-p[t])
    return np.asarray(xs), np.asarray(ys)


def fit_predictor(train_windows, *, seed=101, epochs=30):
    if len(train_windows) < 2:
        raise ValueError('predictor requires separate train fit/validation windows')
    torch.manual_seed(seed)
    cut = max(1, int(len(train_windows)*.8))
    x, y = prediction_samples(train_windows[:cut])
    vx, vy = prediction_samples(train_windows[cut:])
    scale = float(max(np.std(x), 1e-3))
    x, y, vx, vy = [torch.as_tensor(a / scale) for a in (x, y, vx, vy)]
    model = ResidualMobilityGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for index in torch.randperm(len(x), generator=generator).split(512):
            loss = (model(x[index])-y[index]).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        learned = float(((model(vx)-vy)*scale).square().mean().sqrt())
        cv = vx[:, -1:, :] * torch.arange(1, 4)[None, :, None]
        constant = float(((cv-vy)*scale).square().mean().sqrt())
    bundle = {'contract': 'causal_residual_gru_v1', 'state_dict': model.state_dict(),
              'scale': scale, 'horizon': 3, 'history': 8, 'seed': seed, 'epochs': epochs,
              'fit_offsets': [w['frame_offset'] for w in train_windows[:cut]],
              'validation_offsets': [w['frame_offset'] for w in train_windows[cut:]],
              'validation_rmse_position': learned, 'constant_velocity_rmse_position': constant}
    return bundle


class LearnedMobilityPredictor(CausalMobilityPredictor):
    prediction_source = 'causal_residual_gru_v1'
    def __init__(self, bundle):
        super().__init__(horizon=bundle['horizon'])
        self.model = ResidualMobilityGRU(self.horizon)
        self.model.load_state_dict(bundle['state_dict']); self.model.eval()
        self.scale, self.history_size_gru = bundle['scale'], bundle['history']
        self.position_margin = 3*bundle.get('validation_rmse_position',0.)
        self.history = {}

    def reset(self):
        self.previous.clear(); self.pending.clear(); self.scores.clear(); self.history.clear()
        self.last_step = -1; self.last_result = {}

    @torch.no_grad()
    def predict(self, vehicles, associations, rsus, mapper, step_index):
        if step_index == self.last_step:
            return deepcopy(self.last_result)
        old = deepcopy(self.previous)
        result = super().predict(vehicles, associations, rsus, mapper, step_index)
        histories = []
        for v in vehicles:
            previous = old.get(v.vehicle_id)
            delta = [0., 0.] if previous is None else [
                (v.position_x-previous[1])/(step_index-previous[0]),
                (v.position_y-previous[2])/(step_index-previous[0])]
            history = self.history.setdefault(v.vehicle_id, deque(maxlen=self.history_size_gru))
            history.append(delta)
            histories.append([[0., 0.]]*(self.history_size_gru-len(history)) + list(history))
        forecasts = self.model(torch.tensor(histories, dtype=torch.float32)/self.scale).numpy()*self.scale
        final, sequences, next_rsu, dwell = {}, {}, {}, {}
        for v, forecast in zip(vehicles, forecasts):
            hops = []
            for offset in forecast:
                future = deepcopy(v); future.position_x += float(offset[0]); future.position_y += float(offset[1])
                hops.append(mapper.associate([future])[v.vehicle_id])
            changes = [(i+1, rid) for i, rid in enumerate(hops) if rid and rid != associations[v.vehicle_id]]
            sequences[v.vehicle_id] = hops
            next_rsu[v.vehicle_id] = changes[0][1] if changes else None
            dwell[v.vehicle_id] = float(changes[0][0]) if changes else 0.
            final[v.vehicle_id] = hops[-1]
        self.pending[step_index+self.horizon] = final
        counts = Counter(final.values())
        result.update(next_rsu_sequence=sequences, predicted_next_rsu_by_vehicle=next_rsu,
            predicted_first_handoff_rsu_by_vehicle=next_rsu.copy(),
            predicted_handoff_target_rsu_id_by_vehicle=next_rsu.copy(),
            predicted_handoff_vehicle_ids=[v for v, rid in next_rsu.items() if rid], dwell_time=dwell,
            future_load={r.rsu_id: float(counts[r.rsu_id]) for r in rsus},
            prediction_source='causal_residual_gru_v1',
            predicted_positions_by_vehicle={v.vehicle_id:[[v.position_x+float(d[0]),v.position_y+float(d[1])] for d in forecast] for v,forecast in zip(vehicles,forecasts)},
            position_uncertainty_margin=self.position_margin)
        self.last_result = deepcopy(result)
        return result
