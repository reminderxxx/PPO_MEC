"""Causal, public-information planning proxies shared by every stream controller.

These are feasibility/planning estimates, not counterfactual reward labels.
"""
from math import ceil, hypot, inf
from src.envs.core.joint_delivery_planning import node_delivery_utility


def balanced_training_schedule(cases, episodes):
    """Interleave size; cover every condition in both halves of a 64-session run.

    All seeds use the same curriculum, so seed changes initialization only.
    The three original train time sessions rotate on successive 24-case cycles.
    """
    if len(cases) != 72:
        raise ValueError('balanced stream schedule requires 3 sessions x 24 conditions')
    order = [(i % 3)*8 + ((i//3) % 2)*4 + ((i//6) % 2)*2 + i//12
             for i in range(24)]
    return [cases[((ep//24 + ep%3) % 3)*24 + order[ep%24]]['case_id']
            for ep in range(episodes)]


def state_restore_frames(state, target):
    rates = state['control_cost_parameters']
    if not state['execution_state_version'] or target == state['execution_location']:
        return 0
    copy = state['state_copies_by_rsu'].get(target, {})
    if copy.get('version') == state['execution_state_version']:
        return 0
    transfer = state['state_transfers_by_rsu'].get(target)
    if transfer:
        remaining = max(transfer['remaining_frames'],
                        ceil(transfer['remaining_bytes']/rates['state_bytes_per_frame']))
        if transfer['version'] != state['execution_state_version']:
            remaining += max(rates['minimum_state_frames'], ceil(max(
                1, state['execution_state_payload_bytes']-transfer['payload_bytes']) /
                rates['state_bytes_per_frame']))
        return remaining
    return max(rates['minimum_state_frames'], ceil(max(
        1, state['execution_state_payload_bytes']-copy.get('payload_bytes', 0)) /
        rates['state_bytes_per_frame']))


def visible_delivery_timeline(state, target):
    """Only active and already released FIFO workflows, including paid progress."""
    rates = state['control_cost_parameters']; elapsed = 0.; uses = {}; times = []; indices = {}
    workflows = [(state['request_id'], state['workflow']), *[
        (q['request_id'], q['workflow']) for q in state['queued_workflows']]]
    for request, workflow in workflows:
        nodes = {n['node_id']: n for n in workflow['nodes']}
        for nid in workflow['execution_order']:
            if nid in workflow['completed_node_ids']:
                continue
            n = nodes[nid]; current = request == state['request_id'] and nid == state['current_workflow_node']['node_id']
            uses[nid] = inf if current and state['output_pending'] else elapsed
            indices[nid] = len(times)
            if current and state['output_pending']:
                elapsed += ceil(state['output_pending']['remaining_bytes']/rates['output_bytes_per_frame'])
            else:
                if not (current and target in state['input_received_targets']):
                    amount = state.get('input_remaining_bytes_by_rsu', {}).get(target, n['input_size']*rates['profile_unit_bytes']) if current else n['input_size']*rates['profile_unit_bytes']
                    elapsed += ceil(max(1, amount)/rates['input_bytes_per_frame'])
                work = max(1., n['input_size']*rates['profile_unit_bytes']/rates['workload_bytes_per_unit'])
                if current:
                    work = max(0., state['current_node_work_units_required']-state['current_node_work_units_progress'])
                elapsed += ceil(work/rates['rsu_work_units_per_frame'])
                elapsed += ceil(max(1, n['output_size']*rates['profile_unit_bytes'])/rates['output_bytes_per_frame'])
            times.append(elapsed)
    return uses, times, indices


def optional_cache_value(state, row, node, span, selected_target, working_bytes,
                         *, prediction_enabled, horizon, use_after_override=None):
    """Bounded-use, route survival and waste-cost proxy; never future trace truth."""
    site = row['rsu_id']; uses, timeline, indices = visible_delivery_timeline(state, site)
    use = uses.get(node['node_id'], inf)
    if use_after_override is not None:
        use = float(use_after_override)
    result = dict(use_after_frames=None if use == inf else use, survival_weight=0.,
                  net_delivery_proxy=0., admission_span=span, eligible=False)
    if not span or use < span or use > horizon or use >= state['episode_frames_remaining']:
        return result
    pred = state['predictions'] if prediction_enabled else {}
    vid = state['primary_vehicle_id']
    points = pred.get('predicted_positions_by_vehicle', {}).get(vid, [])
    if points:
        if use > len(points):
            return result
        weight = float(pred.get('prediction_confidence_by_vehicle', {}).get(vid, 0.))
        relevant=points[:max(1, ceil(use))] if site==selected_target else [points[max(0,ceil(use)-1)]]
        if any(hypot(x-row['position_x'], y-row['position_y']) > row['coverage_radius']
               for x,y in relevant):
            weight = 0.
    else:
        # A no-prediction controller can still prepare its committed station.
        weight = float(site == selected_target) if not prediction_enabled else .5*float(site == selected_target)
    weight /= max(1., working_bytes/max(1, row.get('capacity_bytes', 1)))
    gain = node_delivery_utility(timeline, 0, state['episode_frames_remaining']) - node_delivery_utility(
        timeline, span, state['episode_frames_remaining'], indices[node['node_id']])
    net = weight*gain - .01*span*(1-weight)
    result.update(survival_weight=weight, net_delivery_proxy=net, eligible=net > 1e-9)
    return result
