"""Train-only mobility cohorts; common wide/tight coverage with no score selection."""
from collections import Counter
from copy import deepcopy
from math import hypot


def coverage_counts(frames, rsus):
    previous = {}; exits = uncovered = 0
    for frame in frames:
        for v in frame['vehicles']:
            covered = {r.rsu_id for r in rsus if hypot(v.position_x-r.position_x,v.position_y-r.position_y)<=r.coverage_radius}
            exits += len(previous.get(v.vehicle_id,covered)-covered)
            uncovered += int(not covered)
            previous[v.vehicle_id] = covered
    return dict(coverage_departures=exits, uncovered_vehicle_frames=uncovered)


def tight_sites(rsus):
    result = deepcopy(rsus)
    for row in result:
        row.coverage_radius *= .65/1.2
    return result


def mobile_cohort(frames, rsus, count=8):
    """Use only a training interval's external geometry, never a policy reward."""
    common = None
    for frame in frames:
        counts = Counter(v.vehicle_id for v in frame['vehicles'])
        valid = {v for v,n in counts.items() if n==1}
        common = valid if common is None else common & valid
    ranked = []
    for vid in sorted(common):
        track = [dict(frame,vehicles=[next(v for v in frame['vehicles'] if v.vehicle_id==vid)]) for frame in frames]
        c = coverage_counts(track,rsus)
        if c['uncovered_vehicle_frames']:
            continue
        first,last = track[0]['vehicles'][0],track[-1]['vehicles'][0]
        distance = hypot(last.position_x-first.position_x,last.position_y-first.position_y)
        ranked.append((-c['coverage_departures'],-distance,vid))
    if len(ranked)<count:
        raise ValueError('not enough continuously covered train vehicles for a mobile cohort')
    selected = sorted(r[2] for r in sorted(ranked)[:count])
    return [{**f,'vehicles':[v for v in f['vehicles'] if v.vehicle_id in selected]} for f in frames],selected


def build_scenarios(splits, rsus, raw_frames):
    """Raw extra vehicles are accessed solely at original training timestamps."""
    timeline={(f['source_segment_id'],f['time_index']):f for f in raw_frames}
    train=[];dev=[];tight=tight_sites(rsus)
    for w in splits['train']:
        frames=[timeline[(f['source_segment_id'],f['time_index'])] for f in w['frames']]
        selected,ids=mobile_cohort(frames,tight)
        for cohort,window,sites in [('base',w,rsus),('mobile_tight',dict(w,frames=selected,vehicle_ids=ids),tight)]:
            train.append(dict(window,scenario_id=f"train_{w['frame_offset']}_{cohort}",panel=cohort,rsus=sites,
                              **coverage_counts(window['frames'],sites)))
    for w in splits['dev']:
        for panel,sites in [('base',rsus),('tight',tight)]:
            dev.append(dict(w,scenario_id=f"dev_{w['frame_offset']}_{panel}",panel=panel,rsus=sites,
                            **coverage_counts(w['frames'],sites)))
    return dict(train=train,dev=dev)


def public_scenarios(scenarios):
    return {split:[{k:([r.to_dict() for r in v] if k=='rsus' else v)
                    for k,v in w.items() if k!='frames'} for w in windows]
            for split,windows in scenarios.items()}
