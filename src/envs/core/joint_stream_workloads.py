"""Score-blind workload strata and heterogeneous declared (unmeasured) catalog."""
from copy import deepcopy
from math import ceil
from src.envs.core.joint_model_cache import controlled_catalog_from_workflows, ModelCacheObject
from src.envs.core.multi_request_stream_env import WorkflowArrival

SIZE_BINS={'short':(5,9),'medium':(10,19),'long':(20,40)}
ARRIVAL_MODES=('single_batch','queued_batch','stream')


def request_plan(pool, vehicle_ids, size, mode, families=2, horizon=480):
    if size not in SIZE_BINS or mode not in ARRIVAL_MODES or families not in (1,2):
        raise ValueError('unknown workload stratum')
    lo,hi=SIZE_BINS[size];chosen=[w for w in pool if lo<=len(w.nodes)<=hi][:4]
    if not chosen:raise ValueError('no source DAG in declared size stratum')
    result=[]
    for wave in range(1 if mode=='single_batch' else 2):
        for i,vid in enumerate(sorted(vehicle_ids)):
            w=deepcopy(chosen[(i+wave)%len(chosen)])
            w.workflow_id=f'{w.workflow_id}_r{wave}_{i}'
            names={n.node_id:f'r{wave}_{i}_{n.node_id}' for n in w.nodes}
            w.edges=[(names[a],names[b]) for a,b in w.edges]
            w.execution_order=[names[n] for n in w.execution_order];w.current_node_id=w.execution_order[0]
            for n in w.nodes:
                n.node_id=names[n.node_id];n.predecessors=[names[k] for k in n.predecessors];n.successors=[names[k] for k in n.successors]
            for n in w.nodes:
                n.required_base_model=f'veh_base_{i%families}'
                n.required_adapter=f'f{i%families}_{n.required_adapter}'
            result.append(WorkflowArrival(0 if mode!='stream' else wave*(horizon//2),vid,w))
    return result


def catalog_from_pool(pool, families):
    result=[]
    adapters=sorted({n.required_adapter for w in pool for n in w.nodes})
    for family in range(families):
        base=f'veh_base_{family}';size=2048+family*1024
        result.append(ModelCacheObject(f'base:{base}','base_model',size,size,base_model_id=base))
        for adapter in adapters:
            aid=f'f{family}_{adapter}';amount=512*(family+1)
            result.append(ModelCacheObject(f'adapter:{aid}','adapter',amount,amount,(f'base:{base}',),adapter_id=aid))
    return result


def capacity_lower_bounds(arrivals, horizon, rsu_count, rates):
    compute=io=0;finish={};violations=[]
    for a in sorted(arrivals,key=lambda a:(a.frame,a.request_id)):
        c=sum(ceil(max(1,n.input_size*rates['workflow_profile_unit_bytes']/rates['workload_bytes_per_unit'])/rates['rsu_work_units_per_frame']) for n in a.workflow.nodes)
        data=sum(ceil(max(1,n.input_size*rates['workflow_profile_unit_bytes'])/rates.get('input_bytes_per_link_frame',1024))+ceil(max(1,n.output_size*rates['workflow_profile_unit_bytes'])/rates.get('output_bytes_per_link_frame',1024)) for n in a.workflow.nodes)
        compute+=c;io+=data;finish[a.vehicle_id]=max(a.frame,finish.get(a.vehicle_id,0))+c+data
        if finish[a.vehicle_id]>horizon:violations.append(a.request_id)
    return dict(optimistic_compute_utilization=compute/(rsu_count*horizon),optimistic_io_utilization=io/(rsu_count*horizon),
        fifo_serial_bound_violations=violations,all_delivery_necessary_bound_pass=(compute<=rsu_count*horizon and io<=rsu_count*horizon and not violations),
        interpretation='necessary only; ignores model/state/shared overhead and coverage, not sufficient feasibility')
