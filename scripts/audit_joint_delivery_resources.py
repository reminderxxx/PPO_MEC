"""Independent raw-event checks, without trusting a producer's safety boolean."""
def audit_optional_cache_evictions(event, peers, workflows):
    c=event['control'];rid=c['cache_target']
    if rid is None:return
    optional=rid!=c['execution_target'] or c['cache_node_id']!=event['node_id'] or event['output_source_rsu_id'] is not None
    if not optional:return
    required=set()
    for other in peers:
        if other['control']['execution_target']!=rid or other['output_source_rsu_id'] is not None:continue
        workflow=workflows[other['vehicle_id']];node=next(n for n in workflow.nodes if n.node_id==other['node_id'])
        required.update((f'base:{node.required_base_model}',f'adapter:{node.required_adapter}'))
    assert not required.intersection(event['cache_request']['evicted_object_ids']),'optional prefetch evicted a current team model dependency'
