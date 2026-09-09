import sys
from pathlib import Path
sys.path.insert(0,'/private/tmp/ppo_mec_g14r20_b_62f432c/scripts')
from continuation_executor.identity import read_json,canonical,digest,git
from continuation_executor.scientific import load_native,loaded_origins
b=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_b_continuation_20260908/e6a73357d08220d34ddc2d6537d140131c60520f')
r=read_json(b/'readonly_compatibility.json');p=read_json('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json')
n=load_native(p['worktree_root'],p['execution_commit']);ctx=read_json(Path(p['run_root'])/'resolved_execution_context.json');protocol=read_json(ctx['runtime_location']['protocol_path'])
m=n['execution'].validate_command_templates(protocol['execution_contract']['command_templates'],ctx['resolved_expansion_context'])
assert m['command_matrix_sha256']==ctx['command_expansion']['resolved_command_matrix_sha256']
report=dict(status='pass',execution_authorized=False,executor_commit='e6a73357d08220d34ddc2d6537d140131c60520f',scientific_commit=p['execution_commit'],observed_main=git(Path.cwd(),'rev-parse','main'),observed_origin_main=git(Path.cwd(),'rev-parse','origin/main'),
 full_original_matrix=m,persisted_expansion_identity=ctx['command_expansion'],phases=r['plans'],cwd=p['worktree_root'],environment=r['environment'],
 isolation_probe_flags=['-I','-B'],frozen_child_argv_modified=False,source_origins=loaded_origins(Path.cwd()),
 equivalence_evidence='test_planner_projections_equal_frozen_original_main_expressions in full_pytest.xml; evaluates frozen original identity and staging expressions without invoking public main')
(b/'compatibility_details.json').write_bytes(canonical(report)+b'\n');print('matrix verified',m['phase_count'],m['command_count'])
