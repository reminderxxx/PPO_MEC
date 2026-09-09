import sys,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
root=Path('/private/tmp/ppo_mec_g14r20_b_62f432c');sys.path.insert(0,str(root/'scripts'))
from continuation_executor.identity import read_json,canonical,digest,file_hash,git
base=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_b_continuation_20260908');b=base/'e6a73357d08220d34ddc2d6537d140131c60520f';fixture=b/'synthetic_acceptance_01'
s=read_json(fixture/'acceptance_report.json');r=read_json(b/'readonly_compatibility.json');protection=read_json(b/'protected_final.json');start=read_json(base/'startup_identity.json')
proposal_path=Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json');proposal=read_json(proposal_path)
assert digest(proposal)==start['proposal_canonical_sha256'] and file_hash(proposal_path)==start['proposal_file_sha256']
assert s['serialized_authorization_verified'] is True
assert file_hash(fixture/'proposal_test_only.json')==s['synthetic_proposal_file_sha256']==read_json(fixture/'contract_test_only.json')['proposal_file_sha256']
assert s['status']=='pass' and all(x['status']=='completed' for x in s['phase_results'])
assert s['gate']['passed'] and not s['gate']['holdout_opened'] and not s['gate']['paper_claims_permitted']
assert r['checkpoint_audit']['pass_count']==150 and r['active_resource_count']==44 and r['registry_audit']['resource_count']==6
results={}
for name in ('full_pytest','targeted_pytest','synthetic','readonly','smoke'):
 d=read_json(b/(name+'_command_result.json'));assert d['exit_code']==0 and not d['executor_status_before'] and not d['executor_status_after']
 assert d['executor_commit_before']==d['executor_commit_after']==b.name
 results[name]=d
suites={};cases=[]
for name in ('full_pytest','targeted_pytest'):
 suite=ET.parse(b/(name+'.xml')).getroot().find('testsuite');assert all(int(suite.get(k,'0'))==0 for k in ('failures','errors','skipped'))
 suites[name]={k:int(suite.get(k,'0')) for k in ('tests','failures','errors','skipped')}
 if name=='targeted_pytest':
  cases=[dict(name=c.get('name'),class_name=c.get('classname'),seconds=float(c.get('time','0')),status='passed') for c in suite.findall('testcase')]
assert suites['full_pytest']['tests']==1431 and suites['targeted_pytest']['tests']==71
reg=read_json(fixture/'synthetic_continuation/generated_checkpoint_resource_registry.json')
phase_rows=[json.loads(line) for line in (fixture/'synthetic_continuation/phase_state.jsonl').read_text().splitlines()]
freeze=next(x for x in phase_rows if x['phase']=='checkpoint_freeze' and x['status']=='completed')
anchor=reg['source_phase_committed_ledger_identity']['terminal_record_sha256']
assert anchor==freeze['current_record_hash'] and anchor!=phase_rows[-1]['current_record_hash']
assert s['prefix_unchanged'] and len(s['monitor']['child_monitors'])==25
assert s['monitor']['synthetic_child_dispatch_count']==25
for key in ('scientific_rollout_count','real_v16_dispatch_count','real_v16_write_count'):assert s['monitor'][key]==0
assert digest(read_json(fixture/'contract_test_only.json'))==s['approval_contract_sha256']
report=dict(version='1.0.0',reviewed_at=datetime.now(timezone.utc).isoformat(),status='technical_acceptance_pass',
 executor_implementation_commit=b.name,initial_implementation_commit='54cf2e7db6b924680f896ac3efbe530eb721e373',development_baseline=start['development_commit'],scientific_execution_commit=proposal['execution_commit'],
 evidence_commit='The separate Git commit containing this report; full hash is recorded after commit without self-reference.',
 states=dict(implementation=True,synthetic_compatibility=True,read_only_technical_compatibility=True,independent_real_qualification='unavailable',launch_approval='unavailable',release_attestation='unavailable',continuation_approval='pending',production_signer_count=0,real_execution_authorized=False,real_v16_resumed=False,holdout_opened=False,g14d_g15_started=False),
 suites=suites,phase_results=s['phase_results'],monitors={k:s['monitor'][k] for k in ('synthetic_child_dispatch_count','scientific_rollout_count','real_v16_dispatch_count','real_v16_write_count')},
 cli_trust_refusals=read_json(b/'cli_trust_refusal_report.json'),native_staging_refusal=read_json(fixture.parent/'synthetic_staging_negative/report.json'),consumer_case_count=len(s['consumer_cases']),synthetic_checkpoint_gate_count=s['checkpoint_audit']['pass_count'],producer_main_calls=2,preexisting_phase_child_dispatch_count=0,
 real_readonly=dict(checkpoints=150,active_resources=44,generated_resources=6,reconciliation=r['reconciliation']),
 protection=protection,proposal_unchanged=dict(canonical_sha256=digest(proposal),file_sha256=file_hash(proposal_path)),
 registry_anchor=dict(freeze_terminal=anchor,final_tip=phase_rows[-1]['current_record_hash'],registry_still_uses_freeze=True),
 warnings=['16 existing pytest record_property/xunit2 warnings; zero skipped tests.','Synthetic payloads and the controlled consumer boundary establish execution semantics only, not performance or paper claims.','The required repository smoke is a separate toy-environment check; continuation monitor counters describe the isolated B acceptance run.'],
 required_independent_materials=['Original launch approval and release attestation, independently authenticated; presently unavailable.','Exact executor identity, implementation commit, source manifest and the separate evidence commit.','Unchanged A proposal, protection inventory, original a6d1fd8 source/environment/context/binding and checkpoint/registry evidence.','Explicit continuation approval for the eight-phase contract and write/coordination scope, with expiry/revocation and any required quiescence evidence.','Independent production trust provisioning and revalidation of its exact implementation identity; current empty signer map cannot be unlocked by supplying approval JSON or a test key.'])
(b/'acceptance_summary.json').write_bytes(canonical(report)+b'\n');(b/'fault_and_boundary_cases.json').write_bytes(canonical(dict(implementation_commit=b.name,cases=cases,evidence='targeted_pytest.xml; exact test source in implementation commit'))+b'\n')
shutil.copyfile(fixture/'acceptance_report.json',b/'synthetic_acceptance_report.json')
shutil.copyfile(b/'synthetic_staging_negative/report.json',b/'staging_refusal_report.json')
helpers=b/'validation_helpers';helpers.mkdir(exist_ok=True)
for name in ('run_g14r20_recorded.py','run_g14r20_b_readonly_cli_final_e6a7335.py','prepare_g14r20_b_readonly_final_e6a7335.py','g14r20_compile_import_e6a7335.py','g14r20_final_protection_e6a7335.py','g14r20_compatibility_details_e6a7335.py','g14r20_build_report_e6a7335.py','g14r20_cli_trust_refusal_e6a7335.py','g14r20_staging_refusal_e6a7335.py'):
 shutil.copyfile(Path('/private/tmp')/name,helpers/name)
phases='\n'.join('| '+p['phase']+' | '+str(len(r['plans'][p['phase']]['commands']))+' | completed |' for p in s['phase_results'])
text=f'''# G14R20-B 技术验收报告

验收日期：{report['reviewed_at']}。实现提交：`{b.name}`。原科学提交：`{proposal['execution_commit']}`。
证据记录使用后续独立 Git 提交，不把该提交的自身 hash 写回文件；完整 hash 由交付记录给出。

独立 executor 的实现与隔离合成技术验收通过。真实 v16 未恢复，独立批准未签发，原始启动/发布证据仍 unavailable；v16-B 尚未获执行许可。

## 三层证据

1. 实际新 CLI：八个 phase 的缺批准拒绝、越权 phase、固定身份漂移、生产入口拒绝 synthetic trust；71 项定向测试均通过。完整用例见 `fault_and_boundary_cases.json` 与 `targeted_pytest.xml`。
2. 原生产/消费入口：原 artifact main 实际调用两次，生成 synthetic selection/freeze/companion；原 benchmark main 的正例及 13 个负例均经过真实 loader/gate，在 rollout 前停止。另有 150 个合成 checkpoint gate 全覆盖。每例实际调用和模块来源见 `synthetic_acceptance_report.json`。
3. 原事务完整链：五类 cell 共 22 个，后接原 statistics、integrity/formal gate 和 completion。实际 25 次 child 调度对应 25 份子进程监测；scientific rollout / real v16 dispatch / real v16 write 都为 0。

| phase | 原冻结命令数 | 合成结果 |
|---|---:|---|
{phases}

`compatibility_details.json` 保存原 15-phase / 186-command 完整展开、八阶段 argv/order/coordinates/output/hash、cwd/env 与原 context 的一致性。
定向测试直接求值原固定源码中的 identity 和 staging 表达式，逐项比较；没有调用原 public main 或以恒成功 validator 替换检查。
合成前五类 cell 使用明确的 test payload adapter；其命令替换与原科学命令分别记录，不把这些替换称为科学 argv 等价。

合成授权在落盘后重新读取并核验 proposal 原始字节 SHA-256、command plan digest 和批准签名；四个序列化正负例通过。

## 恢复与隔离

实际覆盖 exit 75 同一原命令的一次重试、非 75 终止、缺失/损坏 payload、descriptor/provenance 错配、publication 后 append 前进程崩溃、candidate finalize-only、重复 committed ID、并发单写者、锁 owner 崩溃与精确批准恢复、重复启动幂等、阶段中撤销/过期、UTC 回拨、乱序/分叉/截断/跨账本、immutable payload 损坏及 false/missing gate。
实际生产 CLI 的两个 test-trust 拒绝例见 `cli_trust_refusal_report.json`，均在写入/dispatch 前拒绝；独立原生 staging-root 错配记录见 `staging_refusal_report.json`。
原账本前缀保持不变；registry freeze anchor `{anchor}` 与最终 tip 不同，仍经原 consumer 验证。
来源验收覆盖旧 src/scripts __file__/hash、实际 Python/cwd/sys.path、5 个关键依赖来源、关闭 user-site、无 editable install，以及当前 main、外部 src、错误解释器/依赖污染拒绝。

## 真实只读与保护

真实只读复核：150 checkpoints、44 active resources、6 generated resources、174 committed payloads、167 immutable evidence identities；15 phase / 348 cell records。
原保护清单校验：{protection['verified_entries']:,} 条、{protection['verified_bytes']:,} bytes；检查旧 run/worktree 新增文件，七个用户文件完整 SHA-256 起止一致。
原 proposal 与原 inventory 不改写。没有真实锁、running record、staging 或 child；没有 G14D/G15。

## 验证与使用边界

- `python -m pytest tests -q`：1431 passed，0 failed/error/skip。
- `python -m pytest tests/test_continuation_executor_boundaries.py -q`：71 passed，0 failed/error/skip。
- `python scripts/smoke_test.py` 与 compile/import：通过。Smoke 是单独的仓库 toy 检查，不合并进 continuation monitor 的计数。
- 原测试的 V2.4 临时 context builder 不再要求测试分支等于 origin/main；生产发布 gate 保持不变。
- 保留 16 条既有 pytest record_property/xunit2 warning；不把 warning 或未执行项写成通过。
- 命令、stdout/stderr、JUnit、运行前后 clean Git 身份与实际耗时均保留。格式/hash 检查见 `format_validation.json` 与 `artifact_integrity.json`。

当前生产 signer map 为空。独立原始启动/发布凭据、可信批准及其信任根仍需另行核验；仅提供 JSON 或测试密钥不会开放执行。
未来批准必须绑定经过验收的精确 executor 身份、未改写的 A proposal、原科学来源和固定八阶段合同。若安装生产信任导致代码身份变化，必须对新精确身份重新验收。
本报告不作论文效果、统计优势或 paper-ready 判断。完整独立材料清单见 `acceptance_summary.json`。
完整合成夹具（含 test-only checkpoint/输入表）保留在本机验收目录；Git 只保存报告、命令、JUnit、验证脚本和 hash 清单。`versioned_evidence_manifest.json` 定义随提交交付的文件子集，`artifact_integrity.json` 覆盖本机完整验收夹具；不提交真实数据、checkpoint 或旧 run 内容。
'''
(b/'acceptance_report.md').write_text(text)
print('acceptance report prepared')
