from pathlib import Path
import json
root=Path('/private/tmp/ppo_mec_g14r20_b_62f432c')
commit='e6a73357d08220d34ddc2d6537d140131c60520f'
rel='artifacts/analysis/g14r20_b_continuation_20260908/'+commit
report=root/rel/'acceptance_summary.json'
r=json.loads(report.read_text());assert r['status']=='technical_acceptance_pass'
link='../../'+rel+'/acceptance_report.md'
status='2026-09-09 G14R20-B：独立 executor 实现及隔离合成验收完成；真实 v16 未恢复，独立批准未签发，原始启动/发布证据仍 unavailable，v16-B 尚未获执行许可。'
for name in ('README.md','docs/project/README.md','docs/project/RUNBOOK.md','docs/project/DIRECTORY_STRUCTURE.md','docs/project/CODE_MODULE_MAP.md','docs/project/PROGRESS.md'):
 p=root/name;s=p.read_text();first,rest=s.split('\n',1)
 assert 'G14R20-B' in first
 if name=='README.md': first='> '+status+' 固定实现 `'+commit+'`；[验收报告]('+rel+'/acceptance_report.md)，[合同](docs/project/continuation_executor_contract.md)。'
 elif name.endswith('PROGRESS.md'):first='> '+status+' 固定实现 `'+commit+'`；全仓 1431、定向 71 项通过，均 0 skipped。八阶段合成 child 25 次，科学 rollout、真实 dispatch/write 均 0；真实只读核验 150 checkpoints / 44 active / 6 generated。证据见 [验收报告]('+link+')。下方 G14R20-A 的“执行器未实现”为当时历史状态。'
 else:first=first.replace('（实现验证中）','（2026-09-09 技术验收通过）').replace('实现合同（验证中）','实现合同（2026-09-09 技术验收通过）').replace('目前不得将开发验证称为最终验收','精确实现提交为 `'+commit+'`，真实执行批准仍不可用')+' [验收报告]('+link+')。'
 p.write_text(first+'\n'+rest)
p=root/'docs/project/BUGS.md';s=p.read_text();p.write_text('> '+status+' 技术通过不替代原始授权凭据和生产信任配置；当前 production signer map 为空。若后续配置改变源代码身份，需重新验收精确实现提交。合成结果不证明算法性能。证据见 [验收报告]('+link+')；下方 A 状态为历史记录。\n\n'+s)
p=root/'docs/project/DECISION_LOG.md';s=p.read_text();s+='\n## 2026-09-09 G14R20-B 固定实现验收闭合\n\n实现固定为 `'+commit+'`，证据独立提交。最终验收使用落盘后重新读取的 synthetic proposal/contract/approval/command plan，并核对原始字节 hash；此前 `0d25796a56e1cc9d48fe74093877e429e2da432e` 的合成 proposal 换行字节问题已修复，其证据不作为最终交付。科学来源仍为原 a6d1fd8，未改写原 run、context/binding 或 A proposal；当前生产 signer map 为空，不签发真实批准。完整三层及恢复证据见 [验收报告]('+link+')。\n';p.write_text(s)
p=root/'docs/project/continuation_executor_contract.md';s=p.read_text();s=s.replace('Status: implementation under validation; no production signer installed. Acceptance\ncompletion and exact commit identities must be supplied by the separate evidence\nrecord.','Status: isolated technical acceptance completed on 2026-09-09 for implementation\n`'+commit+'`; no production signer installed. Exact\nidentities and validation results are in the separate [evidence report]('+link+').')
s=s.replace('This\nimplementation draft is not the final acceptance report.','The separate evidence report records the completed technical acceptance; this\ncontract does not establish independent production authorization.')
p.write_text(s)
print('final evidence documentation updated')
