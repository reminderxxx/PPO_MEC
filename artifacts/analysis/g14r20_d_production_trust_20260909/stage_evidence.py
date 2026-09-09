import pathlib,json,shutil,hashlib
O=pathlib.Path('/private/tmp/g14r20_d_evidence');D=pathlib.Path('/private/tmp/ppo_mec_g14r20_d_delivery/artifacts/analysis/g14r20_d_production_trust_20260909')
s=json.loads((O/'acceptance_summary.json').read_text());assert s['isolated_acceptance']=='pass' and s['real_execution_authorized'] is False
D.mkdir(parents=True,exist_ok=True)
names=['scope_and_diff_check.json','acceptance_report.md','acceptance_summary.json','executor_identity.json','executor_verification.json',
 'unsigned_execution_contract.json','frozen_command_plans.json','unsigned_handoff.json','fixed_input_provenance.json',
 'protected_before.json','protected_after.json','full_pytest.xml','targeted_cases_from_full_pytest.json',
 'synthetic_acceptance_report.json','synthetic_file_inventory.json','compile_import_report.json',
 'authorization_dependency_identity.json','format_validation.json','requirement_evidence_matrix.md']
for name in ('full_pytest','smoke','compile_import','readonly','synthetic_final'):
 names += [name+suffix for suffix in ('_command.json','_result.json','_stdout.log','_stderr.log')]
for n in names:
 p=O/n;assert p.is_file() and p.stat().st_size<5*1024*1024,n;shutil.copyfile(p,D/n)
for p in (O/'serialized_test_only').iterdir():
 assert p.suffix=='.json';(D/'serialized_test_only').mkdir(exist_ok=True);shutil.copyfile(p,D/'serialized_test_only'/p.name)
for n in ('verify_delivery.py','protect.py','run_recorded.py','freeze_identity.py','read_only_compatibility.py','compile_import.py','handoff.py','build_report.py','format_validation.py','stage_evidence.py'):
 shutil.copyfile(O/n,D/n)
(D/'development_notes.md').write_text('''# 开发与验收边界记录

开发期间针对已发现的最小合同问题依次补充了累计撤销集合、重启 nonce 回执以及非空撤销 ID 校验。早期候选的部分验收在代码更新后被停止，未计入最终通过证据；其临时日志仍保留在 /private/tmp/g14r20_d_evidence/prior_candidate_*。首次沙箱内原生进程测试因 ps 权限受限拒绝，之后在允许进程身份读取的环境执行；首次合成启动因 PYTHONPATH 不等于原科学根而在写入前拒绝，后续使用只向父进程追加授权依赖的明确 wrapper 保持科学 PYTHONPATH 不变。

最终报告只接受 command/result 中前后 commit 相同、工作区 clean 且 exit_code=0 的最终精确提交结果。当前提交尚无真实 trust、release 资格或 continuation 批准；测试域伪造的 production 标签用于负例，不能视为生产批准。

新增授权依赖仅安装于 /private/tmp/g14r20_d_dependencies，原科学 venv 未修改。requirements_continuation.txt 固定 cryptography 版本，完整依赖文件摘要另行提供。原 native statistics 的 bootstrap_samples=10000 及全部 gate 保持原配置，没有为缩短验收而降采样。

启动挑战接口是持有同一 TrustContext 的进程内交接：先生成 startup_request，独立 custody authority 根据保管的当前 checkpoint 对 nonce/PID/scope 签回执，再在同一进程继续验证/调用 executor。另起进程或 fork 必须重新取得回执；不得复用旧回执或从本地快照自举。当前生产 installation=None，不执行这一真实安装流程。
''')
print(D)
