# 开发与验收边界记录

开发期间针对已发现的最小合同问题依次补充了累计撤销集合、重启 nonce 回执以及非空撤销 ID 校验。早期候选的部分验收在代码更新后被停止，未计入最终通过证据；其临时日志仍保留在 /private/tmp/g14r20_d_evidence/prior_candidate_*。首次沙箱内原生进程测试因 ps 权限受限拒绝，之后在允许进程身份读取的环境执行；首次合成启动因 PYTHONPATH 不等于原科学根而在写入前拒绝，后续使用只向父进程追加授权依赖的明确 wrapper 保持科学 PYTHONPATH 不变。

最终报告只接受 command/result 中前后 commit 相同、工作区 clean 且 exit_code=0 的最终精确提交结果。当前提交尚无真实 trust、release 资格或 continuation 批准；测试域伪造的 production 标签用于负例，不能视为生产批准。

新增授权依赖仅安装于 /private/tmp/g14r20_d_dependencies，原科学 venv 未修改。requirements_continuation.txt 固定 cryptography 版本，完整依赖文件摘要另行提供。原 native statistics 的 bootstrap_samples=10000 及全部 gate 保持原配置，没有为缩短验收而降采样。

启动挑战接口是持有同一 TrustContext 的进程内交接：先生成 startup_request，独立 custody authority 根据保管的当前 checkpoint 对 nonce/PID/scope 签回执，再在同一进程继续验证/调用 executor。另起进程或 fork 必须重新取得回执；不得复用旧回执或从本地快照自举。当前生产 installation=None，不执行这一真实安装流程。
