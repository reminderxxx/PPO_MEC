# G14R20-I5-A 验收基线勘误

原 I5 `qualification_stop.json` 保留且不回填。其两个 task-start 常量不能作为文件证据：硬编码给
`train_sa_ghmappo_real_sample.py` 与 `sa_ghmappo_agent.py` 的两个 SHA-256 实际分别来自旧 E02 held lock 与
I3 held lock。原 I5 会话 ordinal 294 在 `2026-09-14T11:43:34.895+08:00` 对七文件执行了逐文件
`stat + shasum`，ordinal 672 与 1319 又取得相同内容哈希。精确路径、时间、哈希和限制见
`g14r20_i5_a_historical_qualification_correction.json`。

未找到任务开始时刻的可靠七文件快照，因此 `historical_start_evidence=unavailable`。11:43 的观测晚于任务开始约
8 分钟，只能与 12:06、12:38 观测共同证明该日志区间内内容相同；不得据此追认 I5 历史保护通过，影响交中央窗口判断。

原“缺失 I4 包”检查解析的是 main/I5 worktree 下的相对 `artifacts/analysis/...`。I4 原件实际位于独立 I4
worktree 的绝对路径。工程 review 在原 I5 stop 前存在；当前 independent rerun 与更新后的 manifest 在 stop 后生成，
只能证明当前三文件完整，不能倒推当时三文件字节已经齐备。

验收器现只消费显式 start snapshot，并在构包前采集 end snapshot；历史基线、当前重验起点和结束观察值分开保存。
JUnit 由 commit-bound receipt 绑定；failure、error、空集、错误 commit/tree、报告 hash/summary 不匹配、缺件以及
未逐项解释且无同范围替代证据的 skip 均 fail-closed。旧 62 failures、84 errors、1 skip 必须逐 nodeid 映射到本轮
通过的重验结果。

本勘误与验收不修改 restricted recovery 执行机制或科学参数，不启动真实恢复，不签 grant，不清锁，不运行正式
rollout；状态上限仍为 `READY_FOR_RESTRICTED_RECOVERY_AUTHORIZATION`。
