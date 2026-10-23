# 持续请求长训的冷启动兼容性

更新日期：2026-10-23。

旧四方法长训在首个 learner 的 Adam 构造阶段失败，尚无训练会话或更新。
`Runtime` 曾将多个 PyTorch API 指向同一个禁止函数；Adam 首次延迟导入
Dynamo 时，不同 API 的追踪规则在同一函数对象上冲突。修复仅为各 API
生成独立守护函数，保留原 CPU、反向传播、优化器和采样边界；不修改算法、奖励或数据。

`scripts/qualify_stream_startup.py` 在独立冷进程中核验四方法和三个固定 seed：
生产 `Runtime → make_agent` 顺序、空优化器、504 预算、初态参数和 RNG、
有界 checkpoint 保存及 `from_checkpoint` 加载、守护有效性和退出恢复。
与无守护的同 seed 初始化独立对照。资格阶段不运行环境、forward、backward 或优化器 step。
首次 Adam 构造会增加 PyTorch 自身的性能统计包装；验收仅放行该固定包装代码，
并确认其直接闭包和 `__wrapped__` 仍指向生产守护，其余 API 与退出恢复继续逐对象检查。
它验证启动与存取接口，不能代替训练后 checkpoint、长期稳定性或收敛证据。

```text
python scripts/qualify_stream_startup.py run --output-root <fresh_private_qualification>
```

新授权绑定完整资格回执的 SHA；条件衔接器复核精确四方法 × 三 seed 网格、
源码和 Python 绑定、零训练计数及每格验收后才允许启动。资格不齐或失败时拒绝启动。
旧无资格字段的历史授权保持原解析规则，不因此追溯获得新资格。

用户已授权修复验收后重新执行一次四方法 × 三 seed × 504 会话。
使用新的源码提交、manifest、授权、handoff、run 和绘图目录，保留旧失败原件，
不能覆盖或恢复旧 run。精确运行指针与结果存于本地 Git 忽略的
`artifacts/analysis/stream_startup_repair_20261023_v1/`；实际是否通过及启动以回执为准。
每20分钟只检查当前任务，状态不变不通知；完成后独立核验并交付真实曲线。

训练与评估日程、固定最终模型、full48/probe12 和尾稳规则沿用
[长训说明](stream_long_training.md)。后续失败仍保留现场并停止，不自动追加预算或重试。
