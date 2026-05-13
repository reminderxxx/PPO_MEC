# Status Tags

在文档、计划和交接说明里使用以下标签，避免把临时状态写成稳定结论。

- `[active]`：当前推荐使用的主线或入口。
- `[canonical]`：长期事实来源，其他文档应引用它。
- `[experimental]`：可运行或可讨论，但不能作为正式结论。
- `[smoke]`：只做链路联调，不能代表算法有效性。
- `[paper-grade]`：满足正式论文实验口径的结果或流程。
- `[blocked]`：有明确阻塞原因，继续前需要先处理。
- `[deferred]`：保留但当前不优先推进。
- `[deprecated]`：不再推荐使用，仅为兼容或历史追溯保留。

当前建议：

- `NGSIM + Alibaba`：`[active]`
- `benchmark_main_results.py` 多窗口多 seed 主表：`[paper-grade]`
- `smoke_run`：`[smoke]`
- `LuST` 主线联调：`[deferred]`
- 旧通用模板目录：`[deprecated]`，内容已整理进 `docs/project/`



