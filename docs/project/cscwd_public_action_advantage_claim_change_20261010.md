# A 论文线主张与证据变更：公共动作优势候选（2026-10-10）

本文件供 A 论文线只读消费；不修改主论文稿。

> 2026-10-11 追加澄清：2.2 s raw 窗口的零 prepare 只说明执行资格不足，候选状态为 `UNTESTED`，不构成算法否证；不得据此
> 推荐结果驱动扩窗。当前优先补公共 estimator 与 executor 的 fail-closed phase accounting，详见
> `cscwd_public_estimator_executor_consistency_20261011.md`。
> 只读阶段表还确认存在“prepare 可在 current contact 内完成、但整步原子准入回滚”的见证，故历史零 prepare 不能被写成
> 学习候选失败；raw exact decision clone 会读取未来物理接触，只能保持 privileged reference 能力标签。
> 2026-10-11 后续状态：public estimator `v2` 已完成阶段核算纠错和合成验收；既有合法 development 区间资格为 `0/40`，
> 因此训练仍未启动，论文线只能记录接口纠错与 `UNTESTED`，不得写算法收益。

- 新增可写入内部研究记录的事实：共同、因果、无副作用的 public action estimator 与 default-off
  `causal_public_prepare_advantage_v1` event auxiliary 已实现；未知状态 abstain，不做 action mask，不改变 slow/fast、reward、
  network、PPO 或 baseline 权限。
- 新增 source-grounded 边界证据：raw NGSIM 三个已暴露开发窗口均只有 2.2 s 决策余量；15/15 episode 截断、0 workflow
  完成。30 个决策中 10 次请求被 mask 改写，实际 action4 执行 2 次、0 成功 prepare。当前窗口不足以评估准备机制，匹配
  训练已在启动前停止。
- 不变主张：没有 SA 稳定领先、算法效果、正式/holdout、真实无线或 paper-ready 证据；原 exact two-step 仍须标为 privileged
  model-based reference，不能与 public rule 做同能力排名。
- 禁止新增主张：不得把 estimator 合同测试写成性能收益，不得把 default-off 实现写成创新已验证，不得用 24-frame raw 结果
  宣称真实 VEC 不可完成，也不得从当前零 prepare 事后扩窗或挑实例。
- 原“较长窗口双标签门”保留为历史设想，不是当前自动下一步。先修复并验收 estimator/executor phase contract；之后是否另立
  数据协议须独立预注册，不能由本批零 prepare 结果驱动。

证据入口：`cscwd_public_action_advantage_candidate_20261010.md`；A 原件 run ID=
`cscwd_raw_ngsim_event_time_20261010_v4`，最终运行 commit=`0704937742ec9218094f380a1b541c4bd2159bcb`，B default-off commit=
`5f88785ae4c2951e3826a468d3c17e389365f39c`。
