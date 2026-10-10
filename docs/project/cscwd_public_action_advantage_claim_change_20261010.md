# A 论文线主张与证据变更：公共动作优势候选（2026-10-10）

本文件供 A 论文线只读消费；不修改主论文稿。

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
- 下一证据门：先冻结结果盲、原始区间互斥的较长 development window，只验 prepare/serve 可达性；通过后才允许单变量匹配
  pilot。若仍无双标签/成功 prepare，候选被否定而不是继续延长预算。

证据入口：`cscwd_public_action_advantage_candidate_20261010.md`；A 原件 run ID=
`cscwd_raw_ngsim_event_time_20261010_v4`，最终运行 commit=`0704937742ec9218094f380a1b541c4bd2159bcb`，B default-off commit=
`5f88785ae4c2951e3826a468d3c17e389365f39c`。
