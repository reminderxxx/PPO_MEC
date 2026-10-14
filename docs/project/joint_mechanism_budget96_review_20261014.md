# 八车联合控制：多帧主机制、96 回合曲线与容量可辨识性审查（2026-10-14）

- `reviewed_at`: 2026-10-14（Asia/Shanghai）
- `literature_cutoff`: 2026-10-14（沿用同日 `joint_cold_mask_budget96_cache_baseline_review_20261014.md` 的近邻核验；本报告不作新 novelty 判断）
- `target_venue`: IEEE Transactions on Mobile Computing（内部证据门槛）
- `artifact_run_id`: `20261014T074430Z_development`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `6d66ec5ca8191587b48a846eb70ac1aa8046155d`（运行时 clean checkout）
- `evidence_level`: 开发集原件已独立核对；完整论文证据缺 formal/holdout/support，按政策为 `E0_UNAVAILABLE`（开发核对不等于政策 `E2`）
- `verdict`: `Unverifiable`（SA-GHMAPPO 相对 MAPPO 或缓存方向 PPO 的最高奖励、收敛、model-cache 友好性、paper-ready）

同日补充：`joint_reward_identity_audit_20261014.md` 已核验 9 个不同网络的实际加载/参数更新，并独立重放全部 72 个 greedy 开发执行。精确同分可复现；容量不足解释低完成率，同分还由本批实际动作效果退化与固定执行调度解释，不能仅凭容量上限归因。

## Evidence inventory 与协议

- 运行原件：`artifacts/eight_vehicle_joint_v2_mechanism_budget96_20261014/20261014T074430Z_development/`，包含 `manifest.json`、`train.jsonl`、`updates.jsonl`、最终与曲线 checkpoint、`dev_curve.jsonl`、`dev.jsonl`、`dev_sampled.jsonl`、`dev_events.jsonl`；启动命令在 manifest，上级 `training.log` 记录 `RUN_COMPLETE`。NGSIM / Alibaba 原始文件 SHA-256 也在 manifest。
- 预先固定 3 算法 × 3 seed × 96 episode，更新间隔 4，曲线 checkpoint 间隔 24；同一 20 train、8 dev 窗口及 120 帧，strict cold-cache mask。仅开发运行，formal/holdout 均未评估。多帧工作量采用 `workflow_profile_unit_bytes=1`、`workload_bytes_per_unit=4`；70 个节点工作量均为 2–5.75 单位，所有节点至少需要多帧。每站 1 个计算槽和 1 个链路槽，typed cache 容量 4096 B，合成模型工作集 4608 B。
- `independent_audit.json` 的 14 项检查均通过：864 train、216 update、9 最终 checkpoint、72 greedy 与 72 sampled dev、117,864 条事件及 17,280 份帧收据；最终 checkpoint 哈希、唯一键、帧数与奖励守恒成立，所有更新执行且 formal/holdout 文件不存在。`independent_curve_audit.json` 另核对 36 个曲线 checkpoint 的 SHA-256、288 条曲线行、每 agent/seed/窗口末端与最终开发评估严格相同。
- 训练窗口的机制机会先用 `plan.json` 在 128/256/512 合成输入比例中按多帧与未完成换站条件选 256，再因其同时放大状态字节，用 `prepare_probe_decoupled_plan.json` 固定等工作量、较小状态代理的对照。该两阶段设计用的是**训练窗口**，没有借 dev 奖励挑 checkpoint，但完整物理成本仍是探索性合成假设；不得追认为预注册正式协议。

## 开发结果和学习曲线

每个算法在相同 3 seed × 8 窗口上确定性评估。窗口跨 seed 重复，24 行不是 24 个独立时间窗口。

| 算法 | 贪心平均奖励 | 完整请求 / 8 | 完成节点 / 70 | 24/48/72/96 回合曲线 |
|---|---:|---:|---:|---|
| `sa_ghmappo` | 43.279593 | 2.625 | 52.000 | 四点全部相同 |
| `mappo` | 43.279593 | 2.625 | 52.000 | 四点全部相同 |
| `cache_offload_drl` | 43.279593 | 2.625 | 52.000 | 四点全部相同 |

- SA 对 MAPPO、缓存方向 PPO 的 24 个配对 greedy 行均为 0 差；sampled 的三算法均值也同为 43.279593、2.625/8。所有 greedy 轨迹的完成节点、缓存拉取（平均 10.75 次/窗口）、模型链路（15 帧/窗口）、最终驱逐（2.25 次/窗口）、状态与共享字节、逐状态计数均相同。
- SA greedy 的 19,644 次请求帧动作中 action 0 为 11,882，action 3 为 7,762；MAPPO 为 13,357 / 6,287；缓存 PPO 全部为 action 0。三者均没有贪心 action 1、2、4；未完成请求的关联换站、到站就绪、状态传输和车端计算均为 0。严格 cold mask 在当前 adapter 未就绪时禁止 action 3，热态下 action 0 请求返回 `already_cached`，因此不同的 0/3 混合仍产生同一轨迹。
- SA 的每窗口奖励可由 52 节点完成、未完成等待 `−8.158750`、模型链路 `−0.150000`、共享代理字节 `−0.411657` 复算；完整 DAG 请求仅 2.625/8。平均 15,027 次 `compute_queue` 事件跨 24 个贪心执行，等待是主要损失。
- 96 回合未提升开发贪心奖励，但不等于训练器完全没学到。SA 三 seed 的 critic `explained_variance` 从前 6 次更新接近 0 升至末 6 次约 0.718–0.770，MAPPO 末期约 0.729–0.765；SA 后期策略熵约 0.560–0.652。只能说这四个**行为端点平台**未改善，不能证明参数或策略已统计收敛。训练 episode 相同，活跃请求数随轨迹变化，逐决策交互量仍需另报。

## 一阶根因与机制机会

`artifacts/joint_mechanism_opportunity_20261014/capacity_audit.json` 从开发窗口的原始位置、站点覆盖和 DAG 工作量独立计算，不读取学习策略奖励：

1. 8 个开发窗口中 6 个有一组全窗口固定关联的车辆，其合计工作量超过单 RSU 在 120 帧的 120 单位理论最大量。车端 adapter 未声明，动作 2 被禁；现有五动作也不能指定第二个 RSU 执行。因此这 6 个窗口在当前执行合同下**不可能** 8/8 全部完成，即使忽略模型装载、状态传输和 DAG 排序。
2. 全部 7,680 个开发车辆帧均有至少 2 个覆盖内 RSU（7,404 帧有 2 个、276 帧有 3 个）。若乐观地把每辆车完整 DAG 静态分配给其全窗口持续可达的站点，8 窗口最小单站工作量峰值为 57.75–98.5，低于 120。这仅是忽略远站输入传输、模型装载和排队的容量可行性证据，不是可执行控制结果或奖励上界；它说明增加候选站执行动作存在真实设计空间。
3. 在全部 7,680 个开发车辆帧的因果预测扫描中，动作 4 只有 13 帧合法，实际 greedy 执行期关联换站为 0；这批开发窗不能辨认预测与状态预迁移收益。训练窗固定 action-4 规则可在少数事件上提前就绪，但比 action-0 反应式规则的平均奖励只高约 0.055，属于训练窗口的机制探针，不能与本表合并成算法优势。
4. Alibaba 节点在本选择器中均使用 `veh_base_v1`，五个 adapter 标签为受控合成映射。typed cache 字节和少量驱逐可见，但跨 base cache 放置/替换及未来 DAG adapter 预置没有动作。`cache_offload_drl` 是第一方 PPO 方向基线，旧槽位占用特征与 v2 字节容量不匹配，`cache_demand` 特征在当前 predictor 下缺失；三算法同分不证明该基线已经公平代表外部论文方法。
5. 当前预测器是公共的 3 帧常速度代理、共享状态是 1 帧延迟的 JSON 字节代理；本运行 SA `auxiliary_coef=0`。不存在经此运行验证的 LSTM/GRU、校准不确定性或独立训练 surrogate。输出/状态大小和 RSU 算力均非真实设备测量。

## 审查结论与下一实现任务

- **Hard blockers**：完整正式/封存/support 证据缺失；SA 相对两个对照的开发配对奖励差全为 0；预测迁移与完整 model-cache 控制没有被当前环境兑现。论文主张仍 `Unverifiable`，不更新 canonical 或主表。
- **Major concerns**：6/8 开发窗口在旧合同下容量不可行；action 4 合法机会仅 13/7,680 帧；单 base、五个合成 adapter；缓存 PPO 字节特征错配；共享与状态成本为合成代理；仅 8 个外层开发窗口且多个相邻窗口的结果完全相同。
- **Scorecard**：七维均 `N/S (not scored)`；本 run 不含 formal/holdout/support，不能造 TMC 内部总分。**Safe claims**：固定开发合同下三算法四个曲线端点和最终奖励完全相同；候选站负载有可验证的改造空间。**Prohibited claims**：SA 奖励最高或已收敛；更大训练量已解决性能；本 run 证明 SA 的 model cache、学习型预测或状态共享优于 MAPPO；当前缓存基线忠实复现外部 DDPG 方法。
- **Required actions before re-review**：实施 `joint_control_v3_design_20261014.md` 中的共同候选站执行、显式 base/adapter admission 与状态版本合同；以训练数据预定机制活跃且容量可行的时间窗，先过固定规则与成本账本；再在同动作/输入/预算下重训 SA、MAPPO、缓存方向 PPO 和强规则。预测器与 surrogate 独立消融，报告完整请求、延迟、缓存/迁移真实事件及等交互学习曲线。随后冻结候选再评未查看的 formal 和 holdout。
- **TITS/TVT fit note**：本开发实验不支持判断期刊适投性；novelty 与最近邻比较未在本报告重新审查。
