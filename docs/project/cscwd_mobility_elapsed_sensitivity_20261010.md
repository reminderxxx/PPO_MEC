# 合成 elapsed-time 移动语义：有界固定策略敏感性

## 身份、来源与证据边界

- `reviewed_at`: 2026-10-10 Asia/Shanghai；`literature_cutoff`: 2026-09-28（未作新文献评价）；`target_venue`: IEEE TMC；`policy_version`: `tmc_review_policy_v3_20260621`。
- `artifact_run_id`: `cscwd_mobility_elapsed_sensitivity_20261010_v1`；事前计划 commit `428cdc48a9791523d68c57fa8ca7f6372ef256f1`；旧科学原件 `cscwd_causal_prepared_state_visibility_matched_20261010_v1` 与候选 `cscwd_event_aux_abstention_ab_20261010_v1`；诊断消费 B 的冻结实现源码（执行时 B HEAD `89be4d673bef7ecaf00c94d76e23f5eaa71cc90b`，五个关键源码 SHA 在 manifest 中）。本轮报告与脚本最终 Git commit 以提交记录为准。
- `evidence_level`: **development-local fixed-checkpoint synthetic sensitivity**，产物完整且逐步成本核验；没有真实逐帧接触时段、共同新环境匹配训练、formal/holdout/support，TMC-ready/优秀基线优势 `Unverifiable`。原条件弃权 `MIXED_STOPPED` 不变。未修改论文、主表、原 checkpoint、原环境默认或旧 artifact。
- 新原件 `artifacts/analysis/cscwd_mobility_elapsed_sensitivity_20261010_v1/`：`run_manifest.json` SHA `440b3af1b76430a7080475291fb0fb98fa1be82c1da92b4d30d109ea03ccfa34`，`aggregate_summary.json` SHA `a418a042509111754461fa8ce7870db2e101e2364f5986da4e5a595f40083ac1`，`legacy_replay_receipt.json` SHA `4eb3289cedc53c9fd553f427807face52caa50be0dfe11e4deff31e522776bba`；episode/transition JSON 的 SHA 在 manifest 中。旧/新原件、四个 seed7 selected checkpoint 和协议的 SHA 见冻结计划；运行前逐项检查。

## 两个时间合同

**旧默认**：`decision_step_index` 每做一次真实动作便令位置索引加一，包括 2s 失败等待；deadline 时钟按动作的 modeled seconds 累加。它与既有代码合同一致，本轮没有把它标为实现 bug。选定的 regression/frozen 六实例×四 learned 方法，旧 profile 24 episode、**144 条逐 transition** 对原 selected ledger 的 action、起点 RSU、服务/迁移、model/state bytes、recompute、service seconds、clock、deadline event 全部 0 mismatch；另在局部测试直接以固定动作序列比较原类的 observation、mask、cache/state、reward。

**新 opt-in `synthetic_elapsed_5s_atomic_v1`**：只在诊断脚本中派生原环境类，不改原类。把 frozen `rsu_sequence` 的每个元素解释成固定 5s 合成 slot：动作前 `slot=floor(clock_seconds/5)`，超过序列长度则钳在末元素；动作成本可一次跨多个 slot，**整步在起点 RSU、起点链路原子结算**，下一决策才采用末端位置。2s 失败、8s vehicle fallback、model load 和长 compute 都按同一 modeled clock 推进。action4 是阻塞动作，当前节点完成后才提交目标 prepared state；无中途转发、异步预取或跨界中断。action1/4 的真实可行性仍使用起点 current RSU 的连续 slot 剩余接触预算，目标 RSU 接触时间**没有**单独建模；公开预算及 two-step 决策预览使用因果预测序列。deadline 在整步结算后判断，max-step 仍按真实决策次数。

这是明确的**atomic 合成假设**：冻结 workload 有 NGSIM 来源窗口的 frame/time 身份，但没有每个决策位置的实际进入/离开时刻；5s/slot 不是 NGSIM 物理测量。跨界仍在旧 RSU 完成服务尤其不能当作无线可行性证明。新 profile 的独立局部测试覆盖短动作无跨界、2s 失败、成功长动作、>15s 多 slot fallback、前缀预测无未来泄漏；四 learned checkpoint 在篡改未观测实际未来后，公开 observation/semantic/mask 与动作/概率相同。

## 冻结矩阵、权限与完整性

按初始 cache readiness、首节点耗时及合成接触条件预选 `dev_01`,`regression_00`（短动作对照），`regression_05`,`frozen_check_05`（初始缺 bundle/失败等待），`dev_00`,`frozen_check_02`（成功 compute>5s），`frozen_check_04`,`frozen_check_06`（合法 fallback 加 compute>10s）。这些类别通过固定动作的局部测试触发；正式固定策略若未选择相应动作不换样本。固定 seed7 的旧 SA、条件弃权 SA、PPO、controller-MAPPO selected checkpoint，加原 two-step rule；旧/新 profile 各一次，**8×5×2=80 episode、528 个真实 env.step**。two-step 另外做 **2,094 个模型预览 step**，总真实+预览 `2,622<2,880`（不含局部测试）；0训练、0 optimizer step、0选模、0新增 seed。所有真实动作逐步匹配公开五动作 mask，参数前后哈希不变。

四 learned 方法共用相同原生五动作、cache/state、link/费用和各自 v4 公共输入；固定参数跨语义评估会改变状态分布，不能当匹配训练比较。two-step 保留两步环境 clone 能力，且原规则的第二步预览在未来 preview state 上用 `instance.rsu_sequence` 取当前 RSU，虽然接触可行性在 clone 中改用因果预测预算；因此它具有**实际未来路由预览权限**，超出 learned 方法的公开单步观测。原代码如此，本轮未偷偷改规则。two-step 行仅作为能力不同的系统敏感性参照，**从公平方法排名和 learned 归因中排除**；若要求严格 public-only two-step，需另立规则合同和验证，不能拿此原件直接晋级。

528 个真实 transition 全部满足 `finish_clock-start_clock=step_cost=failure_seconds(如失败)+service_operation+recompute+transfer`；逐 episode `Σmodel/state/input bytes` 与原生 metrics 完全一致，cache 无超容量/重复/未知 resident，prepared-state 前缀为实际完成节点子集。模型字节在 action2 上仍为零，vehicle fallback 的 8s 固定服务费用和 input network 继续计入；没有给 PPO 另加车型 model bytes。

## 机制读数与方法边界

| 固定策略（各 8 实例） | workflow complete 旧→新 | on-time 旧→新 | service failures 旧→新 | 平均 elapsed s 旧→新 | model MB 合计 旧→新 |
|---|---:|---:|---:|---:|---:|
| 旧 SA | 8→7 | 2→3 | 3→19 | 81.85→85.84 | 0→0 |
| 条件弃权 SA | 8→8 | 6→2 | 0→0 | 65.17→73.47 | 2724.88→5198.92 |
| PPO | 8→8 | 4→5 | 0→0 | 63.30→61.67 | 501.84→328.13 |
| controller-MAPPO | 8→8 | 5→3 | 2→5 | 66.32→85.68 | 5430.48→6580.64 |
| two-step（额外未来路由预览权限） | 8→8 | 6→6 | 0→0 | 43.99→45.01 | 820.33→665.90 |

这是**同一 checkpoint/实例在两种假设环境的描述读数**；8 个已消费实例、单 seed、非匹配训练，不能把行间数值当新的 baseline 排名，也不能把某个方法的方向当普遍物理结论。平均 elapsed 包含未完成 episode 的截断时钟；各行 completion 需并列。开发实例重复源时间窗口不能当独立统计 cluster。

机制上，旧 profile 的 `4` 个纯 2s 失败后 RSU 切换中有 `3` 个**没有**跨 modeled 5s 边界；新 profile `24` 个纯 2s 失败只有 `3` 个 RSU 切换，且全部确实跨 5s 边界。旧 SA `dev_00` 是直接反例：step2 在 `7.826→9.826s` 的 action4 失败后旧语义从 `rsu_0→rsu_1`，新语义仍在 `rsu_0`；旧 episode `1` 次失败、53.65s 按期完成，新语义随后多次失败并在 154.61s 才完成、逾期。它证明**某些路径的失败等待收益方向依赖步数推进假设**，但后续策略/状态联动，不是单步收益的普遍估计。旧 SA 在另两个初始缺 bundle 实例的 on-time 反而 `0→1`、`0→1`，其八实例总数 `2→3`；不能只挑 `dev_00` 写成总体优势反转。

长动作同样改变移动：固定 `frozen_check_04` 首步合法 action2 的旧/新 event、缓存、成本与奖励完全相同，耗时 >15s；旧下一步只到 slot1 的 `rsu_0`，新跳过多 slot 到 `rsu_1`。新 profile 的 273 次真实决策里有 **22 次在合成序列完整时长之后**，被钳在最后 RSU（旧 profile 为 255 次中 11 次 modeled clock 已超该时长）。末端驻留、起点原子服务及 5s contact 均无真实轨迹支持，尤其影响长/重复失败 episode，不能把此敏感性改写成“修正物理真值”。

## 决定

旧合同确定允许 2s 失败按决策次数推进位置；合成 elapsed profile 消除了未跨 5s 边界的这类切换，固定策略后果却因方法、实例和长动作而混合。**没有证据把旧实现判为 bug，也没有足够来源把新原子语义判为物理正确。** 若未来获得逐决策真实接触时段，需先冻结：跨边界服务继续/中断规则、目标 RSU 接触与 action4 阻塞语义、轨迹末端处理、共同 actor 公共时间字段及 two-step 权限；再由所有 learned 方法在同一环境重新匹配训练与独立评估。当前只交付已冻结的敏感性与明确候选假设，不自动训练、不扩大队列/多车/跨 workflow。
