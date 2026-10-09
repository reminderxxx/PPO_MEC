> **2026-09-08 追加勘误（G14R20-A）**：G14R18 补验任务新增正式执行为零，不表示既有 v16 未创建。
> 既有 v16 已完成 150 train cells、24 dev cells、1,200 candidates、150 selected/frozen checkpoints 与 6 个 generated resources；
> 无 failed terminal，尚无后续 formal 阶段。恢复门禁受阻不等于 run 失效。下文历史正文保留；涉及项目当前状态时以本勘误为准。
> 独立合同、原声明/hash 及证据指针见 [fixed_commit_continuation_contract.md](fixed_commit_continuation_contract.md)。执行器未实现、独立批准未签发，`execution_authorized=false`。

﻿# Bugs And Risks

## 2026-10-09：新前缀版本已消除公共实际后缀回退；确认性证据仍缺

- `RESOLVED IN V3 PREFIX-ONLY PROFILE`：新版本对 36 实例的 474 个决策前缀篡改未来后缀，公共观察、语义、mask 均不变；旧版本的阻断继续有效。物理接触判定版本已独立分离，因此旧/新结果不能作算法因果比较。
- `OPEN`：训练数据拟合的小计数预测器与人工轨迹规则可能相关，预测误差、unknown 与误差分层须由逐步原件报告。two-step 精确 clone 能力高于 learned/Popularity；36 实例全已消费，跨 run/车辆独立确认与真实无线仍缺，paper-ready=`Unverifiable`。

## 2026-10-09：公共预测含实际未来迁移序列（旧版 OPEN SCIENTIFIC CONTRACT BLOCKER）

- `BLOCKED`：36/36 已消费 development 实例没有独立 `predicted_rsu_sequence`；`_predicted_sequence` 回退到环境将执行的 `rsu_sequence` 后缀，继而公开 next/target/dwell/contact。DT 与 Popularity 明确消费这些字段，其他 learned 方法也可访问；同信息不等于无未来泄漏。本轮科学比较在 run root 前停止。
- `REQUIRED`：另行冻结因果 prefix-only 预测来源、生成时点、source hash 与逐步消费测试；不能只复制实际序列。现有原始区间独立性缺口和 B PopArt 不晋级结论均不因此改变。详见 `cscwd_2027_strong_baseline_prediction_permission_blocker_20261009.md`。

## 2026-10-09：强基线工程已接通，科学公平性和独立数据仍开放

- `OPEN / method fairness`：仅完成 synthetic fixture 的公共输入、mask、DT checkpoint 与 Popularity 逐实例 reset 验收；尚无四 learned 方法完整开发比较、逐字段预测权限核对、完整计算成本或基线排序。默认 `--run` 禁止，不能从接线成功推断算法优势。
- `OPEN / data independence`：36-instance manifest 全部已消费为开发，尚无跨 run/车辆独立的新确认性 split；旧 holdout 不重开。详见 `cscwd_2027_strong_baseline_wiring_20261009.md`。
- `BOUNDARY / PopArt`：下文“尚未实现/未训练”保留为当时诊断记录；B 的开发 A/B 后续已执行，本分支未审查其科学原件或把 PopArt 晋级。raw/disabled 仍为新基线草案默认。

## 2026-10-09：critic 尺度与共享裁剪为首选候选，因果仍待匹配 A/B（OPEN LEARNING BLOCKER）

- `NO IMPLEMENTATION DEFECT CONFIRMED`：当前 hash-matched 版本的 termination、bootstrap observation、episode GAE 和
  executed-action loss 接线通过只读核对；不以旧分支缺陷代替当前证据，也不提交算法修复。
- `SUPPORTED CANDIDATE`：service-aligned 下三方法的 critic value loss/EV、value:policy gradient ratio 与 clip scale
  一致异常；尺度归一化的无更新梯度 probe 显著降低 ratio。该证据支持最小 PopArt A/B，不证明因果或创新。
- `UNRESOLVED`：训练/评价 truncation 目标边界、小 dev 早选和固定 episode 导致 steps 不等仍是混杂；没有现成 later
  checkpoint evaluation，禁止事后补评选最好模型。
- `REJECTED PRIORITY`：policy/aux 梯度冲突不跨 seed 一致，no-aux 已失败；不得自动进入 auxiliary-target 消融。
- `DATA/CLAIM BOUNDARY`：当前 36-instance manifest 已全部消费，只能作 development；confirmatory 需新 interval ledger、
  原始 frame/time 互斥 split 和窗口身份校验。PopArt 配置尚未实现、未授权执行、paper-ready=`Unverifiable`。

## 2026-10-06：服务对齐 reward 公式通过但策略退化（VERIFIED FORMULA / REJECTED CANDIDATE）

- `RESOLVED FORMULA GAP`：candidate 对按期/逾期/未完成、成本和 truncation 的单元排序通过；deadline 不再只在完成时可见，
  外部截断仍 bootstrap。reward profile 不改变状态、动作、数据或网络。
- `NEGATIVE BEHAVIOR`：候选使三个 learned 方法在 regression/frozen 的 completion 全部下降；MAPPO/PPO 的 service
  failure、invalid prepare 和 no-progress 显著增加。PPO on-time 单项上升不能抵消总完成和可靠性退化。
- `TRADE-OFF`：SA transfer/invalid prepare 下降，但 completion/coverage 降低且无进展上升；禁止写成成本—服务共同改善。
- `OPEN`：大终局效用与 PPO 稀疏 credit、不同 policy parameterization 的交互尚未因果分离；本轮禁止按结果搜索权重、
  加 seed 或延长预算。候选已拒绝，不进入 auxiliary-target 消融。
- `BOUNDARY`：regression 已暴露，frozen check 仍是开发验证；无 formal/holdout/真实无线。paper-ready 继续 `Unverifiable`。

## 2026-10-06：删除 auxiliary 未修复过度准备，训练 target 设计仍开放（REJECTED FIX / OPEN METHOD BLOCKER）

- `DIAGNOSED`：`event_prepare_margin_boost` 与 sharpening 在当前 `raw_policy` 训练/评价路径实际 dormant；temperature 不改变
  deterministic binary argmax。不得继续把这两项写成当前 action-4 过多的直接原因。
- `CONFIRMED DESIGN CONFLICT`：auxiliary hard event target 不检查 current bundle readiness；17 个 current-missing dev
  状态中 12 个 target=1。但 target=0 的 current-missing/infeasible 状态也会选 action 4，故该冲突不是充分根因。
- `REJECTED FIX`：中性化整个 auxiliary 后，completion 不变，frozen action 4 `110→149`、invalid prepare `56→86`、
  target-infeasible prepare `34→59`；source-window CI 也指向无效准备增加。不得把删除项作为保留候选。
- `OPEN`：event hard target、temporal soft target、temperature、executed-action PPO 与有限训练的交互尚未因果分离。
  本轮停止。若另立任务，唯一合理方向是把 current service readiness 与 target feasibility 写进 event auxiliary target，
  不能用 SA 专属强制 action 0、reward 调整或追加预算代替。

## 2026-10-06：action-4 实现循环已修复，SA 事件策略偏置仍开放（RESOLVED INTERFACE / OPEN METHOD BLOCKER）

- `RESOLVED`：旧确定性路径把 event-prepare 全部概率集中给 action 4，却把 event-keep 分摊到 action 0--3；再叠加
  canonical-head/executed-action likelihood 错位与 failure-time mobility 冻结，形成相同状态下的 action-4 循环。v3 的独立
  head argmax→aggregate→mask projection、executed-action PPO 和 decision-step mobility 已由测试及 1,340-row replay 闭环。
- `RESOLVED`：flat encoder 的 adapter-count/byte-capacity 单位错误，以及 flat/graph 对 typed base、bundle/model/state/input
  bytes、estimated link、contact budget 等公共字段的漏用，已在显式新 profile 中修复；旧 profile/checkpoint/结果不改写。
- `NEGATIVE`：修复后 SA completion 仅恢复到 regression `0.944`、frozen development `0.958`，PPO/MAPPO/rule 均为
  `1.000`。SA 在 current bundle missing 时仍约 50% 选择 action 4，并有最多 4-step 的短暂无进展；这已不是状态冻结，
  而是合法但低效的 policy 行为。未做 event 消融，不能把因果归给某个单独增强。
- `CAPABILITY BOUNDARY`：two-step rule 不读 execution-time actual link，但拥有 exact decision transition clone 与字典序目标；
  learned actor 只优化标量 reward。它是应保留的强 model-based baseline，但不能称与 learned policy 具有相同信息使用能力。
- `OPEN`：固定预算下学习不足不能排除；不过三 seed 均完成 192 episodes、训练 completion 为 0.983，且规则更省时省流量。
  当前证据更支持 SA policy/event design 不稳且该 workload 下两步规划已足够。新检查不是独立 holdout，paper-ready 仍
  `Unverifiable`；不得自动追加训练或调参。

## 2026-10-06：校准 pilot 中 SA-GHMAPPO action-4 过度激活（OPEN ALGORITHM CLAIM BLOCKER）

- `NEGATIVE`：冻结匹配训练不支持主方法优势。SA-GHMAPPO 12-window completion `0.528`，相对 two-step rule 差
  `-0.472`，窗口 bootstrap 95% CI `[-0.722,-0.250]`；不得进入支持优势的论文主表。
- `OBSERVED`：SA 的 36 个 seed-window run 共 action-4 `243` 次、service failure `191`；handoff prepare 在当前
  bundle 未 ready 时仍为 contract-legal action，策略未学会先恢复当前服务，导致截断。PPO/MAPPO 完成但主要 action-0，
  每次 handoff 重算，亦未兑现状态迁移机制。
- `BOUNDARY`：这是固定 192-episode、3-seed non-formal pilot 的策略行为，不等于实现 bug；同一审查轮不修改 reward、
  mask、guard 或预算。若修复，必须先另立诊断/实现任务，再用新结果盲 protocol 验证并保留本轮负结果。
- `OPEN`：formal/holdout/support、BCa/sign/Holm、跨数据组合、共享资源争用、compute accounting 和真实无线均缺失；
  paper-ready 仍为 `Unverifiable`。证据见 `calibrated_continuous_workflow_pilot_20261006.md`。

## 2026-10-06：路径不对称已修复；算法增量价值缺口确认（RESOLVED IMPLEMENTATION / OPEN CLAIM BLOCKER）

- `RESOLVED`：新 v2 ledger 为 restart/recovery 分别建立 cache state，双方都执行 current→next 合法 transaction；每个 model
  admission、dynamic transfer、restore/recompute 只计一次。decision preview 与 realized score 使用 4 个独立环境。
- `WITHDRAWN`：旧 action 0 跳过 target 当前准备，旧 `two_step` 又复用同一 scorer，offline reference 也比较不匹配分支；
  因此 `10/12 vs 7/12`、17.993 s / 972,125,684 B 优势及 d10/d11 原归因不能作为纠正后证据。
- `NEGATIVE`：纠正后四方法 12/12 全同；新边界三在线规则 6/6 全同，并共同在两个固定 link estimate mismatch 点错选。
  新方法相对正确简单阈值和两步前瞻没有新增能力。
- `OPEN BLOCKER`：若论文坚持算法创新，需要外部动机充分且真正 action-specific 的 lifecycle 差异及独立现实证据；本轮不扩建
  跨 workflow 平台、共享队列、RL 或模型测量。当前只能定位为系统机制与经验性失败边界研究。
- `BOUNDARY`：synthetic 时间来自 Mbps/fixed-latency 模型，不是网络实测；6 点边界不是现实误差分布或统计泛化。

## 2026-10-05：真实 victim→reload 已闭环，但不是差异化决策证据（RESOLVED BOUNDED / OPEN CLAIM BOUNDARY）

- fixed commit `f31024d…` 的 12/12-call witness 已确认 native 合法 ALPR victim、PEFT tensor/object 实际卸载、后续
  local-file reload 和 n2 执行；无驱逐对照 action load/unload=0。
- restart/recovery 请求相同当前/未来 adapter，故各自实测 lifecycle `1.022245/1.026997 s` 是共同路径成本；旧公式
  若只向 recovery 收费会制造不对称，不能用于真实 action reversal claim。
- 两条件均继续选择 recovery，且预测明显偏高；决策边界、真实无线/queue、任务正确性和统计泛化仍 OPEN。
- 结果：`real_cache_victim_reload_results_20261005.md`；artifact：
  `artifacts/analysis/real_cache_victim_reload_20261005_v1/`。

## 2026-10-05：历史 recovery 公式对当前模型准备存在路径不对称（FROZEN SCIENTIFIC DEFECT）

- `eviction_aware_recovery.py` 的 frozen development rule 从原 resident 集 `C` 计算 rerun 的未来缺失，却只在 recovery
  显式计 `D0\C` 与 victim 后 `C'`；它不是一般 restart/recovery 全路径公式。
- 既有 12 点是按该已冻结方法定义和评分，原 artifact、动作和数值不重算、不覆盖；这项缺陷限制其可解释范围，而不是
  授权事后换公式改善历史结果。
- 既有 real-model runner 的两臂各自新进程加载相同模型，但没有真实 typed resident→PEFT object 生命周期，因此也不能
  证明 victim 成本只属于 recovery 或已由“共同成本”抵消。
- 新的 `real_cache_victim_reload_v1` 只用逐事件全路径账本补最小真实见证；两臂共享同一 lifecycle 能力，并仅抵消对象、
  时点和调用均实际相同的 setup。该接线不包装为原创算法，也不改变正确两步前瞻等价结论。

## 2026-10-05：独立检查同侧通过但未触及驱逐项和决策边界（OPEN SCIENTIFIC BLOCKER）

文稿 v1.1 追加口径风险：冻结开发公式仅显式向 recovery 写入当前依赖准备项，rerun 路径仍从 C 投影；泛化应用前必须核对两条实际路径的准备与缓存转移，不能默认共同成本相等。当前仅识别文稿适用边界，未据此判定代码错误或重算结果。见 `manuscript_evidence_progress_20261005.md` P01/P02。

- 新测量 6/6 方向匹配，但全部是 recovery，不能据此声称阈值校准、分类稳定性或泛化；不追加条件制造翻转。
- 绝对成本统一偏高，prepared/recovery 相对误差达 `45.9%--49.7%`，说明旧 child-wall calibration 与新 action-window
  边界之间仍有固定开销差异；方向正确不能替代成本校准。
- 真实模型生产路径没有可合法构造的 dependency-safe victim→later reload；中央“驱逐代价”项仍只有 native synthetic
  witness，未获独立真实执行支持。
- 网络为 100 Mbps/0.02 s 假设，queue/task quality unavailable；证据只支持同机机制/经验性研究。若仍以算法创新投稿，
  与正确两步 12/12 等价和既有 cost-aware caching 文献共同构成 blocker。

## 2026-10-05：驱逐代价修正受估计误差和两步等价限制（OPEN SCIENTIFIC BLOCKER）

- 新规则与同信息、正确计 victim cost 的两步前瞻在 12/12 点动作/成本完全相同，纯函数中位开销仅
  `7.875` vs `7.896 µs`；当前没有算法或效率优势。
- d10 将 recompute 低估 50% 时，新规则错误重跑并多 `2.300247 s`；d11 高估 2× 时错误恢复并多
  `9.128260 s`。结构成本修正不能替代独立校准和 uncertainty/margin contract。
- 12 点是 synthetic validation instances，不是独立现实数据；没有无线、共享链路/计算队列、跨 workflow cache
  生命周期、任务质量或统计区间。aggregate 只能作 bounded diagnostic。
- 当前 paper claim 只能是实现修正/组合；若投稿主张算法创新，缺少相对正确两步前瞻的新增能力或证据是 blocker。

## 2026-10-05：workload v0.1 原 local/exact 证据存在自证循环（FROZEN SCIENTIFIC DEFECT）

- local 直接读取与执行模拟器相同的 low/high `restore_cost` 真值，不读取逐事件 cache 或 target-ready；exact 的动作不改变
  后续模型/cache 状态，三次二选一主要由同一阈值决定。原 24/24 一致性不是独立在线方法证据。
- seed 只改变 arrival jitter 和派生 ID；按决策有效输入仅 low/high 两类，不能把 3 seed 称为成本独立复测。
- sharing-off 实际是 `base_a→base_b→base_a` 两组 family identity，A0/A1 仍共享 base_a；不得写成三个完全不兼容 base。
- 12 点成本失配出现 4/12 local/oracle 分歧；边界/高恢复点最大时间代价 17.696701 s。target-ready 错误在当前共同
  model-prepare 合同下不改变动作，只改变绝对完成时间。
- 三次固定实现同机复测都支持 recovery，但 OS cache 未控制、没有真实网络/队列/任务质量，不能升级为部署稳定性结论。
- 当前保留简单阈值且不启动 RL；外部独立成本校准前，RL 必要性和复杂方法优势均为 unsupported。

## 2026-10-05：action 4 / workload v0.1 后的剩余边界

- `RESOLVED / production technical state path`：显式 opt-in action 4 已实际 export/import、校验 target model 与
  workflow/model/input/next-node identity，正例只执行后缀；缺模型负例 fail-closed。默认/正式 Protocol 不变。
- `OPEN / production state completeness`：2,185 B 仍是 explicit-text technical package，不是 KV/tensor 或完整应用状态；
  task quality unavailable，不能外推任意 DAG/model。
- `OPEN / system realism`：没有无线、RSU queue、丢包/重传或多租户测量；100 Mbps、容量、deadline 和 high restore
  都是 synthetic，单机 load/I/O 不称 VEC deployment latency。
- `BOUNDARY / workload evidence`：8 点半分数和 3 seed 是 bounded development diagnostic，不是独立现实测试或最终
  统计；公开价值、许可、代表性和独立复现未完成，旧 holdout 未使用且仍永久不可复用。
- `NEGATIVE / high restore`：高恢复点下 local/exact 都选择 restart；恢复并非普遍有益。`ALPR adapter` 负结果不变。
- `BOUNDARY / algorithm`：local=exact 于 24/24；当前不授权 RL。未来只有外部有效性出现可解释、公平 gap 才可另立
  最小方法任务。
- `RESOLVED_FOR_FUTURE_RUNS`：普通 frozen-plan loader 的身份缺口已以 fail-fast 检查封闭，覆盖主 benchmark、算法池及 SA 训练入口；旧 v70 身份错配和重叠统计不因代码修复而消失。新独立数据及同预算强基线仍 `OPEN`。

- v70 mixed/full 的 `selected_window_plan` 是 Peachtree，原始 rows/episode receipt 是 Lankershim，ID 交集为 0；普通 frozen-plan 路径传入的 `expected_window_id` 未被普通 mobility loader 校验。需要独立实现任务修复身份链和 rollout 前错段负例。
- full 48 个窗口存在 177 对原始 frame 区间重叠，mixed 20 个窗口有 61 对；原 window-outer CI 与 Holm p 不可按独立窗口解释。
- SA 与 PPO/DT 继承池的更新计数及 checkpoint 选择规则不同。新数据、同条件重训和可审计命令是下一轮门槛；详见 `cscwd_2027_baseline_contribution_audit_20261009.md`。

## 2026-10-05：机制闭环后的有效边界

- `OPEN / production action 4`：native action 4 只做 prepare/history，未调用已验证 state export/import；独立恢复脚本
  不能记为 production migration 已实现。
- `OPEN / state cost identity`：2,040 B 是固定 technical package，160 MiB 是 synthetic stress point；均不是完整
  production runtime/KV state。没有真实无线传输测量。
- `OPEN / decision matrix activation`：12 格中 sharing 和 136/320 MiB 没有改变结果，说明该工作流/动作轨迹未激活
  两个轴，而不是证明它们普遍无效。
- `NEGATIVE / ALPR adapter`：旧 12 样本 locked-check 与 base 持平，development 更差；不得晋级或重命名为独立测试。
- `BOUNDARY / algorithm`：完整枚举只在失败数与状态字节间形成 Pareto 交换；差距可由局部 action-4 规则表达，
  不授权 RL 或扩大模型搜索。

## 2026-10-03: 两节点显式文本状态恢复已技术闭环，production/VEC 语义仍开放（PARTIAL CLOSURE）

- 新 artifact 已证明独立 target 消费 source 的 `n0` 文本并只执行 `n1`；因此“本技术链没有 workflow-state
  recovery witness”的缺口关闭。历史 9/30 失败校准和 10/3 旧顶层 FAIL 均不改写。
- 输出 ` 2010 - 2` 没有 ALPR/Helmet 标签且语义质量不佳；任务正确性仍 unavailable。单次同机 CPU 成本不能外推
  RSU、无线链路、排队、稳态均值或净收益。
- 状态边界是显式应用文本，不包含 KV/tensor；production action 4 仍未序列化/传输/导入该状态。多 adapter DAG、
  真实车联网输入、故障恢复和统计重复仍是下一阶段缺口。
- 详见 `two_node_workflow_suffix_recovery_acceptance_20261003.md`；不得据此晋级算法或论文结论。

## 2026-09-30: 真实 adapter 和独立进程状态恢复仍不可用（OPEN BLOCKER）

- 本地没有与固定 SmolVLM base 配套的真实 adapter；公开 ALPR/Helmet LoRA 共需 164,065,376 bytes 新权重，固定
  环境也没有 PEFT/accelerate。本轮未授权下载或安装，故没有 A→B→A load/switch/execute 或任务质量证据。
- 两节点状态 calibration 的 source 写出 3,085-byte 声明状态，但冻结 base 输出为 `lanestatus` 而非 `clear`；
  target 独立进程在模型加载前按正确性合同拒绝。正式测量为 0，该字节数和预热 save 时间不得作为完整迁移成本。
- action 4 的 `migration_realized` 只兑现当前 adapter prepare；原账本 0 state bytes 表示缺 payload/object 记账，
  不是零成本。真实状态、adapter switch、target load/rebuild、waiting 和网络成本未知，104 MiB 合成字节收益仍 conditional。
- 下一轮需中央复核新增资源授权，并另立预冻结的状态任务；不得在本轮失败后换 prompt 重试或直接进入方法比较。
  详见 `adapter_state_recovery_calibration_20260930.md`。

## 2026-09-30: 剩余信息见证受状态账本与动作可达性限制（OPEN）

- `semantic_discrete_5` 只能为当前节点adapter做current/next/handoff-target placement；不能直接准备future adapter，
  ActionAdapter不输出`migrate`，也没有继续旧RSU执行并转发结果的动作。
- bounded action4 prepare在两个长尾case同完成/失败下少104 MiB synthetic模型传输，但
  `migration_prepare_realized=true`时`state_migration_size_mb=0`；这是payload未记账，不是零成本迁移。
- 本机已测SmolVLM base和732-byte显式应用状态，但无真实兼容adapter，且应用payload不是完整runtime/KV state；不得
  回填native ledger或外推RSU/车辆/无线网络。
- 两步前瞻在全部预注册实例与full-tail首动作一致；在出现预注册两步不足case、冻结扩展动作合同和完整state成本前，
  不支持更强oracle或算法修改。详见`remaining_workflow_decision_value_audit_20260930.md`。

## 2026-09-30: native typed 跨底座替换已有 opt-in LRU 候选，正式扩展仍受限（PARTIAL CLOSURE）

- `sequential_dependency_recompute_lru_v1` 已在 shadow state 上逐 victim 重算依赖，固定 microtrace 的合法 action、
  保护负例、四配置和两节点 `env.step()` 见证通过；旧 `static_dependency_safe_v1` 默认与历史 hash 不变。
- 候选仅定义原生 LRU；FIFO/LFU/Aging-LFU/Random 显式拒绝。它对所有共享 action/runtime 的算法可用，不是 SA
  专属能力，但尚未形成五policy公平性合同。
- 正式 Protocol 2.9、oracle/replay仍使用v1.0静态语义；不得把candidate诊断产物混入formal/canonical或历史解释。
- callback异常恢复、复杂度/scalability、真实模型I/O与真实联合trace仍未验证。报告见
  `native_typed_cache_replacement_witness_20260930.md`。

## 2026-09-30: native typed dependency-safe feasible set 与 closure eviction 不同（OPEN DESIGN ISSUE）

- 固定四配置的288请求原生审计确认：当 `b0` base 仍被 resident adapter 依赖时，原生
  `_typed_evictable_residents()` 把 base 排除在单次静态 victim set 外；选中 adapter 后不会在同一原子事务中重算
  base eligibility。
- sharing-on/interleaved 请求1 `b1.a0` 需要释放104 MiB，但原生唯一 eligible victim 只有8 MiB adapter，因而
  `insufficient_dependency_safe_evictable_capacity`；reference 可用 dependency-closure eviction 删除 adapter+base。
- 当前规则保持原子拒绝、无 orphan、容量与状态不变量，但使可执行 placement/action space 小于 reference；固定
  microtrace 中 sharing-on 有36/72服务失败，sharing-off有60/72服务失败。
- 原审计轮只读且没有修改规则；后续独立任务已实现显式、LRU-only顺序重算候选，但旧规则仍是默认，五policy与
  oracle/replay正式扩展仍未冻结。不得把旧语义的低传输误写为效率收益。

## 2026-09-29: DriveLM demo 不提供可直接使用的依赖边（OPEN）

- 单场景引用包已生成但仍无真实推理；DriveBench公开研究提示Q中文字/坐标可能形成猜测线索，
  须检查视觉依据与上游错误传播，不能以答案非空晋级。许可确认、模型实载和状态恢复实测仍待完成。

- 实读790 QA的五个关系字段均null，不能把任务类别排序当成原始执行DAG。
- 下一步须显式派生模板并验证数据依赖；不能外推完整语料或宣称已形成新数据贡献。
- 证据：`drivelm_sample_qualification_20260929.md`。

## 2026-09-28: AI工作负载融合真实性与数据贡献（OPEN）

- Alibaba DAG拓扑/任务类型到adapter的映射是受控标签；统一base及CPU/memory到I/O尺寸转换尚非AI测量。
- 模型家族兼容、输入/中间数据有效性、状态内容/恢复成本和移动—到达联合关系需要校准；不能全部假定共享base。
- 新数据集尚处设计阶段：本地profile、独立真实性验证、许可及可再分发范围未闭环，不宣称已发布贡献。
- 文献边际参数拼接不构成真实联合分布；按candidate胜负选择生成分布会造成循环论证。
- 生成器新seed不替代真实独立来源；旧holdout永久消费边界不变。
- 方案及证据池：`vec_ai_workload_dataset_design_20260928.md`。

## 2026-09-28: 开发矩阵的效率解释与设计实现差距（OPEN）

- CRDCM开发矩阵只有3个outer windows；SA 18/36与PPO 27/36不能支持SA总体优势。
- 保留失败成本后MB/完成workflow为SA412.889、PPO378.667；按成功request归一化方向不同，禁择分母。
- 当前五动作实现不能选择任意future critical object，不得用对象级联合优化的设计文字代替实际方法。
- 机制任务的首因诊断报告behavior/update distribution缺口，性能归因待数学复核及新匹配修复实验；
  中央本轮只独立复算成本与完成数，未重跑梯度审计。不得因缺陷存在即断言修复后必优于PPO。
- 详见 `research_problem_and_evidence_plan_20260928.md`。

## 2026-09-28: 核心机制机会、因果消融与独立测试缺口（OPEN）

- formal workload 的 1,860 个 SA request 全部只请求 `adapter_batch_type_1 + veh_base_v1`，没有多 adapter
  共享 base 的机会；因此 base-sharing 代码虽已启用，当前不能提供其收益证据。
- `no_base_sharing`、`no_workflow_state_migration`、`fixed_no_eviction` 等预注册 typed-semantics 消融不可用；
  handoff/migration 事件稀疏。核心机制属于“已实现但不可公平归因”，不能写成已证明贡献。
- G12 supervised predictor 在正式路径关闭，predictor availability mask 与 causal accepted snapshot 均为 0；
  若未来加入，应视为新机制并重新训练/评估，不能回溯归入旧 formal 方法。
- 576/864 MB 均高于 observed reachable working-set peak 472 MB，eviction 为 0，primary outcomes/action 序列
  相同；当前容量设计只能识别 288 MB 绑定与 ≥576 MB 非绑定，不能支持饱和曲线或 scalability claim。
- G14R22D holdout 已永久消费，在第一档 rollout 前因 sealed-window validation interface 失败，没有性能结果，
  永久禁止 retry/resume/reopen。G14R23 修复接口后仍未找到跨多 run 的可靠 unused range；当前没有有效独立
  测试，paper-ready 继续为 `Unverifiable`。
- 当前成本统计存在宿主/批次混杂，缺统一训练 wall-clock、CPU/GPU-hours、峰值 RSS 与能耗；SA inference 的
  描述性额外开销不能升级为效率结论。修复路线与预注册实验见
  `docs/project/g14s01_innovation_algorithm_diagnosis_20260928.md`。

## 2026-09-21: G14E07 formal 统计解释与 holdout readiness blockers（OPEN）

- `formal_gate.json` 的 claim map 对已经统一为“正数有利 candidate”的 signed CI 又按 lower-is-better 二次
  翻转，把 12 个 transfer 劣势错误标为 `supported`。原 artifact 不覆盖；修复必须另立实现任务并生成带
  provenance 的纠正统计附录。
- `analyze_top_journal_statistics.py` 的 hierarchical CI 按 window outer cluster 计算，但 sign test 仍按 540 个
  paired rows 计 wins/losses，导致伪重复；Holm 也继承该错误 p-value。G14A01 窗口级复算后没有 84-family
  Holm 显著项。
- delay 只在完整且全部 request 成功的 workflow 上 finite：每个 agent 为 `220/540`，59.26% unavailable，
  共同 finite pair 全部零差。它是 survivor-conditioned、当前无策略区分力的 endpoint。
- ready/continuity gain 只出现在 2/12 窗口；transfer/backhaul 代价覆盖 9/12 窗口。576/864 primary outcomes
  完全相同；五个 reactive policies 的 primary outcomes 也完全相同，capacity/eviction discrimination 有限。
- typed-semantics 仅 `typed_full` 与 `no_prediction` 可执行；4 个机制消融 unavailable。G12 supervised predictor
  disabled；oracle baseline gap unavailable，实际 visited states 仅 20–63。
- 当前 G14E07 contract 无 holdout capability、无 holdout command。Holdout 前必须先冻结并批准独立 opening/
  execution contract、候选 hash、window-level statistics、signed claim rule 与一次性消费记录。详见
  `docs/project/g14a01_formal_results_independent_review_20260921.md`。

## 2026-09-06: provenance envelope 被误当作 shared identity（RESOLVED）

- 根因：producer 正确写入 17 字段 companion，但 benchmark 将完整 envelope 直接传给只应比较 8 字段 shared
  identity 的 typed validator，导致身份完全正确的正式 checkpoint 也被确定性拒绝。既有测试只传纯 8 字段，
  没有覆盖真实 producer→file loader→benchmark gate。
- 修复：完整 envelope 与 capability-aware identity projection 分层验证；expected identity 只从已验证 active
  Protocol/context/binding 链推导。保留 top-level/nested、nullable、bundle/order/Protocol/commit、context/binding、
  checkpoint SHA-256、Git、window、runtime/capacity、agent/seed、path/registry 的 fail-closed 检查。
- 剩余风险：本轮使用 test-only checkpoint 和受控输入，在环境 rollout 前终止；未执行 G14C v16 的正式训练或
  性能实验。Readiness v21 不是算法优势、TMC-ready 或 paper-ready 证据，holdout 仍 sealed/unopened/unconsumed。
  G14C v15 及更早 invalid 状态不变，G14C v16 只是启动授权暂缓而非运行失败。

## 2026-09-06: checkpoint nullable identity top-level omission（RESOLVED；G14C v15 永久 invalid）

- 根因：nullable hash 已进入 resolved/nested formal training contract，但 checkpoint 顶层 metadata 与 summary
  没有使用共享 projection；dev 前置 validator 未要求 nullable 字段，导致昂贵 dev 全部完成后才在 selection 拒绝。
- 修复：producer/read-back/pre-benchmark/pre-sort/freeze/typed provenance 共用 capability-aware identity 字段；active
  路径要求 top-level+nested+trusted expected 一致，旧缺字段 checkpoint 不回填、不获得 formal 资格。
- 历史边界：v15 为“training 和 dev evaluation 后、selection 发布前”失败，不得误写为 before-dev 或零执行；其
  150 train cells、24 dev cells、1,200 candidates 仅供审计，禁止 resume/retry/finalize/salvage/reuse。
- 剩余风险：验收使用 test-only checkpoint 与确定性合成 dev 输入，未执行 G14C v16 的 256-episode 正式训练或
  formal performance；Readiness v20 不是算法优势、TMC-ready 或 paper-ready 证据，holdout 仍 sealed/unopened。

## 2026-09-05: active nullable training resolver NameError（RESOLVED；G14C v14 永久 invalid）

- 根因：`resolve_training_contract(formal_protocol=...)` 的 active nullable 分支读取未定义 `protocol`；历史 v1.6
  测试未进入 `nullable_metric_contract_required=true`，而 compile/import 不执行该运行时路径。
- 修复与门禁：读取已验证的 `formal_protocol`，并要求 Readiness v19 消费 commit/bundle/nullable-bound 的
  150/150 production training-entrypoint 初始化证据；缺失、旧版本、跨 commit/bundle 或未进入 nullable 分支均拒绝。
- 历史边界：G14C v14 在 episode 0 前失败，所有有效训练与性能计数为0，不得 resume/retry/finalize/salvage/
  checkpoint reuse。Protocol 2.6 降为 historical/audit-only；唯一 live 版本为 Protocol 2.7。
- 剩余风险：G14R16 只验证训练初始化。完整正式训练、checkpoint、formal performance 与算法优势仍未执行；
  Readiness v19 不是 TMC-ready 或 paper-ready 证据，holdout 保持 sealed/unopened/unconsumed。

## 2026-09-05: formal cell publication/recovery/completion gaps（RESOLVED）

- G14R15 已关闭 formal support producer/outer consumer 路径分叉、freeze 后 registry 发布恢复，以及 gate=false
  仍可能被 phase 标记成功的执行合同缺口。Protocol 2.6 是唯一 live 版本；Protocol 2.5 保留为历史审计，不能
  resume/finalize 为新正式结果。
- Readiness v18 只证明非正式执行合同闭环，不是 formal performance、算法优势、TMC-ready 或 paper-ready
  证据；完整 G14C v14 训练矩阵、formal performance 和 sealed holdout 仍未执行。

## 2026-09-05: generated checkpoint downstream identity gap（RESOLVED）

- 根因：pre-run static registry 无法合法包含未来 checkpoint；cache wrapper 未消费/透传同一资源参数；support
  generator 缺少 capacity label 时回落 medium。三者均为执行合同问题，不是算法或数据结果问题。
- 修复：checkpoint-freeze committed terminal 后原子 create-only 发布 6-resource generated registry；所有真实
  downstream consumer 先验证 current-run/protocol/bundle/context/binding/path/hash/size/role/schema/capacity；
  capacity mapping 从生成器按 label 派生，outer/child 参数逐项相等。
- 剩余风险：仅完成 tiny non-formal rehearsal，不能替代 150 training/1,200 candidates/84 rows/12 clusters 的
  正式执行；Readiness v17 不是算法优势、formal gate、TMC-ready 或 paper-ready 证据。

## 2026-09-05: Protocol 2.3 nested preflight capability dispatch omission（RESOLVED）

- 根因：nested restart validator 的本地 resolved-context 版本集合停在 2.2，导致 active 2.3 即使收到 persisted
  context 参数仍回退 unresolved default context；这是 wrapper-level dispatch 错误，不是数据、依赖或算法错误。
- 修复：Protocol 2.4 由共享显式 registry 路由；只有 2.4 可 live execution，historical/unknown/default fallback/
  relative Python/cwd guessing/identity drift/holdout capability 均 fail-fast。
- 历史边界：G14C v13 是 pre-execution stop，不虚构 run/ledger/failure artifact，不登记 executed invalid run，
  不新增 checkpoint denylist。Protocol 2.3/Readiness v15 均 superseded/audit-only。
- 剩余风险：Readiness v16 没有正式性能证据；G14C v14 尚未运行，holdout 仍 sealed/unopened/unconsumed。

## 2026-08-29: v1.8 active resource schema misread（RESOLVED；G14C v8永久invalid）

- 根因：v1.8 index的唯一目录是`active_bundle_resources`，dev selector却仍直接读取旧顶层
  `runtime_configs`和`dev_fairness_manifests`，故第一条dev命令在任何performance row前抛出`KeyError`。
- 修复：v1.9 active consumer只接受`validate_active_formal_bundle()`结果，通过Resource Resolution Contract
  `1.0.0`解析单资源、group、三档capacity pair和support setting；missing/duplicate/extra/role/hash/size/path/
  symlink/scope/CLI mismatch全部fail-fast。nonformal与formal dev共用同一capacity resolver。
- 历史边界：G14C v8的150 training cells与1,200 candidates只保留失败事实，任何checkpoint/candidate/dev
  input均不可复用；v1.0–v1.8全部audit-only。
- 仍开放：Readiness v11不是正式performance或paper-ready证据；G14C v9必须未来另立任务从pushed clean
  Commit A10运行。Holdout仍sealed/unopened。

## 2026-08-28: v1.7 active index / Readiness / environment split（RESOLVED；v1.7 audit-only）

- 根因：v1.7 generator从v1.6 index深拷贝后遗漏`execution_environment_manifest`更新，并固定写pending；
  Readiness v9 finalizer未审计或冻结index。outer runner只验证调用者给出的Protocol/environment，能绕过错误index。
- 修复：Protocol v1.8以Active Formal Bundle Contract `1.0.0`绑定index、Protocol、environment、Scientific、
  Order、binding/context schema、portable/split/window/catalog/runtime/fairness、Readiness evidence、command matrix、
  execution commit rule和holdout seal。ready index最终SHA-256=`793f5106...38bd`。
- 门禁：outer runner在任何run-root写入前自动解析唯一index；CLI只可作一致性断言。pending/ready分叉、旧路径、
  content/fingerprint/commit/shared-resource/evidence/symlink/cwd漂移与dry-run bypass均fail-fast。
- 历史边界：v1.0–v1.7全部audit-only；G14C v1–v7 invalid roots/checkpoints继续拒绝。v1.7 Readiness v9只作
  历史不一致证据，不再授权执行。
- 仍开放：Readiness v10仅证明pre-execution合同，不包含正式checkpoint、formal performance或算法优势；
  G14C v8必须未来另立任务，holdout继续sealed/unopened。

## 2026-08-27: G14C v7 agent mapping-order identity drift（RESOLVED；run永久invalid）

- 根因：dev selector曾直接遍历`training_budget.agent_configs`，使sorted-key JSON mapping从
  `cache_offload_drl, controller_mat, ...`开始；这与fairness冻结的`sa_ghmappo, ppo, ...`顺序冲突，
  在任何dev performance前失败。
- 修复：Formal Agent Order Contract `1.0.0`是active Protocol v1.7的唯一顺序权威；Protocol、scientific
  config、fairness、commands、checkpoint、benchmark rows、statistics/Holm、claim/display及provenance均
  使用共享resolver和exact-order验证。Mapping insertion order与alphabetical sort明确禁止作为身份。
- 历史边界：G14C v7保留150个training cells和1,200 candidates的失败事实，但所有旧checkpoint/candidate/
  dev input/ledger/marker均不可复用。active execution、dev、freeze、benchmark均硬拒绝其run root引用。
- 仍开放：Readiness v9只证明Protocol v1.7执行合同与non-formal链路可执行；没有任何v1.7正式checkpoint、
  formal performance或holdout evidence，不能产生算法优势、G14完成或paper-ready结论。
- SEALED：holdout仍`sealed=true/opened=false/consumed_permanently=false`；G14R7未运行G14C v8/G14D/G15。

## 2026-08-25: G14C v6 legacy agent-config identity coupling（RESOLVED；run永久invalid）

- 根因：v1.5 train template仍消费 v1.3 `agent_training_configs.json`；该文件把稳定超参数与旧 Protocol
  semantic hash耦合，导致 active v1.5 在首个cell、episode 0前fail-fast。
- 修复：Protocol v1.6使用execution-neutral scientific config `2.0.0` 与runtime execution binding `1.0.0`；
  缺config/binding/context、跨Protocol/commit/environment/data/runtime/command配对、checkpoint identity缺失
  或content drift均在消费前拒绝。
- 历史边界：G14C v6 及 v1–v5 invalid roots/checkpoints继续硬拒绝；不得将 v6 已完成的preflight/tests复用
  为未来G14C v7证据。
- 仍开放：尚无任何Protocol v1.6正式checkpoint或performance evidence；Readiness v8不是formal或paper-ready。

## 2026-08-20: G14R2 closes G14C v2 window/ledger blockers

- `RESOLVED / source range`：formal mobility commands 现在必须显式传入冻结 prefix 11,850,526 raw rows；
  1500 默认值、不同 source path/size、selector/length/RSU/vehicle override 均在 episode/checkpoint 前失败。
- `RESOLVED / window identity`：60 个 evaluation units 绑定 segment、raw frame/time、provider offset、
  preprocessing、vehicle coverage、RSU mapper、source interval 与 fingerprint；training/benchmark 使用同一
  loader/fingerprint contract，60/60 实际 reachability 通过。
- `RESOLVED / ledger`：ledger `2.0.0` 使用 running+terminal append records 和 SHA-256 chain，补齐
  started/completed/wall-clock/return/retry/failure fields；G14C v2 return 1 被确定性分类为
  `data_window_unreachable`，不再误重试或需要事后改账本。
- `PERMANENT INVALID / G14C v2`：旧 run 0/150 training、0 checkpoint、0 formal，holdout 未开，禁止
  resume、覆盖或删除；Protocol v1.1 状态为 `invalid_before_performance_execution`。
- `OPEN / formal evidence`：Readiness v4 只证明 Protocol v1.2 execution contract 可执行；G14C v3 仍须
  未来从 Commit A3 clean worktree 另立任务。当前没有正式 checkpoint、formal raw result、statistics 或
  paper-ready evidence。
- `OPEN / full-source I/O`：精确 provider-offset 语义要求完整 11,850,526-row prefix；direct loader 只
  materialize 60 个目标窗口，但每个新的进程仍需扫描 2.12 GB CSV。后续 G14C v3 必须保留资源审计，
  不得因耗时删窗口、seed 或 baseline。
- `SEALED / holdout`：只完成 12 个窗口 identity/interval reachability；`sealed=true/opened=false/
  consumed_permanently=false`，没有 agent、episode 或 performance 访问。

## 2026-08-20: G14R closes G14C Phase-0 execution blockers

- `RESOLVED / B01`：checkpoint cadence 由 shared runner 真正消费；formal=4、legacy omission=1、resume
  mismatch fail-fast，selection 不包含每 update 的 `latest.pt`。
- `RESOLVED / B02`：SA frozen `auxiliary_coef=0.06` 经非 dirty 共享配置层传入并记录；其他 agent 隔离。
- `RESOLVED / B03`：metrics 1.2 生产 `full_service_ready_byte_hit_rate` 与
  `transfer_mb_per_request`，raw/summary/row/aggregate 对账完成。
- `RESOLVED / B04-B06`：support/scalability 数值或 explicit unavailable、typed support provenance、完整
  commands/dev freeze/integrity 与 append-only phase runner 均已实现。
- `OPEN / unavailable support claims`：object-size、transfer-cost、reuse、base-sharing 与四个 system
  scalability dimensions 缺少 fingerprint-safe runtime transformer，因此不能形成对应 claim；不得把
  `unavailable_pre_execution` 解读为零效应。
- `OPEN / formal evidence`：G14R 只有 E2 execution-contract 与 non-formal rehearsal，正式 checkpoint、
  formal raw results、statistics 仍不存在；必须另立 G14C v2 clean execution。
- `SEALED / holdout`：`sealed=true/opened=false/consumed_permanently=false`；普通 phase/support runner
  无 token 或 holdout capability。

## 2026-08-20: G14B protocol ready；正式执行与证据仍开放

- `RESOLVED / G14B split`：历史 frame/time/segment-run 排除账本和 24/12/12/12 split 已冻结；1,770 个 pairwise relations 全部 safe，minimum gap=24，formal/holdout outer count 均为12。
- `RESOLVED / protocol readiness`：G14A typed runtime 与 G14B agent/seed/budget/capacity/metrics/statistics/claim pre-registration 通过；readiness v2=`READY_FOR_G14C_CLEAN_TRAIN_AND_FORMAL`。
- `OPEN / G14C execution`：尚无 clean typed formal checkpoint、formal raw result、checkpoint selection record 或正式统计；必须从 Commit A clean worktree 另立任务执行，当前证据不能替代性能结果。
- `SEALED / holdout`：sealed holdout `opened=false`，一次性 token 未签发；只能在训练/eval/protocol/provenance/fairness/integrity/infrastructure gate 通过后开启，不能使用 formal performance gate。
- `OPEN / compute and I/O`：协议冻结每 learned agent/seed/capacity 最大12小时与总2,500 CPU-hours，但完整 2.12GB NGSIM 读取的实际 G14C wall-clock 尚未实测；不得因运行耗时事后删 seed、窗口或 baseline。
- `OPEN / realism`：NGSIM + Alibaba + controlled typed catalog 仍是跨源受控组合，不是真实联合 model-cache trace；latency saved 继续 unavailable，HF metadata/KV/G12 supervised predictor 不进入 formal。
- `BOUNDARY`：G14B formal episode=0、checkpoint=0、performance result=0、paper_ready=false；不得据此宣称算法优势或自动进入 G15。

## 2026-08-19: G14 typed MB formal pipeline blocked at Phase 0（历史状态）

- `RESOLVED BY G14A`：当时 formal benchmark/training/fairness/CacheEvent typed runtime plumbing 不完整；G14A 已补齐并保留原 blocked artifact 作为历史证据。
- `RESOLVED BY G14B`：当时新的 split 与全历史排除审计未冻结；G14B 已冻结独立 split 和 protocol，并重新执行 readiness v2。
- `HISTORICAL BOUNDARY`：该 `E0_UNAVAILABLE` 只描述 2026-08-19 审查时点，不覆盖 2026-08-20 G14B；历史 artifact 不删除或改写。

## 2026-08-19: G13 typed model-cache contract frozen; real workload/latency evidence remains open

- `RESOLVED / semantics`：vehicle base capability、RSU base resident、adapter resident与workflow-state migration payload现已分层；partial hit不再算full service。
- `RESOLVED / transaction`：base+adapter dependency bundle为single-plan atomic commit；oversized/no-victim rollback不改变resident/policy且orphan始终0。
- `RESOLVED / compatibility`：legacy profile仍默认；CacheEvent 1.0–1.2继续读取，typed-only metrics对旧trace为unavailable。
- `OPEN / real trace`：controlled catalog不是production model/adapter request trace；G11 joint VEC/adapter trace缺口未解决。
- `OPEN / latency`：仍无request-aligned load/inference/transfer/counterfactual latency，latency saved不可用。
- `OPEN / scale`：typed exact oracle只对小状态保证exact；state limit返回unknown，G14前需按冻结规模审计。
- `BLOCKED / HF formal`：qwen/cbow/bert license未知，BERT size provenance异常；diagnostic profile不得用于formal。
- `BOUNDARY`：workflow state是migration-only，不是长期共享cache；KV prefix仍disabled。G13不授权G14、训练或任何formal/holdout/hidden运行。

用途：记录当前有效问题、风险和禁止误读项。

## 2026-08-19: G12 predictor causality/calibration contract frozen; policy use remains blocked

- `RESOLVED / causal runtime`：snapshot现验证generated/as-of/consumed、history/source cutoff、age/expiry；delay读取历史对象，episode/window reset清空状态，oracle/supervised identity分离。
- `RESOLVED / calibration reuse`：原v112在dev选threshold并同split报告的问题不再用于G12；新binary temperature和abstention threshold仅fit/select于独立calibration段，不读取evaluation label或RL reward。
- `OPEN / multiclass calibration evidence`：历史quality rows没有next-RSU/target全量logits；真实multiclass Brier/NLL/ECE仍unavailable。handoff-positive eligible target hard top-1仅`0.001823`，旧all-row `0.950481`主要受no-target多数类口径影响。
- `OPEN / vehicle dependence`：frame/time跨split零冲突，但同一vehicle ID可在不同不相邻区间重现；当前明确记录相邻轨迹/group依赖风险，不把window ID当独立性证据。
- `OPEN / real acceptance`：最小NGSIM trace在threshold `0.95`下0/6 accepted；它证明fail-safe链路，不证明可用coverage或policy收益。需要新独立标签对齐材料才能报告real staleness drift/error。
- `BLOCKED / canonical policy`：v112 predictor-policy collapse仍有效；G12只提供默认关闭接口。未经G13独立协议、训练与新split验证，不得把supervised snapshot启用进canonical profile。

## 2026-08-19: G11 public model-cache registry frozen; joint-trace gap remains open

- `RESOLVED / semantic classification`：HF模型文件、内容cache、KV/prefix trace、generic AI workload与真实model request trace已由固定taxonomy和hard gate分离；rejected来源不能进入live projection。
- `OPEN / joint trace`：未发现公开来源同时观测vehicle/client、RSU/location、model/adapter/object、bytes、reuse/cache outcome、latency与mobility/handoff；任何NGSIM+外部request拼接只能称exogenous/synthetic alignment。
- `OPEN / adapter trace`：未发现公开可验证的真实LoRA/adapter请求序列；S-LoRA、DistServe、ServerlessLLM等artifact不提供可消费production adapter trace。
- `BLOCKER / license and provenance`：qwen/cbow/bert HF候选无明确license，BERT aggregate size另有provenance异常；Alibaba PAI data license未核验，均不得进入formal benchmark。
- `BOUNDARY / G12`：G11只冻结metadata registry和mapping plan；未下载payload、实现raw importer、校准predictor或启用新正式数据源。

## 2026-08-19: G10 information audit frozen; entity-level MARL remains unverified

- `RESOLVED / architecture naming`：源码级审计已冻结 PPO=single controller、MAPPO/SA-GHMAPPO=controller-level CTDE；多个controller head、参数共享或centralized critic不再视为vehicle/RSU-level actor证据。
- `RESOLVED / diagnostic leakage`：G10拒绝reward、aggregate、hidden、oracle future、事后service/cache outcome和不同request流拼接；固定bucket/projection/plug-in estimator不能按结果调节。
- `OPEN / decision trace`：现有 G07–G09 artifact缺逐request action前 observation trace，不能真实计算 recoverability、aliasing、NMI/CMI；禁止从CacheEvent或oracle future反推当前observation。
- `OPEN / sample independence`：真实validation只有1 request/1 evaluation unit；1×4H×5baseline的20行不提供20个独立样本，cross-RSU信息价值、实体级MARL必要性和GNN/GAT必要性均`UNVERIFIABLE`。
- `BLOCKER / entity MARL claim`：当前无两个真实entity actors、独立local observation、entity-owned concurrent/coupled actions；即使未来证明global information有用，也只支持centralized-information benefit，不自动支持entity-level MARL。
- `BOUNDARY / G11`：G10只冻结诊断与接口，不授权扩展observation/encoder/agent/reward/action、训练或执行G11。

## 2026-08-19: G09 analyzer contract frozen; evidence scope remains diagnostic

- `RESOLVED / request attribution contract`：G09现从matched external replay、exact oracle action trace和baseline raw request outcome重算机会；fingerprint/capacity/initial/H/objective/ID mismatch及taxonomy不守恒均fail-fast。
- `RESOLVED / censoring and taxonomy`：first request、right-censored tail、initial-cache natural hit、same-step admission、rolling baseline-hit/oracle-miss已分开；eviction-choice要求victim与后续request证据。
- `OPEN / sample coverage`：当前真实controlled artifact只有1 request、1 window；zero gap、100% capture、Gini null与small-sample warning不能证明五baseline接近真实上界或机会不集中。
- `OPEN / causality and latency`：reuse/frequency/topology/capacity分层和victim evidence均为diagnostic association；无逐request latency counterfactual，latency saved/gap继续unavailable。
- `BOUNDARY / G10`：information labels不证明MARL必要性或信息充分性；本轮未执行G10、训练、formal、holdout或hidden。

## 2026-08-18: G07 fairness manifest closes classical-baseline protocol drift; runtime scope remains bounded

- `RESOLVED / protocol identity`：五个reactive baseline现在由机器可验证manifest绑定相同dataset/window/request/seed/capacity/catalog/initial-cache/metrics合同，10组pairwise diff只允许eviction policy身份差异。
- `RESOLVED / silent override`：manifest runtime拒绝agents、seed、workflow/window、capacity、max-step、vehicle-selection和reward CLI漂移；未传manifest的legacy benchmark显式标记fairness unavailable。
- `OPEN / MB runtime consumer`：schema/validator已冻结slot与MB独立stratum，但`benchmark_main_results.py`当前G07消费者只执行`adapter_slots`；MB manifest在该runner fail-fast，不能宣称已完成MB runtime benchmark。
- `OPEN / exogenous long-horizon requests`：pre-run static DAG workload与同unit observed request stream均受审计；本轮controlled unit只有1 step。若长episode因cache outcome改变workflow推进，cross-baseline observed fingerprint会fail，必须先冻结外生request replay contract，不能绕过校验。
- `BOUNDARY`：G07不是G08 oracle、causal regret、latency-saved、formal或paper-ready evidence；hidden未消费。

## 2026-08-18: G05 unavailable/null aggregation regression resolved

- `FIXED / nullable aggregate`: benchmark aggregate 曾把 `None`/missing 通过通用 default-zero 聚合成 `0.0`，违反 G03 Cache Capacity Contract；现全 unavailable group 保持 JSON `null`，mixed group 只统计 available finite numeric values并记录 available/unavailable counts。
- `FIXED / consumer compatibility`: pairwise、win/tie/loss、robustness、checkpoint sweep 与 transaction summary 已能读取 nullable mean；不可用值不参与 delta 或 best-value 排名。
- `BOUNDARY`: 该修复不追溯改写历史 artifact，也不新增 G06 byte-hit/pollution/regret/latency-saved 指标；历史 capacity-disabled artifact 仍不能支持 cache-efficiency claim。

## 2026-08-14: cache observability contract resolved; metric/baseline work remains open

- `RESOLVED / event observability`: request-level `CacheEvent` v1 now records lookup, mutually exclusive hit source, admission, eviction victim, transfer/migration, capacity before/after and execution result in raw episode summaries.
- `OPEN / derived metrics`: byte hit, pollution, eviction regret and latency-saved metrics are not implemented in this task; v1 events only provide their future raw input contract.
- `OPEN / algorithm comparison`: byte capacity, capacity-matched LRU/LFU/FIFO/Random and future-horizon oracle remain absent. Capacity-disabled historical artifacts still cannot support cache-efficiency claims.

## 2026-08-11: planner distillation is not a stable native-policy improvement

- `FIXED / RNG fairness`: transition-ensemble initialization previously shifted the global PyTorch RNG before policy construction. Historical `no_learned_dynamics` retraining results are confounded and must not be cited causally. New enabled/disabled runs share identical initial policy parameters.
- `REJECTED / v118 full`: conservative planner distillation regressed RNG-aligned v100 raw reward on LuST by `-2.2450` and NGSIM by `-2.4081`; NGSIM BCa `[-6.0801,-0.2678]`. Planner execution masked most of the loss but did not improve v100.
- `REJECTED / early checkpoint`: v118 update8 improved only seed7 on LuST and catastrophically reversed on NGSIM. Fixed shorter training or checkpoint selection is not a valid recovery route.
- `REJECTED / v119-v121`: realized-GAE gating removed the negative tail but produced no material deterministic policy change. Stronger coefficients and logit-margin projection remained effectively tied to v100 in three-seed dual-domain probes.
- `BLOCKER / contract`: PPO and MAPPO currently share one global semantic observation and one environment action; MAPPO only factorizes the controller heads and centralizes the critic. A large generic MAPPO-over-PPO claim is structurally unsupported until a vehicle/RSU-level decentralized observation/action contract is frozen and benchmarked.
- `OPEN / selection bias`: NGSIM formal and LuST future outcomes were consumed repeatedly during v118-v121 development. These profiles cannot be promoted on the same splits; any new architecture requires an untouched frozen holdout.

## 2026-08-09: v100 reward lead confirmed on independent future split, mechanism dominance unresolved

- `RESOLVED / cross-split reward`: frozen v100 SA-GHMAPPO reached `33.342` versus Popularity `29.350` on the one-time v20 future-validation package; reward BCa CI was fully positive and SA won `75/15/0` paired comparisons.
- `OPEN / metric trade-off`: SA's mechanism realization (`0.600`) is above Popularity (`0.567`) on this split but not uniformly above all learned baselines; backhaul and migration costs remain trade-offs. Do not compress the result into “all metrics improved.”
- `OPEN / paper readiness`: one mobility/workflow combination, complete component ablations and unified compute accounting remain absent. The independent future split strengthens evidence but does not make the package TMC-ready.

## 2026-08-10: LuST support is positive but not an independent generalization claim

- `RESOLVED / external direction`: v100 wins on the available LuST support package (`34.200` vs Popularity `27.215`, MAPPO `28.946`).
- `OPEN / support power`: only 4 outer windows are available and the historical plan metadata is `outcome_blind_selection=false`; do not use this as primary cross-city evidence or pool it with the NGSIM future rows.
- `OPEN / workflow scope`: LuST support still uses Alibaba workflows; a second workflow source remains unavailable in the current data root.

## 2026-08-11: LuST independent split confirms reward lead but exposes regime trade-off

- `RESOLVED / cross-mobility reward`: the new outcome-blind 12-window LuST split gives SA `-25.638` versus Popularity `-32.961`, with BCa reward CI `[+2.324,+15.186]` and paired `24/48/0`.
- `OPEN / regime coverage`: all methods are negative in the aggregate; SA's lead concentrates in mechanism-activating windows (`31.218` vs Popularity `13.644`), while active/idle strata remain negative. The algorithm must not be described as uniformly robust across regimes.
- `OPEN / paper readiness`: component ablations and unified compute accounting remain incomplete even after cross-mobility evidence was added.

## 2026-08-11: inference ablation contract fixed and planner attribution recorded

- `FIXED / ablation reward protocol`: `scripts/benchmark_ablation.py` previously inherited a `+5.0` reward offset by omission; the invalid diagnostic is excluded, and the corrected runner defaults to explicit zero offset.
- `RESOLVED / planner attribution`: full v100 beats the no-online-planner inference variant by `+2.084722` with a positive BCa interval on all 12 LuST windows.
- `OPEN / training ablation`: this is an inference-side component ablation. Matched retraining ablations for every loss/constraint component are still required for a full causal training claim.

## 2026-08-09: v113-v117 native-policy internalization boundary

- `REJECTED / factorized target`: v113's joint-to-head marginal target is a valid MAPPO training loss but destabilized the event head and reduced reward; it is not enabled by the canonical v100 profile.
- `REJECTED / hard teacher distillation`: v114 generated exact model-improved labels with positive support, but its rollout behavior and native PPO returns were misaligned. Nonzero teacher support must not be interpreted as policy improvement.
- `REJECTED / training-only planner`: v115-v117 kept planner execution out of evaluation, but the native policy did not internalize the online planner advantage within the tested budget. Any future result must report training behavior and native evaluation separately.
- `OPEN / metric interpretation`: v100 formal uplift was `+0.3475`, while the one-time independent v20 future split produced `+3.992`; the larger cross-split reward gap is now established for the frozen checkpoint, but mechanism/cost metrics are not uniformly dominant. No reward-only selection or runtime wrapper may be used to manufacture an advantage.

## 2026-08-09: learned predictor integration remains blocked

- `BLOCKED / predictor-policy mismatch`: the supervised predictor has strong next-RSU classification but low handoff probability under the current class imbalance. Feeding its low-confidence targets into the v100 planner caused `mean_total_reward=16.366` and mechanism collapse on the dev probe. Do not enable `predictor_kind=supervised` in the canonical v100 run until a separately frozen calibration and policy-gating protocol is designed.
- `FIXED / variable RSU slots`: v71 windows expose different RSU slot counts. The predictor now accepts a runtime RSU subset of the checkpoint slot map and pads newly observed slots during train-sample construction. This is an input-contract fix, not evidence of algorithmic gain.
- `FIXED / threshold-selection complexity`: supervised predictor threshold selection previously rebuilt predictions for every candidate score, creating quadratic behavior on full mobility plans. It now uses one sorted cumulative scan.

## 2026-08-09: CAMA follow-up did not widen the reward gap

- `OPEN / candidate performance`: native CAMA v102/v103 remains below Popularity on the dev protocol (`18.4101`/`18.4093` vs `21.1033`). Do not claim that counterfactual head credit has already improved end-to-end reward.
- `OPEN / planner dependence`: v100's positive margin is produced by the agent-side online counterfactual planner; CAMA head distillation did not reproduce that margin when planner execution was disabled. Any future claim must report planner-enabled and native-policy ablations separately.
- `RESOLVED / negative stress test`: strong CAMA target distillation was tested and rejected after mean reward `-15.038` and continuity `0.256`. A non-triggered `collapse_detected=false` flag is not evidence of safe learning; the raw probe remains a negative boundary condition.
- `OPEN / paper evidence`: v101-v105 have no untouched independent hidden result, complete component ablation package or unified compute audit. They remain development evidence, not paper-ready proof.

## 2026-08-08: v100 paper-readiness risks remain

- `BLOCKER`: v100 has no untouched hidden holdout. The historical v71 hidden split was consumed by v98 and must not be reused for v100 tuning or claimed as v100 evidence.
- `BLOCKER`: only the NGSIM + Alibaba controller-level combination is covered. Cross-mobility, cross-workflow and larger system-scale validation are absent.
- `MAJOR`: formal mechanism realization, continuity, handoff-ready ratio and backhaul/migration outcomes tie Popularity; the reward margin is mainly delay-based and small (`+0.3475`).
- `MAJOR`: v100 and v98 formal reward rows are identical, so the v100 formal package is a replication/winner confirmation rather than proof of a new formal uplift.
- `MAJOR`: full noise sweep, complete component ablations and unified wall-clock/compute accounting are not yet available. Compact prediction support is support-only and must not replace these packages.

## 2026-08-08 v98 hidden 已消费后的剩余风险

- `RESOLVED / formal-hidden independence`: `scripts/audit_window_independence.py` 对 v71 formal 与 hidden 计划通过，20+20 windows 的 frame/time/segment intervals 均无重叠；hidden 已一次性消费，不能再用于调参或换 checkpoint。
- `RESOLVED / reward main claim`: v98 full formal 与 hidden 均相对 Popularity 为正，且 window-outer BCa CI 完全高于 0；结果不能解释为所有机制指标全面领先，因为 continuity、handoff-ready、mechanism realization 与 Popularity 持平。
- `RESOLVED / policy-improvement attribution`: v97 calibration-only 三 seed full formal 未超过 Popularity；v98 第二层 exact one-step policy improvement 后超过，v97 collapse seed29 未被筛除。
- `OPEN / support scope`: prediction robustness 已完成 5-window frozen support subset；未完成 full 20-window noise sweep、system robustness 和 scalability 的当前 v98 package，不能宣称完整 support suite 已闭环。
- `OPEN / compute audit`: 训练/评估包含 exact branch samples、UCC ensemble 和 online planner；当前 artifact 有训练 summary 与 update count，但尚未形成统一 wall-clock/peak-memory/branch-cost table。
- `OPEN / metric tradeoff`: v98 的 reward 胜出主要来自 total_reward；与 Popularity 的机制 realization、ready、continuity 持平，backhaul/migration 等指标必须随主表一起报告，不能只给 reward。

## 2026-08-08 v98 candidate risks

- `OPEN / multi-seed evidence`: v98 目前只有 seed7、48 episode 的 formal probe，不能与 v94/v70 的 multi-seed full result 等量齐观；必须完成 seeds `[7,13,29]`、256 episodes、formal aggregate 和 paired statistics。
- `OPEN / holdout freeze`: hidden holdout 尚未消费。v98 formal candidate、checkpoint manifest、stats protocol 和 claim 必须先冻结，再做一次性 holdout；不得根据 holdout 结果继续调参。
- `OPEN / compute cost`: v97/v98 每个训练 step 使用 five-action one-step branch calibration，训练 wall-clock 高于 v94；最终必须记录 branch transition count、runtime 和 checkpoint provenance，并做 compute-aware comparison。
- `OPEN / ablation`: v98 的增益尚未拆分为 UCC calibration、exact policy improvement、policy prior 和 no-model/no-calibration ablations；在补齐前不能声称每个组件独立显著贡献。

## 2026-08-08 v94 UCC-MAPPO 开放风险

- `OPEN / evidence pending`: UCC-MAPPO 的正式多 seed benchmark 尚未完成；当前不能把首个 seed、真实 smoke 或历史 v93 开发结果写成全量优胜结论。
- `OPEN / calibration`: ensemble uncertainty 在当前真实小跑中出现较高 validation error 和 clipped calibration scale；正式结果必须报告 calibration、no-uncertainty、no-policy-prior、no-model ablation，不能只报告 reward。
- `OPEN / contract scope`: 该 model 是 controller-level action-conditioned TD surrogate，不是 vehicle-level / RSU-level multi-agent world model；论文中不能扩大 MARL 或 digital-twin claim。
- `OPEN / protocol`: hidden holdout 仍 sealed；formal candidate、checkpoint manifest、statistics 和主 claim 冻结前不得开启 hidden，也不得用 formal 结果继续调参。
- `RESOLVED / data-loading`: baseline 长跑曾由纯 Python `csv.DictReader` 解析 5M-row NGSIM 造成首个 episode 长时间无输出；`NGSIMProvider` 现用 bounded-chunk pandas C-parser 并保留 fallback，已用 segment contract、真实 1500-row sample 和 5M-row read benchmark 验证。
- `RESOLVED / frozen-plan startup`: `train_algo_pool_real_sample.py` 在冻结训练计划存在时曾重复执行全量窗口扫描；现直接消费冻结 plan，与 SA 入口保持一致。低行数 smoke 若覆盖不到冻结 plan 的 frame offset 会明确报错，不能用缩短数据集伪造正式协议。
- `RESOLVED / evaluation checkpoint contract`: 首轮 v94 benchmark 的 SA evaluator 白名单漏掉 UCC ensemble/planner 配置，导致恢复时 planner 默认关闭；已补齐字段、补充 checkpoint compatibility regression test，并用真实 checkpoint 验证 model state/replay 恢复。首轮 aggregate 降级为 diagnostic，需用同一 frozen manifest 重跑 benchmark。
- `RESOLVED / learned action propagation`: evaluator 修复后发现 planner 的 `applied` 只写入 action metadata，未同步更新 `env.step()` 使用的 action；已修复并用 fake environment 验证 planned action 到达环境。此前 repaired-evaluator aggregate 仍只能作为诊断，需再次重跑。

## 2026-08-08: v93 开发集已领先但仍有训练预算与顶刊证据 blocker（OPEN）

- v93 在两个独立 seed 的同窗口 zero-offset full-stratified dev benchmark 中均高于 PPO、controller-level MAPPO 和 Popularity；seed13 为 `17.901250` vs `16.947000`，seed7 为 `18.068000` vs `16.947000`。这只能作为开发集事实，不能替代 formal/holdout 统计。
- 当前最大 blocker 是训练预算不对称：v93 seed13 checkpoint 仅 16 episodes，seed7 的独立训练在 episode19 / update2 后因 branch `deepcopy` 反事实计算过慢中止；PPO/MAPPO checkpoint 为 96 episodes。后续 paper comparison 必须统一 episode/update budget，或给出预注册的 compute-matched protocol。
- v93 仍未完成未消费 formal/hidden、window-outer hierarchical CI、paired sign test/Holm、prediction/system robustness、scalability、完整 ablation package、command log 和 artifact integrity manifest。按 `tmc_review_policy_v3_20260621` 当前 verdict 仍为 `Not TMC-ready` / evidence below `E2_ARTIFACT_AUDITED`。
- v93 机制目标已修复 delayed prefetch validation 和当前步 handoff alignment，但机制指标仍不是所有 strata 全面领先；完整结果必须同时报告 validated hit、handoff ready、continuity、backhaul 和失败率，不能只报告 total reward。
- v93 在线 planner 需要每个候选动作做 digital-twin branch replay，训练计算成本高。后续应先做不改变 objective 的 branch reuse / state snapshot 优化，并记录 wall-clock、branch count 和 peak memory；不能通过减少候选或缩短 rollout 后把结果称为等预算。

## 2026-07-29: v70 formal-min all-baseline reward winner 仍非 paper-ready（OPEN）

- v70 `top_journal_mechanism_v70_sparse_tail_option_mappo` 已在当前 formal-min mixed/full 全量 benchmark 中让 SA-GHMAPPO total reward 排名第一。full_stratified artifact 为 `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730/benchmarks/full_stratified_config_loaded/main_results_full_stratified_20260730_010523_184238/aggregate_summary.json`：SA `32.385729` > DT `31.426667` > popularity `29.969271` > PPO `27.301597` > MAPPO `16.261458`。
- 当前最关键正向证据是 full-only window-outer hierarchical statistics：`artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730/statistics/full_stratified_hierarchical/paired_statistics.json`。SA vs DT total reward delta 为 `+0.959063`，95% CI `[0.554171, 1.691468]`；SA vs popularity/PPO/MAPPO 的 reward CI 也均为正。
- 不得把 v70 写成 paper-ready / TMC-ready：本轮是 formal-min benchmark + statistics，缺独立 hidden/future holdout、support suite、artifact integrity/command-log package、ablation、prediction/system robustness、scalability 和完整 readiness audit，证据等级低于 `E2_ARTIFACT_AUDITED`。
- 不得声称所有系统指标全面优于规则。v70 相对 DT 的 reward 与 mechanism readiness/realization 为正，但 workflow continuity delta 微弱为负且 CI 触及 0；backhaul cost 相对 DT 略高，SA vs DT backhaul signed benefit 为 `-0.444444`，95% CI `[-1.515039, 0.0]`，Holm p=`0.07812`。相对 popularity 的 reward/continuity/readiness/realization 为正，但 backhaul cost 仍更高：raw delta `+1.5`。
- v70 的安全表述边界：提升来自 policy-side sparse-tail option prior 修复 `idle_or_sparse` option boundary，使该 strata reward 从 v67/v69 的 `27.1015` 提到 `30.7365` 并反超 DT `29.29275`；不是 reward shaping、environment/action schema、baseline contract 或 evaluator filtering。后续若冲击投稿，需要冻结未消费 holdout 并补 support/ablation，不得继续在已查看 formal-min 结果上反复调参后再把它当独立验证。

## 2026-07-27: v55 coverage-recovery dev evidence 与后处理审计风险（OPEN）

- v52/v53 已确认存在策略路径接入问题：net-advantage prepare gate 没有进入 `SAGHMAPPOBaseAgent` 覆盖后的 `_apply_policy_adjustments`。本轮已在真实 policy path 中修复，并用单元测试覆盖；历史 v52/v53 结果不得解释为 gate 已实际改善主策略。
- v54 虽修复 gate 路径并加入 service-completion gate，但 no-current-RSU 覆盖缺口仍大量选择 action2 vehicle fallback，导致主算法低于 MAPPO。v55 的 coverage-recovery MAPPO credit/option candidate/guard 在三组 full dev 训练中把 no-current action 稳定转为 `{4:1064}`，这是当前 reward 提升的主要机制证据。
- v55 仍不是 paper-ready：`artifacts/training/top_journal_v55_coverage_recovery_full_dev_summary/sa_ghmappo/` 已补齐三 seed `train_summary.json`，full-dev mean reward `9.102656`，高于同协议 PPO/MAPPO dev 对照，但缺 formal/holdout/support suite、窗口外层统计、完整 manifest/command log 和 paper-grade full checkpoint consistency audit。禁止写成“足以发论文”“TMC-ready”或“显著高于全部算法”。
- 训练脚本后处理风险已部分修复：`scripts/train_sa_ghmappo_real_sample.py` 新增 compact post-training audit，v55 默认先生成完整 `train_summary.json` 并把 compact scope 明确标为非 paper-grade；compact 审计不会修复 best checkpoint record，也不能作为 checkpoint family 的最终一致性证据。剩余风险仍然存在：正式 `E2` / paper-ready package 必须补跑 `--post_training_audit_mode full` 或独立 full checkpoint consistency audit，并生成完整 manifest / command log / formal-support package。

## 2026-07-27: v47--v51 policy-attribution blocker（OPEN；门禁已实现）

- `top_journal_mechanism_v47`--`v51` 目前只有 dev-stage single-seed training evidence，且各 update 的 deterministic evaluation 指标不变。v51 update 1--16 的 reward `15.358`、continuity `0.706148`、handoff-ready `0.45`、mechanism realization `0.525` 相同；`best_by_reward` 位于 update 1。不能把 checkpoint selection、physical-transfer feature 或 reward 数值解释为 MAPPO 学习改进。
- v51 diagnostics 的 `deterministic_event_prepare_rate_on_valid_target=1.0`、`event_margin_mean=5.998730`、`guard_action_delta_rate=0.684751` 表明当前推理路径混合了 learned logits 和 runtime policy interventions。任何主 claim 必须先比较 raw-policy 与 safety-projected-policy；环境 action mask 可以保留，heuristic logit bias/guard/option replacement 的贡献必须作为独立机制消融报告。
- Policy-Learning Gate 已实现：raw learned policy 与 safety-projected policy 分开评估，自动 checkpoint selection/consistency audit 固定使用 raw-policy metrics，且 raw action/metrics invariant 的 run 被拒绝选择。这只消除了“看不见策略是否改变”的测量缺口，不证明 v47--v51 有效。
- 在完成 multi-seed dev ablation 和新的 frozen formal/holdout 协议前，v47--v51 仍为 `Unverifiable`，不得晋级 canonical、paper-ready 或 TMC-ready。

## 2026-06-21: strict-full statistical blocker（RESOLVED）

- v7 的负向结论保持有效，但 v8 已按修复条件完成：四个 split 各 20 个互斥 outer windows、minimum gap 24 frames、5 seeds、window-outer hierarchical BCa/Holm、候选冻结后 formal 与一次性 hidden。
- v8 对 DT 的 full reward 与 continuity 在 formal/hidden 的 BCa 95% CI 均为正；原 strict-full blocker 标记 resolved，不再用 v7 legacy gate 充当修复证据。
- 修复不等于全面 TMC-ready。最新判定仍为 `Major revision`，详见 `top_journal_readiness_audit_20260621.md`。

## 2026-06-21: v8 system-tradeoff 与泛化缺口（OPEN）

- hidden 相对 PPO 的 handoff failure 显著更差；formal/hidden 相对 PPO 的 backhaul cost 显著更高。任何“failure-safe”或“降低回传开销”主张当前都会成为 blocker。
- formal/hidden 相对 popularity heuristic 的 reward CI 均跨 0；不能声称显著优于 strong heuristic。
- v8-current prediction robustness、system robustness、scalability 和逐机制消融已有统一入口，但尚未完成 full run 与 raw-row/statistics 审计；旧 v7 support suite 不能替代。
- LuST 修正 grid 只有 4 个独立 outer windows，低于 12-window 门槛；只能作为低功效辅助证据。
- hidden 已开启一次并永久 consumed。后续优化只能使用 dev 或新冻结 future validation split，不得再次读取现有 hidden 做候选选择。

## 当前限制

- 2026-07-22 v46 是当前 offset-free dev full-pool 最强候选，但仍不是 paper-ready。`top_journal_mechanism_v46_net_utility_constrained_mappo` 在 20 outer windows / 40 paired rows / seed7 / full_stratified 上得到 SA-GHMAPPO `38.791`，高于 PPO `38.3375` 与 `popularity_cache_heuristic=38.0`；相对 popularity 的 reward BCa CI 为 `[0.18, 2.239]`，但 Holm p=`0.210924`，相对 PPO 的 CI 为 `[-0.09125, 2.330993]` 且 Holm p=`1.0`。禁止把它写成显著高于 PPO、显著高于全部 baseline 或 TMC-ready。
- v46 的 `sa_advantage_diagnosis` 仍有 `backhaul_cost_above_popularity` blocker：SA backhaul `170.8` vs popularity `170.0`，mechanism realization 与 readiness 均与 popularity 持平，reward gap 主要来自少数窗口 reward / migration-overhead 侧收益。论文表述必须同时报告 backhaul、migration overhead、mechanism success gate 和 PPO trade-off，不能只报告 total reward。
- v42-v46 是 offset-free protocol 修复后的 dev-probe 线索：`reward_positive_offset=0.0` 已进入训练、benchmark 和 aggregate reporting，但只有 seed7 的 full-pool dev evidence。任何 formal/holdout/support 结论必须重新训练或冻结 manifest 后按 top-journal protocol 运行，不得把 v46 dev statistics 当 final package。
- 2026-07-21 v39/v40/v41 dev-probe 不能证明“主算法高于全部算法”。当前最强 SA full-pool 复核是 v39 update_0005：SA-GHMAPPO `106.041`，高于 MAPPO `105.5875`、popularity `105.25` 和 PPO `94.77375`，但低于 `cache_offload_drl=119.14875` 与 `dt_handoff_drl=119.22625`。v41 conservative recovery 为 `105.686`，同样只略高于 MAPPO/popularity，未超过 DT/cache。禁止将 v39/v41 写成 all-baseline winner 或 paper-ready 结果。
- 历史 v39/v41 的 `VecWorkflowCoreEnv.reward_positive_offset=5.0` 是已确认 artifact risk：它按 step 累加，导致未完成但拖到更长 horizon 的策略可能比更快完成 workflow 的策略得到更高 `total_reward`。v42+ 已用 `reward_positive_offset=0.0` 修正该 ranking 协议；旧 v39/v41 只能作为目标不一致诊断，不能再作为论文主 reward ranking。
- `top_journal_mechanism_v40_advantage_weighted_behavior_mappo` 是负向算法探索：train-window update 5/6 reward 很高，但 frozen dev targeted benchmark 只有 `85.788`，说明 positive-deviation advantage-weighted behavior cloning 过拟合窗口行为。它只能作为 AWR/AWAC-style behavior regularization 的失败消融，不能晋级。
- `top_journal_mechanism_v41_conservative_recovery_mappo` 修复了 v40 的严重 dev collapse，但没有扩大 MAPPO gap：SA `105.686` vs MAPPO `105.5875`，差距只有 `+0.0985`；对 popularity 差距 `+0.436`，对 DT 差距 `-13.54025`。它可以作为 conservative recovery ablation，不是论文主候选。
- checkpoint audit best-source fallback 已修复，但历史 v40/v41 修复前生成的 `best_by_reward` 记录需要看 `source_checkpoint_path` 和 `expected_best_sources`，不要仅凭 `best_by_reward.pt` 文件名判断是否真来自最佳 update。正式重跑前应保留 `tests/test_sa_checkpoint_repair.py` 覆盖。
- 2026-07-19 v23-v26 reward-gap 扩大实验没有形成比 v22 更强的主结果。v23 counterfactual constrained PRD 在 dev 上低于 popularity；v24 tail-risk PRD 只取得 dev `+0.195`、formal `+0.05625` 的 popularity reward delta 且 CI 跨 0；v25 opportunity PRD 退化到 dev `+0.100`；v26 safe-counterfactual PRD 为 dev `+0.16425`，仍低于 v22 `+0.268`。这些 profile 只能作为负向探索和机制诊断，不能写成 canonical 晋级或 paper-ready 改进。
- 当前 strict/full dev 和 time-audited formal 中，`popularity_cache_heuristic` 是非常强的 supplementary rule reference；formal split 上 SA、popularity、PPO、MAPPO 的 `mechanism_realization_rate` 均为 `0.0`，因此很难从 handoff/cache mechanism 上拉开大 reward gap。若继续要求显著高于 popularity，需要新冻结更能覆盖 validated mechanism opportunities 且未消费的 formal/hidden split，不能在已查看 formal 上继续筛 profile。
- v23-v26 的失败说明“增加机制尝试”不是单调有效：v23 会产生 failed mechanism tail loss，v25/v26 降低 validated mechanism realization 或压低 v22 的正窗口收益。后续算法设计必须先提供可检验的一阶机制假设，例如可学习 counterfactual evaluator、offline safe policy improvement 或更强 predictor，而不是单纯调大 PRD/auxiliary 系数。
- `top_journal_mechanism_v21_efficiency_prd` 与 `top_journal_mechanism_v22_validated_utility_prd` 未能形成 paper-ready stronger-heuristic 优势。v21 dev 对 popularity reward delta 为 `+0.29125`，BCa CI `[0.01425, 0.811673]`，但 Holm p=`0.0789`；formal delta 只有 `+0.17275`，CI 跨 0，且仍有 backhaul 和 mechanism-attempt-without-validated-success blocker。v22 dev 提升了 mechanism realization（delta `+0.05`，Holm p=`0.046872`），但 reward delta 降至 `+0.268` 且 Holm p=`1.0`；formal reward delta 降至 `+0.10025`，blocker 未消除。hidden holdout 未开启，不能把 v21/v22 写成论文主 claim。
- v20/v21/v22 formal 诊断已经使用 `configs/experiment/top_journal_v20_formal_time_audited_20260717/future_validation_window_plan.json` 反馈算法设计，因此该 formal split 不能再作为未调参消费的最终 holdout。若后续候选看起来达标，必须重新冻结未消费的 formal/hidden 或只在预先冻结、未读取的 hidden 上做一次最终验证，并清楚记录开启条件。
- v20 formal/hidden 诊断 split 在排除历史窗口后没有可用 idle/sparse 窗口，只包含 10 mechanism / 10 active non-mechanism。它适合暴露 mechanism/active-heavy blocker，但不能替代 balanced strict split，也不能证明 idle/sparse 泛化。
- `top_journal_mechanism_v20_idle_execution_prd` 是当前最强算法候选：在 frozen dev 上 SA-GHMAPPO reward `79.7195` > popularity `79.46875`，在 time-audited v20 future-validation 上 reward `67.561867` > popularity `65.754`、PPO `65.866133`、MAPPO `64.0888` 和全部其他对照；相对 popularity 的 future reward delta `+1.807867`，BCa 95% CI `[0.373706, 3.793892]`，Holm sign-test p=`0.047208`。但它仍不能写成 TMC-ready / paper-ready：future split 只有 15 个 outer windows 且 `minimum_gap_frames=0`，还缺新 formal/hidden/support suite，训练 summary 仍有 collapse flags，future 上 `mechanism_realization_rate` 低于 popularity、adapter migration overhead 略高。
- v20 的收益来自 learning-side idle-execution partial-reward-decoupled MAPPO credit：改动 PPO/MAPPO actor advantage 与 option-gate advantage，让低风险 idle/current-RSU delay 与 local fallback 的 credit 分离。不得写成环境 reward shaping、evaluation wrapper、baseline 削弱或 window 筛选收益；也不得声称每个机制指标都全面优于 heuristic。
- `top_journal_mechanism_v19_handoff_risk_prd` 是 v17 之后的 handoff-risk PRD 中间候选，frozen dev reward `79.69925` 高于 popularity 但低于 v17 `79.70825` 和 v20 `79.7195`，当前不作为主候选。它的 handoff-risk credit / dual-cost 逻辑只作为 v20 的组成和后续风险约束依据。
- `top_journal_mechanism_v18_counterfactual_option` 是一次算法性 counterfactual option-credit MAPPO 尝试，但 frozen dev 结果低于 v17，且出现 continuity / handoff failure / mechanism readiness blocker；当前不得把 v18 写成主算法改进成功，只能作为负向探索和后续 credit-assignment 依据。
- `top_journal_mechanism_v17_dag_aware_option` 在 time-audited future-validation 上均值仍为第一，但相对 `popularity_cache_heuristic` 的 reward margin 只有 `+0.04815`，BCa 95% CI `[-0.396869, 0.636962]` 且 Holm sign-test p=`1.0`；不能声称显著优于 strong heuristic，不能据此判为 TMC-ready candidate。
- future-validation split 必须使用 `future_validation_split_v2_time_audited_20260717` 或后续更严格版本；只按 `frame_offset` 审计的 split 可能漏掉 `time_index_start/end` 重叠。任何旧 `top_journal_v17_future_validation_20260717` 结果不得作为 independent holdout / future evidence。
- `top_journal_mechanism_v17_dag_aware_option` 已在 frozen dev full_stratified 上完成 5-seed / 20-window / 2-workflow / 12-agent 全量 benchmark，并把 strongest-other reward margin 提升到 `+0.2395`，同时清除 v13/v16 的 `backhaul_cost_above_popularity` blocker；这仍不是 hidden/future-validation 证据。当前 hidden 已 consumed，v17 不得用现有 hidden 做进一步筛选；promotion 必须新冻结 future-validation split 并按 top-journal review policy 重新审查。
- v17 改进来自 policy-side DAG-aware MAPPO option termination，不是环境 reward、action contract、baseline contract、window plan 或评估包装改动。论文或汇报中可以说 v17 是当前 dev 主候选，但不能声称已 TMC-ready、全面优于 PPO 的所有系统指标，或已经通过独立 holdout。
- v17 的 mechanism realization `0.195` 低于 v16 `0.265`，说明 DAG-aware gate 用更保守的机制动作换取 reward / backhaul trade-off；不能把它写成“机制触发越多越好”，必须同时报告 validated success、continuity、backhaul 和 total reward。
- `top_journal_mechanism_v13_prd_option` latest 已在 frozen dev full_stratified 上完成 5-seed / 20-window / 2-workflow / 12-agent 全量 benchmark，并把 strongest-other reward margin 从 v12 `+0.12465` 扩大到 `+0.17590`；这仍不是 hidden/future-validation 证据。当前 hidden 已 consumed，v13 不得用现有 hidden 做进一步筛选；promotion 必须新冻结 future-validation split 并按 top-journal review policy 重新审查。
- v13 的 `best_by_reward` checkpoint 会停留在 warm-start update 0，不能代表 PRD 学习后的策略；本轮正向结果来自 `latest_checkpoint_path`，必须在论文或汇报中如实说明 checkpoint policy。不得把 latest-after-training 包装成 hidden-validated 或 reward-oracle selection。
- v13 改进来自 policy-side partial-reward-decoupled MAPPO event/option credit，不是环境 reward、action contract、baseline contract 或 window plan 改动。PPO 在 handoff failure/backhaul trade-off 上仍需单独报告，不能因为 reward margin 扩大而声称系统指标全面优于 PPO 或 heuristic。
- `top_journal_mechanism_v12_learned_option` 已在 frozen dev full_stratified 上完成 5-seed / 20-window / 2-workflow 全量 benchmark，并超过 `popularity_cache_heuristic` 和全部 learned baselines；这仍不是 hidden/future-validation 证据。当前 hidden 已 consumed，v12 不得用现有 hidden 做进一步筛选；promotion 必须新冻结 future-validation split 并按 top-journal review policy 重新审查。
- v12 的 full-dev 胜出来自 regime-aware 组合：mechanism window SA `82.758` > popularity `82.3425`，active non-mechanism 与 popularity/PPO 持平，idle/sparse 与 popularity 持平。不能声称每个 window class 或每个系统指标都全面优于规则；PPO 在 handoff failure `0.02` 和 backhaul `100.64` 上仍优于 v12 的 `0.075` / `110.72`。
- v12 的 learned option gate 是 policy-side MAPPO option head + contextual prior，不是环境 reward 改动、action contract 改动或 baseline contract 改动。若论文要作为算法创新表述，必须说明 `window_class` 的 outcome-blind 来源、mechanism window preserve-MAPPO 规则、option loss/prior 的训练角色，以及和 v11 evaluator-side hard gate 的区别。
- `top_journal_mechanism_v11_mappo_reward` 已在 frozen dev full_stratified 上超过 `popularity_cache_heuristic` 和全部 learned baselines，但这不是 hidden/future-validation 证据。当前 hidden 已 consumed，v11 不得用现有 hidden 做进一步筛选；promotion 必须新冻结 future-validation split 并按 top-journal review policy 重新审查。
- v11 的 full-dev 胜出不是所有 window class 全面胜出：机制窗口 SA `82.788` > popularity `82.3425`，active non-mechanism 持平，但 idle/sparse 仍为 SA `77.2175` < popularity `77.3975`。论文或汇报只能说总体 full-dev reward 过线，不能声称 idle/sparse 已彻底优于规则。
- v11 的 window-context no-RSU local fallback 由 outcome-blind `window_class=idle_or_sparse` gate 触发；它是推理期 regime-aware safety option，不是环境 reward 改动，也不是 learned predictor。若论文要将其作为算法创新，需要把 window class 的可观测来源、非 reward 选择边界和对 baselines 的公平性说明清楚。
- `top_journal_mechanism_v9_pareto_safe` 目前只是 dev/future-validation 安全候选 profile 和 checkpoint-ranking 入口；在完成 5-seed train/dev、learned-baseline 同窗口比较、future-validation split 互斥审计和新 readiness audit 前，不能替换 v8 canonical，也不能声称已解决 handoff failure / backhaul blocker。
- v9 的 `best_by_pareto_safe_score.pt` 是 checkpoint selection heuristic，不是新 reward function 或环境约束；论文必须把 reward、DT continuity、handoff failure 和 backhaul non-inferiority 分开报告，不能把 safety guard 收益写成纯 learned policy 收益。
- `top_journal_mechanism_v10_mappo_rl` 目前只是把 MAPPO controller-level credit / entropy floor 迁入 SA-GHMAPPO 的 RL 候选 profile；v11 dev result 才是本轮 reward-first follow-up。若未完成 future-validation、window-class gap、learned-baseline、failure/backhaul non-inferiority 与新 readiness audit，不得声称 MAPPO 路线已 paper-ready 修复 popularity gap 或系统 trade-off。
- 当前 `sa_ghmappo` 预测层默认仍是 `baseline_predictor_v2`。代码已新增 `predictor_kind=supervised` 和 `supervised_handoff_predictor_v1` checkpoint runtime，但在正式冻结 checkpoint、quality report、SA-GHMAPPO v9 重训和 formal/future-validation benchmark 前，不能把当前主结果写成已经使用 learned predictor。`predictor_kind=learned_or_calibrated` 仍只表示 calibrated baseline surrogate interface。
- `supervised_handoff_predictor_v1` 的安全定位是短时 next-RSU / handoff-target / ETA anticipation；不得写成完整 digital twin、轨迹预测 SOTA 或独立解决连续 cache 的核心算法。
- 当前 action contract 仍是 `semantic_discrete_5`，DAG graph encoder 与 DAG pressure diagnostics 已接入，但环境动作不选择 DAG frontier / target node；不能声明 DAG-level parameterized decision，除非后续冻结 `action_type + target_node + target_rsu/adapter` contract。
- `mechanism_exploration_bonus` 已标记为 shaping/diagnostic，但历史 reward 字段仍存在；正式机制收益必须优先用 validated prefetch hit、realized prepare、handoff ready、continuity 和 mechanism success gate，避免把 prepare/prefetch 尝试次数解释为机制兑现。
- `action_mask_info`、`ControlAction.metadata.invalid_reason`、action projection 和 guard delta 已进入新链路；历史 artifacts 没有这些字段，跨版本比较时必须显式标注 protocol version 或缺失字段。
- 旧 `final_submission_controller_mappo_qmix_20260509_v1` 中 `mappo` 是 pre-head-credit MAPPO。该结果里 `mappo` 在 continuity / handoff / backhaul 上更保守，但 total reward 弱于 `ppo`，作为“主算法基于 MAPPO 增强”的顶刊主表存在审稿风险；新的 MAPPO claim 必须改用 controller-level CTDE + `aggregation_reason_weighted_controller_ppo_v3` MAPPO 重跑。
- `paper_claim_summary.json` 中部分中文说明存在历史编码乱码；正式记录以 `docs/project/ARTIFACT_RECORDS.md` 的整理版为准。
- `smoke_run` 和 early toy benchmark 不能用于论文结论。
- `LuST` 场景仍保留 provider 价值，但当前不作为 `NGSIM + Alibaba` 主线的阻塞项或正式结论来源。
- 部分 ablation 记录使用早期 baseline checkpoint，适合作为机制对照，不适合单独声明最终 SOTA 结论。
- robustness 最新保留记录早于主表 frozen rerun，应该作为辅助压力测试，不应压过 frozen main table。
- 历史混合 aggregate 可能仍包含已删除算法记录，只能作为归档快照；当前 live 论文表格需要重新生成主方法单算法结果。
- `td3` / `sac` / `maddpg` 仍未进入当前 live registry；当前动作空间是 `semantic_discrete_5`，不应强行改写为纯连续控制实验。
- `mappo` 当前是 controller-level CTDE baseline：flat semantic encoder + cache / execution-offload / handoff-event controller actors + centralized flat semantic critic，并启用 `aggregation_reason_weighted_controller_ppo_v3`。它可以进入当前 paper-grade learned baseline gate，但 checkpoint 必须通过 `baseline_protocol_versions.mappo` 审计；pre-v3/pre-head-credit MAPPO 只能作为历史归档。它不是 vehicle-agent / RSU-agent full MAPPO；若论文声称 full multi-agent MAPPO，仍需 future multi-agent wrapper/action contract。`flat_mappo` 只表示历史 artifact run 名称，不再是 live agent 名称。
- `qmix` 当前是 controller-level value-decomposition baseline：三 controller Q heads + centralized monotonic mixer。它可以进入当前 paper-grade learned baseline gate，但不是 vehicle-agent / RSU-agent full QMIX；若论文声称 full multi-agent QMIX，仍需 future multi-agent wrapper/action contract。
- `controller_mat` 当前是 controller-level MAT-style transformer baseline：三 controller tokens + centralized transformer critic。它可以进入后续 paper-grade learned baseline gate，但不是 vehicle-agent / RSU-agent full MAT；`final_submission_controller_mappo_qmix_20260509_v1` 尚未包含它，不能把该旧 package 写成含 Controller-MAT 的最终对比。
- `dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl` 是围绕主线新增的领域专项 learned baselines：分别覆盖 DAG offloading、model/adapter cache offloading 和 Digital Twin handoff/service migration。它们按当前 controller-level `semantic_discrete_5` contract 实现，不是 full vehicle-agent/RSU-agent wrappers，也不使用 SA-GHMAPPO 的 graph message passing、calibrated surrogate gate、uncertainty-aware event scaling、mechanism auxiliary loss、heuristic imitation 或 policy guards。旧 `final_submission_controller_mappo_qmix_20260509_v1` 不含这些新增 baseline，不能把该旧 package 写成已覆盖 DAG/cache/DT 领域对照。
- `reactive_greedy` 和 `popularity_cache_heuristic` 是非学习 heuristic baseline，只用于提供规则对照，不应解释为 RL 训练结果。
- Hugging Face `model-cache` 候选全集当前只是审计、metadata 和 file-size reference；在实现文件级 importer、adapter 映射和独立 benchmark profile 前，不能声称 benchmark cache events 直接采样自这些数据集。
- formal v2 支撑实验已补齐 current-contract ablation；但 `no_dag_dependency_aware` 的 reward CI 跨 0，`no_uncertainty_signal` 不体现独立 reward 正贡献。论文中不能把这两项写成单独显著 reward 来源，只能作为机制设计组成或辅助稳定性因素谨慎描述。
- `reactive_greedy` 和 `popularity_cache_heuristic` 已降级为 supplementary heuristic reference；顶刊主 claim 应优先引用 canonical clean-retrain final comparison package，不要再把手写规则当主对照或 gate 阻塞项。旧 `top_journal_learned_baseline_formal_20260505_v1` 未覆盖当前去重后的 final-submission 口径，只能作为旧 baseline set 记录。
- DQN-family learned baselines（`dqn`、`ddqn`、`dueling_dqn`、`dueling_ddqn`）只适配当前 `semantic_discrete_5` 动作 contract；它们不能替代 TD3/SAC/MADDPG 这类连续控制 baseline。连续控制类 baseline 仍需先改变或扩展动作 contract。
- `train_sa_ghmappo_real_sample.py` 默认不再全量审计 `update_*.pt` 中间 checkpoint；如需复现完整 checkpoint consistency audit，必须显式加 `--audit_update_checkpoints`，并预期可能遇到损坏中间 checkpoint 需要容错记录。
- `top_journal_mechanism_v2` 和 clean retrain `top_journal_mechanism_v3` 虽然 learned-baseline gate 通过，但相对 supplementary `popularity_cache_heuristic` 未形成稳定优势，不能替代 formal_v2 主结果。
- `top_journal_mechanism_v3_eval_bias` 已补齐 formal/holdout 主表、latency fallback 消融、robustness 和 scalability；但它仍是在 formal_v2 权重上启用 inference calibration，不是 clean retrain，论文中必须如实说明。
- `top_journal_mechanism_v3_eval_bias` 的 prediction robustness 不满足“全面优于 heuristic upper-bound”：四类 prediction setting 汇总 SA `89.927917` 低于 popularity `90.94375`，主要由 `oracle_prediction` setting 拖累；不能写成 oracle 条件也全面领先。
- `top_journal_mechanism_v4_prepare_eval_bias` 是负向筛选结果；predictive prepare hard override 没有修复 oracle setting，反而使 prediction robustness 总 reward 低于 v3，不应推广或写入主结果。
- 当前 pre-Controller-MAT canonical `final_submission_controller_mappo_qmix_20260509_v1/comparison_report/paper_ready/paper_ready_report.md` 的自审没有 blocker，但有 4 个必须随论文主张保留的限制：`popularity_cache_heuristic` 与 SA-GHMAPPO reward 很接近；`no_prediction` / `oracle_prediction` 不支持全面预测条件优势；`mechanism_realization_rate` 不是每个 split 上的独立正向优势；holdout backhaul 对 PPO 不具备正 CI。新增 `controller_mat` 后需要重跑 final-submission loop 才能升级 canonical。
- `final_submission_clean_retrain_repaired_baselines_20260507_v1` 是 pre-MAPPO/QMIX-controller-level historical package；所有 legacy canonical 标签均受 2026-06-18 strict window audit 结论约束。

## 当前风险

- G08 遵循当前环境“cache action 先于 service lookup”的同 step admission-hit语义；短 replay 可能让baseline与oracle都达到100% hit并出现零gap。这是合同事实，不代表算法已接近常见pre-request cache oracle上界。若未来改成lookup-before-admission，必须升级major contract并重跑比较。
- `adapter_slots` 中每个incoming adapter只占1 slot，且valid initial state不超容量，因此一次admission最多需要1个victim；“slot multi-victim”在当前合同下结构性not applicable。multi-victim有效验证位于MB heterogeneous-size stratum。
- G08 solver支持MB不代表 `benchmark_main_results.py` 已完成五baseline的MB runtime公平比较；G07 runner仍只消费slot stratum。
- G08 validation只有controlled单request unit，不能支持oracle gap大小、算法接近上界、causal regret或latency gain结论。request-level observed/counterfactual latency仍缺失。

- 2026-06-21 最近邻审查确认：TMC 2026 已有 DAG timing/data dependency + MADDPG，以及 mobility-aware parallel-task cross-RSU collaborative offloading；IoT Journal 2025 已有 dependency-aware hierarchical VEC offloading。论文不能把 `DAG + mobility + MARL`、`DAG + hierarchy` 或 graph-assisted VEC offloading 单独写成 novelty。
- `Dual Dependency-Aware Collaborative Service Caching and Task Offloading in VEC` 已覆盖 DAG/task dependency、service dependency、hierarchical cache 和 PPO；若 PPO_MEC 缺少 adapter size/load/warm/migration latency 或 serving-profile 证据，adapter cache 容易被审稿人视为 service cache 重命名。
- 当前可守 novelty 是完整联合 contract，而非单组件：跨 RSU continuous workflow state、adapter warm-state lifecycle、predictive handoff preparation/state migration 和 cache/execution/event 三时间尺度控制。任一元素拆开都已有强近邻，相关绝对首次表述会形成 novelty blocker。

- 2026-06-18 rebuild 已达到 `E3_REPRODUCED`，但旧 `window_rank_offset=3` formal/holdout 时间区间重叠；历史 offset holdout 只能作为 near-window sensitivity，不能作为 independent holdout。
- v7 严格非重叠协议下，SA 对 `dt_handoff_drl` 的 formal-full 与 holdout-full total-reward 95% CI 均跨 0；该历史 blocker 已由 v8 冻结 formal/hidden 修复，但不得反向把 legacy v7 gate 写成有效 strict evidence。
- mixed/full 会复用部分窗口，必须按 mode 分开报告，不能把 mode 当独立 cluster 合并扩大样本量。
- `no_prediction` 与 `no_adapter_prefetch` 消融高度耦合，不能解释为正交因果贡献或将 delta 相加。
- LuST grid 外部迁移的 reward 对 learned baselines 为正，但 backhaul 高于 popularity heuristic；不得声称全面改善系统指标。
- paired comparisons 尚未做 family-wise/FDR 校正，论文必须说明 multiplicity 策略。
- 若继续删除 artifacts，需要先确认对应路径没有被 `ARTIFACT_RECORDS.md` 的保留记录引用。
- 若重新生成主表，必须同步更新 `paper_main_table.json`、`paper_claim_summary.json` 和本目录下的整理记录。
- 若更换 checkpoint，必须同步检查 benchmark 消费端、manifest 和训练审计字段。
- 当前 reward 中 `mechanism_exploration_bonus` 会奖励预测 handoff 信号下的 prepare/prefetch 选择，未区分 prepare 是否最终成功；分析 heuristic reference 时需要同时报告 `migration_success_count`、`migration_failed_count`、handoff ready 和 continuity，避免把失败 prepare 尝试误读成真实系统收益。
- `--window_rank_offset` 只能用于 ranked-window sensitivity；独立 holdout 必须同时使用 interval exclusion、非重叠选择和 `scripts/audit_window_independence.py` 校验。
- 根目录下少数 `pytest-cache-files-*` 临时目录在本次清洗中被系统权限锁定，内容无法枚举；它们不属于项目 live 逻辑，但仍需在句柄释放后删除。
- 若后续启用 MADDPG 或 full vehicle/RSU-level QMIX，需要先冻结 multi-agent observation/action schema，再接入训练和 benchmark；当前 `qmix` 仅是 controller-level value-decomposition baseline。

## 已清理或不再阻塞

- 通用模板目录不再作为 live 文档入口。
- 旧阶段文档不再作为事实来源。
- toy / tmp / quickcheck / 单次 dry-run 产物不再参与当前结论。

# 2026-08-14: cache observability / baseline / oracle blockers

- `BLOCKER / baseline identity`: live registry 没有独立 LRU、LFU、FIFO、Random cache agents；环境中的 LRU 仅是 capacity-enabled admission 时的 eviction primitive，不能当成已评估 baseline。
- `BLOCKER / capacity protocol`: v100 formal/future 主 rows 的 cache capacity 未启用，aggregate occupancy/eviction 为 0；这些 artifacts 不支持 cache pollution、turnover、eviction regret、byte efficiency 或容量受限 placement claim。
- `BLOCKER / request observability`: 当前 raw rows 没有 request-level object bytes、hit source（local/neighbor/cloud）、delay decomposition、future reuse 与 counterfactual cloud latency，无法计算 byte hit、P95/P99、useful-cache ratio 或 latency-saved-per-MB。
- `OPEN / cache oracle`: `oracle_prediction` 不是 future-request capacity oracle；尚未建立固定 horizon、相同容量/transfer cost 下的 offline optimal placement/eviction upper bound。
- 完整审计与 P0--P3 路线见 `docs/project/full_system_cache_algorithm_audit_20260814.md`。
# 2026-08-17 MB capacity implementation status

- `RESOLVED / live capacity semantics`: 环境已支持 slot/MB、initial enforcement、原子 multi-victim 与 oversized 拒绝。这不追溯改变 v100 capacity-disabled artifacts，也未解决独立 cache baseline、正式 byte metrics 或 oracle blocker。

# 2026-08-18 Classical baseline boundary

- `RESOLVED`: FIFO/LFU/Aging-LFU/Random、明确 baseline identity 与 agent-policy mismatch 检查已实现。
- `OPEN`: 正式 byte-hit/pollution/regret/latency-saved、capacity oracle 与 paper fairness manifest 不属于本 Goal；controlled validation 不能替代 formal evidence。
## 2026-08-18: G06 cache-efficiency contract frozen; causal latency/oracle remain unavailable

- `RESOLVED / derived metrics`: request/byte efficiency、lifecycle/churn/transfer、slot/MB capacity、operational pollution + right censoring 与 fixed-horizon future-reuse proxy 可从 raw CacheEvent 1.2 + trace context 独立重算。
- `RESOLVED / multi-victim and initial state`: optional per-victim MB、admission identity/MB 与 initial/final per-RSU snapshots 消除 multi-victim/pollution reconstruction ambiguity；旧 trace 缺证据时明确 partial/unavailable。
- `OPEN / latency saved`: 缺 request-aligned observed/cold-counterfactual/transfer/stall latency，正式值保持 null；reward 和 workflow delay 不得代替。
- `OPEN / causal regret and fairness`: future reuse 仅为 proxy；G08 oracle regret 与 G07 fairness manifest 未实现。历史 capacity-disabled artifact 仍不能支持 capacity/pollution claim。

## 2026-08-19 G14A status

- `RESOLVED / typed runtime plumbing`：typed catalog、MB capacity、initial/dependency/pinned contract、training/checkpoint provenance、fairness 1.1、benchmark、CacheEvent 1.3与metrics 1.1已端到端接通；legacy slot/MB未回归。
- `RESOLVED / controlled Alibaba adapter mapping`：repository controlled catalog已显式声明`adapter_batch_type_1 -> base:veh_base_v1`，不再在NGSIM+Alibaba入口因未知typed adapter失败。
- `RESOLVED BY G14B / split protocol`：正式 split exclusion/final protocol 与 readiness v2 已冻结；但正式 checkpoint、formal raw trace、holdout/support仍不存在。G14A artifact和tiny checkpoint不得进入论文主表。
- `OPEN / latency saved`：metrics 1.1继续输出null；G14A没有新增request-aligned counterfactual latency，不能将reward/delay替代。

## 2026-08-24 Closed: clean-worktree external-resource identity mismatch

G14C v3 dev selector 未显式传递 workflow path，benchmark fallback 指向 clean worktree，而 fairness 仍冻结主
worktree absolute path，导致在首次 dev performance 前失败。G14R3 通过 content-addressed logical identity、共享
resolver、explicit workflow propagation、portable fairness companion 与 checkpoint location contract 关闭该问题。

残余风险：Readiness 不等于正式结果；G14C v4 仍需从头训练并可能暴露运行时间/基础设施问题。任何自动下载、
cwd 猜测、无 registry 的正式命令、旧 run checkpoint reference 或 holdout capability 都仍是 blocker。

## 2026-08-25 Closed: long-phase terminal timing and clean-worktree Python portability

- `RESOLVED / timing`：phase ledger v2 错把 UTC delta 与 monotonic elapsed 要求在 2 秒内一致；ledger v3 改为
  monotonic authoritative duration，并单独审计 UTC adjustment、child 与 finalization duration。
- `RESOLVED / terminal transaction`：commands 完成后先提交 completion candidate；terminal append 失败可在同一
  v1.4 run 通过 finalize-only 重验 output hashes 后完成，不重跑 commands。
- `RESOLVED / per-cell resume`：cell ledger、unique staging、immutable commit marker 与 committed-only consumer
  防止完整 cells 重跑或 partial output 进入 selection/aggregate。
- `RESOLVED / Python`：v1.4 templates 使用 resolved absolute interpreter，clean worktree 无需 `.venv`；项目 import
  必须来自 clean source，依赖可来自共享 venv。

残余风险：G14C v5 尚未启动，新的长正式执行仍可能暴露未覆盖的 OS/磁盘/进程级故障。当前 cell transaction
不支持单 cell 内 checkpoint 续训；未提交 cell 只能新 attempt 从头运行。Readiness 不是性能或 paper-ready 证据。

## 2026-08-25 G14C v5 preflight context failure / G14R5 resolution

- `RESOLVED / nested expansion`：v1.4 outer runner 已解析绝对 Python，但 preflight child 从未覆盖的
  `default_expansion_context` 二次展开，首条命令因缺 `python_executable` 失败。G14C v5 永久 invalid，不得恢复。
- `RESOLVED / context divergence`：v1.5 outer runner唯一生成 resolved context `1.0.0`；nested consumers只加载并
  hash-verify该 artifact，preflight强制 outer/nested 186-command expansion SHA相等。
- `RESOLVED / implicit fallback`：active v1.5要求显式绝对 Python与environment manifest；cwd猜测、相对 `.venv`、
  child `sys.executable`补位、跨 run/commit/environment/path复用均 fail-fast。
- 残余风险：Readiness v7只验证 preflight/tests 与既有 transaction regressions；G14C v6 长训练仍可能暴露新的
  OS/磁盘/进程级故障。该风险不授权复用任何 G14C v1–v5 invalid artifact。
# 2026-08-31 — G14C v9 request divergence（已合同修复，run 永久无效）

- 根因：legacy service success/failure 会改变 workflow progression 和下一 request，因此比较策略的 observed request denominator 不是外生 matched estimand；这不是 path、resource 或 checkpoint plumbing 故障。
- 永久边界：`typed_model_cache_formal_20260830_113339_g14c_v9` 以及其 checkpoints、candidates 和 partial dev outputs 禁止 resume、retry、finalize、salvage、复制或进入新 manifest。
- 修复：Protocol 2.0 在 agent action 前生成统一 request exposure，outcome 不再反向改变 progression；跨 agent request 与 outcome fingerprint 分离。
- 剩余风险：本轮只有 non-formal tiny/controlled rehearsal，不构成算法性能、论文结论或 holdout 证据；G14C v10 必须从全新 clean formal training 开始。

# 2026-08-31 — G14C v10 environment projection mismatch（已在执行前合同修复）

- 已确认：Protocol 2.0 manifest full identity=`acc61f8f...8196`，A11 runtime small projection 重算为
  `3858b1ba...76de`；三个 Protocol-bound extension 未进入 resolver producer。
- 已修复：Protocol 2.1 由一个 strict projection producer 生成 manifest/runtime identity，并将 full normalized
  projection 传入 binding、resolved context、training/checkpoint provenance；active validator 逐字段验证。
- v10 不是 invalid formal run：durable run、ledger、checkpoint、row 均不存在，禁止伪造 run/failure artifact。
- 剩余风险：readiness 仅授权未来独立 G14C v11，不是 formal/performance/paper-ready 证据；holdout 继续 sealed。

# 2026-09-02 — G14C v11 formal request subject lifecycle drift（已合同修复）

- 已确认：Protocol 2.1 exposure producer 在 reset 冻结 `i_80:1116`，但没有证明该车辆覆盖完整 request horizon；
  frame 4 缺失后 legacy runtime 重选 `i_80:1120`，造成 request trace 与 runtime/CacheEvent 身份分叉。
- 已修复：Protocol 2.2 在 exposure 前过滤 frame `0..request_count` 持续存在且物理连续的车辆，随后按原
  `handoff_pressure` 选择并冻结一个 subject；runtime 禁止 formal episode 内静默重选并独立复算候选/RSU/time。
- 永久边界：v11 run/root/checkpoint/candidate/partial output 全部禁止 resume、retry、finalize、salvage、复制或
  复用。Protocol 2.1 仅保留 audit compatibility，不得作为 active execution。
- 剩余风险：G14R11 只证明 execution contract 与 non-formal rehearsal；尚未运行 G14C v12、formal performance
  或 holdout，不能形成算法优势、formal gate、G14/TMC/paper-ready 结论。

# 2026-09-04 — G14C v12 nullable aggregation failure（已修合同，run 永久无效）

- 根因：`scripts/train_algo_pool_real_sample.py` 的旧 `metric_means()` 对每行执行 `float(row[name])`；Endpoint 2.0
  合法产生的 215 个 `end_to_end_workflow_delay=null` 触发 `TypeError`。数据、episode、request/Event alignment
  没有损坏，这也不是算法性能失败。
- 永久边界：`typed_model_cache_formal_20260902_162203_g14c_v12` 的 256 episodes、32 updates、8 candidates、
  latest checkpoint 及全部 staging 引用不得 resume/retry/finalize/salvage/reuse，也不得进入 selection、freeze、
  benchmark、statistics、gate 或 formal manifest。
- 修复：Protocol 2.3 统一严格 nullable reducer、availability counts、Dev finite-first ordering、paired availability、
  finite-only Holm 与 zero-pair `UNAVAILABLE` gate/claim semantics；共享 denylist覆盖完整 active command graph。
- 剩余风险：长时 G14C v13 尚未执行；non-formal exact/phase-chain 验收不构成 performance 或 paper-ready 证据。

## 2026-09-08 G14R18 入口证明边界

既有 main 源码顺序断言不等价于实际入口执行，空 rollout 列表也不构成调用计数。补充的 main 正负
用例保留真实 generated registry、companion loader、checkpoint read 和 strict gate，并监测实际 rollout
符号。数据准备/fairness 使用 test-only 替身，因此仍不构成完整正式 benchmark 或性能证据。

## 2026-09-28 Open: 顶刊 novelty 压缩与 manuscript/evidence 闭环缺口

- 最新近邻已分别覆盖 `DAG/dependent task + VEC`、`service/content cache + offloading`、`DT + prediction`、
  `trajectory-aware migration`、`mixed/dual-timescale control`、`MARL` 以及 `agentic DAG + model loading`；上述任一
  单点均不能继续作为 PPO_MEC 的核心 novelty。
- 当前可守边界是“跨 RSU 连续 DAG + typed base/adapter/workflow-state cache + handoff prepare/state migration +
  可审计三时间尺度控制”的交集，但尚缺 formal/independent holdout/support、机制 endpoint、完整 manuscript 与
  claim-to-artifact 映射，故 TMC-ready 只能判为 `Unverifiable`。
- 后续必须保持 controller-level MARL、`semantic_discrete_5`、受控 adapter mapping、policy guard 和 learned predictor
  的归因边界；不得把 `NGSIM + Alibaba` 写成真实 adapter request trace，或把 readiness/checkpoint inventory 写成性能证据。

## 2026-10-05 shared cache × recovery remaining blockers

- `OPEN / cross-workflow cache lifecycle`：原生 typed resident set 只在同一 env episode 内持续；`reset()` 和当前逐-workflow benchmark 实例化会恢复 template resident，尚无跨 workflow 共享缓存生产生命周期。
- `OPEN / shared resource contention`：当前 bytes/time 是 accounting；action 不会消耗共享无线带宽、compute queue 或 waiting-time resource。不得把本轮结果写成多请求无线/计算争用收益。
- `OPEN / external validity`：A/B/C 使用 contract-legal synthetic audit objects，不是真实模型权重；100 Mbps、同机 state overhead 和 technical-workload recompute 是冻结估计，非真实无线测量。
- `RESOLVED WITH BOUNDED WITNESS / within-episode state coupling`：C 已证明合法 action 4 可经有限容量 LRU 改变后续 resident set、reload bytes 和 action mask；这不是不同 `restore_cost` 造成。但仅为 E2 bounded，不授权 RL、formal/holdout 或 paper-ready claim。
- `CORRECTED / method identity`：frozen runner 将 resident-aware estimate 标为 `current_simple_threshold`，与 workload-v0.1 原方法信息权限不一致。原规则按预存冻结输入复算仍在 A/B/C 全选 action 4，故 native outcome 不变；原 frozen decision record 只保留审计，不可用作方法权限证明。

## 2026-10-06 SA-GHMAPPO v2 remaining blockers

- `OPEN / policy and aggregation collapse`：full SA 三 seed 中 seed 29 completion 仅 0.25；另两 seed 退化到与 MAPPO
  相同的高 transfer/high recompute 行为。raw evaluation action 0/3/4=`180/10/120`，未学出正确规则的 0/2/4 条件切换。
- `INTERFACE-BLOCKED / dependency-message hypothesis`：no-dependency completion 0.917，高于 full 0.750；paired delta
  `-0.167 [-0.250,-0.083]`，但 optimized/executed action likelihood 错配使其只能作为负向敏感性信号。不得把 DAG
  message passing 写成已验证贡献或已证伪机制，也不得用更少的截断 elapsed 掩盖未完成。
- `OPEN / matched feature contract`：flat actor 不消费 cache readiness/occupancy，flat critic occupancy 单位错误，graph/flat
  未对等消费 typed base、byte occupancy、link/state/model cost；规则另有 exact transition clone + lexicographic objective。
- `OPEN / failure-time mobility`：失败只推进 step/clock，RSU/vehicle/prediction 仍由 node index 派生，会冻结 mobility 并形成
  重复失败状态；在修复前不得把 v2 称为公平序列控制 benchmark。
- `OPEN / external validity`：link rate/error、deadline、adapter mapping 和 trajectory-workflow pairing 是合成因素；没有真实
  adapter request trace、共享 queue/bandwidth/compute、跨 workflow cache 或真实 RSU。
- `OPEN / statistical maturity`：只有 3 seeds、12 evaluation windows、percentile bootstrap；没有 formal/hidden holdout、
  support、BCa/Holm、收敛证明或独立复现，paper-ready 维持 `Unverifiable`。
- `BOUNDARY / historical v1`：192-episode v1 早于 action-4 commit 与 actual/estimated link 修正，只保留审计，不得与 v2
  checkpoint、逐行结果或 aggregate 混用。

## OPEN / causal learned baseline budget sufficiency

- `cscwd_causal_strong_baselines_dev_20261009_v1` 每个 learned cell 只有 1,440 环境步；是否仅因共同训练曝光不足而限制
  SA/PPO/MAPPO/DT 尚不能区分。
- 已预注册统一 4× budget + 等比例选模时点的联合干预；这不是 SA 专属追加预算，也不能把二者效应拆开解释。
- 现有 36 个实例全部已消费为 development；即便改善也不构成 independent/formal 证据。若 SA 不改善，本轮停止，
  不自动搜索新算法或奖励。

## OPEN / calibrated workflow critic scale candidate 仍只是假设

- 诊断见 `calibrated_workflow_service_reward_learning_diagnosis_20261009.md`：service reward 下 raw critic loss 和 value-to-policy gradient ratio 异常大，但尚无 development A/B 因果证据。
- PopArt 实现只能检验“critic scale 干扰共享更新”这一机制；它不修复 reward 设计、样本不足、auxiliary target 或任务局部可分性。
- 当前 36 个实例无独立 holdout，任何正结果最多是 development candidate；任一机制、行为或服务 gate 失败即不得晋级。
- checkpoint 含本地训练权重，禁止提交或上传；科学 run 不允许 retry、追加 seed、改预算或用最终评价反选方案。

## OPEN / PopArt A/B pre-run launch failure

- 2026-10-09 唯一授权后台启动报告 PID 后，在即时快照前结束；runner 未创建 run root，stderr/stdout 日志为 0 bytes，OS 级原因不可恢复。
- 这不是算法失败，也不是训练结果；可观测 scientific steps/updates/checkpoints/evaluation rows 均为 0。
- one-launch/no-retry 已耗尽，本轮禁止以 foreground、resume、新 PID 或新 output root 补跑。未来必须获得新的显式授权和新的 create-only run id。

## IN REPAIR / persistent launcher v2

- 新授权允许以新 run ID 独立执行，不恢复 v1；旧空日志和 failure receipt 保持不变，OS 退出原因仍为 unknown。
- 风险控制改为 child-owned entry receipt + detached supervisor + exit/terminal sidecars。两次宿主验收任一未通过即停止，不以训练任务测试 launcher。
- 此修复不改变 PopArt、reward、data、budget 或科学 gate；若 diff 出现科学变量变化，启动授权失效。
