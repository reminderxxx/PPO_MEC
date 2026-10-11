# CSCWD phase-v2 action4 公共可观测性与候选边界审查（2026-10-11）

## 身份、范围与结论

- `reviewed_at`: 2026-10-11；`literature_cutoff`: 2026-10-11（沿用本支线，本轮不作 novelty 判断）；
  `target_venue`: IEEE TMC；`policy_version`: `tmc_review_policy_v3_20260621`。
- `artifact_run_id`: `cscwd_phase_v2_public_observability_audit_20261011_v1`；审查源码 commit
  `6ce0a583d2b6853a69cecf4e0e37d060b878a81b`；A 分支结果 commit
  `dabd57dced3f166bc93d4af8e6ddfcc484ed7603`。
- `evidence_level`: `E2_ARTIFACT_AUDITED / bounded development diagnosis`。本轮只读既有 pilot、A 的 3 个公共起点／
  6 个局部分支和当前源码，并新增 1 个确定性合成单元见证；训练、checkpoint 选择、正式评价、holdout、模型调用、
  下载均为 0。
- verdict：**当前公开输入不足以逐状态识别“可避免的准备”**。A 的三处第二次 action4 确为事后局部成本支配，
  但分叉前只知道新预测目标 `rsu_2` 尚未准备，并不知道目标交付会失败。当前 candidate 的一个可干预问题是：
  它将未校准的有限 uncertainty 转成点等待估计，并把 target delivery 判为 `yes`，从而给这些状态
  `prepare, supervision_weight=1`。这不是执行器 bug，也尚未被证明是总体退化的充分原因。
- 本轮不改生产策略。唯一冻结候选是 **positive-prepare calibration abstention**：仅当公共
  `causal_provenance.confidence_calibrated is True` 时才允许正 `prepare` auxiliary 监督；缺失、false 或 unknown 一律
  `abstain`。不把其实现或收益写成既成事实。

机器可读字段表与裁决见 `cscwd_phase_v2_public_observability_audit_20261011.json`；A 论文线的独立边界见
`cscwd_phase_v2_public_observability_claim_change_20261011.md`。

## 1. A 分支结果复核

A 的 `starts.json`、`paired_branches.json`、`summary.json` SHA-256 分别为
`10aa0537…a68`、`2dc40ac1…d39a`、`be57ec84…e9a1d`。三条件在共同首步 action4 成功把状态交付给
`rsu_1` 后，第二个决策的公开状态均为：current=`rsu_1`、predicted target=`rsu_2`、current prefix valid、target
prefix missing。故第二次 action4 是向**新目标**准备，不是向同一 RSU 重发相同状态。

| 条件 | 重复 seed | candidate 在第二步的 action4 概率 | 公共 auxiliary 标签 | 事后 action3 相对 action4 | 服务结果 |
|---|---:|---:|---|---|---|
| default | 3 | `0.512011–0.547348` | `prepare`, weight `1` | `−9,782,424 B`, `−0.411255 s` | 完成/按期/失败相同 |
| adverse_link | 3 | `0.512021–0.547360` | `prepare`, weight `1` | `−9,782,424 B`, `−0.724293 s` | 完成/按期/失败相同 |
| ample_cache | 5 | `0.673038–0.700591` | `prepare`, weight `1` | `−140,480 B`, `−0.025536 s` | 完成/按期/失败相同 |

第二步相对共同首步的 action4 概率最大绝对变化仅 `0.000541`。这是冻结策略的相邻状态现象，不是在线参数更新；
它只说明成功迁移后的新目标缺失状态仍得到强 prepare 倾向。三条件 action4 随后均为
`target_departed_before_delivery`，这是执行后 outcome，不能倒灌成在线标签。

## 2. 决策时字段权限与缺失语义

“公开存在”和“网络实际消费”必须分开。PPO/MAPPO 的 flat encoder、SA 的 graph/surrogate encoder、public
candidate/rules 并不消费完全相同的公开字段。

| 信息 | 公共字段与来源 | PPO/MAPPO actor | SA actor/critic | candidate / public rule | 缺失或 unknown 语义 |
|---|---|---|---|---|---|
| 当前 RSU | `vehicles[].associated_rsu_id`；当前已观测位置与圆形 RSU geometry 的关联 | 仅编码 current/target readiness 与 target-diff，不编码 ID | 用 ID 选 current RSU embedding | 直接消费 ID | 无覆盖时 action mask 仅保留 fallback；不得补目标 |
| 预测目标 | `predicted_first_handoff_rsu_by_vehicle`；最近两点速度的 10 s 常速外推 | 编 target 是否不同，不编码目标 ID | 选 target embedding，并编码目标存在/切换 | 直接消费目标 ID | `None` 时 target prepare=`no`；不读取真实未来 |
| 当前/目标模型就绪 | `required_bundle_ids` + `object_catalog` + `typed_resident_object_ids` | 两个 bundle-ready 标量 | DAG/RSU/vehicle 多处分目标消费 | 计算 readiness、bytes、load；私有 eviction order 不可见 | RSU/catalog 缺失→`unknown`；需 eviction→admission `unknown`，不是 infeasible |
| 已交付状态 | `prepared_state_prefix.current/predicted_target`：`known/exists/valid/missing_completed_fraction`；只含过去已提交状态 | v4 以 7 个标量消费，target unknown 有独立 mask | RSU embedding 消费 current/target 三元组 | exact status：valid/stale/missing/unknown | target `known=false` 时 exists/valid/gap 必须为零；estimator→`unknown` |
| 当前 contact 点估计 | `calibrated_context.contact_budget_seconds`；最近两点速度与当前 RSU 圆交点 | 归一化标量 | 归一化标量，另有当前位置到当前边界的派生量 | phase-v2 只判 input upload 是否能在当前 contact 完成 | estimator 缺失→fit `unknown`；actor 的 `or 0` 会编码为 0；当前生产字段实际为 finite |
| contact 置信区间 | **不存在**；无 lower/upper bound、目标 entry/dwell interval | 无 | 无 | 无 | 不得以 `confidence=.5` 冒充 contact bound |
| 预测 confidence / uncertainty | raw 固定 `0.5/0.5`；`causal_provenance.confidence_calibrated=false` | 消费两标量 | 形成 reliability/gate；仍不读 calibrated flag | phase-v2 estimator 只读 uncertainty | uncertainty 缺失→target delivery unknown→abstain；finite 值当前被当作可计算点值 |
| 校准 provenance | `predictions.causal_provenance.{source,observed_sample_count,actual_future_used,confidence_calibrated}` | 不消费 | 不消费 | 不消费 | `confidence_calibrated=false` 当前不会阻断正 prepare 标签 |
| input/compute/state 大小 | current node 与 `calibrated_context` | 消费 bytes/compute | 消费 bytes/compute/DAG | 估算各 phase | 非 finite→相关 phase unknown |
| radio/backhaul/queue/timeout | `phase_contract` 提供 estimated rate、fixed latency、timeout、queue=0；来源均明确为 simulation assumption | 不消费 phase contract；仅消费 legacy `link` radio 点估计 | 同左 | 完整消费 phase contract | 必需字段缺失→unknown 或 schema fail-fast；不是实测链路 |
| 条件成本估计 | estimator 派生 `estimated_phase_seconds/bytes/deadline_fit` | 不直接消费 | auxiliary label 间接施加梯度 | 直接用于 label/rule score | 任一必需 phase unknown→total/deadline unknown；actual seconds 永远 `None` |
| 实际 delivery/成本 | `transition.migration_status`、phase ledger、actual Mbps、未来轨迹 | 决策时不可见 | 决策时不可见 | 明确 forbidden | 只能用于事后审计或奖励，不可生成同一步在线标签 |

关键代码事实：raw public predictor 使用最近两个已观测位置，但 semantic state 只公开位置、速度大小、预测目标和点 contact；
没有公开二维速度、目标进入/离开区间或经过校准的 target-delivery probability。PPO/MAPPO 的 calibrated flat 路径甚至不编码
位置/geometry；SA 虽有当前边界派生量，仍没有目标交付区间。phase-v2 public two-step 因此明确
`phase_v2_second_step_mobility_unknown`，退化为 immediate score。

## 3. 当前 candidate 的可干预环节

`public_prepare_advantage_label()` 仅在 action4 的 current service、target prepare、deadline 和 future readiness gain 均为
`yes` 时给 `prepare`。phase-v2 estimator 对 target delivery 的处理是：

1. 从公共 uncertainty 与 10 s delivery timeout 计算 `predicted_delivery_wait = uncertainty × timeout`；
2. 只要 target 存在且该乘积 finite，就把 `predicted_target_delivery` 设为 `yes`；
3. raw fixture 的 uncertainty=`0.5`，所以得到 5 s 点估计，即使 provenance 同时声明 `confidence_calibrated=false`；
4. A 三个第二步起点由此全部得到 `prepare, weight=1`；public immediate/two-step 也都选 action4。

这定位的是 **positive prepare auxiliary 的 epistemic abstention 缺口**，不是“奖励太小”、不是重复向同一目标传输，也不是
action4 执行语义错误。PPO 主损失仍可能从回报学习 action4；本轮没有历史 minibatch/optimizer 顺序，故不能声称该标签是
冻结策略概率的唯一成因。

## 4. 同观测异最优的确定性见证

新增 `tests/test_cscwd_phase_v2_public_observability_audit.py`，构造两条前两个样本完全相同、之后分别继续前进／反向的
合成轨迹。四个 action3/action4 分支在决策时满足：

- 完整 `_info()`、semantic state、mask、public estimate、candidate label 完全相同；
- label 均为 `prepare, weight=1`，action4 `predicted_target_delivery=yes`，同时 `raw_trace_fit=unknown`；
- public immediate/two-step 在两个后缀给出相同动作；
- 前进后缀 action4 交付成功，按“当前服务→迁移成功→时间→字节”的声明局部序优于 action3；
- 反向后缀 action4 交付失败，action3 在同一局部序优于 action4。

这证明任何只对当前公开状态作确定性映射的方法，都不可能在所有合法隐藏后缀上识别可避免准备。该见证不是科学 rollout、
不是 NGSIM 频率估计，也不证明 A 的 US101 起点必然不可预测；它只否定“现有单帧公共字段足以逐状态确定最优动作”。

## 5. 唯一冻结候选与下一轮 A/B

候选名：`calibration_provenance_positive_prepare_abstention_v1`。

- 唯一变量：在现有 public-causal v3 label 上增加一个布尔门。仅对拟输出的正 `prepare` 标签检查
  `predictions.causal_provenance.confidence_calibrated is True`；false、缺失、类型错误或 unknown 均改为
  `abstain, supervision_weight=0`。现有 `serve` 标签、slow/fast auxiliary、PPO loss、reward、observation、action mask、
  optimizer、rule objective 均不变。
- 与旧候选的区别：旧候选把 finite uncertainty 当作 target delivery 可估；新候选不改变 target，也不禁止 action4，只在
  交付置信度未校准时撤销正 event CE/temporal 梯度。它不同于 current-bundle-missing abstention，也不同于关闭全部 auxiliary。
- 公平能力边界：只读已经向所有方法公开的 provenance；不引入真实未来路线、actual link、exact clone 或 SA 私有字段。
  PPO/MAPPO/rules 保持同一公开能力，候选只改变 SA 的训练监督使用方式，因此论文只能称 learning-design ablation。
- 预期：在 A 类未校准状态不再人为强化 prepare；action4 仍可由 PPO 回报选择。不能预承诺完成率或成本改善。

若另行授权，探索性 A/B 只比较 current public-causal v3 与该 gate：两臂从头训练、同 5 seeds、每 cell 1,152 env steps、
24 updates、192 optimizer steps、同 checkpoint cadence 与 result-blind selection；总计 11,520 env steps。因 US101 已参与本候选
形成，该 A/B 最多是 development mechanism check；任何晋级还需 source-disjoint、事前冻结的公共前缀证据。

主要指标固定为 workflow completion、on-time completion、service failure；共同机制指标为 action4 probability、migration
success/failure、state-ready、model/state bytes、recompute。停止/否证条件：

1. 任一 selected/fixed 视角 completion 或 on-time 下降，或 failure 上升；
2. 未校准且拟为 prepare 的样本仍有非零 event supervision（实现失败）；
3. 服务不降但 failed migration/transfer/recompute 没有形成预注册的共同 Pareto 改善；
4. 改善仅来自少数 seed、已暴露 US101 起点或 checkpoint 反选。

本轮不实现、不训练。当前 raw profile 全部为 calibrated=false，因此该 A/B 可检验“只撤销不可靠的正 prepare 监督”，
但不能宣称形成了自适应置信机制；若要晋级为后者，仍须先有可审计的 calibrated=true 生产者及 source-disjoint calibration
record，不能把 false 改成 true 或调阈值追逐收益。

## 6. 论文边界与未定位项

已确认：A 三处是新目标准备的真实事后浪费；当前正 auxiliary 标签未检查 confidence calibration；同公开观察可对应相反
局部最优动作。否定：同目标重复发送、现有 public fields 能确定识别所有浪费、public two-step 拥有真实二步 mobility、
规则或 candidate 使用 actual route。未定位：该 abstention 是否改善总体训练、其与 PPO 梯度的净方向、目标交付概率的真实
校准误差、SA 相对 PPO/MAPPO 的总体差距。

因此安全主张仅为“发现并冻结一个可证伪的 epistemic abstention 候选”。禁止写成算法改进已有效、服务退化根因已证实、
action4 应在成功迁移后禁用、SA 获得新优势或 paper-ready。formal/holdout/support 仍缺，论文级 verdict 保持
`Unverifiable`，本轮不评分。

## 7. 验证与未覆盖风险

执行：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest -q \
  tests/test_cscwd_phase_v2_public_observability_audit.py
```

结果：`1 passed`。后续提交前另执行相邻 estimator/raw phase-v2 tests、smoke、JSON 解析与 `git diff --check`。

未覆盖：真实 NGSIM suffix 的概率校准、source-disjoint 频率、任何训练因果效应、formal/holdout、无线/queue 实测、完整策略
长期最优性。A 的另一个最大 36 分支／288 步结果在本轮一次性检查时没有已推送终稿，故未等待、未轮询、未用于裁决。
