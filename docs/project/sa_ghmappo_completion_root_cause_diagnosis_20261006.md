# SA-GHMAPPO 完成率下降根因定位

## 审查身份与结论边界

| 字段 | 值 |
|---|---|
| `reviewed_at` | `2026-10-06` |
| `literature_cutoff` | `2026-10-06`；本轮未新增网页或文献检索 |
| `target_venue` | `IEEE TMC` |
| `artifact_run_id` | `sa_ghmappo_completion_root_cause_20261006_v1` |
| `policy_version` | `top_journal_review_policy_v3` |
| 被诊断 Git commit | `728040c55866a3dd27e5091fd8f2186050179e2d` |
| 冻结 pilot source commit | `b418eb4dbfa012e271cb97880693c714e6abf79e` |
| Evidence level | `E1_DOCUMENTED_WITH_REPRODUCIBLE_NONFORMAL_DIAGNOSIS` |
| Paper-ready verdict | `Unverifiable` |

本轮只读原 v2 checkpoint、训练记录、逐行评估和失败原件；新增内容只有隔离诊断脚本、测试、诊断
artifact 和文档。没有训练、模型调用、下载、旧 holdout/formal、自动重试或按结果追加样本。历史排名继续只作
描述性诊断，不能据本报告恢复公平排名。

## 最短结论

完成率下降的最直接执行原因是：首次不能服务当前节点后，`node_index` 不变，而 mobility、当前位置、接触预算和
未来 RSU 序列都由 `node_index` 派生，导致失败步骤虽然增加时钟和 `step_index`，却不推进移动状态；策略随即在近似
同一状态再次选择不能完成当前节点的 prepare，形成失败循环。18 个冻结诊断 episode 中有 60 个
`service_failure` 步骤没有 mobility 推进。

使策略难以从这些失败中正确学习的首要接口错误是：真实采样对象是 mask 后五动作环境分布，但 rollout 保存和 PPO
更新使用环境动作反解出的单一 canonical head tuple 概率。小空间枚举确认环境动作 pushforward 与实际采样概率一致，
因此不能把“聚合概率错误”写成事实；错误发生在训练 likelihood/执行身份。147 个诊断步骤中 142 个保存概率与真实
执行动作概率不一致，最大绝对差为 `0.405241`。失败负 advantage 会降低真实动作的相关概率，但同时更新本不应由该
动作识别的 latent heads，造成错误 credit assignment。这个局部因果链已确认；它对历史完成率差值的定量贡献仍需修复后
按新预注册协议验证，本轮不重训。

奖励确有连续服务目标错配，但不是本轮首次失败循环的唯一或已量化主因。DAG edge 已进入 SA encoder 和 handoff prefix
recompute，但执行节点固定、准备对象固定；四个冻结剩余依赖配对中，深度 3 合法枚举、立即规则、两步规则和 SA 首动作
全部不变。因此当前工作负载/action contract 尚未证明需要依赖推理，不能通过增加节点数或网络复杂度解释当前下降。

## 方法与预算

- 输入固定为 v2 config/manifest；二者 SHA-256 与原 `run_manifest.json` 一致。
- 三个 selected checkpoint 均逐文件与原 `artifact_integrity.json` 核对：seed 7/17/29 分别为
  `4ceae438…8bb`、`43b12fd…f64b`、`c5dc06fc…b994`。
- 按 manifest 顺序选取 `train_00`（当前 bundle ready）、`train_05`（当前 bundle missing）、`train_02`
  （其余首个最小 contact budget 的不同实例），不按策略胜负选样本。
- 固定三 seed × 三实例 × `deterministic_env_argmax`/`fixed_env_sample`，共 18 episode、147 steps；每 episode
  上限 24 steps，无自动重试。
- 奖励确定性案例 6 个；DAG 配对 4 对，深度 3，共消费 1,277 个分支，低于冻结上限 2,048。
- 完整逐步记录见 `h1_episode_traces.json`；integrity manifest SHA-256 为
  `40d2a95a9cf6df24bbccf55b2f82c39283d81bf1470cb810355f852085349994`。

## 故障树与证据分级

| 分支 | 状态 | 证据 | 对当前下降的判断 |
|---|---|---|---|
| 真实环境动作概率计算错误 | 已排除 | 3×2×2 全枚举 pushforward 最大误差 `5.75e-8`；执行动作 log-prob 最大误差 `2.77e-7` | 不是主因 |
| 确定性选择不符合声明策略 | 已排除，限声明为 env-action policy | 确定性为同一 mask 后环境分布 argmax；head 独立 argmax 在 55/147 步不同，但两者是不同合法策略定义 | 需明确 contract，不构成现有概率 bug |
| rollout/PPO likelihood 与执行动作不一致 | 已确认接口错误 | 142/147 步不一致；PPO 使用 canonical head tuple，`env_action_ppo_enabled=false` | 必须最先修 |
| 失败后移动状态冻结 | 已确认直接循环条件 | 60 个失败步 `node_index`、position、contact、RSU sequence 均不推进 | 第二优先级；与失败循环直接相乘 |
| 奖励未表达按期连续完成 | 已确认目标错配 | completion bonus 不区分按期；deadline 仅完成时检查；外部截断无终局代价 | 潜在放大因素，未量化历史差值 |
| DAG 需要更强依赖推理 | 当前未获支持 | 4/4 配对均无合理首动作翻转；局部/两步已覆盖 | 不是当前优先修复 |
| encoder cache 单位适配正确 | 已否定 | flat critic 用 adapter count 除 byte capacity；`cache_used_bytes` 单字段扰动对已用 encoder 为零 | 公平特征 blocker，未直接归因完成率差值 |
| planner 与 learned policy 同能力 | 已否定 | planner clone 保留精确未来 DAG/order/RSU sequence、全缓存状态和转移模型，并用 lexicographic objective | 排名公平性 blocker |

## H1：首个失败如何形成循环

首个失败见证来自固定顺序中的 `train_00 / seed 7 / fixed_env_sample`；不是按结果挑选。

| 项目 | step 2：首次失败 | step 3：再次失败 |
|---|---:|---:|
| 当前节点 / 已完成 | `2` / `[1,4]` | `2` / `[1,4]` |
| 当前→目标 RSU | `rsu_0→rsu_1` | `rsu_0→rsu_1` |
| 当前 / 目标 bundle ready | `false / false` | `false / true` |
| slow 概率 | `[0.37125,0.51692,0.11183]` | `[0.37158,0.51682,0.11160]` |
| fast 概率 | `[0.59185,0.40815]` | `[0.59446,0.40554]` |
| event 概率 | `[0.77360,0.22640]` | `[0.77372,0.22628]` |
| mask 后环境动作概率 | `[.39989,.08651,.11722,.16998,.22640]` | `[.43766,0,.12761,.18706,.24766]` |
| 实际动作 | `4: event prepare` | `4: event prepare` |
| canonical head tuple | `slow=0, fast=0, event=1` | 同左 |
| 保存概率 / 真实执行概率 | `.049746 / .226404` | `.049984 / .247664` |
| reward / diagnostic advantage | `-3.389234 / -9.322492` | `-3.160000 / -6.325652` |
| 结果 | prepare 目标 adapter，但当前节点服务失败 | 目标已 ready，仍 prepare，当前节点再次失败 |
| position / contact | `20→20 / 5→5` | `20→20 / 5→5` |
| clock | `8.0246→11.9912` | `11.9912→13.9912` |

完整链为：当前节点在 `rsu_0` 缺 bundle → 随机采到 action 4 → 只为 `rsu_1` prepare，当前节点未完成 →
`node_index` 不变使移动状态冻结 → 下一步仍看到同一当前节点/RSU/contact，仅目标 cache 变为 ready → 再次选 action 4 →
继续失败。这里 action 4 的环境概率是所有 `event=1` 组合的总和 `.226404`，而 buffer 只保存 canonical
`(slow=0,fast=0,event=1)` 的 `.049746`。

对 step 2 的负 advantage 做局部梯度符号检查：正确的 executed-action likelihood 对 slow/fast 的梯度约为 0，只更新
event；当前 canonical loss 还给 slow `[0.62875,-0.51692,-0.11183]`、fast `[0.40815,-0.40815]` 梯度。
因此失败信号并非完全消失，而是额外污染 action 0--3 内部的概率分配。该结论同时避免把“关闭全部增强后更好”误写为
单一模块因果结果。

尚未获得的因果证据：本轮没有修复后重训，故不能给出 likelihood 修复、mobility 修复各自能恢复多少 completion，也不能
断言历史 full-vs-ablation 差值完全由某一项造成。

## H2：奖励计分与连续服务目标

逐行重算 168 条原 evaluation rows，六项 reward 分量与总 reward 最大误差为 `2.84e-14`。节点完成奖励按完成节点
一次发放，workflow bonus 仅完整完成时发放；实现与记录一致。问题是目标定义：completion bonus 不区分按期与逾期，
deadline penalty 只在 workflow 完成后检查，外部时间截断和未完成 episode 没有终局未完成／超期代价。

| 确定性案例 | 原 reward | 原排序 | 与服务目标的关系 |
|---|---:|---:|---|
| 按期完整完成 | `9.000` | 1 | 符合 |
| 低传输但未完成（4/5） | `6.400` | 2 | 高于有中断的完整完成，冲突 |
| 同完成量、更多中断 | `3.000` | 3 | 中断被罚，但仍低于未完成 |
| 已超 deadline 后外部截断（4/5） | `1.600` | 4 | 无 deadline/unfinished 终局代价，冲突 |
| 相同完整完成但更晚 | `-0.600` | 5 | 时间和完成后 deadline 被罚 |
| 重复 prepare 且移动不推进 | `-19.035` | 6 | 已由时间/传输/失败惩罚；不存在直接奖励 prepare |

在冻结 168 行中没有观察到“未完成 row 的 reward 高于某个完整 row”的成对反转，也没有未完成且记录时间已超过 deadline
的 row；所以当前 artifact 支持的是实现级目标缺口和确定性反例，不是该反例在 evaluation split 的发生率估计。
`handoff failure` 表示 handoff 后重算/输入/时间路径，`service_failure` 才触发 `-3`；prefix recompute 是工作重做成本，
不应与服务中断同名合并。

若下一轮另行修 reward，只建议一个最小目标：终局效用以“按期完整完成”为主项；对环境内部终止的未完成/超期给予明确
终局代价；每步仍只对真实 service interruption、elapsed time 和传输资源计费；不奖励 action 4、prepare 次数、状态
传输行为。外部采样时间截断与环境终止必须保持不同：现有 buffer 对非 terminated truncation 保留 bootstrap，这一边界
不能通过把所有 truncation 标成 terminated 来绕过。本轮不启用该目标，也不搜索权重。

## H3：DAG 的实际信息价值

输入链核对如下：SA graph encoder 消费节点、前驱/后继和 frontier 信息；MAPPO/PPO 的 flat 路径不做 edge message
passing。环境用 `execution_order[node_index]` 固定串行推进，edge 会改变 encoder 传播和 handoff 时 prefix recompute 的
祖先集合，但不决定合法 next node。策略不能选执行节点、future prepare 对象或独立 prepare 时机，只能对当前 adapter
选择五个环境动作。

四对诊断只改后续 edge/order/model demand，当前节点、cache、capacity、link 与当前成本不变，参数来自冻结 manifest：

| 配对变化 | 深度 3 合理首动作 | 立即 / 两步规则 | SA 首动作 | SA encoder |
|---|---|---|---|---|
| 删除首条未来 edge | `[0,3]→[0,3]` | `0→0 / 0→0` | `0→0` | 可区分，embedding 最大差 `.03178` |
| 增加首条合法未来 edge | `[0,3]→[0,3]` | `0→0 / 0→0` | `0→0` | 可区分，最大差 `.00942` |
| 交换首对无依赖未来执行顺序 | `[0,3]→[0,3]` | `0→0 / 0→0` | `0→0` | 不区分 |
| 交换首对不同 future model demand | `[0,1,3,4]→同集合` | `0→0 / 0→0` | `0→0` | 可区分，最大差 `.06455` |

没有配对产生合理首动作翻转，且局部规则已覆盖所有选择。将节点 ID 做一致无语义重命名并反转输入 node list 后，枚举最优
集合和 SA 动作不变；encoder 最大数值差为 `4.77e-7`。这不是“DAG 永远无用”的证明，而是说明当前冻结任务的 action
空间和固定顺序使依赖结构近似局部可分；在出现预注册的 edge-induced action reversal 前，不应把更强依赖网络列为首修项。

## 两项适配核对

### 编码单位

`cache_capacity` 与 `cache_used_bytes` 均为 bytes。`train_00/rsu_0` 的真实占用率是 `.876197`，array observation 也为
`.876197`；但 flat centralized encoder 计算 `len(adapter_ids)/cache_capacity_bytes=.000000000855`。单字段扰动中，
`cache_used_bytes` 减半对 flat actor、flat centralized 和 graph shared 全部零响应；capacity 翻倍只让错误 flat 比值变化
`1.43e-10`，adapter count +1 才改变 graph shared。当前 agent 丢弃正确 array occupancy，使用 semantic encoder。
这是明确的单位/消费适配缺陷和 baseline 公平性 blocker，但本轮没有证据把它单独量化为 completion 下降主因。

### planner 权限

`clone_for_decision_model` deep-copy 完整内部环境：全部 future nodes/edges/`execution_order`/`rsu_sequence`、cache
resident/LRU/capacity、完成集合、时钟、prepared state 和精确转移。preview 把 link rate 换为 `estimated_mbps`，不读取
`actual_mbps`，但未来 mobility sequence、DAG demand 和成本仍是冻结真值；目标是“最大完成节点、最小 failure、最小
deadline violation、最短时间、最少 bytes”的 lexicographic tuple。learned actor 使用被 encoder 丢弃一部分字段的公开语义
输入和 scalar reward。planner 应保留为强 model-based baseline，但必须标注模型能力与目标差异，不能把其结果当 matched
ranking，也不应为追求表面公平而自动削弱规则。

## 下一轮唯一优先修复决定

下一实现轮只修同一条不可分的执行身份/likelihood 接口，不同时改网络、reward、数据、guard 或 DAG：

1. rollout 以 mask 后真实采样的环境动作作为唯一 PPO action identity，保存其 summed pushforward log-prob；
2. 更新时从当前各 head logits 重建同一 mask 后五动作分布，对同一 executed env action 计算 ratio/entropy；
3. deterministic 明确定义为该环境动作分布 argmax；若 projection/guard 改动作，buffer 必须保存实际执行身份及与其一致的
   behavior likelihood，不能再静默反解 canonical tuple；
4. 用 3×2×2 枚举、alias 多对一、action 4、mask 和负 advantage 梯度测试冻结 contract。

该修复完成并通过接口测试后，第二顺位才是把 mobility/contact 推进从 `node_index` 解耦到 decision time，以及统一 bytes
occupancy 特征；第三顺位才是预注册连续服务 reward；最后才讨论 DAG 扩展或新算法。修接口后如要重训，必须另立任务、
冻结协议并保留本轮负结果。本报告完成即停止，不授权重训。

## 可复现入口与未覆盖风险

复现命令：

```bash
PYTHONPATH=. /Users/howen/Projects/PPO_MEC/.venv/bin/python \
  scripts/diagnose_sa_ghmappo_completion_root_cause.py \
  --pilot_artifact_root /Users/howen/.codex/worktrees/calibrated-workflow-training/PPO_MEC/artifacts/calibrated_continuous_workflow_pilot_v2_20261006
```

脚本强制校验冻结输入和 checkpoint hash；输出不包含 checkpoint、原始数据或模型权重。未覆盖风险包括：没有修复后重训、
没有 evaluation/formal/holdout 因果复验、配对 DAG 只有 4 对且深度 3、reward 反例是确定性计分案例、没有真实无线/RSU、
共享资源争用、独立现实 adapter-request trace 或 paper-grade 统计。因此结论只定位接口和机制因果链，不是算法性能结论。
