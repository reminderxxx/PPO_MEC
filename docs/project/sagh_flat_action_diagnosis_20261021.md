# SAGH 初始平线的独立动作诊断

- reviewed_at: 2026-10-21；literature_cutoff: 2026-09-28（未重新评价 novelty）
- target_venue: CSCWD 2027 development；policy_version: tmc_review_policy_v3_20260621
- artifact_run_id: sagh_flat_action_diagnosis_20261021_v1
- Git: 算法固定 clean `8e15418`；父实验 runner `ba1b04e`，诊断脚本 hash 见 protocol.json
- evidence_level: MATCHED_DEVELOPMENT_CHECKPOINT_REPLAY_AND_PAIRED_ACTIONS；formal/holdout claim: UNVERIFIED

## 范围与方法

用户明确授权先查平线及动作变化，再统一扩大十算法预算；不修改优化主线。诊断只读父实验
`fair_reward_training_20261021_v2` 的 SAGH 三 seed `[7,13,29]`，各取事前固定的 update 0/12/24。
在全部12 validation windows × 2 workflows 上重放，每个检查点24 episode，每seed368实际状态。
另外将三个检查点逐一放到完整初始策略状态及 algorithm_memory 上配对评分，不只挑奖励有差异的状态。
透明 observer 记录原网络 logits、现有各 prior/gate 后的输出及最终实际执行动作；返回原对象，
不改变任何算法开关、输入奖励、权重或优化器，不调用 learn/backward，不做去规则的替代实验。

## 直接证据

| seed | 初始→最终原网络 masked env-score argmax 改变 | 相同状态上的最终动作改变 | 原网络概率平均 L1 变化 | 执行后回报初始/中间/最终 |
| --- | ---: | ---: | ---: | --- |
| 7 | 258/368 | 0/368 | 1.247434 | 18.871250 / 18.871250 / 18.871250 |
| 13 | 262/368 | 0/368 | 1.168266 | 18.871250 / 18.871250 / 18.871250 |
| 29 | 269/368 | 0/368 | 1.205080 | 18.871250 / 18.871250 / 18.871250 |

原网络比较采用层级 heads 导出的五个 env-action score、相同合法动作 mask；这不是移除全部规则后的
另一次轨迹回报。三 seed 的中间检查点也均为0个最终动作改变。所有实际轨迹顺序逐step相同，
每检查点动作计数为 action3=233、action0=24、action4=111。216条重放 episode 的 reward/完成标志
与父实验原始验证CSV逐条完全相同，排除了仅比较总体平均值掩盖动作不同的情况。

逐层观测表明现有规则把变化映射回相同执行行为：seed7初始机会约束层改变274/368个进入该层的
argmax，最终改变76/368个；数字孪生 prior 分别改变67/68个。后续 idle popularity fallback、
option gate 和层级聚合继续影响执行。seed7初始有49次 popularity fallback 和32次 option机制准备，
最终为38次和43次，路径归因改变但最终动作不变。规则层次数是顺序相关的描述，不可相加作为独立因果效应。
所有状态都至少有3个合法动作，不能解释成环境只允许一个动作。

因此本轮平线的已证实近端原因是：**网络偏好与概率在变，但本验证分布下，经已有先验、门控、
回退和聚合后的最终确定性动作保持相同，导致相同原生奖励。**
这不是训练从未执行，也不是通过训练逐步上升后收敛；不能凭此宣布网络没有学习、算法已最优，
或断言增加预算一定解决。观测到的 final env 概率同样有变化；概率是门控前的模型统计，
不能直接当成后置覆盖后的最终执行分布。训练使用随机采样，与本次确定性验证的边界应区分。

## 证据与验证

产物：`artifacts/analysis/sagh_flat_action_diagnosis_20261021_v1/` 下 protocol.json、9份逐step
trajectory JSONL、episode_replay.csv、paired_states.csv、stage_counts.csv、summary.csv、
complete.json 和 verification.json。初始/中间/最终 checkpoint 的路径与 SHA-256 完整保留。

执行 `scripts/diagnose_sagh_flat_reward.py`，CPU单线程/nice19，遇其他训练自行等待；216 episode
重放及3,312次相同状态评分完成。离线逐行匹配216条原验证数据通过；七个用户文件保护diff仍为
`1b10c6e1739314267d12bf86b56069efb4efd2a348ecdd0c519a132733908bb2`。

局限：只覆盖父实验12个开发验证窗口、两个workflow、三个固定检查点；不代表全部训练状态、
未接触数据或更长训练后的行为。未修改任何规则来消除平线，也未开启 formal/holdout 或改论文。
后续扩大训练属于用户另行明确授权的独立实现阶段，预算与图形规则先冻结，保留所有负结果。
