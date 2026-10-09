# CSCWD 长预算 SA 逾期与成本只读定位计划

- `frozen_at`: 2026-10-09（Asia/Shanghai）；本诊断实施前冻结
- `scientific_commit`: `d25ebcded6b43b69b82bae825b035adc1d6f19c4`
- `source_run_id`: `cscwd_causal_strong_baselines_budget_extension_20261009_v1`
- `scope`: 400 learned episode、2,811 行行为原件的已消费开发数据；不重训、不写论文、不更改 B 工作树
- `target_venue`: CSCWD 2027（拟投）；不生成论文表或作投稿晋级判断
- `policy_version`: `tmc_review_policy_v3_20260621`

## 固定输入与停止门

只读 B run 的 terminal、run manifest、integrity 清单、400 行 evaluation、2,811 行 behavior、20 个 selected checkpoint 的 hash、selection curve 与训练/优化记录。核对 B 的环境/encoder/agent/base config 与因果实现 `a088693` 对应文件无科学语义改动，source interval/predictor 身份一致；不加载旧 oracle。科学结果已由 B 完成，`analysis_failure_receipt` 是后处理 Python `zip(strict=True)` 兼容故障，不重新训练或在本分支修 B。输入身份或逐事件重放不一致则停止，不修原件。

## 全实例代价与配对口径

固定使用原 400 个 episode、20 个实例、五 seed，不挑结果。重放记录的动作以补齐逐事件分量，并逐步对齐原 behavior/episode：节点 compute、vehicle fallback、失败等待、current bundle model load、模型/状态/输入网络传输、state restore、DAG recompute。所有分量之和必须等于真实 step cost 和 episode modeled completion time。另保留 action/mask、当前/目标 bundle readiness、已发生 handoff、预测 handoff 距离、目标缓存 resident before/after、admission/victim/reload、准备 commit、节点进展与首次 deadline crossing。区分合法准备的当前服务、目标缓存延迟副作用和下一 RSU 实际可复用性；不以 action 4 数量、低 transfer 或“无 migration”单项下结论。

对 SA 全部逾期完成与未完成 episode，与**同 split、实例、seed** 的 PPO 和 DT 分别成对；计算全实例与双方共同完成子集，列 completion/on-time coverage、时延、模型/状态/输入字节、重算、失败等待及其分解。首个动作分叉只有此前动作前缀和公开状态 hash 相同才作同状态差异；之后仅描述各自轨迹成本，不用后续不同缓存状态推出单动作因果收益。逐 seed、split、deadline 区域（按期、逾期完成、未完成）保留正负样本。

检查 B 现成 checkpoint selection curve、selected checkpoint 的 dev 服务指标和训练曲线；冻结规则本来按 on-time 第一、completion 第二排序，不能凭结果反推“未重视 deadline”。若 dev 四实例无法解释 frozen 结果，标注 selection/generalization 未识别，不追加 checkpoint 评估或选择。

## 可选诊断上限

若原轨迹仍无法区分动作选择与状态分布/表示，最多选 60 个共同状态。只从 B 记录的**完整 train rollout episode** 重建：各 `(method,seed,train design_id)` 取最早完整 episode 的已记录动作前缀（只要唯一 design/episode 映射可复核），加 B 已保存的 dev 轨迹若其动作也完整可追溯；不按 check 结果或收益选状态。按 `current_ready×target_ready×已发生handoff×公开预测距离(unknown/1/2/3+)×estimated_cost_bucket` 分层，各层按 `(method,seed,design_id,training_episode_index,step_index)` 排序轮转，最多 60。20 selected checkpoint 每状态 deterministic forward 一次，上限 1,200；单列 raw head、五动作概率、mask 投影、最终动作，网络/optimizer/normalization/checkpoint hash 前后不变。无完整动作轨迹则停止可选前向，不伪造 train/dev 状态。

若仍需要即时机制反例，最多 12 个从上述非 check 状态按同一顺序预先选定的局部状态；每状态检查全部合法动作，各分支最多 2 步，第二步固定 action 0 若合法否则 action 2。只作离线局部诊断，真实未来仅在克隆环境内部执行，不提供给 policy；不作新性能矩阵。若重放已定位，记 `0/12`。

## 输出与止步

输出 create-only 本地完整事件/episode/配对机器表、输入和输出 hash 回执；Git 只提交脚本、报告、小型摘要和必要测试。最终只能提出至多一个有观察支撑的最小干预候选，注明使用哪些**公开**信息、是否所有算法共享、属于纠错/已有技术/待文献核验创新，并给公平 A/B 设计与否证门；本轮不实施、不训练、不改 reward、actor、network、guard、environment 或 B 原件。如资料仍不足，明确“未识别”。不触及论文、主稿或论文表。
