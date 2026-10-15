# 协同控制严格门控：完整开发结果独立审查

- reviewed_at：2026-10-15T03:06:02.510037+00:00；literature_cutoff：2026-10-15；target_venue：IEEE TMC；policy_version：tmc_review_policy_v3_20260621。
- artifact_run_id：`joint_coordinated_control/seed{7,17,27}/20261015T015625Z_development`；Git：84028c1146d4b21681e5d46a408c1880f38dd518；evidence level：开发原始记录/最终checkpoint首窗口重放已审计；论文E0_UNAVAILABLE / Unverifiable。

## 1. Artifact 完整性

七配置×128episode×3seed完整，共2688训练episode。84份中间/最终checkpoint哈希、predictor来源、145441条评估请求事件及21945帧费用收据通过；21个最终checkpoint逐一重新加载并重放第一个dev窗口，完整事件相等。独立机器重放、formal/holdout/support正式闭环仍缺。

审计：`artifacts/joint_coordinated_audit_20261015/joint_coordinated_control_audit.json`；命令：`python scripts/audit_joint_coordinated_control.py joint_coordinated_control`。

## 2. Protocol / provenance

固定20train/8dev、seed7/17/27、最终128主结果，四个学习曲线点不选峰值。实际frame/time interval核验互斥；GRU仅16train拟合/4train验证，dev不参与拟合。共同原始奖励/费用/信息/预算及公共负载先验。formal/holdout封存，canonical不变。

## 3. 统计与baseline公平性

| 配置 | 奖励均值 | seed均值样本标准差 | 完成请求/8 |
|---|---:|---:|---:|
| sa_ghmappo | 58.218345 | 0.532152 | 7.125000 |
| mappo | 58.070594 | 0.709366 | 6.958333 |
| cache_offload_drl | 57.644831 | 1.084308 | 6.875000 |
| sa_no_prediction | 58.232348 | 0.699428 | 6.958333 |
| sa_no_coordination | 45.112272 | 8.023728 | 4.083333 |
| sa_team_credit | 57.877332 | 0.793042 | 6.958333 |
| sa_ungated_prediction | 57.813081 | 0.874970 | 6.958333 |

SA对MAPPO奖励+0.147751、完成+0.166667，仅2/3seed奖励更高；对去预测版奖励−0.014004、完成+0.166667。未达“完整版同时超过两者”。24个seed-window单元不是24个独立cluster：同一批车辆/DAG、相邻时间片、前三片多项指标完全相同，禁止据此显著性推断。MAPPO/cache为本项目合同匹配实现，不冒充外部论文原版复现。规则辅助参照reward56.415528、完成5.875，每帧cache控制与学习器每4帧有差别，不列主排名。

## 4. 机制兑现

- 协同：SA对no-coordination奖励+13.106072、完成+3.041667，24单元23胜1负，支持当前共同预约/公共队列先验有效，但尚不能拆解每个子机制贡献。
- 信用：SA对team-credit奖励+0.341013、完成+0.166667，24单元14胜10负，仅描述性支持。ledger逐帧守恒；平均分摊共享cache费用不等于因果边际信用。
- 预测：SA最终dev候选和prepare均0。预先声明的固定权重预测开关24单元完整事件流逐一相等。不能把训练后两个checkpoint差异当作dev迁移收益。
- ungated共475条prepare、49次显式copy完成，却没有显式准备带来的ready arrival；不能把泛化prepared_on_arrival计数当作预测贡献。这支持避免盲目准备，但不能证明代理净收益已校准。
- 覆盖影子：有9个预测离站请求帧，实际上为同一外生离站在三seed下复现，ETA1–3小于约7帧状态传输。严格完整复制gate排除了部分准备；已先独立报告再立partial修复任务。

## 5. Claim边界

本轮已修正此前全站拥挤/行为退化的关键工程缺口，但不等于TMC-ready或最高奖励/收敛证明。四点dev曲线只能显示训练后的稳定范围，仍须独立测试、多seed、样本效率与真实成本校准。新旧轮奖励不作机制净增益：本轮另记意图共享费用。后续partial/lifecycle修正版独立同预算重训，禁止拼表、筛seed或改奖惩追认优势。

## 6. 文献novelty

本轮无新文献检索，不作novelty结论。3帧预测、零延迟控制意图总线、合成模型/状态成本、serial DAG控制及观测型surrogate仍是工程与外推边界。

补充只读真实轨迹核验：`python scripts/audit_joint_prediction_opportunities.py joint_coordinated_control`，产物`artifacts/joint_coordinated_audit_20261015/joint_coordinated_control_opportunities.json`。在confidence门限内、当前执行站仍覆盖且3帧真实未来存在可达替代站的9请求帧，预测覆盖影子也为9帧，均为同一离站事件三seed复现；只说明此小样本没有观察到3帧内漏报，不外推预测器总体准确性。真实未来只用于审计，不进入策略。
