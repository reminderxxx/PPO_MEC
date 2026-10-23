# 修复后SAGH目标与周期波动只读审查索引

用户本轮要求先只读查因并给独立实验分支方案。本轮未改奖励、曲线、算法或优化主线，未新建实验分支、训练、调参、追加预算或打开formal/holdout。

已对完成续作的三个SAGH完整cell进行原生奖励分项、CSV/更新账本、初末动作轨迹、固定初态探针及冻结源码交叉核验；概率修复有效不等于已解决目标/执行语义问题。周期形态与固定重复的窗口/workflow日程高度一致；验证下降集中于少量机制窗口动作变化。训练/验证fallback差异、辅助监督动作标签语义、潜变量多对一映射和目标方向差异分别记录，未把标量loss或相关性冒充梯度因果份额；旧故障轮的分项归因不能套用新run。

详细未公开实验结果、源码定位和优化方案只保存在Git忽略的`artifacts/analysis/sagh_repaired_objective_audit_20261023_v1/`：`report.md/json`、`supplement.json`、`optimization_plan.md`及两份离线归约脚本。独立方案拟从科学45c8c88另建`codex/sagh-objective-credit-alignment`，先处理可确认的语义问题和映射一致性，再依据梯度证据决定有限辅助目标候选；本轮只有设计，没有实施授权或新增计算预算。

验证：使用主`.venv/bin/python -B`执行`audit_source.py`和`supplement_source.py`，原生分项恒等式、receipt/CSV/预算、验证差额、phase归约、概率/KL边界、输入hash及七文件保护hash检查通过；没有新rollout或完整算法回归。现有日志缺验证分项和分项梯度，相关因果份额未验证。原图、原交付包、数据、checkpoint和两处冻结source不覆盖；详细包不上传公开仓库。

reviewed_at：2026-10-23（精确时间见本地JSON）；literature_cutoff：2026-09-28（未新评novelty）；target_venue：CSCWD2027；artifact_run_id：repaired_fair_reward_continuation_20261022_v1；policy_version：tmc_review_policy_v3_20260621；科学Git：45c8c88e390f49fc4e66c70871c2318e18883f24；launcher Git：71ca85dcac7f709bc27fc1c3e115f771f50a4cb4；evidence_level：E2_ARTIFACT_AUDITED_MATCHED_DEVELOPMENT_ONLY。非独立holdout/paper-ready，不作算法优势或策略收敛判断。
