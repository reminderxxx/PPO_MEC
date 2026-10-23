# SAGH C/T counterfactual一致性预检（2026-10-23）

用户在独立只读报告后授权C/T单因素实现和一次小预算比较。专用分支codex/sagh-cf-teacher-consistency-ablation，基线9cb156a（前轮科学实现864add6）。不合并优化主线，不改reward、环境、encoder、critic、原teacher credit或任何辅助数值系数。

C为counterfactual_consistency_contract=none_v1；T为native_positive_executed_action_v1。T仅当冻结的原生归一化GAE>0且CF目标不同于该row最终执行动作时，使该row CF贡献为0。暂停行仍进入原eligible分母；所有原weight、target、执行前像均保留。原生GAE在mechanism/其他credit之前冻结，不能用credit后优势替代；缺失/非有限值fail closed。默认C与旧源码的loss及梯度匹配。checkpoint生产/推理消费端同步合同，推理行为不因该训练开关改变。

独立入口scripts/preflight_sagh_cf_consistency.py只支持C/T×seed7/13：4fresh cell，每cell768训练、3次learner、4checkpoint及4×6固定验证episode。训练3072，验证上限2112，总真实环境交互硬上限5184。逐次core env.step前记账，失败调用不退款；create-only、无重启/继承/预算扩展。单CPU线程/nice19，遇其他训练让出资源。匹配构造测试与真实数据预检分别报告，不把构造监督注入真实run。

沿用原plan36f01993的三个train/validation类别首窗口、workflow j_3/j_8、原生offset0、128步unit/22步episode。每seed初态tensor与首256步轨迹必须相同，配置仅训练合同不同。未做critic消融、正式训练、formal/holdout或canonical晋级。

输出仅本地主工作区artifacts/analysis/sagh_cf_consistency_preflight_20261023_v1/comparison，含逐次交互ledger、identity、原始step/episode/update、checkpoint和completion/failure。CF coverage按实际loss评估次数及last-evaluated row分别保存；不冒充optimizer步或独立样本。必须同时报告监督覆盖/非零梯度、所有分组方向/幅度、reward分项、机制兑现与全部四固定验证点，不仅比较负cos次数。详细数值及未公开产物不推送public仓库。

运行前先过匹配/相关回归及记录状态重放，固定科学源码commit再执行一次；缺依赖、资源扫描失败或运行故障保留现场，不自动修科学规则或重试。当前结果UNVERIFIED，完成回执只更新实际验证结果和限制。
