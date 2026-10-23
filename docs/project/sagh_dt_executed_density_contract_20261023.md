# SAGH DT环境动作概率D/E单因素合同（2026-10-23）

独立实验分支codex/sagh-dt-executed-density-ablation，基线科学c12a27b；不合并算法优化主线。D的digital_twin_env_supervision_contract=proposal_v1保留原环境动作CE，E=executed_marginal_v1只改该项为最终执行概率NLL。两臂同保留DT各头CE、teacher/原标签、positive-advantage加权/cap、各scale/schedule、CF门与原分母、argmax、block_value_to_encoder_v1、clip/Adam/KL和奖励。默认仍proposal_v1，checkpoint config生产及推理filter同步。E要求shared_train_eval_v1和executed_marginal_v1辅助合同，缺少配对rows/masks或mapper history失败关闭。

最终执行概率由同一production mapper枚举合法base/conditional option，恢复每条pre-decision history，aliases经logsumexp求和；合法动作0保留。不可达DT目标为连接计算图的零，仅环境动作项零贡献，DT各头CE继续训练；该row仍占原DT eligible均值分母，不重归一化、不换teacher target。可达全alias产生零梯度须与可达性分开报告。

新增dt_env_supervision_coverage及digital_twin_env诊断区分unique/实际重复loss访问、reachable/unreachable、原/active weight和env scale权重、最后目标与原分母。梯度分解digital_twin_components保存环境、全部heads、fast head的带系数probe，与optimizer terms分开，不重复计入total。诊断只观察不改变前向、RNG或mapper memory；D须逐参数匹配冻结原D，E所有非DT env目标及DT head梯度保持。

限定入口scripts/preflight_sagh_dt_executed_density.py，仅有--execute-preflight模式，无正式训练模式。新output root create-only、禁止resume/retry/覆盖；D/E×seed7/13四fresh cell，每cell768训练交互、6个原128-step unit、3次learner机会、4checkpoint；四固定验证点0/256/512/768×6episode×22步上限。训练3072+验证最多2112=总硬上限5184，每个core env.step前计费（失败调用不返还），没有额外真实smoke/试跑/预算补充。沿用原冻结plan36f01993的window_class/原始区间、j_3/j_8、数据hash及原native offset0账本；CPU单线程/nice19/单cell顺序且其他训练优先。

先匹配测试与12个原记录snapshot的冻结D/新D/E逐参数检查，通过并提交冻结科学Git后才运行一次预检。所有旧checkpoint/曲线/数据只读保护，不继承旧训练权重。审计失败停止且保留现场；负向、零收益和监督覆盖损失全部报告。冲突次数、loss或探索bonus不代替真实服务完成、continuity、delay/miss、backhaul和realized prepare/strict success。此为matched development，非formal/独立holdout、收敛、算法优势或paper-ready证据；novelty不重评。

本地本轮根artifacts/analysis/sagh_dt_executed_density_preflight_20261023_v1；详细结果、样本、checkpoint和真实数据不随public仓库提交。审查完成记录实际reviewed_at，cutoff沿用2026-09-28、target CSCWD2027、policy tmc_review_policy_v3_20260621、源Git及evidence。仅本轮有界执行获授权，完成后停止，不自动开下一轮。
