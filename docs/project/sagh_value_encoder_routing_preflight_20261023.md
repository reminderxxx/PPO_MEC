# SAGH T/D单因素预检完成索引（2026-10-23）

按用户授权在codex/sagh-value-encoder-routing-ablation完成一次T/D比较，科学Git c12a27b7932410c5c20d3e92c9b49c4ce1e057b3、基线1627f96；未合并算法优化主线。D只阻断原weighted value loss对encoder.*的梯度，保留value前向、value head训练及全部非encoder梯度（含DT/memory shared_other）；显式一阶路由，不整体detach critic输入。checkpoint生产/推理消费合同和raw/routed梯度诊断同步。

389专项及相关回归、12个既有真实记录snapshot的26,400项参数—目标匹配PASS；前向、其他目标、CF targets/gate/原分母/weight不变，value encoder精确0，非encoder value梯度逐项保持。最初系数乘法顺序的浮点差异在真实预检前由匹配拦截并更正，失败匹配日志保留；没有因此消耗真实交互或科学重试。相同输入的图匹配与历史全精度GAE重现分别界定。

4fresh cell/16checkpoint审计PASS，3072训练+1632验证=4704真实交互，低于5184硬上限；剩余480不使用。T两seed全部轨迹/网络精确重现旧T，仍是fresh训练。84组独立梯度投影、原生奖励账本、source/旧产物/曲线及七用户修改hash通过；D的value head与其余encoder更新均实际发生。

局部encoder冲突减少没有转化为一致的跨头改善或真实服务收益；全部seed、正负结果、探索bonus与机制兑现分别报告，CF覆盖和自然remap缺口继续保留。原CF门、分母、所有辅助系数、reward、teacher与DT规则均未改。无正式训练、holdout、阈值/权重搜索、追加预算或下一轮自动实验。

详细未公开结果仅本地artifacts/analysis/sagh_value_encoder_routing_preflight_20261023_v1：入口report.md、augmented_result.json、recorded_matching.json、comparison/preflight_audit.json、supplemental_audit.json、independent_check.json、figures与command_receipt.json。三张新图保留全部原值，训练/验证分开，无平滑、插值或历史曲线覆盖；本地交付zip不含checkpoint或原始数据，不上传public仓库。

reviewed_at：2026-10-23T16:46:35.005059+08:00；literature_cutoff：2026-09-28（未新评novelty）；target_venue：CSCWD2027；artifact_run_id：sagh_value_encoder_routing_preflight_20261023_v1；policy_version：tmc_review_policy_v3_20260621；scientific_git：c12a27b7932410c5c20d3e92c9b49c4ce1e057b3；evidence_level：MATCHED_PER_PARAMETER_GRADIENTS_AND_BOUNDED_DEVELOPMENT_COMPARISON_ONLY。matched development非正式/独立holdout，长期因果、稳定收益、策略/参数收敛及paper-ready均UNVERIFIED，novelty N/S。本轮完成后停止。
