# 生命周期修正版：同预算三组完整结果审查

- reviewed_at：2026-10-15T03:31:34.659007+00:00；literature_cutoff：2026-10-15；target_venue：IEEE TMC；policy_version：tmc_review_policy_v3_20260621。
- artifact_run_id：`joint_coordinated_partial_control_v2/seed{7,17,27}/20261015T030133Z_development`；Git：e96f6ac3e6dd6284d22c01c0994637c0545d506a；evidence level：开发原始数据/重放已核验，论文 E0_UNAVAILABLE / Unverifiable。

## 1. Artifact完整性

三配置各128episode×3seed，1152训练episode完整回收。36份checkpoint、71308条最终评估事件、11000帧费用收据通过只读核验；9个最终模型首dev窗口重放，完整事件相等。

原始manifest、training/dev/events、predictor和checkpoint位于上述三个run目录；独立审计`artifacts/joint_coordinated_audit_20261015/joint_coordinated_partial_control_v2_audit.json`，完整日志保存在各seed根目录`command.log`。未执行formal/holdout。

本次有效主比较预算为严格七配置2688 + 生命周期三配置1152 = 3840episode；中止旧partial516episode及四次入口smoke共48episode单列，总训练4404episode（不含此前轮次）。旧partial带run_abort.json，禁止参与排名；四次smoke不是效果证据。没有为获胜选择seed、窗口或checkpoint。

## 2. Protocol / provenance

来源固定NGSIM+Alibaba原manifest，20train/8dev；原始frame/time interval核验互斥，同一批DAG/车辆的相邻dev片不能视为独立样本。三个方法共同使用partial恢复、output-only无强制模型、inflight模型/状态剩余费用、协同预约与prefix信用；共同控制网络尺寸、资源、费用、GRU与训练次数匹配，算法编码和辅助损失按各自定义保留，去预测仅关闭预测消费。主结果固定最终128。

旧严格版84028c1与本版e96f6ac行为约束不同，不混为同一主表；旧版七配置仅作为先前独立机制研究。cache基线未参与本版，不能用旧版cache数值与本版三组作公平排序。

## 3. 最终比较与公平性

| 配置 | 奖励均值 | seed均值样本标准差 | 完成请求/8 | seed7/17/27奖励均值 |
|---|---:|---:|---:|---|
| sa_ghmappo | 57.660698 | 0.830527 | 6.708333 | 58.093779/56.703141/58.185174 |
| mappo | 57.375365 | 0.323439 | 6.750000 | 57.008898/57.496237/57.620961 |
| sa_no_prediction | 57.521120 | 0.436546 | 6.916667 | 57.401288/58.005068/57.157005 |

| 完整SA减比较组 | 奖励差 | 完成请求差 | 24单元奖励胜/负/平 |
|---|---:|---:|---|
| mappo | +0.285332 | -0.041667 | 9/15/0 |
| sa_no_prediction | +0.139577 | -0.208333 | 10/12/2 |

均值优势集中于offset3600：该窗SA对MAPPO/去预测平均差为+2.676134/+2.018752，其他七窗差值均值为−0.056211/−0.128876（仅作影响诊断，主表保留所有窗口）。因此奖励均值第一不等于多数窗口领先或稳健优势。

目标‘完整版奖励和完成率同时超过MAPPO与去预测版’：本轮未达到。不能据此承诺最高奖励、收敛或论文优势。MAPPO为本项目共享合同/先验下实现，不声称外部论文原版复现。

### 开发学习曲线（各点都是固定8dev×3seed）

| episode | SA奖励/完成 | MAPPO奖励/完成 | 去预测奖励/完成 |
|---|---|---|---|
| 32 | 58.562567/7.291667 | 57.962175/6.875000 | 58.356116/7.000000 |
| 64 | 57.902758/6.916667 | 57.927456/6.791667 | 58.552115/7.250000 |
| 96 | 58.021530/7.041667 | 58.007018/6.916667 | 58.300103/7.208333 |
| 128 | 57.660698/6.708333 | 57.375365/6.750000 | 57.521120/6.916667 |

只有四点评估，不足以证明收敛。完整版从32到128开发均值下降，三个seed训练首/末20episode奖励却从58.3580/58.7522/58.5522增至59.9940/59.8656/60.1353（各20块都覆盖全部20train窗口）。训练回报上升与开发回退并存，不能简单解释为训练量不够，也尚未识别唯一原因。

## 4. 机制兑现与剩余缺口

固定最终full权重预测开关、零训练的24单元结果：奖励差+0.0295703、完成差+0.0416667，1胜1负22平。seed7/offset3120实际提前传1024B、迁移请求早6帧完成，整个干预使节点+1/请求+1、奖励+0.7196875；另一个单元多耗state帧而−0.01。不是所有窗口都有机会，也不是所有正net proxy都带来真实净收益。

原始请求账本收支等于团队奖励，输入/输出/模型/状态均核对共享链路上限。完整SA最终dev未使用车端计算兜底。机会计数是候选目标×请求帧，不能当独立迁移次数；prepared_on_arrival也不能直接当预测预复制贡献。

奖励均值已超过两比较组，但请求完成率未超过。SA平均完成67.833333节点，MAPPO67.541667、去预测67.708333；现行正奖励是节点送达数，完整请求不是单独的正奖励项（仅有未完成等待惩罚）。因此两指标并非严格同序，不能用奖励第一替代workflow完成优势；本轮保持原奖励，不为追认结果添加completion bonus。

主要剩余风险有实际witness支持：

1. 服务预约有效，但没有预约目的站未来数据链路；9个prepare命令只有2帧真正推进，其他被queue挡住。需要把model/state/input/output的观测在途量、近期开销及可用时隙纳入共同预约与机会净收益。
2. 代理对正/负真实收益未校准；当前请求账本+prefix baseline不等于三个动作头的因果边际信用，surrogate仍为观测回归。需用独立训练内反事实/干预样本验证收益排序，不能看dev追选。
3. 首dev初始图中输出特征缩放25–50，对比输入0.25–0.71875，图encoder近饱和55%–91%；应在下一独立合同中共同改为service-frame或train-only尺度，并先核验表示。当前只诊断，尚未证明它是排序差距的原因。

详见`joint_partial_prediction_support_review_20261015.md`和`joint_feature_scale_review_20261015.md`。本审查轮未为消除blocker继续调参或改算法，当前比较保持冻结。

## 5. Claim边界

本轮已使部分预复制真正执行、低机会可不准备，并保留了负例。协同的较大收益来自独立严格版消融；本版没有重训协同/信用消融，不能把旧消融差值直接归给本版。不能宣称model-cache机制已比所有缓存基线优秀：本版缺独立缓存消融、learned eviction和硬件标定。重复使用开发集、外生迁移少、合成成本、零延迟意图总线、serial DAG/单批请求仍限制外推。canonical不更新，paper_claim_ready=false。

## 6. 文献novelty

本轮未检索新文献、不作novelty判断；无公开材料上传。

## 执行过的验证

```bash
python -m pytest tests/test_coordinated_control_learning.py tests/test_parameterized_control_learning.py tests/test_multi_request_target_control_env.py tests/test_verify_joint_target_control.py tests/test_joint_model_cache.py tests/test_multi_request_joint_v2_execution.py tests/test_joint_experiment_launcher.py tests/test_algo_pool_contract.py tests/test_checkpoint_compat.py -q
python scripts/train_joint_parameterized_control.py --source_manifest artifacts/eight_vehicle_joint_v2_mechanism_budget96_20261014/20261014T074430Z_development/manifest.json --episodes 4 --eval_every 4 --seeds 7 --agents sa_ghmappo mappo sa_no_prediction --coordinated_controls --partial_preparation --output_root artifacts/joint_lifecycle_smoke_20261015
python scripts/audit_joint_coordinated_control.py joint_coordinated_control
python scripts/audit_joint_coordinated_control.py joint_coordinated_partial_control_v2
python scripts/probe_joint_coordinated_prediction.py joint_coordinated_partial_control_v2
python scripts/audit_joint_prediction_effects.py joint_coordinated_partial_control_v2
python scripts/audit_joint_prediction_opportunities.py joint_coordinated_control
python scripts/audit_joint_prediction_opportunities.py joint_coordinated_partial_control_v2
python scripts/audit_joint_feature_scale.py joint_coordinated_partial_control_v2
```

结果：346项相关回归通过；4episode×3配置真实数据入口smoke完整；两轮完整审计/重放通过，fixed-weight probe及逐项费用重建通过，3个图尺度witness精确复现；修改模块py_compile及git diff --check通过。完整主训练命令在各manifest.command/command.log。没有覆盖formal/holdout、硬件/真实无线/大规模泛化、因果surrogate与独立缓存优势。
