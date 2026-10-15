# 高移动、多站可选支持场景：独立零训练协议

日期2026-10-15，前置独立容量审查a1b6048。原base/tight冻结主表不变，本支持不是重排权重或替换负例。

1. 原8dev时间区间、原RSU布局/1.2覆盖系数、资源费用和八个DAG保持不变。只从对应真实区间取持续车辆，要求每帧至少2站可达，再按站覆盖离开次数、首末位移、ID排序选8车。这个条件按完整外生轨迹定义，属于条件化探索支持，不是自然分布无偏估计，真实未来不传入actor。
2. 单一定义先写mobility_support_plan.json，再生成数据、检查必要容量上界；条件不足即Unavailable，不为获胜搜索覆盖参数或cohort。source仅旧dev，不评估formal/holdout。
3. 公共predictor仍为原b78be998...，不重训、不微调。全部八主配置与两support使用各自同预算训练的最终128、seed7/17/27 checkpoint；增加训练为0。actor/环境/episode代码必须与1667da3一致，checkpoint/tensor hash前后相同。
4. 共240个默认policy单元，另24个full固定权重prediction-off单元。保存逐帧原始事件、费用/结果、模型hash；消费者重放每模型首窗口，共30次。新24单元与主表48预测开关分开报告，不能合并当独立样本。
5. 该支持定义晚于主训练启动及首批原panel结果读取，故明确exploratory，不能回写主计划或据支持得分声明普遍优势。相同8时间区间及cohort仍相关；必要容量条件通过不证明实际8/8调度存在。

入口`python scripts/joint_tail_mobility_support.py prepare|evaluate|collect --root artifacts/joint_tail_control`，evaluate另加`--agent <name>`。产物在root/mobility_support及独立plan；这里的evaluation.json不是训练manifest，不让原主审计误读。数据producer如有未提交的只读报告/审计编辑如实记录dirty_paths，不冒充干净训练；actor源文件与1667da3逐项Git diff校验，producer自身hash绑定。
