# 部分预复制与模型单独预取：独立修复合同

日期：2026-10-15。前置独立审查 `joint_coordinated_gate_review_20261015.md`（9352f1b），保留84028c1七配置整轮。状态：IMPLEMENTED_VALIDATION_IN_PROGRESS。

## 修复范围与旧版本

- `--coordinated_controls --partial_preparation` 选择新分支，旧flag默认false、旧checkpoint与严格完整复制逻辑保持可重放。网络维度不变，checkpoint options显式保存partial_preparation_enabled，manifest protocol为joint_parameterized_ppo_coordinated_partial_v2。
- 不再要求完整state在ETA前传完。按目的站单链路预算，先估计模型预取量、可提前推进的state前缀、到站后剩余老版本恢复及增量恢复，比较未提前准备的reactive路径。源端可能完成一个节点时，把声明输出大小计入增量增长保守估计。
- 模型预取与state prepare各自计算净收益代理；state只按相对“已有模型预取计划”的边际收益触发，不能把cache收益算成state收益。模型单独有收益、state无收益时仅保留cache机会，prepare保持noop。
- 已有正确版本的copy不会被当作可再次推进的state；未完成在途前缀按剩余量继续，ETA前推进量有限，传输剩余部分可以到站后正常收费续传。成本仍为声明参数，机会/完成收益是代理估计而非已知真未来。
- 额外约束：预测实际执行站将失去覆盖、目标覆盖有裕量、confidence>=.5、剩余episode至少容纳当前节点恢复/服务；拥塞代价或增量成本抵消收益时不准备。仍不读取真实未来。
- 后续执行站选择使用自己已经付费传输的在途前缀/正确版本，修正只看完整copy导致忽略partial进度的问题。所有三方法共同使用此表示和费用；它不是SA专有奖励。

## 冻结再训练

- 本阶段三配置：full SA、MAPPO、SA-no-prediction；每配置seed7/17/27、128episode、固定20train/8dev、每32评估、主结果固定最终128，共新增1152episode。
- lr、gamma、epoch、batch、head-credit、原始reward/资源、GRU训练和3帧horizon保持上一阶段同一设置。新flag对所有三配置一致，无预测消融仅关闭预测使用，仍具有partial恢复的公共真实状态表示。
- 输出 `artifacts/joint_coordinated_partial_control/seed<seed>/<run_id>/`。旧七配置完成后独立审计，不混入新阶段三配置主排名；不选择最好checkpoint或dev窗口，不打开formal/holdout。
- 源发起cache-only或partial-copy的次数、到站残余传输、实际覆盖离开、提前完成copy的贡献与原始奖励需分开报告。完整/无预测同分也保留，不能为差距强制prepare或加bonus。
- 3帧horizon仍限制提前预算；state增长、队列、滚动confidence和模型就绪存在估计误差。该阶段不解决模型/状态真实硬件成本标定与完整动力学surrogate、learned eviction。

## 验证

343项相关回归通过；包含全量传输不及ETA但有部分复制价值、模型单独预取、负收益/期限拒绝、已有copy不重复准备、inflight进度进入选站费用、概率/更新/checkpoint，以及旧默认合同。生产链smoke与干净commit下的128episode主运行分开记录。


## 2026-10-15 生命周期修订 v3（独立审查后实现）

独立审查 `joint_output_lifecycle_review_20261015.md` 已先提交。58796cc试跑中止并保留各run_abort.json与所有旧产物，不能拼入新主比较。新增output_aware_cache checkpoint选项默认false以重放旧模型；新partial入口统一为true，protocol为joint_parameterized_ppo_coordinated_partial_v3。已计算结果交付不再强制加载模型；模型机会与选站费用使用观测到的在途剩余字节。所有三组共同修复，不改奖励、资源、网络和128×3seed预算。

新输出：`artifacts/joint_coordinated_partial_control_v2/seed<seed>/<run_id>/`。旧partial试跑额外训练成本单列，不作为正式预算或最佳checkpoint候选。训练开始前冻结主比较为最终128；不选择中间最好结果。

生命周期v3验证：346项相关回归通过，py_compile/diff检查通过，真实冻结数据入口4episode×3配置smoke完整结束（joint_lifecycle_smoke_20261015/20261015T025720Z_development）；旧partial中止共516episode（seed7/17/27分别174/170/172），单列为缺陷定位成本。
