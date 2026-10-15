# 参数化控制两轮训练与机制回收

- `reviewed_at`：2026-10-15（Asia/Shanghai，客户端日期）。
- `literature_cutoff`：2026-10-14；`target_venue`：IEEE TMC（内部审查边界）。
- `policy_version`：`tmc_review_policy_v3_20260621`。
- `artifact_run_id`：初轮 `joint_parameterized_learning/seed{7,17,27}/20261014T105504Z_development`；修复轮 `joint_parameterized_learning_productive/seed{7,17}/20261014T111122Z_development`、`seed27/20261014T111123Z_development`。
- Git：初轮 `a7a06a0`；修复轮 `b6a0dbe9af3d1dac3035bb623012a1f7260636ba`，全部 manifest `git_dirty=false`。独立表示审查 commit `899c6b3`。
- Evidence level：开发原始值已审计、首 dev 窗口全部最终模型独立重放；**论文完整证据 `E0_UNAVAILABLE / Unverifiable`**，没有 formal/holdout/support，不作 canonical 晋级。

## 结论

工程接入和同预算开发对比已经完成。新版 SA、MAPPO、仓库内缓存 PPO 都能通过同一参数化控制合同训练、保存、独立加载和评估；训练窗口拟合的冻结 GRU、延迟服务状态共享、DAG 编码、return/cost surrogate 已接入。公共 identity/冷缓存条件修复消除了 SA 的全等待退化，但 **SA 主机制目前没有带来奖励或完整请求完成率优势，也没有稳定收敛**。没有修改奖励、选最好 checkpoint、筛掉负窗口来制造胜出。

## 1. Artifact 完整性

每轮五配置 × 三 seed × 128 episode，共 1,920 episode；两轮共 3,840。每32 episode保存一次，总计120个checkpoint。每配置最终checkpoint评估8个dev窗口，三seed共24个配对结果；每轮规则另外重复执行三次，汇总去重后只计8个窗口。

产物包括 manifest、command log、training.jsonl、dev.jsonl、最终逐请求 events.jsonl、GRU 和 checkpoint。审查机器输出：

- `artifacts/joint_parameterized_learning_audit_20261014/audit.json`
- `artifacts/joint_parameterized_learning_audit_20261014/audit_productive.json`
- `artifacts/joint_parameterized_learning_audit_20261014/matched_rule.json`

初轮核验131,533条事件/17,148份帧收据；修复轮127,967条事件/17,148份帧收据。所有checkpoint文件hash与tensor hash通过，初始/训练tensor确实不同。30次独立重放（每轮每配置每seed的首dev窗口）逐事件一致。没有把独立首窗重放写成全窗口重新运行。

## 2. Protocol / provenance

- 源数据与冻结来源为八车v2机制开发manifest `20261014T074430Z_development`，训练入口按数据SHA和窗口逐字段重建；旧v2与v3的动作/IO物理链不同，奖励不能直接跨版本当算法优势比较。
- 每轮内部共同使用20个train、8个dev窗口、相同synthetic成本、同共享延迟、同冻结GRU、相同训练预算和最终128 checkpoint。训练顺序固定，dev不回写梯度，评估恢复训练RNG。
- 初轮缺站点identity导致执行→缓存条件歧义，独立审查后另开productive修复轮。修复轮所有方法共同获得站点identity、自身位置/roster、上帧声明服务负载的延迟共享，以及冷站当前adapter closure条件mask；无执行后覆盖动作。两轮负结果全部保留。
- 检查原始time_start/time_end互斥，并将全部dev事件time_index逐条对齐重建源帧。窗口相邻、同源且部分共享车辆，不能假设24个结果独立同分布。
- formal/holdout未执行。dev反复用于工程诊断，不能当独立外部验证。

## 3. 同预算结果与公平性

### 修复轮：固定最终128 checkpoint，8 dev × 3 seed

| 配置 | 平均奖励 | seed均值标准差 | 完整请求 / 8 | 完成节点 / 70 |
|---|---:|---:|---:|---:|
| MAPPO | 32.537 | 8.230 | 1.625 | 44.208 |
| 缓存 PPO | 28.483 | 9.459 | 1.083 | 40.708 |
| SA，无 surrogate | 28.186 | 8.567 | 1.250 | 40.458 |
| **SA，完整机制** | **27.333** | **5.720** | **0.833** | **39.458** |
| SA，无预测特征 | 34.537 | 5.231 | 2.083 | 46.000 |

SA相对MAPPO奖励差−5.204、完整请求差−0.792；24个配对reward中6胜18负，8个窗口先跨seed平均后全部为负。相对缓存PPO奖励差−1.150；相对无surrogate差−0.853；相对无预测差−7.204、完整请求差−1.250，24个reward配对4胜20负。仅描述当前开发结果，不作显著性或novelty推断。

SA无预测保留DAG与surrogate，相对MAPPO的开发均值为奖励+2.000、完整请求+0.458，是值得保留的候选信号。三个seed的奖励差约+0.075/+0.466/+5.460；完成率差为−0.375/−0.125/+1.875，改善主要由seed27贡献。没有独立DAG消融及新的冻结验证，不能据此认定稳定算法优势或把预测从研究问题中删除。

初轮固定最终结果同样保留：SA −7.594 / 0.208请求；MAPPO 4.994 / 0.875；缓存PPO 1.858 / 0.583；SA无surrogate 2.316 / 0.250；SA无预测 −10.370 / 0。初轮的不同站状态特征混叠与waiting/adapter starvation使其不适合作为成熟算法能力排名。

### 规则参照的公平性说明

原规则每帧admission，学习策略每4帧，原规则56.476 / 5.875仅是效率参照。为排除周期差另做只读开发诊断：规则同样每4帧、只在未就绪且未加载时请求当前adapter，奖励57.308、完成6.250/8。该规则仍采用集中顺序选站，是**实验后效率诊断**，不是预冻结外部论文baseline，也不是拿来选学习checkpoint的teacher。

三种学习策略内部输入和动作约束相同；SA额外启用DAG/surrogate，cache PPO额外启用cache-context编码。模块虽统一实例化，baseline存在未使用模块，总参数数不能直接当有效学习容量；缓存基线仍是first-party PPO，不是外部论文独立DRL复现。

### 收敛检查

| episode | SA奖励 | MAPPO奖励 | 缓存PPO奖励 | SA无surrogate | SA无预测 |
|---:|---:|---:|---:|---:|---:|
|32|31.618|28.779|32.345|31.545|30.372|
|64|33.903|30.423|36.167|30.640|28.836|
|96|34.699|35.407|31.691|32.356|31.523|
|128|27.333|32.537|28.483|28.186|34.537|

SA的完整请求均值依次1.333、1.708、1.750、0.833；不能称为收敛。未采用96的最好点替代预先冻结的128最终结果。继续增加episode可能改变结果，但现有曲线不支持“只差训练量”这一定位。

## 4. 机制是否兑现

### 协同选站仍退化

修复轮SA多窗口把所有请求帧集中在一个RSU：seed7的2760/2880/3000/3120/3600窗口100%在rsu_c；seed17/27也有100%在rsu_d的窗口。其他窗口单站占比约61%–93%。SA最终24个episode共有7,519个compute_queue、4,911个output_link_queue事件。

这是观察到的拥塞与选站集中。原因推断：共享actor的逐请求采样、团队统一return、直到节点变化才释放的站点提交，尚未学出避免同时挤站的协同决策；已共享的服务负载尚未被有效利用。不能把该推断写成已完成的因果消融。

### 预测准确率改善没有转成净收益

GRU在训练窗口内留出的4个验证窗口上，位置RMSE 0.154922，常速度0.176728，下降约12.34%。三seed使用同一冻结模型。该指标不是跨站迁移收益或校准置信概率。

最终SA平均每个120帧episode只有0.417次association handoff，却发出513.333条prepare命令，实际状态链路占用100.375帧，prepared_on_arrival为0。SA无预测对应178.208条prepare、49.042状态链路帧，奖励与完成率更高。命令数包含重复命令，不等于513次实际迁移。当前证据显示预测驱动的准备没有兑现到达收益；具体由哪些预测特征造成需后续独立消融。

### cache有成本信号，但不能宣称“处理更优秀”

SA每episode模型链路18.292帧，MAPPO33.667、缓存PPO22.708；按完成节点归一化分别0.464、0.762、0.558帧/节点。但SA完成节点和完整请求更少；集中使用单站也会减少模型复制，并放大计算排队。规则相同cache周期仍有更高完整完成率。故低模型传输量不是当前主问题整体优势的充分证据。

SA有42.5条future-cache、0.25条base-only命令/episode，实际会发起已知未来DAG的模型准备；尚未给出独立timeliness/eviction收益消融。**显式victim仍不学习，所有方法共用typed LRU**，不能宣称完整学习了更优缓存替换。

### surrogate仍是第一版

接入的是实际训练rollout的团队return、命令初始cache槽和state槽回归，三个条件控制阶段分别预测，再以bounded residual影响logits；没有完整状态转移imagination、因果反事实或可信区间。持续模型fetch费用进入团队return，但不全部进入命令初始槽label。

独立首dev窗口重放中，SA三seed的return RMSE约5.892/3.443/6.593 reward单位；state cost MAE约0.137/0.276/0.344 link slot。该小样本诊断不能证明用于未选动作规划的准确性。完整SA相对无surrogate没有奖励/完成率优势，不能把“存在辅助loss且下降”当机制有效证据。

## 5. Claim 边界与后续实现任务

当前仅能声称工程动作合同、因果GRU、共享服务负载、DAG/观察性surrogate可运行；不能声称SA领先、model-cache更优、预测有调度收益或已收敛。所有费用仍为未实测synthetic profile；串行DAG、单批到达、控制共享不占数据链路、状态存储无上限仍未覆盖。

本审查轮只读，不因负结果继续修改算法。建议独立实现任务按次序处理：

1. **请求间协同与credit**：显式传递同轮服务意图/预约量，审查共享actor是否实际能区分并发请求；用请求/控制头的边际服务贡献和队列代价构建可核验credit。先解决一站拥塞及节点级提交锁定，再讨论训练放大。
2. **状态准备的触发与净收益**：将实际覆盖将失效、迁移ETA、copy版本/增量字节、剩余工作、竞争链路放入触发特征与目标；低机会时允许稳定noop，分别验证迁移事件/无事件窗口。不能用人为奖励bonus强推prepare次数。
3. **更完整surrogate与cache**：预测可核验的ready时刻、排队/进展和完整传输费用，并验证误差/非负成本/不确定性；将未来adapter的及时命中、避免重复拉取、依赖安全victim选择单独做机制验证。
4. **机会与规模**：先用train窗口量化覆盖交接和cache竞争机会，再冻结覆盖/负载/到达扩展；不得挑dev有利窗口制造机制优势。完成这些修复后同预算重训；统计晋级前补足外部baseline、独立formal/holdout与校准。

源码还显示两处明确的表示欠缺：共享payload虽提供 `service_work_units`，当前actor只消费服务请求数；DAG编码只做全图池化，cache候选未接入各自节点的GNN表示/拓扑位置，同尺寸未来节点可能再次混叠。这些需要在后续独立实现中补齐，不能把本轮“所有开关开启”当完整研究设计兑现。

这份任务清单不是已经实现或验证的算法收益。

## 6. Literature novelty

本轮未新增文献检索，也未评价顶刊水平。相关公开参考与差异仍见 `literature_reference_table.md`。当前机制实现、内部基线与工程负结果不足novelty或paper-ready结论。

## 7. 验证与复现

项目解释器：`/Users/howen/Projects/PPO_MEC/.venv/bin/python`，工作目录为八车研究worktree。

- `-m pytest tests/test_parameterized_control_learning.py tests/test_multi_request_target_control_env.py tests/test_verify_joint_target_control.py tests/test_joint_model_cache.py tests/test_multi_request_joint_v2_execution.py tests/test_joint_experiment_launcher.py tests/test_algo_pool_contract.py tests/test_checkpoint_compat.py -q`：327 passed。
- `-m py_compile`：变更的agent、环境、预测器、训练和审查入口通过；`git diff --check`通过。
- `scripts/train_joint_parameterized_control.py --source_manifest artifacts/eight_vehicle_joint_v2_mechanism_budget96_20261014/20261014T074430Z_development/manifest.json --episodes 128 --eval_every 32 --seeds <7|17|27> --output_root artifacts/joint_parameterized_learning/seed<seed>`：三seed全部complete。
- 上条增加 `--productive_controls` 并使用 `artifacts/joint_parameterized_learning_productive/seed<seed>`：三seed全部complete。
- 两种控制各自4 episode生产链smoke通过，不用于上表。
- `scripts/audit_joint_parameterized_control.py` 与 `scripts/audit_joint_parameterized_control.py joint_parameterized_learning_productive`：原始收据、checkpoint、预算、时间边界及独立重放审查。
- 后验同周期规则见artifact `matched_rule.py` / `matched_rule.json`；与训练无交互。
