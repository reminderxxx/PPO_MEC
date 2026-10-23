# SAGH动作语义修复与有界预检（2026-10-23）

本轮用户授权：在独立实验分支修复动作0误标、训练/验证映射不一致，补奖励/门控/辅助目标诊断，执行匹配测试与小预算预检。未授权正式训练、参数搜索或算法优化主线合并。此前只读诊断已经完成，此轮为另立实现任务。

分支`codex/sagh-objective-credit-alignment`从科学Git45c8c88建立，新worktree与主工作区、原科学source和资源launcher分别隔离。不修改环境原生奖励、已有配置数值、原图、旧run/partial/checkpoint。三个旧本地产物目录及交付zip在实施前形成逐文件SHA清单；主工作区七处用户修改独立保护，不重置或纳入此分支。

## 实现合同

- 合法动作0在行为目标producer/consumer、argmax/counterfactual及gate信用路径中保留；只对None/字段缺失使用默认动作，不以truthiness代替缺失判断。没有修改奖励规则或目标权重。
- `execution_mapping_contract=shared_train_eval_v1`显式启用共同映射：随机/贪心仅负责从同一base/option分布选择，fallback和option选后改写对两者执行相同规则。采样option与执行option分开保存，联合logp继续对应采样路径；不把前置proposal当最终动作边缘概率。
- 默认保留显式legacy映射模式以识别旧checkpoint行为；新预检配置显式写入shared合同并贯通save/load/inference过滤。该模式目前拒绝raw-policy绕过、空合法mask、deterministic smoothing、model-critic option improvement/环境critic及learned planner等未匹配组合；没有静默放宽安全条件。
- 逐训练/验证步保存原生reward分项、真实动作、semantic state、窗口/任务身份、网络及各先验阶段、mask、采样base/option和完整guard/fallback理由。exact最终动作边缘概率只在事前固定的同状态探针枚举，不伪称每步都已边缘化。
- 每次learner调用仅在第一minibatch、优化器执行前记录实际加权目标分项的encoder/actor-head/critic梯度范数及相对native-return actor的cosine；逐transition保存native advantage、各PRD/dual及行为目标。native actor和credit差额仅为诊断probe，不加入优化目标。测试验证诊断不改变权重、Adam、随机数或原loss；无辅助系数调节。

## 预检预算冻结

候选为两项语义修复组合的工程预检，不是单因素效果消融。4算法SAGH/PPO/MAPPO/QMIX × seed7/13；每cell fresh网络/optimizer/replay，768训练交互、3次256步learner机会、6个128步window/workflow unit。合计6144训练交互、24次learner机会、32checkpoint（每cell initial与3次更新）。learner机会与实际接受Adam步单独核对。

原科学plan哈希36f019938092eed36bb5bd34e23d88532156db37aae88c2704fa2f689dd4eef4只读引用配置/数据身份。在train与validation各按冻结offset顺序选择三个window_class中第一个窗口，不读reward选择；两workflow原样保留。每次initial/更新后验证6个unit，合计192个验证episode，最多4224验证交互；总环境交互上限10368。固定初末同状态边缘概率探针不执行环境step。未开启新formal/holdout。

小预算runner无正式训练模式，无扩大steps/seeds/algorithms的CLI；output及cell create-only，拒绝resume或覆盖。CPU单线程、nice19、单cell顺序；复用原71ca85d安全资源扫描，遇其他训练/benchmark只让本预检等待，不影响其他任务。数据文件全量SHA、原奖励/环境/config字节和source commit/hash在启动前冻结。

```bash
PPO_MEC_EXPERIMENT_PROJECT_ROOT=/Users/howen/Projects/PPO_MEC PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python -B scripts/preflight_sagh_execution_repair.py --parent-root /Users/howen/Projects/PPO_MEC/artifacts/analysis/repaired_fair_reward_continuation_20261022_v1 --output-root /Users/howen/Projects/PPO_MEC/artifacts/analysis/sagh_execution_repair_preflight_20261023_v1 --execute-preflight
PPO_MEC_EXPERIMENT_PROJECT_ROOT=/Users/howen/Projects/PPO_MEC PYTHONDONTWRITEBYTECODE=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python -B scripts/audit_sagh_execution_preflight.py --run-root /Users/howen/Projects/PPO_MEC/artifacts/analysis/sagh_execution_repair_preflight_20261023_v1
```

以上v1命令已经执行，root现已冻结，不得重复。失败保留现场、独立报告，不自动转正式训练。完成预检也不授权正式训练或收益声明。

## v1检查器故障与有界剩余预检

候选2eb5263的两个SAGH cell完整通过各768步/3调用/4checkpoint；随后PPO seed7在256步后触发预检检查器。独立只读重算256条初始logp误差为0，确认检查器把配置`target_kl=0`错当零上限，而原生`_guarded_policy_step`一直使用既有回退0.01。只修预检/auditor的核验口径，不改原生KL守卫、配置或任何算法数值；学习返回值改为先记录后验证，避免失败丢失诊断。

v1完整保留。v2仅只读引用全部两个完整SAGH cell，原生运行时（全部src及fair runner/诊断observer）逐字节核验相同；其余6cell全部fresh，失败PPO partial不拼接、不加载。v2接受集合仍8cell/6144训练交互（2继承+6fresh），实际新增4608；v1失败PPO的256训练步与其验证交互另列开销，因此本轮实际训练交互合计6400。原partial/checkpoint及全部v1文件hash写入v2操作回执，不按结果筛选继承。

```bash
PPO_MEC_EXPERIMENT_PROJECT_ROOT=/Users/howen/Projects/PPO_MEC PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python -B scripts/preflight_sagh_execution_repair.py --parent-root /Users/howen/Projects/PPO_MEC/artifacts/analysis/repaired_fair_reward_continuation_20261022_v1 --output-root /Users/howen/Projects/PPO_MEC/artifacts/analysis/sagh_execution_repair_preflight_20261023_v2 --reuse-completed-sagh-from /Users/howen/Projects/PPO_MEC/artifacts/analysis/sagh_execution_repair_preflight_20261023_v1 --execute-preflight
```

审计入口同上，`--run-root`仅替换为v2。新root和剩余cell仍create-only，无正式训练模式。修复后的候选先运行包含零target原生KL上限和拒绝不匹配继承的匹配测试，再冻结commit执行。

## 验证状态

349项匹配/相关回归测试通过：`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python -B -m pytest tests/test_sagh_execution_repair.py tests/test_repaired_probability_contract.py tests/test_window_selection_contract.py tests/test_env_contract.py tests/test_algo_pool_contract.py tests/test_checkpoint_compat.py tests/test_long_fair_reward_training.py tests/test_fair_reward_training.py tests/test_training_reward_plots.py -q -p no:cacheprovider`。六个实现/测试文件AST及scoped diff检查通过。真实预检尚待候选commit冻结，完整回执将在预检后追加。只检验动作语义、完整采样密度、共同映射、诊断透明性、native账本、fresh初始化、checkpoint/hash与预算，不以短跑reward改善作为通过条件。

reviewed_at：2026-10-23；literature_cutoff：2026-09-28（未新评novelty）；target_venue：CSCWD2027；artifact_run_id：sagh_execution_repair_preflight_20261023_v1；policy_version：tmc_review_policy_v3_20260621；Git基线：45c8c88；实际候选Git见execution_receipt；evidence：实现与有界预检待验，正式效果/holdout/paper-ready均UNVERIFIED。
