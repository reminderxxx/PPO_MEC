# 5C数值资格独立回收：固定矩阵通过，覆盖仍未齐

- reviewed_at: 2026-10-20T15:50:24.082649+00:00；本地日期2026-10-20。
- literature_cutoff: 2026-10-17沿用，本次无novelty刷新。
- target_venue: IEEE TMC internal software diagnostic；policy_version: tmc_review_policy_v3_20260621。
- artifact_run_id: `sa_credit_numerical_stage5c_20261020`。
- algorithm_source: `2c59719d31c49faaff2158bdd94bcd429008f174`；clean runtime Git: `d16ac102c0e12055f456ba26191eeeb8c9b3c17c`；文档回收commit/push另见私有git_sync.json。
- contract SHA: `0503b913a610f8efe18cdcaf8416c1bea42ac20b6a0815c164c8c2d1cc853f1a`；runner binding SHA: `4291019682b7daac6cc38bef881c99d7faacd818ad7d5e3544dd9b7d0f6bb095`。
- verdict: **LIMITED_PASS_COVERAGE_OPEN**。七组on/off固定数值对照均通过；全头/idle运行资格尚未完成。正式论文证据仍E0_UNAVAILABLE / Unverifiable、N/S；没有退化修复、候选晋级或性能结论。

## 1. 授权、实际费用与封存

用户“继续下一步”承接上轮提出的有界数值资格。先冻结[独立软件合同](sa_credit_numerical_contract_20261020.md)，静态验证105项通过后提交固定runner，bind纯只读核验120个源码绑定、18个原checkpoint和已完成5B元数据/输入；唯一run于15:36:50–15:37:43 UTC执行，未重跑或追加。

| 计数 | 实际 / 上限 |
|---|---:|
| 私有模型实例 | 14 / 14 |
| 原_evaluate调用 | 602 / 1024 |
| 内部nn.Module forward调用 | 24050 / 100000 |
| Categorical sample调用（含forced） | 864 / 2048 |
| backward | 24 / 32 |
| 主/副Adam.step合计 | 48 / 64（各24） |
| 私有合成update调用 | 14 / 14 |
| 环境、autograd.grad、科学新训练 | 0 / 0 |

两侧各做六组正常update及一组KL拒绝。on日志只有其中一侧：144唯一行为/144信用、272跨epoch采样出现、13mini（12接受/1拒绝）、7update，对应主12+副12=24次Adam；不得把此24与整个资格两侧的48混淆，或把epoch重复当新独立行为。

旧checkpoint、源资产和科学run均只读，源/旧资产157次绑定核验覆盖147个唯一路径，前后SHA一致；35份私有核心binary证据SHA一致。原版SA及已有整体/seed7/27保护、seed17自身初始化恢复要求、固定64和尾稳门禁保留；分组候选不晋级。formal/holdout封存、canonical不变、提醒PAUSED。

## 2. 固定矩阵和精确结果

固定全部六个seed×arm的64点已填充Adam checkpoint，另一个seed7原版仅在私有update中加candidate-index×4 logit的控制流test double；它不代表实际科学KL。每组off/on从相同Python/NumPy/Torch RNG出发，使用原dev_046九分位索引0/4/8，即frame0/239/479。逐帧恢复已保存控制memory，使用显式合成监督；不存在环境连续性或历史stochastic MC重建主张。

| seed | 版本 | 每侧行为行 | 每侧正常mini/Adam |
|---|---|---:|---:|
| 7 | 原版 / 分组 | 16 / 24 | 各2 / 4 |
| 17 | 原版 / 分组 | 24 / 24 | 各2 / 4 |
| 27 | 原版 / 分组 | 16 / 24 | 各2 / 4 |
| 7 | 原版KL test double | 16 | 0 / 0（拒绝1） |

独立消费者采用自有逐位比较，55,629对tensor、227,379,712原始字节、949对array通过；包括完整原forward输入/输出、原controls/rollout、裁剪前梯度、原Adam边界裁剪后梯度/参数delta、所有参数/buffer/modes、两optimizer状态/LR、controller及derived cache、更新指标、三类RNG和序列化载荷。仅原rollout附加credit_audit_uids从原输出比较中排除。源checkpoint到初始私有载荷亦精确相同。旧pre-hook算法完整AST一致沿用独立核验，本轮没有另外运行历史代码模型。

KL拒绝保留原randperm RNG消耗、LR设置与completed_updates递增；两侧状态彼此相同，不要求拒绝前后所有状态不变。first-pass执行采样与second-pass强制重评、实际cache/priority采样顺序和原调用计数相同。独立从raw trace重建602/864/24/48/14各项计数；内部module-forward总数依赖冻结guard账，保存原件不能凭空证明所有未记录调用不存在。

## 3. 独立信用账和实际覆盖缺口

第二独立消费者核对7份JSONL与raw binary，信用MC/10/8单位、合成逐帧targets、stored B、raw/normalized advantage、条件logprob/mask、UID/mini索引/epoch、ratio/KL、两Adam/实际delta及gradient账一致。行内4320、跨frame15360个额外padding槽的f1=0、m1=False、node1=-1及未填充prefix均通过。

| head | forced行 | 多选行 |
|---|---:|---:|
| execution | 69 | 75 |
| cache | 144 | 0 |
| prepare/priority | 0 | 144 |

**缓存自由actor未被此次实际模型输入覆盖；21个team-frame全部nonidle，idle实际覆盖0。** 合同写明保留idle，但冻结0/4/8没有提供idle见证；不能把这次矩阵PASS泛称完整运行资格通过。先前纯callback idle测试不能补位为实际模型idle验收。未为补覆盖追加帧或新环境。

仅CPU单线程（Python3.9.6、PyTorch2.8.0、NumPy2.0.2）及已填充Adam受测；空Adam、GPU、多线程、长run容量/计时和真实stochastic长期信用仍OPEN。7份on日志约34.9MB，单份3.65–6.17MB，均低于单组64MiB/10000行上限；这不是整轮训练日志预算/耗时资格。

## 4. 消费端修订与失败边界

科学资格一次成功，failure.json不存在。独立只读consumer经历保存dataclass导入适配，以及OrderedDict→clone dict、JSON排序键→binary插入序两类跨schema投影修订；原off/on binary类型、键序、标量和tensor字节比较保持严格。失败attempt、适配/helper/hash与未注释PASS档单列；首次解码失败派生JSON在归档前已被成功输出覆盖，原bytes未保留的边界也如实记录，不伪称所有失败原件仍在。consumer修订未改变源证据、未重跑资格，新增模型/forward/backward/Adam/环境均0。

## 5. 下一任务与未识别项

[补齐覆盖的下一步方案](sa_credit_numerical_followup_20261020.md)已按既定顺序从现有5B定位四个真实cache多选参考见证，另两个seed/arm无见证；不按reward选。先明确conditional reference prefix与完整随机act的差别，再冻结独立、零环境的软件补验。idle须单独明确已有真实空输入或显式synthetic empty-input来源；不补写成当前覆盖。

本次没有发现日志开关改变已覆盖数值路径的缺陷，也未识别分组裁剪退化的唯一因果。下一步优先完成这些软件覆盖缺口；算法候选、新环境/实际行为采集及新学习均未启动。不得据合成监督或本次bitwise一致性声称seed17恢复、缓存机制改善或学习收敛。

## 验证与证据入口

`PYTHONDONTWRITEBYTECODE=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_sa_stage5c_numerical.py tests/test_sa_stage5c_credit_audit.py tests/test_sa_stage5c_observer.py tests/test_sa_stage5_credit_audit.py tests/test_sa_stage5_same_state.py` → **105 passed in0.68s**；全部为纯软件测试，不额外执行上述模型矩阵。

`python scripts/sa_stage5c_numerical_qualification.py bind --contract <root>/contract.json --output_root <root>`和`run --output_root <root>`已各执行一次；run exclusive claim拒绝重跑，status只读。不得复制命令重新执行已用root。

私有根：`artifacts/joint_stream_learning/sa_credit_numerical_stage5c_20261020/`。contract/runner_binding/run_started/result、7组完整binary/原日志、root_integrity_review、两份独立raw/consumer审查与其helpers/失败派生边界、review_inventory、git_sync及continuation共同绑定证据；真实输入、模型和artifact均不入Git。
