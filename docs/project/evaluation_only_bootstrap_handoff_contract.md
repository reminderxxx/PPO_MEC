# G14R20-I2 evaluation-only bootstrap 与阶段衔接

日期：2026-09-11。范围：工程启动修复及隔离非正式验收；不授权正式评估、训练、选模、freeze 或 holdout。
最终验收 commit/tree、clean checkout、命令与 JUnit 以独立验收记录为准。

## 根因及初始化责任

G14E01 executor `9d9f2aeea565ced8d576df3f6a778b4217b58de9` 的 `_initialize_run` 先创建目录、四份身份/state
JSON 和 cell ledger，再构造 `TransactionalPhaseRunner(resume=False)`；后者必须拒绝非空目录，导致确定性
`phase output root conflict`。原 `_load_run`、初始化与 phase 顺序判断还在 SingleWriter 之外。qualification
不执行这些路径，既有 JUnit 不能替代首次公共 execute 证据。

唯一创建责任方是持有外部 SingleWriter 的 evaluation-only `_initialize_run`。SingleWriter 的显式
`allow_missing_root=True` 只允许 evaluation bootstrap 在 run 尚不存在时取得同一物理锁；默认 continuation
仍要求 run 存在。锁路径保持 `run_root.parent/.continuation_locks/<sha256(run_root)>.lock`，授权在 open 前和
取得 flock 后均验证。父目录必须预先存在、可写可搜索，宿主 process identity 必须可取得；不足时不创建 run。
没有删除锁、PID/时间推断放行或 recovery 权限扩张。

锁覆盖：重新授权、run 是否存在与首次 phase 判断、初始化或完整载入、phase 顺序、既有产物复验和本次执行。
非阻塞竞争失败者不写 owner record，不写胜者 run。异常保留 held owner；下一请求不得清锁或自动恢复。

初始化按以下顺序完成，任一中断均保留原现场：

1. 在锁内以 `exist_ok=False` 创建 run；构造仍要求空目录的 phase runner。
2. create-only 写空 phase ledger；写 source reference、execution contract、context 和初始化状态快照。
3. 由原 FormalCellLedger 创建 cell identity 和空 cell ledger。
4. 最后 create-only 写 `evaluation_initialization_complete.json`，绑定根目录、全部不可变初始化文件的字节 hash、
   初始两个空 ledger 的 hash。文件写入 flush/fsync，不删除残留或自动重建。

存在目录必须有完整 marker、身份文件及两个 ledger，逐项校验 marker、批准 package、context 实际字节和 cell
identity，才能显式 `resume=True` 载入。marker 本身不是授权；package/grant 与 ledger 仍独立校验。未知非空、
空但预存在、缺件、无 marker、hash 或目录身份不同都拒绝。没有全局放宽 phase runner 的非空目录保护。
context 序列化只有一个字节生成函数；phase runner 首次构造时预计算该字节 hash，后续进程逐字节比较并
核对所有 phase records 的 semantic/file hash。重新格式化但语义相同的 context 也不能静默接管。

## 阶段与科学活动边界

初始化状态是不可变快照，`formal_execution_started=false` 仅描述初始化完成时还没有 dispatch。初始化完成、
phase running、cell child dispatch、committed cell 和 completed phase 是不同事实；不能用一个 state flag
证明发生科学评估。后续进度由原始 phase/cell ledger、实际 child argv/stdout/stderr 和 payload 证明。

只允许下一个尚未开始的 phase；duplicate、jump、failed terminal、未完成的 running/candidate 均拒绝，不提供
cold recovery、retry 或 finalize。进入下一 phase 前，原 phase runner 复核既有 completed command/input/output
hash，原 cell ledger 重新验证 committed marker、producer manifest 和 transaction inventory 的两层完整性。
已完成阶段没有再次 dispatch 或重新发布。

## 隔离公共入口验收

`tests/evaluation_only_public_driver.py` 调用真实 parser/main → execute_phase。真实 prepare 生成绑定 fixture run
的新申请；main 保留真实 model-source 150 模型校验、execution/command parser、grant/review、身份、初始化、锁、
ledger、descriptor、publication 和 phase transaction。测试授权仅绑定 `synthetic_*` scope 的
`synthetic_evaluation_run`；未为任何正式 run 签发 grant。

源码受控的 scientific child adapter 只能由测试 driver 显式传入，生产 CLI、env、package 没有开关。它只将
科学 child 替换为真实 Python 子进程生成的小型合成产物；原 command/staging builder 保留并记录，原 descriptor
与两层完整性检查及原子提交仍真实执行。child 禁止测试 overlay 路径。该适配器固定只覆盖前两个 phase。

首次和第二进程分别完成 formal_cache_policy、formal_controller，每阶段三 cell；核对第一阶段原文件 hash 和
phase/cell ledger 字节前缀不变。不得称作八阶段完整实跑。其余阶段仅引用既有 I1 tests 与本轮相邻/全仓回归，
不增加 formal_statistics、formal_gate、complete_without_holdout 的本轮完整运行主张。

负例覆盖未知目录、初始化缺件、身份/context 漂移、首次启动竞争、held lock、duplicate/jump/failed 请求、
无效/过期授权、权限不足和各初始化写入边界的故障。注入只在隔离 fixture 的真实写入后制造异常，不替换
initializer/loader/lock/ledger/publication/identity。拒绝请求核验零新增 child dispatch 与 run 内容不变。

## 保护与申请包

G14E01 已发生一次执行器启动失败，科学 dispatch 为 0，原生 phase/cell terminal 不存在；外部进程退出日志
不是原生 failed terminal。其六个残留文件、原 grant、旧申请包、I1 release 与失败记录全部保留；当前不可执行，
不得改写为未启动或复用旧 run ID。旧 v16、150 模型、selection/freeze/provenance 和七个用户修改均只读保护。

新的 unsigned 申请必须由实际 prepare create-only 生成，绑定最终验收 executor commit/tree/持久 clean checkout、
原模型来源、不变科学协议和八阶段命令、新 run ID 和尚未创建的目录。仅运行 validate；不签 grant，不正式 execute。
`formal_execution_authorized=false`、`formal_execution_started=false`、`holdout_opened=false`。
