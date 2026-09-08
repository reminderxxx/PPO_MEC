# G14R20-B continuation executor contract v1

版本：1.0.0。状态：实现中，尚未完成独立验收，不构成真实执行许可。

本对象独立于 G14R20-A 未授权 proposal；不修改 A schema、proposal 或只读 CLI。
开发基线为 `62f432c68b05b8dcd148fde454dd71d1d9621d68`，科学执行身份仍为
`a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d`。待验收的精确实现 commit 与
后续证据 commit 分别记录；当前分支或移动的 origin/main 不是运行身份。

## 能力边界

唯一允许的后续阶段依次为 formal_cache_policy、formal_controller、formal_ablation、
formal_support、formal_scalability、formal_statistics、formal_gate、complete_without_holdout。
train、dev_select、checkpoint_freeze、holdout、未知阶段和第二个真实 run 均不属于能力范围。

本任务仅授权独立合成环境执行及真实 v16 只读验证。原始 launch approval、release
attestation 和 continuation approval 分别记录，不能互相替代。生产信任接口只读取
独立部署的 root-owned 固定配置；实现不生成、不安装生产密钥或批准。测试 HMAC 仅用于
synthetic-only 域，不能进入生产 CLI。生产 CLI 已连接资格 bootstrap 与原事务，但该完整生产路径尚未验收；当前缺少独立批准，不能执行。

## 原检查与接管职责

| 责任 | 实现方式 | 当前验收边界 |
| --- | --- | --- |
| 版本、执行范围、独立批准 | 外部 security 模块 | helper 拒绝用例已验证；生产入口全面验收待完成 |
| 科学代码来源 | 原 commit、Git blob、clean source、实际 module 路径与 hash | 合成调用实际检查旧 checkout |
| phase/run/cell identity | 从原 main AST 取原始 identity 表达式 | 八阶段合成实际执行；真实完整展开报告待完成 |
| child argv 与坐标 | 原 expand_command_plan | 合成模板显式替换 payload producer，不作为真实命令等价证明 |
| 事务 dispatch | 原嵌套 execute 的 publication 分支及 subprocess 尾部 AST 投影 | 不调用 public main；完整源码/选取 AST 哈希进入 receipt |
| descriptor、inventory、原子发布、marker | 原事务实现 | 实际合成 child、成功/失败/恢复均走原实现 |
| phase terminal、monotonic timing | 原 TransactionalPhaseRunner | candidate、finalize-only、exit75、terminal failure 验证 |
| selection/freeze/generated registry | 原生产函数生成 test-only 前缀 | 不训练、不调用真实 rollout、不重选真实 checkpoint |
| statistics/integrity/formal gate | 原实际 main 消费者 | 八阶段合成通过，不是科学效果证据 |
| active bundle、environment、denylist 等资格 | 保留原验证职责 | 原生只读资格链已运行；新增 inventory 后继校验和生产连接尚待完整验收 |
| 移动 origin/main 发布门禁 | 仅计划对既有 run 由独立 continuation 批准替代 | 原 public runner 不改动；真实替代尚未获许可 |

AST 投影仅保留八阶段可达的原 publication 分支和一般 subprocess 尾部，省略 legacy/train
分支。receipt 记录原 source SHA、AST SHA 和行号。差异等价性需要逐阶段命令、嵌套展开、
环境和身份的独立报告，单纯测试通过不构成完整证明。

## 单写者与事务租约

顺序是无写入 admission、打开固定 `.continuation.lock` inode、非阻塞 kernel flock、
再次 admission，随后才进入事务。并发失败不删除或替换锁 inode。进程退出由内核释放
flock；PID、UTC、时间到期和锁文件是否存在都不是强占依据。

锁文件只保存固定版本、run root 和 executor identity。重复启动核验其字节，不能重写
PID/时间导致已生成 integrity inventory 漂移。PID 和 monotonic acquisition observation
只出现在调用者 receipt。不同 executor identity 不得复用同一锁。

批准以阶段为事务租约边界：进入阶段前检查过期和撤销。已获准进行中的阶段可完成原子
发布及 terminal，下一阶段必须重新 admission。duration 始终由原 monotonic clock 实现
计算，不因 UTC 回拨阻止既有事务提交。此租约的全场景验收仍待完成。

合法 continuation 保留冻结 ledger 前缀，允许经原生 validator 和 B 后继核对接受的新
记录。generated registry 继续绑定原 freeze terminal，不能随最新 ledger tip 重绑。
非 75 terminal failure 不可恢复。publication 后真正进程退出与被捕获异常不同：原 phase
runner 会将后者记为 terminal failure，不能把异常注入误报为崩溃恢复成功。

## 合成隔离

fixture root 必须是独立的临时目录、名称以 synthetic_ 开头；拒绝父目录遍历、symlink、
任何 Git worktree 内位置及真实产物路径。校验先于 mkdir 和 sandbox 建立。内核 sandbox
禁止网络、fixture 外写入及真实 data/artifacts 读取，并由嵌套 child 继承。

Python audit hook 和科学调用 profiler 另外记录实际调度、科学调用来源、写入及被拒绝
尝试。被拒绝的真实路径访问尝试单列，不记为实际成功 dispatch/write。成功链必须包含
真实合成 child 调度；父级与嵌套子级计数必须分开核对后汇总，不将空计数当监测证据。

三层验收必须分别完成：新生产 CLI 资格/拒绝、实际 producer/consumer main 与 checkpoint
相关 gate（在 rollout 前停止）、完整八阶段合成事务。当前八阶段合成已通过中间验证；
不代表另外两层完成，也不代表精确 clean 实现 commit 已验收。

## 独立验收及批准材料

批准前必须具备：完整实现 commit 和文件 inventory、版本化 execution contract、完整引用
A proposal 的文件 SHA 与 identity SHA、原始证据独立核验记录、原始环境和不可变资源报告、
八阶段完整 argv/嵌套展开/identity 差异报告、三层验收及故障矩阵、JUnit 和 artifact integrity、
实现与证据 commit 的分别标识。缺失项维持 unavailable/pending，不能自算 hash 补位。

历史隔离事件必须随证据交付：两份新增测试 helper 曾因错误 cwd 短暂写入旧科学 scripts
目录，随后移回开发树；既有文件未变，真实 run 未写入，旧树恢复 clean。此事件意味着不能
宣称整个任务期间旧 worktree 零写入。独立验收必须保留事件记录，不能以结束时 clean 掩盖。


## 独立签名后端与 fixture 子任务范围

冻结科学 Python 未安装 cryptography；B 不向该环境安装包。生产 trust 配置另行固定
`signature_backend`（kind=node_ed25519_v1、executable 绝对路径、sha256），每次核验先检查
可执行文件字节身份，再用独立 Node/OpenSSL 进程验证 Ed25519。进程不继承 NODE_OPTIONS、
NODE_PATH 或 preload 设置，不启动 shell。该配置仍必须由独立部署方提供，本任务未安装。
密码学测试只签署非批准消息，没有为真实 run 产生任何批准或原始证据追认。

当前验收可用后端为预装 Node v24.19.0 / OpenSSL 3.5.7，文件 SHA-256 为
`27db838bb204ef7c21df2931f5656e4c8fb32e6e947f363a402b49714d32b5b1`。
这属于外部执行器依赖，不写入旧科学 execution identity。最终验收材料需记录精确路径、
版本和 hash；后端缺失、变化或签名不匹配一律拒绝。

合成 child 还必须读取 `fixture_dispatch_scope.json`：它明确绑定独立 fixture、synthetic
run、八阶段范围、外部执行器文件 SHA 和有效期。事务 child 的输出路径必须匹配当前原生
ledger 中对应 running cell 的 staging；统计/gate 输出必须落在获准 run 范围。测试 HMAC
密钥仅存在独立 fixture 内，不能由生产信任加载器读取。exit75 使用原 ledger attempt
判断首次尝试，不另写 fixture 根目录 sentinel。该 fixture 有效期只约束合成测试，不替代
生产阶段事务租约和独立批准的验收。
