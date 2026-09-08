# G14R20-B 验收矩阵（工作稿）

当前状态：实现与中间验证仍在进行，不是最终验收报告。所有“通过”仅指所列中间验证；
精确 clean 实现提交上的全量复验、证据提交和 push 尚未完成。真实执行许可为 false。

证据目录：`artifacts/analysis/g14r20_b_continuation_20260908/`。

| 要求 | 当前直接证据 | 尚待补齐 |
| --- | --- | --- |
| A proposal/只读语义独立保留 | 原 A 文件未编辑；B 新增独立入口及合同 | 最终字节核验、发布材料 |
| 精确外部执行身份 | CLI 检查 commit、tracked clean、全部 continuation 文件 hash；临时 clean commit 的 10 项真实 CLI 拒绝用例通过 | 最终实现 commit 和身份文件 |
| 独立生产信任与原始证据角色 | root-owned 固定配置加载、三角色分开；测试 key 未安装 | 真实证据继续 unavailable/pending，不由 B 补造 |
| 签名后端 | 预装 Node/OpenSSL 实际 Ed25519 非批准消息测试；固定 executable SHA；NODE_OPTIONS 注入被排除 | 最终依赖身份与文档复验 |
| 真实 CLI 写入前拒绝 | 缺批准、合约/身份/源码/文件/范围漂移及 synthetic 冒充通过 | 精确最终 commit 再验证 |
| 科学模块源隔离 | 实际 main src、外部 src、cwd、PYTHONPATH 拒绝用例；原 module 路径/hash 回执 | 最终汇总；环境完整覆盖核对 |
| 错误解释器/user-site 声明 | 实际只读 CLI 在科学模块加载前拒绝；9 项资格测试通过 | 最终复验 |
| 原资格检查 | B 只读入口核验 150 checkpoint gates、174 committed cells、原 context/binding/registry/环境 | 新增完整 preserved inventory 与锁内 freshness 分支的实际复验 |
| 第一层验收 | 临时固定提交的真实 production CLI 拒绝用例 | 最终固定实现提交复验 |
| 第二层验收 | 10 学习算法 × 3 容量的实际 benchmark main 到达 rollout 前边界；30 次原 gate | 更多拒绝覆盖及最终复验；不能解读成性能结论 |
| 第三层验收 | 八阶段实际子任务、原事务、统计/integrity/formal gate/completion 已通过 | 最新源码的完整链与最终复验 |
| fixture 授权/隔离 | fixture/run/phase/文件 SHA/有效期；实际 staging 绑定；内核 sandbox；8 项 child 拒绝用例 | 最终完整范围审计 |
| 调度计数 | 父进程 24 次直接调度；新增子进程回执汇总验收中 | 确认包含统计 analyzer 的总数与监测边界 |
| prefix/immutable/registry | 原生 ledger、marker/inventory；冻结 terminal 绑定；分叉/截断/重复/漂移拒绝 | 最终 protected inventory 全范围校验 |
| 恢复 | publication 后实际进程退出恢复、finalize-only、重复启动、75 重试、非75 terminal 拒绝通过 | 生产连接和最终版本复验 |
| 单写者 | kernel flock 并发、崩溃回收、稳定锁字节与身份漂移测试通过 | 最终并发材料汇总 |
| 过期/撤销 | admission 过期/撤销单元用例通过 | 进行中真实合成阶段租约边界测试 |
| 八阶段命令等价 | 原 expand_command_plan 与 AST identity/transaction 投影；真实只读输出保存 argv | nested expansion、expected outputs、cwd/env/command hash 逐项最终报告 |
| 七个用户文件 | 多轮 SHA 与起始值一致 | 最终 SHA 对照 |
| 旧 worktree/run | 旧树当前 clean，真实 run 未执行 continuation | 最终保护核验；下述历史偏差不能隐去 |
| 全仓/交付 | 尚未执行最终 clean commit 全仓验收 | pytest tests、smoke、compile/import、diff check、JSON/JSONL/XML/hash、skip 解释、分离提交并 push |

历史偏差：两份新增测试 helper 曾因错误 cwd 短暂写入旧 worktree 的 scripts，随即移回开发树，
旧有文件未改，真实 run 未写入。事件已向用户披露并保存在 `source_write_incident.json`。
结束时 clean 不能证明全过程零写入，此偏差必须保留在最终审查中。

后续独立批准所需材料至少包括：精确实现 commit/文件清单、外部签名后端身份、独立执行合同、
完整 A 引用、原始 launch/release 证据核验记录、八阶段兼容报告、三层及故障/并发/租约报告、
真实只读报告、完整命令/JUnit/integrity、分别标识的实现与证据 commit。B 不自动启动批准任务。
