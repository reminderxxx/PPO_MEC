# production_trust_gap_review

生产资格：not_ready。以下形成独立修复/资格任务；本轮不改源码、测试、Protocol或A proposal。

| 编号 | 已核实事实 | 后续最小工作与责任主体 |
|---|---|---|
| PT1 空signer | authorization.py:20为MappingProxyType({})；production分支找不到key即拒绝。CLI先verify_approval，后scientific imports/lock/dispatch；独立内存负例拒绝。 | 项目授权主体定义trust owner与签发职责；实现agent完成独立受审安装方案。C不安装、不生成私钥。 |
| PT2 原件验证 | verify_approval:94–130只验证三项evidence存在、state和64hex reference_sha256，加message签名；不打开原件、不判断其作者/时间/run/commit/授权范围。三个值甚至无需不同；key名称分开不保证三份独立原证据。 | 独立证据核验者逐件阅读/认证原件，输出可追溯核验记录；签发者应明确担保认证主体、原时间、范围、身份对应以及原件保管，而不只签state/hash。相同原件若合法覆盖多职责也必须解释，不强制无意义的hash差异。 |
| PT3 生产撤销缺失 | production分支没有查询revocation_id；只有synthetic fixture_authority.revoked_ids检查。现状因空signer全拒绝；未来仅填key会使有效签名即使已撤销也无法被此代码识别。来源不可用/过期/回滚处理未实现。 | 独立实现任务：权威、认证且具新鲜度的撤销来源；每次阶段准入fail-closed，来源不可用、过旧、签名无效或rollback均拒绝新阶段；测试撤销前后、失联和重启。 |
| PT4 expiry与事务 | execute_phase在准入和取得flock前后校验授权；已准入phase内每child查固定源码，但不重新检查expiry/revocation；允许本phase原生事务/同命令一次75重试/finalization结束，阻止下一阶段。 | 治理批准必须明确phase级（可能很长）准入租约，不误说逐child撤销。未来撤销修复保持或另行冻结这一边界；冷启动finalize-only仍需新准入。不得对已合法commit追溯改写账本。 |
| PT5 身份变更 | trust pin在被20-file覆盖的authorization.py；安装key或撤销实现会改变source hash、tree和commit。 | 新精确executor身份、command plan/contract digest、拒绝/授权/撤销/过期/锁恢复/事务/gate/隔离验收必须重做；不能沿用e6a7335批准。科学commit与原binding/context不重绑main。 |
| PT6 锁恢复责任 | persistent-inode flock，owner记录含nonce/host/PID/start/executor；批准contract绑定recovery_owner_sha256及quiescence字段；内核锁可得且旧process identity不相等才能换owner。 | 独立运维核验无live descendants及host/namespace，原件与精确owner摘要由新批准绑定。代码只检查quiescence声明格式和签名覆盖，不自己证明no_live_descendants。需补签名中owner/quiescence篡改、外部证据失效和独立审批的组合验收。 |

空trust使上述生产缺口当前不可达，不代表已实现生产撤销。B合成通过不等于生产资格ready。锁原语测试用lambda:None是单元边界，不能作为“独立批准已验证”的证据；完整synthetic HMAC也不能认证真实运维主体。

本轮未用生成key或monkeypatch PRODUCTION_SIGNERS去模拟生产成功；因此生产正向签名路径和外部来源故障处理尚未验证。真实权限必须由后续独立决策任务给出，C报告只是证据与修复清单。
