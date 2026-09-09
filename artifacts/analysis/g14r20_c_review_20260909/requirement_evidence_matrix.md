# requirement_evidence_matrix

E2=源码与既有原始材料审计；E3=本轮隔离重现。路径 B 指固定 e6a7335 验收包；C 指本目录。

| 要求 | 独立检查/复现 | 证据与等级 | 限制 |
|---|---|---|---|
| 实际生产 CLI 拒绝 | 16个 phase×qualification/execute 无批准、越权 phase、identity drift 测试 | C independent_pytest_retry.xml，E3；B cli_trust_refusal_report，E2 | CLI 的无批准例先于完整 executor/source 加载，不能证明成功生产路径 |
| 空 production trust | 直接调用原 verify_approval，非恒成功 mock，返回 independent production trust unavailable | C independent_probes.json，E3 | 内存负例不落批准文件、不签名、不安装 key |
| producer→save/annotate/read-back→selection/freeze→consumer | 检查 fixture_inputs.build_checkpoints 与 consumer_entry；真实 metadata builder、annotator、serialized validator、artifact main、loader/gate | B synthetic_acceptance_report、完整合成 checkpoint/inventory，E2 | 初始 torch.save 为 synthetic_test_only 元数据容器，不含经训练策略；不是训练 save 路径和科学 rollout |
| consumer 正例/13负例 | 每例 main、provenance loader、benchmark gate 实际计数为1；来源hash独立重算 | C independent_probes.consumer_evidence，E2 | load_window_bundle 为合成替身，trace 停在 run_real_episode 之前；测试可重写合成 registry哈希以进入深层检查 |
| 八阶段原命令/资源 | 原 context 15阶段186命令；phase-plan/staging/cell input 与旧 main AST 表达式比较重现 | C test_planner_projections_equal_frozen_original_main_expressions，E3；B compatibility_details，E2 | 原 scientific argv 不等于 synthetic payload argv；不得把 AST 对比当完整 nested benchmark 调度 |
| 八阶段事务链 | B 3 cache、3 controller、2 ablation、11 support、3 scalability=22 cells，statistics/gate共3 child，总25，完成8 phases；C五类原生事务局部复现 | B monitor/phase ledger/markers/descriptors，C71测试，E2+局部E3 | 前五阶段是 synthetic payload adapter；统计/integrity/gate调用旧真实脚本，但消费合成表且显式 non_formal_rehearsal |
| 单写者 | 两真实进程 flock竞争、重复启动、owner crash | C71测试，E3 | 单独锁测试 authorize=lambda:None 仅测试锁原语，不能证明批准真实性；组合授权由签名fixture事务测试覆盖 |
| 合法恢复与terminal | rc75一次同命令重试、rc9 terminal拒绝、publication/candidate crash、幂等复用 | C71测试，E3 | 只在缩小合成驱动执行，未操作真实 v16 锁 |
| committed-only/跨账本 | missing/corrupt/descriptor/provenance、重复ID、前缀截断/分叉/乱序/跨账本/immutable payload 均拒绝 | C71测试，E3 | 不替换 validator/transaction 为成功常量 |
| registry freeze anchor | 旧 registry loader与phase validator对既有合成包实际只读复核；freeze terminal与最终tip不同仍合法 | C synthetic_registry_audit.json（独立只读validator检查，非重跑chain）；B完整registry，E2+局部E3 | 原run尚无successor，不能拿其当前tip等同未来freeze anchor验证 |
| false/missing gate | 两个全新目录独立调用原native driver，completion不写ledger；报错分别 invalid or failed formal gate / requires a legal formal_gate.json | C independent_probes.json，E3 | preceding statistics/gate phases为受控空命令fixture；不是新统计分析，也不模拟validator成功 |
| expiry/revocation admission | 阶段内事件允许已准入事务结束，下阶段拒绝；UTC调整 | C71测试，E3 | revoked_ids只存在synthetic authority；无生产撤销服务 |
| 固定科学来源 | load_native原worktree；错误cwd/pythonpath/shadow/interpreter/dependency负例 | C71测试+1529来源hash，E3/E2 | 未重新生成完整生产依赖环境报告 |
| 零真实执行 | 已有25子监测重算均0；C入口/驱动仅全新synthetic路径，原保护清单起止核验 | C保护记录、C命令记录，B原始监测 | C计数是限定调用图/隔离路径与完整性佐证，不宣称OS全局监控；没有formal/holdout消费 |

独立测试命令使用 `/Users/howen/Projects/PPO_MEC/.venv/bin/python -B -m pytest tests/test_continuation_executor_boundaries.py -q -p no:cacheprovider`，cwd为 e6a7335 detached worktree；两次分别指定全新 `--basetemp` 和独立 `--junitxml`。首次 rc1 / 52 passed,19 failed（ps Operation not permitted）；原测试不修改，授权环境重验 rc0 /71 passed，106.80秒。两次目录均隔离，不复用 failed fixtures。

另运行 `recompute_integrity.py`（最终rc0）与 `review_probes.py`（rc0）；后者保存两次 driver 的完整argv/cwd/stdout/stderr。最初 integrity 工具不兼容已记录，修正审查脚本后重算，无实现修复。

风险选择：已完整保留且哈希核验的30分钟合成包不盲目重建；把本轮资源用于拒绝、恢复、锁、跨账本与gate负例。不得将B全仓1431或25 child说成C的新执行次数。零真实 rollout 是边界，不能推出科学命令运行成功或算法效果。
