# G14R20-B 固定提交验收记录

记录日期：2026-09-08。实现提交：`6bab4281e19b9798fa78c17ad2628d47b7df1d4c`。
开发基线：`62f432c68b05b8dcd148fde454dd71d1d9621d68`。
科学执行身份仍为 `a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d`。

**状态：实现已提交，B 隔离合成验收通过；整体验收仍有未关闭项，不能宣布完整 Goal 完成。**
全仓为 1431 passed、2 failed、0 skipped（884.73 秒），新增 B 测试 73 项全部通过、无跳过。
定向测试为 61 passed（9.88 秒）。两个失败是既有 v2.4 测试在独立提交上触发原
`HEAD == origin/main` 发布门禁；未移动远程、未替换门禁、未新增 skip。
对应测试及门禁源码与开发基线逐字相同，但这不把失败转写为通过。

真实 v16 未恢复；未签发或安装独立批准、生产信任或测试密钥；`execution_authorized=false`。
原始 launch approval 与 release attestation 继续 unavailable，未经独立核验。
不据此作废旧 run，不重训，不执行 G14D/G15。

## 三层直接证据

| 层次 | 实际执行与结果 | 边界 |
| --- | --- | --- |
| 新 CLI | `test_continuation_production_cli` 实际进程在临时精确 clean Git 身份上检查缺批准、身份/文件/范围漂移、synthetic 冒充；只读 CLI 的错误解释器/cwd/PYTHONPATH/site 声明拒绝通过 | 没有真实批准，未执行生产正向写入；拒绝不能代替科学执行 |
| producer/consumer | 10 学习算法 × 3 容量，30 个实际原 benchmark main 用例；原 companion loader、resource/registry、fairness、checkpoint gate 真执行；94 个科学模块来源记录 | 合成 checkpoint、输入准备；在 `run_real_episode` 前明确截断；cache/support 的完整科学计算由 test-only payload 代替，不宣称正式性能 |
| 八阶段事务 | 3 cache、3 controller、2 ablation、11 support、3 scalability、1 statistics、1 gate，再完成 completion；原 descriptor/inventory/publication/ledger、实际 statistics main/analyzer 与 artifact manager main | 父进程 24 次 dispatch，加 statistics 嵌套 analyzer 共 25；24 条子进程监测回执。analyzer 受继承内核沙箱保护，未单独安装 Python profiler，其启动由父消费者观察 |

八阶段链的 `scientific_rollout_count`、`real_v16_dispatch_count`、`real_v16_write_count` 均为 0。
监测同时报告实际科学函数调用与调度，不使用空列表充当执行证明。
全仓既有 smoke/环境单测是独立回归范围，其 toy 环境操作不混入 B 合成零-rollout 声明。

## 兼容与职责

外部 executor 接管独立授权/固定身份、阶段 allowlist、锁、原资格复核和原事务入口编排。
原 public runner 及 A CLI 不变；只有 continuation 资格路径将移动 origin/main 等式交由独立合同和批准约束。
原 source/clean、bundle、环境、denylist、checkpoint、context/binding、registry、descriptor、inventory、publication 和 ledger 检查保留。
AST 投影只选择原交易 publication 分支与 subprocess 尾部，未调用/monkeypatch 原 public main；原身份表达式和纯函数仍由原提交提供。

`nested_commands_6bab428_readonly.json` 保存 24 条外层 argv、原 canonical command hash、coordinates、expected outputs、cwd/Python/PYTHONPATH，19 条具体嵌套命令，以及原 AST 来源/hash。
统计子命令按原逻辑从未来 `formal_controller/**/benchmark_rows.csv` 排序发现；真实目录尚无这些结果，因此明确保留动态展开源码，不伪造具体 argv/hash。合成链已实际执行此动态路径。
隔离探针使用 `-I/-B`；没有将这些参数插入冻结科学 argv。完整冻结环境见只读报告 `environment_audit`。

## 恢复、授权与隔离

8 个实际事务场景覆盖成功、publication 后进程 exit 86、合法 finalize-only、exit 75 原命令重试、非75 terminal、缺输出、错误 provenance、错误 descriptor。
另有10项 ledger/payload 负例及 gate缺失/失败禁止 completion；原前缀和 immutable committed payload 保持；registry 绑定原 freeze terminal。
原 train/dev/freeze 前缀由合成生产函数准备，continuation 未调度这些阶段，也没有训练。

kernel flock 单写者、第二竞争者拒绝、持有者崩溃后内核释放、稳定锁内容和重复启动均有定向测试。
不按 PID/UTC 到期删除锁。阶段 duration 仍由 monotonic 计算。
实际 expire/revoke 两个合成阶段用例：第一 cache 子任务后注入审批失效，已接纳的 cache 阶段完成3个子任务及 terminal；下一 controller 阶段在记录/dispatch前拒绝。

fixture信任绑定独立临时目录、synthetic run、executor文件hash、phase、实际staging输出、有效期/撤销；调用前递归检查引用。
内核沙箱禁止越界写入和真实数据/权重读取，Python监测补充路径、源码和科学函数观察；覆盖symlink、escape、cross-run、真实引用、错误scope及测试批准冒充生产批准。
Ed25519使用已安装 Node/OpenSSL，固定路径与SHA，清除NODE_OPTIONS/NODE_PATH；只对非批准消息进行实际签名验证测试，生产trust未安装。

## 真实只读覆盖与保护

`real_v16_b_6bab428_readonly.json`：150 checkpoint gates、6 generated resources、174 committed cells、15 phase records、348 cell records、107,320保护项、38,300,782,534原始字节通过。
active bundle原验证器实际执行；A初始只读报告记录44 active resources。
原 ledger prefix未变，无科学树新增；零真实dispatch/write/rollout。
`refresh_6bab428_readonly.json`补充原证据/源码/bundle/registry正向freshness复查，状态由已验证文件独立重建，不重复扫描大inventory。
两次只读探针均在OS禁止写入沙箱内，无lock、running或staging。嵌套命令证据探针有OpenMP临时文件被禁止的warning，未绕过系统限制。
七个用户文件完整SHA起始与当前一致；原A proposal身份未改；科学树当前clean。

**历史偏差不能抹去：** 两个新增测试helper曾因错误cwd短暂写入旧科学树scripts，随后移回开发树；旧有文件与真实run未修改，helper未在旧树执行。详见 `source_write_incident.json`。
结束时clean不能证明全过程零写入；这一事实仍需独立处置，B不能自我豁免。

## 验证、身份与独立待办

完整精确命令见 `continuation_executor_verification_commands.json`；JUnit、逐case审计、完整报告及SHA清单见 `continuation_executor_evidence_index.json`。
smoke通过；315个Python文件compile、5个核心模块import、clean/diff check通过。JSON/XML/JSONL及hash完整性另有清单；没有跳过用例。

未关闭的两项全仓失败：
- `tests/test_formal_protocol_capability_routing_v24.py::test_v24_real_nested_wrapper_consumes_persisted_context`
- `tests/test_formal_protocol_capability_routing_v24.py::test_v24_context_tamper_cross_protocol_and_relative_python_rejected`

它们要求原发布身份等于移动origin/main，与独立executor固定commit的验收checkout不同。未改变原新run发布规则，未把失败静默豁免。需要独立决定该原发布专用测试的验收环境/要求；当前不能声称全仓绿。

`executor_identity_6bab428.json`（21文件）、`execution_contract_6bab428_unsigned.json`、`approval_request_6bab428_unsigned.json`均为未批准材料。后续需精确实现身份、完整A证据、原始launch/release独立核验、独立continuation批准、root-owned角色/有效期/撤销/签名backend配置，以及本报告两个未关闭事项的处置。
本任务不启动该独立批准流程。

实现提交与后续证据提交分开；本报告不自引证据commit。证据commit改动文档后，不应拿它作为executor身份；未来只能检出精确实现commit再由独立授权决定执行。
