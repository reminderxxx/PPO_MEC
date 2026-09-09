# G14R20-D 实现与隔离验收报告

日期：2026-09-09T07:20:49.005320+00:00。本报告是实现 agent 的精确代码验收证据，不是生产批准。

生产分支原先只有签名与 state/hash 形状检查，没有认证撤销来源，也未读取原件。D 沿用 Ed25519 与原 executor，新增固定 installation、签名原件认证记录、固定撤销源及 coordination continuity。每次启动还要求 authority 对新 nonce/PID/scope 和独立保管的当前 checkpoint 签署回执，防止旧回执与 bootstrap 本地状态一起被重放。已知撤销集合累积，更高序号也不能擦除。

## 精确身份

- executor commit：`876c369a1fa8cc33b19d782ab1ba230f021965a7`
- Git tree：`00f4c3e65811aad32b2fb0c1bdd16820262cc1b7`
- 22-file executor identity SHA-256：`8d64e52982e58a56f27ec0348fd7c456acc25fc196c00374290e2f872679e2dd`
- 未签发 contract 2.0.0 SHA-256：`8829706b60fcefa4d6bee077faeb9fc84aded25a38c4617e76e061cf73fc7d19`
- 原科学 command-plan SHA-256：`e68340af18c5195a1c07aa5ef30532ff796ed13f4f4f7e774f2a172960dc9bad`（只读重算一致，科学命令不变）
- 开发仅继承 B `e6a7335`；C `f3ec444` 是审查输入，不是实现基线。`fixed_input_provenance.json` 保存固定输入身份。
- 证据由后续独立 Git 提交保存；该提交的 hash 不自嵌，交付消息给出。原科学 commit、Protocol 2.9、binding/context、A proposal、账本、checkpoint 与 registry 均未修改。

## 合同与职责

详见仓库 `docs/project/continuation_production_trust_contract.md` 的 PT1–PT6 矩阵。软件负责公钥 pin、签名、原件字节、主体/时间/范围/记录身份、撤销序号与新鲜度及启动回执；独立核验者负责来源真实性和语义；批准签发者担保核验主体资格、原件保管与授权范围；运维负责 host/namespace/no-live-descendants。格式合法不等于已证明这些外部事实。

`TrustContext.startup_request()` 只生成挑战；固定 authority 必须依据独立保管的当前 checkpoint 签回执。不能把主机本地文件的自算 hash 当权威。缺失、过期、错误签名/主体/范围、旧回执、回滚或失去连续性依据均拒绝。回执不安装公钥。生产 installation 仍为 None，CLI 不接受任何测试安装或信任参数。

launch/release/continuation 原件允许共用，实际正例共用一份合成原件，并分别签署明确 scope/coverage 的认证记录。launch scope 仅至 checkpoint_freeze；不同职责不靠三个不同 hash 假装独立。同一认证 record ID 不得指向不同内容。owner/quiescence 另绑定精确 owner 原件和认证记录，仍保留原 kernel-lock 与 process-identity 检查。

## 最终 clean commit 验收

`python -B -m pytest tests -q --junitxml=...`：1520 passed，0 failures/errors/skips；其中 160 项为本接口与相邻边界测试，直接来自完整 JUnit，不虚构第二次运行。`python -B scripts/smoke_test.py` 与全仓 311 Python 文件 compile/授权模块 import 通过。保留 16 条既有 record_property/xunit2 警告，未为此扩修范围外测试。所有命令前后均是上述 clean 实现提交，命令/cwd/env/日志/耗时见 `*_result.json`。

正负例包括：有效与错误 Ed25519 签名；原件/认证记录篡改与错误作者、核验者、时间、run、commit、release、scope；state/hash 假证明；原件共用；撤销前后、源失联/不可读/过期/陈旧/错误身份、同序号异内容、低序号与高序号删除已知撤销；重启/本地 bootstrap 回退/旧或缺失回执/可信时间异常；锁前后撤销；阶段内撤销与合法 rc=75 重试；publication/candidate 实际进程崩溃恢复；合法 cold finalize 与撤销 cold finalize 拒绝；owner/quiescence 篡改；全部测试信任在生产 CLI 拒绝；false/missing gate 阻止 completion。

| 合成阶段 | 结果 |
|---|---|
| formal_cache_policy | completed |
| formal_controller | completed |
| formal_ablation | completed |
| formal_support | completed |
| formal_scalability | completed |
| formal_statistics | completed |
| formal_gate | completed |
| complete_without_holdout | completed |

八阶段完整链使用生产共用 v2 Ed25519 核心，不是旧 synthetic HMAC 路径或恒成功 validator。原生产/消费/事务/statistics/gate 代码来自原科学 checkout；合成 payload adapter 明确列于 command mapping。计数：synthetic dispatch=25；scientific rollout=0；真实 v16 dispatch=0；真实 v16 write=0。monitor 子进程记录与调度闭合。pytest 与 toy smoke 不混入此计数。

只读复验 150 checkpoints、44 active resources、6 generated resources、原生账本与 frozen command plan；107,320 项原保护清单无变化、无新增，七个用户文件、main、proposal 和两份账本起止一致。严格 JSON/JSONL/XML、文件内容清单与 `git diff --check` 见格式及完整性报告。大型合成 checkpoint/结果和私钥不进入证据提交；可审计原始合成目录另有完整摘要清单。

## 未批准交接与限制

`unsigned_handoff.json` 明确每个剩余事项的责任主体。生产 trust 未安装、release unavailable、continuation 未签发；原 launch 已由 C 核验且只覆盖 checkpoint_freeze。允许另立“当前日期独立核验历史发布事实”的未批准方案，本轮没有采用或追认历史 release。未来安装任何真实 trust 后必须冻结新 executor 身份、重新验收并由独立主体决定是否签发批准，不能沿用 B 或本轮未签发合同获得 capability。

新鲜认证快照不证明全球最新状态；被攻破的可信时钟、runtime 或错误认证旧 checkpoint 的 authority 不在软件保证内。阶段内租约允许已准入原生事务及一次合法 rc=75 重试/finalization 完成；下一阶段和冷启动 finalize 必须重新准入；不追溯改写 committed 账本。

最终状态：implementation_delivery=complete；isolated_acceptance=pass；origin_launch_evidence=verified（截至 checkpoint_freeze）；origin_release_evidence=unavailable；production_trust_installation=not_installed；continuation_approval=not_issued；real_execution_authorized=false。
