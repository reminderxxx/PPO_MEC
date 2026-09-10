# 一次性项目接续授权合同

- `project_continuation_authorization_contract_version`: `1.0.0`
- 记录日期：`2026-09-10`（本机时间核验；Asia/Shanghai）。
- 范围：G14R20-G 的本地项目执行授权；不是科学 Protocol 升级，也不是论文结果审查。
- 当前状态：实现 `cc32e6e` 已验收，独立技术 review 已通过，精确 grant 已生成并通过 public qualification；
  真实 dispatch、formal phase 和 holdout 均未启动，本文不预报 formal gate 成功。

## 当前用户决定与信任边界

本次会话中，用户先要求“先闭环实验”，在获得简化方式的解释后，对以下问题答复“允许”：

> 是否允许我们把过重的接续授权机制，正式收敛成“你批准范围＋独立核验＋固定输入＋完整追加记录”的一次性实验授权？

因此允许新增明确的 `project` 授权入口，以当前项目 owner 的决定和独立技术核验为依据。它不要求安装
Ed25519 trust store、receipt/challenge 或撤销签名服务。旧 production-crypto 入口及其拒绝规则仍完整保留，
不能把 production 失败静默降级为 project 模式，也不能把 project grant 伪装成历史或密码学批准。

历史独立 release attestation 的缺失仍如实保留。今天核验代码发布、原件、模型和账本的一致性，只证明今天
可复核的历史事实，不补写“当时已经独立批准”。当前聊天授权也不构成用户法定身份认证、抗恶意本地管理员
篡改或权威全局最新状态证明；本机制明确采用单一受信本地项目 owner 的信任模型。

## 唯一获准科学范围

| 项目 | 固定值或约束 |
| --- | --- |
| run ID | `typed_model_cache_formal_20260906_152847_g14c_v16` |
| 科学 execution commit | `a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d` |
| 科学 worktree | `/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847` |
| 原 run root | `/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_formal/typed_model_cache_formal_20260906_152847_g14c_v16` |
| 起点 | 原 `checkpoint_freeze` 已完成；150 个 selected/frozen checkpoints、原 candidate/selection/freeze 和 6 个 generated resources 不变 |
| 科学协议 | 原 Protocol 2.9、context/binding、数据、split/window、agent/order、seed、容量、预算、指标与统计规则不变 |
| 写入 | 只允许 proposal/合同中该 run 的合法后继追加，以及独立授权证据和明确协调目录；不覆盖原件 |
| holdout | `sealed=true`、`opened=false`、`consumed_permanently=false`、`capability=false` |

仅允许以下八阶段按原顺序执行；既有 preflight/tests/train/dev_select/checkpoint_freeze 不得重跑：

1. `formal_cache_policy`
2. `formal_controller`
3. `formal_ablation`
4. `formal_support`
5. `formal_scalability`
6. `formal_statistics`
7. `formal_gate`
8. `complete_without_holdout`

不得开启 G14D/G15、holdout/hidden、新 run、重新训练或重新选模。失败和 unavailable 指标须保留，不吞掉
`null`，不改为零，不减少 seed/window/baseline，不读取正式结果后追胜调参。gate 通过是完整性条件，不是
算法必须领先的条件；没有合法 passed gate 就不能完成。

## Grant 与独立技术复核

授权原件、独立 review、grant 分开保存，不能用 `approved=true` 或实现者自报成功替代审查。独立 review
必须绑定本次决定的记录、精确 proposal/合同、最终 executor identity、原科学源、数据/模型/资源清单、
原 phase/cell 字节前缀及完整命令计划；只验收兼容性和执行边界，不以 dev/formal 性能决定是否准入。

grant 绑定独立 review 的路径、字节 hash 和结论，以及本次精确 executor、run、原科学身份、八阶段 allowlist、
有效期、停止文件和有限写入范围。缺失、错误、过期、漂移、跨 run、扩充 phase 或 holdout=true 一律拒绝。
有效期不得晚于 `2026-09-30T23:59:59+08:00`；这不是对实验一定按期完成的承诺。
不得把新的 executor commit 回填成原科学 commit，也不得改写旧 Protocol、context、binding 或 checkpoint。
实现代码变化后必须重新审查并重新绑定 grant，不因旧测试通过自动继承授权。

公共入口为 `scripts/execute_fixed_commit_continuation.py --project-authorization <grant.json>`；不能同时传
`--approval` 或 synthetic startup binding。`compatibility` 仍只读，`qualification` 与 `execute` 可分别调用，
后者必须重新完整准入，不复用前一进程的通过结论。这不是旧 crypto 同进程 challenge/receipt 接口的改写。

## 执行、取消与故障边界

- 原 source/data/checkpoint hash、clean source、环境/import、top-level/nested identity、static/generated
  resource、完整 command matrix、prefix/successor、committed-only、descriptor/publication 和 failure gates 保留。
- 继续使用单写者锁；grant 的有效期和指定停止文件在每次 phase 准入、实际获锁后的准入复核中检查。
  停止文件存在、状态无法可靠读取或 grant 不满足条件，均不得准入新 phase；不删除停止文件来继续执行。
- 一个已准入 phase 仍按原事务完成合法输出与终结；过期或停止请求阻止下一 phase，不任意杀掉正在原子
  发布的事务，也不把未 committed 产物伪装为成功。
- 正常终结后，可在同一有效 grant 内按顺序准入下一未启动 phase；不能重新执行已 committed cell。
- 冷恢复默认关闭。进程异常退出、遗留 owner/running、崩溃后的 finalize-only 或恢复，不因这个 grant 获得
  权限；须停止并另立有限恢复审查/授权，不能删除锁、清账本、换 run 或循环重试掩盖故障。
- 原 return-code 分类和非 75 terminal failure 边界不变。任何恢复批准均不能把永久失败变成合法重试。

## 交付与证据声明

实施验收至少覆盖：显式 project CLI、缺失/漂移/过期/停止拒绝、阶段与 holdout 扩权拒绝、独立 review 绑定、
获锁后复验、旧 crypto 路径未降级、原事务/账本/资源校验、以及所有拒绝场景的零真实 dispatch/write。
合成验收与真实只读核验分别报告，不混作科学性能证据。真实执行之前应确认原科学 worktree、全部受保护
原件、两份账本和主工作区七个用户文件未改变。

后续只有实际完成的原 run phases、原始结果、统计与 gate 才能改变实验状态；本合同、用户授权或测试数量
均不证明算法优势、formal 完成、TMC-ready 或 paper-ready。原历史文档保留，涉及本次接续的授权方式以本
合同及绑定的独立验收/实际执行记录为准。
