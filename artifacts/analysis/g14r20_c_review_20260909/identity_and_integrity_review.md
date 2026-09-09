# identity_and_integrity_review

固定身份互不替代：

| 角色 | commit / tree | 本轮证据 |
|---|---|---|
| 原科学执行 | a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d / eb8e83c6e4532bc45c22fb7a816388e576753e30 | 原 worktree、保护哈希和来源文件核验 |
| B executor | e6a73357d08220d34ddc2d6537d140131c60520f / 059a9464597b96805b3d6d0eaaca474826b2df31 | verify_executor 实际执行及 Git blob 重算 |
| B 证据 | 33bc736650baf89c26a7a9e4185b8834f90cb3c9 | 唯一 parent 为 e6a7335；差异为 84 个文档/证据文件，无源码/测试变化 |
| 当前 main | 62f432c68b05b8dcd148fde454dd71d1d9621d68 | 实际远端与本地观察；不是科学或 executor identity |
| C 审查 | codex/g14r20-c-review，基于 62f432c | 只新增审查材料及必要 PROGRESS/BUGS 审计索引 |

`git ls-remote origin refs/heads/main refs/heads/codex/g14r20-b-continuation` 实际返回 main=62f432c、B=33bc736。第一次受网络沙箱 DNS 限制失败，获准只读联网核验后成功。不将本地 tracking ref 当实时远端证明。

executor identity 覆盖两个 B CLI、17 个 continuation_executor 模块和 A validator，共20文件；逐项和 e6a7335 Git blob 的 SHA-256 相等，文件集合与 verify_executor 推导完全相同。identity canonical SHA-256=c5ed4aa235494c8875ef8e29abce1a6d0db1d9222a2d3b700830cfa361471427。不是把证据提交当 executor 运行。整个工作树还受精确 HEAD/tree 和 clean 检查约束；被覆盖 A validator 与 main 基线相同，原科学 public runner 未改。

重新计算的完整性：

- versioned_evidence_manifest：72 个列出文件、4550020 bytes，size/hash 全通过；全部与 33bc736 内对应 Git blobs 一致。manifest 自身的身份由 Git commit 承载。
- artifact_integrity：2004 文件、31220335 bytes 全通过；包含本机完整合成输入、checkpoint、结果和监测。其排除项为 .git、该 manifest 本身、delivery_record、versioned_evidence_manifest。不是可离机直接重建全部夹具的 Git 包；未列内容不被该 hash 清单背书。
- delivery_record：本机后置记录、自排除，不能自证其描述；其中 commit/远端/文件数已独立核查。
- A protected_before：107320 项、38300782534 bytes 全量流式重新哈希相等；包括原 run、科学 worktree 和合同定义的保护对象。原 run/科学 worktree 文件枚举无新增；七文件和 B 起止记录相等。
- 原 phase/cell：15 / 348 records，1867526 / 25278634 bytes，prefix hash 分别 aa90b53b8655d8d782f780b9d77d785c4500c9c73a3d8b58a0b32300afa86cc9、509fcb075c2fb59b78ac32e0117fc1cd4b4cc77aecca5705407b0c1f1fc19e4e。字节数没有增长，五阶段均 completed，最后 checkpoint_freeze；没有 formal successor。原 terminal 分别 902d0edf5cc8464890df4e63970527beeb41a80063e1ba7b3584bbd0a936806e、0a9779b4a39d396d76ea0b481547ee655e0a4d0bea90177b5623a67d5e75be65。

原 174 committed payloads、150 checkpoint 语义、44 active/6 generated resource、环境 validator 的完整只读消费来自 B readonly_compatibility/monitor 日志，未重跑约554秒语义验收。本轮全量字节保护不是语义 validator 复现，两者严格区分。

JUnit 解析独立得到 B full=1431、targeted=71，failure/error/skip 均0；7 对 job/result 的 argv/cwd、before/after commit 相等，rc=0。完整 stdout/stderr 和文件哈希也核验。synthetic 25 次调度和25子监测入口 multiset 一致，1529 个来源记录的旧 worktree 路径/hash 重算一致。合成八阶段 rc=0/completed 来自已有运行记录；非单独信任 acceptance_report.pass。C 实际71测试另存 JUnit。

数据分数只按字节哈希保护，不用内容排名或性能指导批准。未独立重新执行所有 producer/consumer、统计或环境解析；未进行生产运行、OS级全局写审计或未来信任安装测试。
