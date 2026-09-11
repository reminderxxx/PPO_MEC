# G14R20-I2 最终验收记录

- reviewed_at: `2026-09-11T19:40:03.367165+00:00`
- literature_cutoff: `null`（本轮没有文献检索或科学水平评价）
- target_venue: `IEEE TMC engineering boundary only; no paper-readiness assessment`
- artifact_run_id: `G14R20-I2`
- policy_version: `tmc_review_policy_v3_20260621`
- evidence_level: `E3_REPRODUCED_SYNTHETIC_PUBLIC_ENTRY_ONLY`
- accepted executor commit: `879b449e817f05d0f1af9483b4abf449a924c015`
- accepted executor tree: `11c5cfe9a840f004065a1a7e22219d770ba33ca9`
- persistent clean checkout: `/Users/howen/Projects/PPO_MEC/artifacts/execution_checkouts/g14r20_i2_executor_release`

本记录是后续文档提交，不改变被验收 executor commit。代码/测试和公共 bootstrap 合同均在上述最终 commit，
终版环境、编译/import、smoke、公共入口、相邻与全仓回归均来自该 checkout。

## 根因与修复

旧初始化先写身份/context/cell ledger，再调用要求空目录的 phase runner，必然出现非空目录冲突。
I2 由锁内唯一 initializer 创建新 root，先建立空 phase runner/ledger，再写身份/context 与 cell ledger，
最后 create-only 写完成 marker。完整 marker、固定身份、context 实际字节和两类 ledger 全通过才显式载入。
SingleWriter 默认 continuation 限制不变，无未知目录接管、清锁、冷恢复或 failed phase 重试。
锁同时覆盖 reload、顺序判断、既有产物完整性检查和执行；初始化 state 明确为快照，不代表科学 dispatch。

## 实际验收

- `python -B -m pytest tests/test_evaluation_only_public_bootstrap.py -x -o junit_family=legacy --junitxml=...`：30 passed。
- 七个相邻测试模块（来源、身份、live consumer、publication、statistics、continuation、cell publication）：132 passed。
- `python -B -m pytest tests -o junit_family=legacy --junitxml=...`：最终 1689 passed、0 failed、0 errors、0 skipped。
- `python -B scripts/smoke_test.py`、330 个 Python 文件 compile 与关键真实 import：通过。
- 冻结 Python 为 `/Users/howen/Projects/PPO_MEC/.venv/bin/python`；环境 identity、31 项依赖 fingerprint 与 I1 相同。
  跨日复核的实际模块路径/hash 也一致，科学 child 无测试 overlay。

首次和第二独立进程通过真实 main/execute、身份/来源/grant 校验、锁、ledger、真实子进程 descriptor、双层完整性
与提交路径，完成两个完整 phase、6 个合成 cell；第一个 phase 的原记录/产物不变。stdout terminal 与原始
ledger 再次逐项比较，6 cells 再经原生完整性消费者复验。9 个初始化写入边界、两种并发交错，以及未知目录、
缺件、身份/context、held lock、权限/授权、duplicate/jump/failed 请求均有公共入口证据。

旧 release 的真实 CLI 在全新 synthetic fixture 复现同一冲突并留下六文件及原始 stderr，未对 G14E01 重试。
本轮不是八阶段完整实跑；后六阶段只引用既有 I1 consumer 证据及本轮 132/1689 项回归，具体文件列表在
`final_acceptance.json.remaining_phases_evidence`。

首次全仓 22 failed / 49 setup errors / 1618 passed：新 checkout 使用 skip-smudge，三项输入为 LFS 指针。
仅从本地已有 LFS 对象向新 checkout 展开 HF 来源 JSON、Alibaba CSV 与 NGSIM CSV，逐项核验 OID；没有联网下载，
未改原始数据或旧 checkout。刷新正常 LFS index 后 staged diff 与 git status 均为空，commit/tree 不变。
相关 96 项回归和随后全仓 1689 项通过。首轮失败 JUnit 保留为 `final/full.junit.xml`，最终为
`final/full_hydrated.junit.xml`，不覆盖失败记录。

## 保护、申请与证据

2154 个保护文件 hash 无变化；旧 v16 的 125964 项文件/目录 size/mtime 清单无变化。150 模型及关键
selection/freeze/provenance/旧锁单独核验字节 hash；另核验 main、I1 release、新 release 共 9 个输入副本
与冻结 OID 一致。全量旧 payload 没有逐字节重 hash，不能把目录清单检查扩大解释。
G14E01 原 run 已创建且首次启动失败，科学 dispatch=0，无原生 failed terminal；旧 grant/申请仍保留且当前不可执行。

新申请：`typed_model_cache_evaluation_only_20260912_g14r20_i2_pending`；request canonical SHA-256：
`c1d501f4cc0a591ed2d037511c04327509a8a48635016e039dcdcfbf65433745`。实际 prepare create-only 与 validate 均 rc=0，绑定被验收 executor/tree/checkout、
不变模型来源、科学协议和八阶段命令；新 run root、ledger、lock、staging 和真实 grant 均未创建。

证据根目录：`/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_i2_bootstrap_acceptance_20260911`。

- [交付报告](/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_i2_bootstrap_acceptance_20260911/execution_report.md)
- [机器验收记录](/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_i2_bootstrap_acceptance_20260911/final_acceptance.json)
- [未签发申请](/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_i2_bootstrap_acceptance_20260911/authorization_request_unsigned.json)
- [八阶段命令指针](/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_i2_bootstrap_acceptance_20260911/future_commands.json)
- [完整性清单](/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_i2_bootstrap_acceptance_20260911/integrity_manifest.json)

限制：没有正式评估、训练或重新选模，没有后六阶段完整科学串行运行；进程/写入边界故障注入不等价于真实断电
或存储控制器故障验收。本 Goal 仅完成启动修复及非正式验收。

formal_execution_authorized=false；formal_execution_started=false；holdout_opened=false；holdout_capability=false。
