# 5C1/2交付审查：因果概率合同与默认关闭信用审计

- reviewed_at: 2026-10-20T06:41:28.507524+00:00；本地日期2026-10-20。
- literature_cutoff: 2026-10-17沿用，无novelty刷新。
- target_venue: IEEE TMC internal diagnostic；policy_version: tmc_review_policy_v3_20260621。
- artifact_run_id: sa_credit_contract_stage5c_20261020；Git baseline: 5c73ace0230c39798b332d122be3b5156972b455；实现commit/push见私有git_sync.json。
- evidence_level/verdict: 源码/有限概率/软件fixture审查通过；真实stochastic运行资格UNVERIFIED，正式E0_UNAVAILABLE / Unverifiable，N/S。没有候选晋级或退化修复结论。

## 1. 授权与实际费用

用户“启动下一步”承接仅5C第1、2项：因果时点/概率合同、默认关闭信用审计工具与零环境软件测试。先独立authorization.json冻结环境帧、真实agent/model构造、forward、backward、Adam和科学训练全部0；实际均0。原提醒保持PAUSED；未创建新run的环境或学习入口。旧5A/5B和阶段四完整结果/正负seed/原版及优势门禁继续封存。

## 2. 数学与因果结论

- 真实链路是全队first execution，再逐owner second cache/prepare-priority。每active owner实际三次Categorical sample，第二遍execution仅重评；forced head仍保留原抽样分支，不能以删除调用冒充无差别。
- 学习使用逐head conditional ratio、分别clip与enabled平均；owner logs.sum或整队joint/class ratio均不是当前actor目标。(.5,.5)→(.9,.1)同类的raw clipped期望=.7、class目标=1，不能当等价重构。
- 固定prepare时priority不进入claims/book/E-memory；最后second owner已知其他controls，更早owner的未来后缀通常未知。受限priority条件恒等式不自动适用于整个prepare+priority联合head；未来Z=R可使naive baseline的prepare-score期望偏差=.25。
- 可以在明确完整支持/固定partition/旧后缀kernel类内不变/乘子类内相同等条件下作受限Rao–Blackwell代数；不等于posterior、整队IS、当前clip/entropy等价或长期return证明。

[完整合同与源码锚点](sa_credit_audit_contract_20261020.md)逐项区分SOURCE_VERIFIED、MATHEMATICAL_CONDITIONAL及UNVERIFIED，没有实施动作合并或信用算法。

## 3. 软件实现与默认兼容

- `scripts/sa_stage5c_credit_audit.py`：纯stdlib四事件sink+独立只读consumer/CLI。fresh目录、有限行/字节、header/terminal SHA、UID/joins/finite检查，KL未采样覆盖缺口单列，不能自认证actual training。
- `scripts/sa_stage5c_credit_observer.py`：只有显式context才绑定SA stream实例；读取原first/second/更新现成输入输出，并观察原主/副step的参数变化。无额外模型调用、梯度或step入口；现有训练CLI、registry、options和checkpoint格式均未新增开关。
- `src/agents/ppo_agent.py`：仅一个private callback查找和八处guard（11新增行/1展开行）。独立narrow stripper与root测试确认删除这九处观察语句后，**完整文件AST**与基线完全相同，包含原loss/mask/概率/clip/日程/保存加载逻辑。没有执行真实模型bitwise对照，不能把源级兼容升级为真实运行资格。
- UID在act返回拼接后附加；按frame/vehicle/request/B/actions/logs/输入核验，T外部MC/10、内部V/Q/credit为MC/80，model/state辅助slot/10。rawA在原归一前保存，原强制行和KL拒绝均保留。
- 对原step成功后日志观察失败、原step部分执行后抛错均写明确失败边界，保留原异常与方法恢复；不声称回滚或自动重试。

## 4. 已执行验证

`PYTHONDONTWRITEBYTECODE=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_sa_stage5c_credit_audit.py tests/test_sa_stage5c_observer.py tests/test_sa_stage5_credit_audit.py tests/test_sa_stage5_same_state.py` → **90 passed in0.31s**。

覆盖：毒disabled路径、fresh/限额/partial/截断/tamper/异常、三head实际mask及forced行、稳定UID和输入错配、KL零step及eligible未采样、跨epoch去重、两个synthetic step callback恰各一次、失败尝试计数与异常恢复、RNG探针语义、MC/fleet单位、class/posterior/baseline/clip反例、consumer CLI和完整默认源AST一致。所有synthetic receipt都注明非真实训练；没有调用原agent的update/learn，也未构造网络或优化器。

独立合同8概率检查、sink6针对检查及observer纯对象/narrow AST审查均保留私有receipt；不把重复软件检查作为独立效应证据。旧B result原始SHA和选定18cp/绑定原件只读复核，未重新运行旧审计器的当前HEAD门禁；旧科学source继续由其历史Git绑定，新HEAD不冒充旧科学源码。

## 5. 未覆盖与后续门槛

缺口：真实stochastic UID/输入/信用/梯度仍未产生；默认off真实模型数值对照、启用后真实RNG/设备/optimizer/controller无扰动和日志完整成本未做实跑；raw learned logits、每head梯度及严格长期反事实仍未识别。当前64MiB/10000行只是软件默认硬限，不能直接拿来启动整轮训练。

**5C第1、2项完成；算法候选与新学习尚未启动。** 下一步须先选择是否另行授权有界软件数值/梯度资格，再另立实际行为/学习合同以采集真实信用，预算与科学训练分开。当前不借旧1/1额度，不把日志工具当性能修复；如无可验证一阶缺陷，交付未识别并保留原版SA。整体及seed7/27门禁、seed17自身0点恢复与费用独立核查、固定64与尾稳规则继续适用。formal/holdout和canonical保持。

## Evidence inventory

私有根：`/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/artifacts/joint_stream_learning/sa_credit_contract_stage5c_20261020`。含authorization、源码anchors、独立概率/sink/observer审查、软件validation/contract、源码基线与本轮脚本副本、Git同步和completion；不入Git。
