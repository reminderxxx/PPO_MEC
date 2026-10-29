# 本地组件成本支持：独立软件实现任务（2026-10-29）

用户已授权自主强化证据与迭代验收。[准备合同](paper_local_cost_quality_support_contract_20261029.md)完成后，
root先保存只读审查，再另立本实现任务；不是在旧失败run中修复重试，也不因head候选失败而追加优化候选。
只实现技术成本T的生产/独立消费端，当前真实模型、generate、原资源传输、训练、环境评价和下载全部为0。
驾驶质量Q仍缺适用冻结图像/独立groundtruth，不能执行；整体论文未完成。

私有 `paper_evidence_strengthening_20261029_v1/local_cost_software_implementation/plan.json`
绑定原准备方案、root独立审查和用户授权；`synthetic_subplan.json`进一步固定12合成family、
最多2轮/24物化实例、7200秒及128MiB。全部失败、重复物化与输出计费；不同family、物化实例和验证轮次数分别报告。
合成假模型/传输不计作实模型结果，不读取旧资源重量文件来制造软件性能。

## 实现职责

- `scripts/run_local_adapter_cost_support.py`：前瞻精确plan、execution grant、clean source、当前硬件/依赖/旧ALPR参考资格，
  单claim、create-only、串行阶段、固定attempt顺序、qualifier-first、完整失败prefix及硬预算。
- `scripts/audit_local_adapter_cost_support.py`：独立核原始事件、身份、token、应用state与传输原件，
  复算计数/字节/嵌套或互斥时间/残差/磁盘/限额，不接受producer只重写summary自证。
- `tests/test_local_adapter_cost_support.py`：仅固定12family合成资格，不启动真实loader、generate、socket资源传输或旧见证。

准许的未来T矩阵仍为12个模型进程、12次base load、16次adapter load、20次activation、28次generate、
4次新应用状态保存/恢复、12次loopback。每项时间与调用从原始事件核验；资格样本另列，失败不补测或选快样。
这些是实现需要验证的上限，**尚未执行，也不是本软件任务的模型调用权限**。

## 不可跨越的执行门

真实T前必须完成独立软件审查，停止全部tracked编辑，提交清洁source，冻结资源/输入/旧token参考、
完整生成参数、当前CPU/float32/offline依赖和硬件/电源状态，以及另一个精确有限执行plan与grant。
参考参数或输出不匹配时停止，不用新输出刷新原参考。无下载、依赖安装、backend替换、自动重试或旧FAIL改写。
5400秒/1GiB等准备预算只有绑定该新执行合同后才可使用，不能从软件资格推导已实测。

## 独立审查后的核心与收尾资格

首实现的两轮资格实际物化13实例，最终12个family通过；不能把它写成真实模型已通过。
独立只读审查随后拦下三项失败闭合缺口：真实T1的input重建/activation顺序与假模型及消费端不一致，
传输已完成后gate失败被误当作空partial，以及两attempt之间失败被强制解释成最后attempt失败。
审查另指出T3/T4的直接调用不能严格中断阻塞I/O，跨进程计时原点尚未见证。
原报告、源码指纹和全部费用保留；没有启动真实T来试错。

另立的`local_cost_core_closure/plan.json`限定1800秒/32MiB、两个新合成fixture、单轮资格，
原13加新2最多15，不重跑原12family。先修生产/消费一致性和失败账单，T3/T4由最多16个非模型helper监督；
T3每次10秒、T4每次60秒且服从剩余总额，所有helper须退出或被清理后才能封存原件。
这16个helper与12个模型进程分账，model/generate/transfer矩阵和5400秒/1GiB不增加。
按[CPython 3.9.6一手实现](https://raw.githubusercontent.com/python/cpython/v3.9.6/Python/pytime.c)，Darwin路径减去进程内`static t0`，
不能假定`perf_counter_ns`具有可跨进程直接相减的共同原点；当前安装版是否另含vendor patch没有二进制认证。
另存`clock_namespace_addendum.json`，事前固定Darwin `clock_gettime_ns(CLOCK_MONOTONIC_RAW=4)`共同系统命名空间，
生产与独立消费端分别实现并绑定；不事后估计offset或动态换clock。
两个旧解释器非模型clock probe只核该固定命名空间的parent/child原始时钟包含关系，失败即停，不作任意offset归一化。
实际模型路径顺序须由新资格明确绑定，不能继续只相信FakeBackend。

实际parent入口必须绑定fresh raw、外置audit和delivery三路径及单次claim。
producer最多4800秒，终端无论成功或失败都只启动一次独立审计，consumer最多600秒且合计不超过5400秒。
原件、所有失败prefix、helper/模型日志与外置收尾产物共同计入1GiB/64MiB日志限额；不能因失败而少记已花费用。
`--child`仅限模型阶段，helper仅限状态/传输阶段，二者均不触发terminal pipeline。

最终增量审查另指出raw日志与外置audit日志缺统一检查。另立`local_cost_log_guard_closure/plan.json`，
只增加合并字节函数与poll/终端两处一致判定；4个纯内存边界例和其余核心AST不变检查通过，新增fixture为0。
旧12family、核心收尾2fixture、此处4个算术例分别绑定各自源码，不合称同一最终版本跑过18项全链路测试。

当前`activate`原生区间还包含状态读取、参数CPU复制、非零核对及hash，必须标作**含完整性核对的activation**。
纯`set_adapter`切换时间没有单独测量；不能把核验开销直接映射为部署切换成本。

全局32小时/4GiB强化范围与既有科学账单继续保留；新T的原生仿真帧、训练/拟合/环境评价均为0。
loopback应用字节不是无线wire流量；process-first不是清空OS page cache后的storage-cold；技术token一致不证明驾驶质量。
当前stream尺寸/时间/费用没有自动得到校准，不输出缺项break-even或论文完成结论。
