# Helmet6：固定ROI的成对质量pilot（2026-10-30）

人工在6图实际交付及“尚未启动模型评价”说明后明确回复“好 继续推进”。据此另立本Q任务，
不是从旧输入资格或技术T结果自动推导模型权限。目标仍CSCWD2027，TMC/ToN只作内部证据参照；paper_goal_complete=false。
私有包为主目录`paper_evidence_strengthening_20261029_v1/helmet_quality_pilot_20261030_v1/`。
既有Helmet6输入audit/terminal/CSV/外部COCO标签及其SHA只读引用，不重跑旧audit、旧T、奖励实验或负候选。

## 精确对象与判分

只用已冻结6张不同原图的原生Helmet/NoHelmet各3个ROI，不读其他图/hidden。
原xywh全部精确整数；Pillow打开原存储像素，不做EXIF转置，转RGB后按原框裁剪，
再以`Image.frombytes`去元数据，PNG compress_level6/optimizeFalse，opaque `q00.png`至`q05.png`。
裁剪不手调、不重标、不额外resize；内存预计算SHA是0模型准备，执行claim后重建6PNG/核验且计执行时间。
模型只见裁剪像素及共同文本，不见原图名、目录类别、ROI、外部GT或累计对话。

固定prompt：`Look at this cropped head region. Is a motorcycle helmet worn? Reply with exactly one label: HELMET or NO_HELMET. No explanation.`
只strip输出外层空白，再精确匹配HELMET/NO_HELMET；小写、解释、拒答、空文本均算错。
只解码generate新增后缀；完整returned IDs和input IDs保留，由消费者独立切分并用固定tokenizer解码，不能把prompt中的选项当答案。
任何语义/格式错仍保留并继续固定全部12调用；基础设施错误保存prefix、停止，不补测、换样本或改prompt。

## 模型、调用与资源

base为本地SmolVLM-500M-Instruct固定revision a7da5b9；adapter为已冻结Helmet 96794cb，rank4非DoRA。
精确资源路径/全部文件size-SHA复用旧T plan并另绑定其SHA；不修改资源、下载、安装依赖或替换loader/backend。
旧Python3.9.6/Torch2.8.0/Transformers4.49.0/PEFT0.14.0/Accelerate1.2.1/Safetensors0.5.3，
numpy2.0.2/Pillow11.3.0/tokenizers0.21.1；CPU float32、intra/inter threads1、batch1、seed0、eval/inference_mode。
base与base+Helmet各1新进程，各6generate、最多32新token；总2processor/2base/1adapter load/1activation/12generate。
两臂同prompt、processor、全部effective generation配置和同PNG；不保留跨case KV，不训练/拟合。
记录所有input tensor的dtype/shape/SHA和真实非空视觉张量、input IDs与crop RGB SHA。
Helmet真实active status及全部LoRA参数CPUfloat32 SHA要与原safetensors逐tensor匹配；不能只看名称或nonzero。

本次新增128MiB（含准备/代码资格/PNG/失败/控制/审计）、metadata及日志16MiB。
独立1800秒从新claim计：producer≤1200、2worker各≤600（含退出清理）、consumer使用剩余≤600。
统一共享`clock_gettime(CLOCK_MONOTONIC)`及stable boot UUID，不用进程相对monotonic/perf_counter做跨进程相减。
子进程独立PG、timeout/异常均terminate/kill/reap并保存控制账；每个operation先落begin，失败实际调用也收费。
原强化科学292286帧等账单不重置；本轮物理环境帧/新训练/拟合/HTTP/ALPR/hidden均0。

## 软件、单次执行与独立接受

`run_helmet_quality_pilot.py`只编排6PNG、两worker与一次consumer；worker只能读取无GT的精确workerplan。
`audit_helmet_quality_pilot.py`只读核原输入引用、原生裁剪像素、来源/资源/激活、每臂全矩阵、token/视觉输入、原始输出及全部调用/磁盘/时钟账单。
软件事前最多producer4+consumer4小合成case、单轮1800秒/64MiB、0真实模型/HTTP；增量只AST/静态另列，不追认旧case最终版重跑。
独立review另拦下QPROD01/02（PG清理/完整P树实时限额）。另立2个非模型增量case，首轮1通过/1因EPERM失败；旧失败和控制收据保留。随后1个有界信号诊断与另立1个回收增量case通过，0真实模型；不是最终版本全部旧case重跑。consumer资源/runtime/失败prefix绑定的最小改动只做AST/独立静态增量接受。
统一P树实时限额含外部准备/控制，受控写前预留1MiB；stdout启动前占位分配并继承文件hardlimit，supervisor只监资源。PG清理屏蔽超时信号，直接子进程无条件reap；无法证明PG消失则失败，不隐藏。
软件与独立复核通过后clean commit/plan/grant/source/input SHA冻结，一次push尝试后才单次启动；运行中冻结tracked源码。
已有audit/terminal复用，不重跑。协议/原件完整则审计可以成功，即使预测0/6；不能把质量失败当软件异常补测。

每臂只有6/6才称“这6个固定fixture质量合格”；报告全部逐例raw文本/格式错/错误、配对差和两臂分母。
不据n6作总体准确率/显著性、训练独立性、检测性能、驾驶安全、无线/RSU成本或系统净收益结论。
公开COCO外部GT足以本pilot来源绑定，不认证每例双人重新标注，模型训练集仍未知。
ALPR文字/NONE样本尚缺，12例全计划未完成；任何结果都不晋级canonical、不改老师稿位/论文主表，不追加主方法候选。

实际执行与保留的质量失败见[结果](paper_helmet_quality_pilot_result_20261030.md)；执行/自动审计已完成，禁止重跑本run或事后改判分覆盖负结果。
