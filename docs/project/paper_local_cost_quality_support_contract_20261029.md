# 本地系统成本与任务质量：有界支持实验准备合同

- `reviewed_at` / `literature_cutoff`: `2026-10-29`。
- `target_venue`: CSCWD2027；TMC 仅作内部证据规范。
- `artifact_run_id`: `paper_local_cost_quality_support_contract_20261029_v1`。
- `policy_version`: `tmc_review_policy_v3_20260621`。
- 准备基线：`fc3f910c18797197b3048d758492925b3308ff3f`。
- `evidence_level`: `PREPARED_FROM_SAVED_RESOURCE_METADATA_NOT_EXECUTED`。
- `paper_goal_complete=false`。本合同交付可审方案，未启动下面的微实验，也不将准备任务的授权转换成模型执行授权。

本轮来自 `followup_queue.json` 的 `local_system_cost_quality_support_contract`。只读已保存清单、小型本地配置和相关报告，准备上限 7,200 秒；训练、环境评价、下载、模型加载、推理、拟合、真实 timing 和旧审计重跑全部为 0。起时、原始边界、来源 SHA 与机器方案保存在主目录 `artifacts/analysis/paper_evidence_strengthening_20261029_v1/local_cost_quality_contract/`，文件仅新增，不覆盖旧产物。下列 T、Q 是两项未来任务，需分别完成软件资格和前瞻冻结后才可执行；当前不启动。

## 1. 已有资源具备什么资格

| 资源 | 已核验的固定身份 | 权重字节 | 当前可守结论 |
|---|---|---:|---|
| SmolVLM-500M-Instruct | revision `a7da5b986cb59b408707209984f360a5f4ad7e47`；weight SHA `d05b567e…91a2` | 1,015,025,832 | 已有本地视觉语言 base，不能据此认证驾驶质量 |
| ALPR adapter | `Hirai-Labs/FT-SmolVLM-500M-Instruct-ALPR`，`edccd0e7ae5cbded1f35c21ea108988e1e32b239` | 154,423,432 | 配置为 LoRA rank64、DoRA；同 base 声明，真实车牌准确性未验 |
| Helmet adapter | `revitotan/FT-SmolVLM-500M-Instruct-Helmet`，`96794cb1fd9abf1636b8f1541095f6f9229e4387` | 9,641,944 | 配置为 LoRA rank4、非 DoRA；头盔识别准确性未验 |
| 保存的两节点 suffix | `two_node_workflow_suffix_recovery_20261003_v3`，固定源 `01c9d972…6707` | 动态包 2,040；state 1,703；manifest337；payload1,090 | 单 adapter、CPU/float32、greedy 技术后缀可独立读取并消费中间状态；不含标签质量 |

两 adapter 配置的 `task_type=null` 保持原样。不得为新测量悄悄改配置、权重、依赖或替换 loader。ALPR 的尺寸差同时伴随 rank/DoRA 差异，不能用这两个点拟合“体积导致时延”的因果曲线。

原 adapter acceptance 的 `terminal_status=FAIL`、`loading_compatibility=fail` 永久保留。2026-10-27 的独立 `compatibility_supplement.json` 从已有原始回执解释：旧汇总使用 `loaded_adapters`，实际状态字段是 `available_adapters`；严格消费者支持 A→B→A 身份/输出复现和独立重载保真。此补证不把旧 FAIL 改成 PASS，也不证明任务质量。

原 suffix 的输入是 LuST 的 TrafficDemands 图，n0 描述为 line graph，n1 输出为 `2010 - 2`；任务标签不存在。它适合技术输出一致性，不能改名为 ALPR、Helmet 或驾驶工作流质量。DriveLM QA 及其待确认图像条款也不能补位成这两类任务的标签。

## 2. 同方向论文要求本项目补什么

复用本轮已核验的公开原论文矩阵，不再检索或下载：[S-LoRA](https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html) 区分加载/驻留与服务质量；[Toppings](https://www.usenix.org/conference/atc25/presentation/li-suyi-toppings) 明示 dummy 权重只能支持系统时延；[Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao) 分开迁移、重算、停顿和队列；[dLoRA](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang) 的 adapter/request-state 协同已有直接近邻。这里最小缺口是本地真实字节、可核验组件成本、任务适用性和当前仿真参数的映射，而不是再画一条 reward 排名。

本次只准备空闲、串行、单主机 CPU 的组件微实验。并发加载干扰、远程 RSU、无线、GPU 扩展均未覆盖，不能引用这些论文的部署规模替本项目背书。

## 3. 未来 T：本地组件成本，先正确性后统计

固定 CPU/float32、batch1、greedy、`num_beams=1`、最多8新 token、intra/inter threads 各1，离线读取、`trust_remote_code=false`。使用保存的 n1 文本及 token 记录，不读源图、不重算 n0、不重跑旧三进程见证。ALPR 输出另与已有 n1 token 参考比较；Helmet 在新 ordinal0 建立技术参考，随后严格复现。此参考只认证保真，不认证语义。

保留既有环境 Python3.9.6、Torch2.8.0、Transformers4.49.0、PEFT0.14.0、Accelerate1.2.1、Safetensors0.5.3。执行前需记录当前硬件型号/CPU/核心数/RAM/OS build/架构/磁盘及文件系统/可用空间/电源及功耗模式、进程负载、依赖版本和 module origins。历史报告的 M5 MacBook Air/24GiB/macOS26.6.2 仅是历史记录，**本次未重新核验当前硬件**；关键身份或电源字段缺失、环境不符时停止，不自动换 MPS/CPU 或安装依赖。温度传感器不可用时记 `unavailable`，保留热状态限制。科学训练与其他模型负载需全部退出。

| 阶段 | 事前固定顺序与调用 | 资格 / 测量 | 正确性门 |
|---|---|---|---|
| T1 process-first / warm suffix | `A0,B0,A1,B1,B2,A2,A3,B3`；8新进程，每进程各一次 processor/base/单 adapter load、一次显式 activation、两次相同 n1 generate | ordinal0 每 adapter1进程；其余每 adapter3进程。总16 generate | active identity/参数来源正确；每 adapter 全部 cold/warm token IDs 相等；A 匹配保存参考 |
| T2 resident A→B→A | 4新进程，ordinal0资格，1/2/3测量；每进程 load processor/base各1、两 adapter各1、activation3、generate3 | 总12 generate，12 activation | A1=A2，A/B 分别与 T1 本 adapter 参考严格相等；不得只信名称 |
| T3 encode / durable save / restore | 保存的固定 envelope，4新目录；每 trial encode1、durable package1、restore1、prompt rebuild1；每包两文件 | ordinal0资格，3测量；总8file fsync、8directory fsync、8rename；新包8,160B | 与原 state/manifest/payload 完整字节和 SHA 相等；身份、节点边界和 rebuilt n1 prompt 相等 |
| T4 TCP loopback | state包2,040B、A权重154,423,432B、B权重9,641,944B，各4次、共12次 create-only transfer | 每对象ordinal0资格，3测量；不重试 | 完整长度/SHA、ACK、目的文件一致；缺字节/多字节/异常均失败 |

资格样本的时间保留且另列；先通过该阶段 ordinal0 才开始其正式3次。任何正式样本失败也不补测、不丢弃、不选择最快。T3 的未来新组件 helper 要将 encode 与 durable save 分开计时并逐字节匹配已有格式；不修改共享旧 runtime。它只恢复应用显式状态，没有 KV/tensor/token continuation，也不建立新的端到端迁移净收益结论。

T4 只绑定 `127.0.0.1` 临时端口。每 attempt 固定64B header（`>8sQ32s16s`）和48B ack（`>8sQ32s`）；payload累计 **656,269,664B**，应用 framing 后累计 **656,271,008B**。内核 TCP/IP headers、TCP重传及物理 wire 字节未测。每次从零请求，单向 payload和双向应用 framing 分账；底层 recv/send 系统调用次数如实记录，不伪定固定次数。单次 socket 阻塞≤10s、attempt≤60s，失败不重试，不 Range/拼接。

T 的拟议硬上限为5,400s、1GiB新增磁盘（含全部副本/日志/失败现场，日志另≤64MiB）、12模型进程、processor/base各12 load、adapter16 load、activation20、generate28、loopback12 attempt。HTTP/外部下载/训练/环境评价/拟合/原生仿真帧均0。每模型进程≤300s；上限消费包括失败资格、失败 attempt及消费者。当前未执行这些调用。

## 4. 时间、字节和消费者必须匹配

每项保留 `perf_counter_ns` 的原始起止、UTC supervisor起止、PID/exit、attempt identity；区分 Python/import、processor/base/adapter load、switch、input rebuild/tokenization、process-first generate、warm generate、encode、durable save、read+validate、receiver durable write、sender→ack。CPU generate 边界含阻塞输出拷贝和 decode；嵌套时间单列，不与其父时间相加。每进程只相加互斥区间，与 supervisor 墙钟的差额报告 startup/idle/未覆盖 residual；负值或不一致阻断性能结论。峰值 RSS 不跨进程相加为整机峰值。

每组3个正式值全部列出，并给均值、样本SD、中位数、min/max；不删离群值、无隐含预热、不称 CI 或跨硬件统计。`process-first load` 不等于清空 OS page cache 的 storage-cold；warm 只指同进程已有资源。无需也不执行全局 cache purge。

新 producer 和独立 consumer 尚未实现，本方案不是可启动软件。两者需共同锁定：clean source commit/模块来源、plan SHA、旧 state/input与模型资源身份、精确 attempt矩阵、active adapter、raw tokens/input IDs、逐项load/generate/transfer计数、全部 bytes/失败 partial、时间/磁盘限额。消费者从原始时间戳、输出和长度/SHA重算，不接受更新 producer manifest 后的 summary 自证。

软件资格至少覆盖12种合成拒绝：missing/duplicate attempt、plan冲突、wrong active adapter、重复输出不同、state stale hash、重新封hash但非法节点边界、注入最终答案、额外文件/路径逃逸、generate超限、应用字节超限、负/重叠时间残差。输出失败原因、截至失败的全部计数与 partial；原始失败现场不删除。软件失败须另立诊断，不在原 run 修复重试。

## 5. 未来 Q：任务质量，当前明确阻塞

Q 必须另立方案；不得让 T 的非空文本或恢复一致性代替标签。最小固定 pilot 是已有本地图像各6例：ALPR 3个清晰车牌+3个无车牌；Helmet 3个明确戴盔+3个明确未戴盔、单指定人员ROI。当前没有适用于这12例的冻结图像/标签manifest，因此 `BLOCKED_NO_APPLICABLE_HASHED_GROUNDTRUTH`，不能执行，也不能用模型卡或 DriveLM QA 补位。

执行前必须有：现有许可适用的12图及 SHA/顺序，模型输出前确定的标注及来源；两名独立标注者与仲裁，或可独立核实的外部 groundtruth；固定 prompt/parser、可读性和适用范围。ALPR 在输出前冻结地区字符规则、NFKC/大小写/分隔符规范化，保留 raw/normalized 和 exact/CER；无牌标签 `NONE`。Helmet 固定 `HELMET/NO_HELMET`，格式错/拒答均失败。不能在看到输出后排除难例。人工标签目前不存在，不自动向任何人发消息索取。

Q 两臂是 base、同base+匹配adapter；共24 generate、4模型进程/4base load/2adapter load，每次最多32新token，batch1/greedy，无重试；拟议≤3,600s、256MiB，0训练/拟合/下载。每任务6例全部符合冻结答案才认定“这个有限fixture质量合格”，保留 base配对全部结果；未通过则不晋级、不换题/改prompt继续。12例只支持可运行性诊断，不支持总体驾驶准确率、优越性或 safety 声明。若做两节点语义链，下游用确定性 validator 消费上游 typed result，并另冻结它的工作流合同；不得把原 technical suffix 暗改为已验证真实任务。

## 6. 校准残差与适用边界

当前 stream 源码有 base2,048B/adapter512B 等默认配置，实际 campaign-specific override仍需绑定；没有已冻结的映射把这套 typed仿真对象对应到上述1GB base、154MB/9.6MB adapter或2,040B状态包。本方案不改单位、环境、reward、算法、费用或拟合参数，不拿旧104MiB合成少传量与2,040B相减当净收益。

未来只有在映射先于结果冻结时，才报告逐组件 `observed_ms−predicted_ms`、字节残差和适用范围；相对残差的分母为0时为 undefined。缺组件保留 unavailable，不拟合相同3次测量再宣称验证。总账必须覆盖 save/transfer/restore/rebuild/缺失静态资源load/switch/waiting，与避免的 transfer/load/switch逐项对应；任一关键项未知时 break-even结论仍 unavailable。新的 calibration/validation分离和仿真参数更新需要独立有界合同。

本方案的 loopback既非无线，也非RSU链路；技术restore既非任务质量，也非production action4实载或算法收益。下一可执行步骤是**先实现新producer/consumer与12项纯软件资格，再冻结T的硬件/计划并单次运行**；Q等待真实许可图像/groundtruth，物理无线/RSU与当前stream成本映射另立任务。论文 formal/holdout/support、强公平对照和完整机制成本仍未闭合，论文目标保持未完成。
