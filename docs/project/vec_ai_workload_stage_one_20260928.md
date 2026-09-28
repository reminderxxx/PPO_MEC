# 车联网AI数据集第一阶段：本地资格核查与最小应用模板

- reviewed_at / literature_cutoff: 2026-09-28
- target_venue: IEEE TMC；policy_version: tmc_review_policy_v3_20260621
- git_base: f8d9bb93376cec28ec9944aa064685f893d14d3f
- artifact_run_id: none_new_scientific_execution
- evidence_level: 源码/本地环境直接核对；新数据真实性仍UNVERIFIED
- status: LOCAL_INVENTORY_COMPLETE_MODEL_CALIBRATION_PENDING

用户已要求启动数据集计划。本阶段仅完成本地核查、应用模板与测量规格，不代表已生成数据集。
总设计及D01–D12证据池见 `vec_ai_workload_dataset_design_20260928.md`。

## 1. 核查事实与范围

| 对象 | 观测 | 边界 |
|---|---|---|
| `src/envs/specs/semantic_objects.py` | WorkflowNode含base/adapter/I/O尺寸/前后继；WorkflowGraphState含拓扑顺序和进度 | 是可用DAG接口，不等于已校准AI应用 |
| 同文件CACHE_OBJECT_TYPES | base_model、adapter、workflow_state、kv_prefix、not_applicable | 未有通用input/intermediate对象类型；不能直接宣称AI数据cache链路已完备 |
| `src/data/workflow/alibaba_dag_parser.py` | 真实依赖＋规则模型/adapter/size映射 | 可复用结构解析；不能继承为真实AI语义 |
| `data/processed/sampled_vec_dags/` | 存在sampled_vec_dags.jsonl及stats | 本轮只确认存在，未审计所有记录或用于校准 |
| `src/data/model_catalog/hf_metadata_diagnostic_model_profile.json` | raw_payload_downloaded=false、metadata_only、license未知/blocked | 不可当作本地模型权重或实测profile |
| `data/raw/model_cache/` | 当前列出huggingface_model_cache_sources.json | 元数据入口，不是已下载AI权重 |
| `data/cache`、`data/processed`中的限定名称/后缀搜索 | 未发现可识别AI服务权重 | 非全机普查；RL policy checkpoint不等于应用模型 |
| 默认HF缓存`/Users/howen/.cache/huggingface/hub` | 目录不存在 | 不排除用户另有模型目录或自定义缓存 |
| `/Users/howen/Projects/PPO_MEC/.venv/bin/python` | 正确venv；torch可发现，transformers/peft/safetensors不可发现 | 未联网安装或修改环境；可发现不替代实际模型加载验收 |

## 2. 最小应用模板（候选，不是现实部署证明）

首版选择非安全关键的“车辆会话中的道路事件辅助信息服务”。输入为已取得使用许可的事件文本与车辆
上下文；模型做事件结构化与面向乘客的信息摘要，不向车辆下发驾驶控制指令。
选择理由：节点I/O易于明确，状态可序列化，能够实测模型加载及连续会话恢复；不依赖算法胜负。
实际文本来源、模型及adapter尚未选定，不用NGSIM运动数据伪装事件文本。

| 节点 | 输入→输出 | 执行类别 | 校准要求 |
|---|---|---|---|
| n1 normalize | 文本/时间/会话ID→标准化事件 | 确定性预处理 | 实际编码bytes、执行时间 |
| n2 extract | 标准化事件→结构化类型/地点/不确定性 | AI任务A | 模型revision、提示或adapter、输出正确性与耗时 |
| n3 summarize | 结构化事件＋会话上下文→辅助摘要 | AI任务B | 模型/adapter兼容；输出质量不可只凭生成成功 |
| n4 persist | 摘要＋进度→可恢复会话状态 | 确定性存储 | 序列化bytes、恢复等价性、传输成本 |

依赖为n1→n2→n3→n4；先证明一条语义正确的链，再扩分支/汇聚。它不替代Alibaba多形态DAG覆盖。
移动车辆关联会话，RSU切换影响执行位置；切换频率来自轨迹/布局，不能强行安排在每个最有利节点。
状态先定义为已完成节点、结构化结果、会话上下文及模型身份；不宣称实现KV迁移。
两AI任务是否可共享同一base、是否使用真实adapter，必须由模型兼容证据决定。若仅prompt切换，
明确标记prompt-based，不制造两个adapter身份来证明adapter缓存收益。

## 3. 旧新字段对应与实施顺序

| 现有字段/机制 | 复用 | 未闭环项 |
|---|---|---|
| node_id、predecessors、successors | DAG结构 | 新增独立sidecar记录任务语义、I/O类型及来源；尚不修改live schema |
| required_base_model、required_adapter | 已确认兼容时映射 | 非AI节点和无adapter模型如何表达需先明确，不能填假adapter |
| input_size/output_size | 保留历史接口 | 单位/缩放与新bytes测量显式转换，禁止静默沿用CPU/mem倍率 |
| workflow进度 | 作为状态的一部分 | 状态内容、版本、恢复一致性需测；不把dataclass自动等同模型内部状态 |
| CacheEvent及catalog | 沿用既有完整性 | 新input/intermediate类型须producer、consumer、记账、统计共同测试后接入 |

不能为了立刻跑通把新模板强塞进旧formal合同。先独立应用执行与profiling，再版本化适配新开发路径。

## 4. 有界profiling草案

在模型、许可证、设备及隔离依赖确认后冻结一次小规模测量包：

- 最多一个base及两个经过验证的任务adapter；没有合格adapter则先做model/state profile，明确共享证据缺失。
- 每个任务三个固定输入长度档；先每档2次warmup、10次测量，上限两任务72次推理调用。
- 另记录模型加载；OS文件缓存未被可靠控制时只能称process-cold，不能宣称storage-cold。
- 会话状态最多10个固定样本，各3次序列化/恢复；验证恢复后内容及任务结果的一致性。
- 计时使用monotonic；记录硬件、精度、线程、输入输出长度、实际存储bytes、内存测量方法和原始样本。
- Mac本机profile仅支持该硬件/软件组合，不直接外推为RSU GPU性能或无线延迟。
- 不在未测量前填写毫秒/MB结果；网络传输模型若仅假设，标为assumed并单独敏感性分析。

上述数量是待执行测量设计，不是已经运行，也不构成精度/功效保证。耗时及模型下载量待候选确定。
真实测量使用独立目录/环境；不修改冻结`.venv`、模型原件、旧正式数据或已消费holdout。

## 5. 下一步输入与完成条件

需要用户提供现有AI应用模型/adapter的本地路径，或允许先选定公开模型并在报告精确下载量、许可证、
依赖和存储位置后单独批准下载。若用户所说AI workflow包另有文件，请提供路径；本轮只找到项目现有DAG接口。

取得输入后才可生成实际profile和小样，完成条件是：任务可运行、输出语义检查、模型兼容、状态恢复、
字段来源、单位和无策略筛选全部通过。不能以通过schema替代模型质量或真实性。
本轮无模型下载/安装、无训练/rollout、无新数据集发布、无后台任务，因此没有需要持续轮询的进程。
