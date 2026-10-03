# 两节点真实 AI 工作流前缀保存与后缀恢复冻结方案

状态：`FROZEN_BEFORE_SCIENTIFIC_EXECUTION`

本方案只验收 technical workflow calibration：证明独立 target 进程消费 source 已计算的 `n0` 文本中间结果，
并仅执行 `n1`。它不验证 ALPR 任务正确性，不扩展 production action codec，不运行训练、formal 或 holdout。

## 固定基线与资源

- 代码起点：`4689403da1d66716f328108d1ef70455563b20ce`；本轮科学执行必须另绑定 clean、已提交的执行器 commit。
- 环境：`artifacts/environments/adapter_state_acceptance_py39_v1`，Python 3.9.6；不得安装或更新依赖。
- base：本地 SmolVLM-500M-Instruct，weight SHA-256=`d05b567e...91a2`。
- adapter：只使用本地 ALPR LoRA，weight SHA-256=`95103f5d...0f83`，激活名固定为 `adapter_a`。
- 技术输入：本地 `TrafficDemands.png`，SHA-256=`83af0966...c307`；没有 ALPR/Helmet 标签，任务正确性固定为
  `unavailable`。
- 完整路径、processor hash、prompt 和生成参数以
  `configs/acceptance/two_node_workflow_suffix_recovery_v1.json` 为唯一机器可读方案。

## 固定 DAG、输入与判据

- DAG：`n0 -> n1`。
- `n0`：固定图片与固定视觉提示，greedy、CPU/float32、最多 8 个新 token。
- `n1`：将 `n0` 原始解码文本逐字嵌入固定模板；只做纯文本处理，不读取图片，greedy、CPU/float32、最多 8 个新 token。
- 连续路径：独立进程按顺序执行 `n0,n1`，2 次 generate。
- source：独立进程只执行 `n0`、原子保存声明状态后退出，1 次 generate。
- target：确认 source 已退出后启动，先校验状态，再加载本地静态模型资源，只执行 `n1`，1 次 generate。
- 恢复保真性：continuous/source 的 `n0` 文本和 token IDs 严格相等；continuous/target 的 `n1` prompt、
  rendered prompt、input IDs 与输入 hash 严格相等；两条路径的 `n1` token IDs 严格相等。
- 逐 token 判据的先验理由：两节点均为同一 CPU/float32 模型、adapter、processor 上的 greedy 解码，后缀输入显式冻结；
  本轮不在失败后放宽判据。
- 任务正确性与恢复保真性分开；输出语义不用于筛选、重试或升级正确性结论。

## 状态与目标端边界

状态 payload 固定声明 schema/workflow/DAG、完成与剩余节点、next node、`n0` 原始文本和 token IDs、后缀实际消费
文本、两节点生成配置、base/adapter/processor 身份、RNG/tensor 状态是否需要及理由；envelope 保存 payload SHA-256。
包只允许 `state.json` 和 `manifest.json`，禁止模型权重和预计算 `n1` 答案。

RNG continuation 不保存，因为 `do_sample=false` 且 `n1` 是消费显式文本的新节点；KV/tensor 状态不保存，因为边界位于
应用节点之间，`n1` 不做 `n0` 的 token continuation，也没有跨边界 tensor 输入。

target 可读取：同一预存 base/processor、单 adapter、冻结方案、版本化执行器、上述两个状态文件。target 明确不得读取
source 图片、source 内存或未声明临时文件。图片路径只属于 source provenance，不属于 target 动态读取白名单。

## 负例与预算

模型加载前必须完成：缺失中间结果、陈旧 hash 的字节篡改、next-node 冲突、completed-node 冲突、模型身份冲突均拒绝；
两份各自合法的不同描述必须产生不同 `n1` 输入内容/hash。旧回执的 adapter 汇总读取
`available_adapters`；缺失及与 legacy `loaded_adapters` 冲突的输入均拒绝。负例不进入科学计时。

科学路径计划严格 4 次 generate，硬上限 6；预列兼容 generate 为 0。每进程 300 秒，累计科学执行 900 秒，失败不自动
重试，不换 prompt/输入/标准。报告单次见证，不称均值、benchmark、RSU 或无线链路实测。

## 成本分层

结果分列：目标端预存静态 base/processor/adapter；目标缺失时才需传输的 base/adapter；每 workflow 的动态 payload 与
完整状态包。记录 serialize/save、package read、integrity validation、state restore/validation、输入重建、target 模型加载、
后缀推理、子进程启动开销和总墙钟。不得与旧 synthetic 104 MiB 直接相减或声称净收益。
