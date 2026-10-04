# 车联网 AI 任务—adapter 适配性审计与下一轮停止门禁

- `reviewed_at`: `2026-10-05`
- `literature_cutoff`: `2026-10-05`
- `target_venue`: `IEEE TMC`（长期研究目标，不表示本轮 paper-ready）
- `artifact_run_id`: `adapter_task_suitability_audit_20261005_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `review_base_git_commit`: `22f0b77`（隔离分支中已保留的 ALPR 负结果）
- `report_git_commit`: `PENDING_BIND_AFTER_REVIEW_COMMIT`
- `evidence_level`: `E1_DOCUMENTED + local metadata audited`；没有新的模型调用或结果 artifact
- `verdict`: `NO_SUITABLE_VERIFIABLE_CANDIDATE`

## 1. 本轮边界与既有结论

本轮只读核查本地模型配置、固定 revision 的公开模型卡、关联仓库、数据卡和公开评估线索；没有下载权重或图像，没有安装依赖、加载模型、`generate`、训练、RL、formal 或旧 holdout 操作。主工作区七个用户修改文件未触碰。

既有 UniqueData 12 图结论保持不变：development 为 base `3/4`、ALPR adapter `0/4`；locked_check 双方是同样的 `4/8`；独立评分复算 24 行 mismatch=`0`。原始产物、提示词、样本和评分不得改写。该结果不支持当前 ALPR adapter 在该协议下的任务增益，也不能推广为所有 adapter 无效。完整边界见 [既有 ALPR 记录](public_alpr_sample_pilot_20261004.md)。

本报告使用以下分级：

- `A`：已确认任务、输入、processor 或输出合同不匹配；
- `B`：底座/PEFT 技术兼容，但任务适配、数据、评估或许可证据不足；
- `C`：模型—任务—输入—数据—许可—评估链足够完整，可进入有界小样本验证。

## 2. 最多三个候选的适配性对照

| 排名 | 候选与精确 revision | 车联网服务语义 | 输入、prompt 与输出证据 | base / processor / PEFT | 数据、评估与许可 | 分级与决定 |
|---:|---|---|---|---|---|---|
| 1 | 本地 `revitotan/FT-SmolVLM-500M-Instruct-Helmet@96794cb1fd9abf1636b8f1541095f6f9229e4387` | RSU 路侧两轮车乘员头盔合规计数；若能力成立，可输出可验证的 `{person-with-helmet, person-without-helmet}` 服务结果 | 发布者同名数据页可见整图输入、固定 JSON-schema 提示和两个整数计数；不是框裁剪输入 | 声明 base=`HuggingFaceTB/SmolVLM-500M-Instruct`；本地 base=`a7da5b986cb59b408707209984f360a5f4ad7e47`；LoRA `r=4, alpha=1, dropout=0.1`，非 DoRA，7 类线性层；`task_type=null`。模型仓库另含 tokenizer/processor 文件，本地下载回执明确未取这些附属文件 | 模型卡写 `unknown dataset`、`results=[]`。同作者数据集 `@296f05ed159c7dcb494962a2a11313180e1015a1` 有 2,190 行、单一 train parquet `204,327,762` bytes，声明 MIT；模型卡没有把该 revision 绑定到训练，数据卡没有上游图像来源/权利说明，也没有未参与训练的检查集。W&B 链接存在，但本轮不能从公开页面核验其 dataset binding、split 或 task metric | `B`。唯一证据补全首选，但**不是**已批准实验首选 |
| 2 | 本地 `Hirai-Labs/FT-SmolVLM-500M-Instruct-ALPR@edccd0e7ae5cbded1f35c21ea108988e1e32b239` | 路侧车辆身份字段读取；隐私敏感，只允许本地、去标识的受控研究 | 关联官方 demo 使用整车图和五字段 JSON schema（type、license plate、make、model、color），没有证据表明必须先裁剪车牌。demo 却用 256M processor 加载声明为 500M-base 的模型，形成 processor 歧义；当前模型卡不冻结原训练 prompt | 同一 500M base；DoRA `r=64, alpha=1, dropout=0.1`，7 类线性层；`task_type=null`。本地 500M processor 技术运行已通过，但“能加载”不等于任务适配 | 模型卡写 `unknown dataset`、无结果。关联组织当前 ALPR dataset 有 train/test 和整车字段，但 dataset 在模型发布后仍更新，模型卡没有绑定训练 revision；当前数据卡缺少许可字段与上游图像来源。Apache-2.0 只覆盖模型/代码声明，不能自动覆盖图像权利 | `B`。不重跑旧 12 图；不把新发现的整车 JSON 合同用于回头改提示词重算 |
| 3 | 远程 `samlucas/smolvlm_500m-parking_occupancy-PKLot-instruct-without-context-without-expert@de341dddb5a082b5c771197e7364a08754e1ee44` | 固定路侧相机的停车位占用识别，可支撑停车诱导/边缘监测 | 仓库名暗示 PKLot 与“without-context”，但模型卡模板为空，未发布训练 prompt、预处理或输出 schema；因此不得按名称猜测是整幅停车场、给定车位框还是单车位裁剪 | 同一 500M base；DoRA `r=8, alpha=8, dropout=0.1`，7 类线性层；`task_type=null`。远程 adapter 约 20.5 MB，尚未下载 | PKLot 官方数据有 12,417 张停车场图、XML 车位坐标和约 695,900 个车位裁剪，CC BY 4.0；但模型卡没有把该 adapter 绑定到官方数据 revision、split 或指标。trainer state 只有训练 token accuracy/loss，无任务检查结果；官方包约 4.6 GB，不符合在证据未闭合前先下载 | `B`。不更换 base，但也不因任务名和训练 loss 判定适配 |

三项都没有达到 `C`。没有发现“ALPR 实际必须裁剪车牌”的一手证据；因此本轮不建立裁剪版新任务，更不会裁剪旧检查集覆盖既有结论。

## 3. 唯一优先补证对象：Helmet（仍未获实验资格）

Helmet 排第一不是因为预期胜率高，而是因为：它已经本地存在、与现有 500M base 的结构声明一致、任务输出是可严格评分的两个整数、服务语义直接对应路侧交通安全，并且 adapter 仅约 9.64 MB，后续 cache resident / transfer 成本容易实测。该专用模型若相对 base 在独立图像上提高严格计数正确率，就提供“头盔合规结构化计数”这一可验证服务能力；系统保留它的理由将是避免每个 RSU 重复传输/加载同一任务增量，而不是假设它在所有视觉任务上优于 base。

当前仍缺三项最小证据，任一未补齐都不得调用模型：

1. 发布者可追溯的训练 manifest：把模型 revision `96794cb...` 明确绑定到数据 repo + exact revision、实际 train/eval row IDs、原始 prompt、processor 来源和 preprocessing。
2. 数据权利链：不仅是顶层 `MIT` 标签，还要给出原图来源及其允许本地研究/下载的许可依据；模型/代码许可不能替代图像许可。
3. 真正未用于训练的带标签检查集，或可证明在模型发布后独立采集的数据；需要 source/capture/video identity。仅对单一 train split 做新的 4+8 划分不能把已见训练图变成未观察检查样本。

因此本轮没有精确下载授权申请。下载现有 2,190 行 train parquet、第三方大型 helmet 包或更多权重都不能单独消除上述 blocker。

## 4. 条件满足后唯一允许的小样本协议

下面冻结的是**条件式协议骨架**，不是当前执行授权；只有第 3 节三项证据齐全、并能把具体数据 revision、文件 hash 与 row IDs 填入 manifest 后，终态才可重新评估为 `WAITING_FOR_EXACT_DOWNLOAD_AUTHORIZATION` 或 `READY_FOR_BOUNDED_TASK_VALIDATION`。

### 4.1 固定资源与任务

- adapter：`revitotan/FT-SmolVLM-500M-Instruct-Helmet@96794cb1fd9abf1636b8f1541095f6f9229e4387`，本地 `adapter_model.safetensors` 为 `9,641,944` bytes，SHA-256=`e899c9c13b53bf80bcfc86048f5678a727e8c1838dc824156b521d5cdaafdd2d`。
- base：`HuggingFaceTB/SmolVLM-500M-Instruct@a7da5b986cb59b408707209984f360a5f4ad7e47`，本地 `model.safetensors` 为 `1,015,025,832` bytes，SHA-256=`d05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2`。
- processor：只允许使用该 base revision 的 `Idefics3Processor` 文件集合；若训练 manifest 指向 Helmet 仓库自带 processor/tokenizer，必须先逐文件比较。任何差异都要先确定双方共同使用哪一套，不能只给 adapter 臂专属 processor。
- 任务：整幅道路图中 `person-with-helmet` 和 `person-without-helmet` 的计数。processor 内部 resize/split/normalize 是任务成本；不生成裁剪文件，不把真值框、类别或人数送入任一模型臂。
- prompt：使用发布数据中的完整 instruction 与 JSON schema，冻结如下；两臂完全相同，不在 development 结果后修改。

```text
Examine the total number of persons wearing a helmet and those without a helmet.
The output should be formatted as a JSON instance that conforms to the schema below.
For example, given the schema: {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
A well-formatted instance would be: {"foo": ["bar", "baz"]} while the object: {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
Here is the required output schema: {"properties": {"person-with-helmet": {"title": "Person with Helmet", "description": "Total count of persons wearing a helmet", "type": "integer"}, "person-without-helmet": {"title": "Person without Helmet", "description": "Total count of persons not wearing a helmet", "type": "integer"}}, "required": ["person-with-helmet", "person-without-helmet"]}
```

- parser：先严格 JSON；为兼容公开标签的单引号表示，仅可回退到受限 `ast.literal_eval`，结果必须是恰含两个字段、值为非负整数的 dict。禁止从解释文本抽数、补字段或重试。

### 4.2 确定性样本与去重

- 合格数据必须先提供 exact revision、许可文本、文件大小、source/capture/video identity 与标签生成说明；这些字段当前为 `MISSING`，所以尚不能写出合法 row ID 清单。
- 在模型输出不可见时，以 source sequence/capture ID 建组；缺 ID 时再用相同字节 SHA-256、64-bit dHash Hamming distance `<=4` 的连通分量作保守近重复代理。人工复核只用于合并更多疑似组，不能拆开代理认定的重复组。
- 按 `(helmet_count, no_helmet_count, source_group)` 分层；组内使用 `SHA256("helmet-adapter-pilot-v1:" + canonical_source_id + ":" + relative_path)` 排序。
- development 取 4 个不同 identity groups；locked_check 取另外 8 个不同 identity groups。任何 source/capture/video group 不得跨 split。该 4+8 只是小样本检查，不称 formal/holdout。
- 对拟用 12 图与发布者训练 manifest 的所有训练文件做 exact SHA-256 比较；可获得训练图时再做 dHash/pHash 近重复筛查。未公开的训练数据只能记 `training_overlap=UNVERIFIED`，不能用“没搜到”当无交叠证明。

能力边界：hash/pHash 不能排除裁剪、颜色变换、压缩、相邻视频帧或同一物理场景的未公开版本；没有可靠 identity 时即使启发式无重复，也不能宣称独立泛化。

### 4.3 公平推理、指标和预算

- 两臂按预先冻结、交错的样本顺序运行；共同使用同一整图字节、processor、prompt、CPU float32、单线程、seed、greedy decoding 和 `max_new_tokens=96`。
- base 臂为纯 base；adapter 臂只增加上述 adapter。不得给 adapter 臂额外 system prompt、few-shot、框、标签或重试。
- 记录图像 decode、processor、model load、adapter attach、每次 generation、parse 的时间与峰值 RSS；预处理成本单列，不能吞入 adapter 收益。
- 主要指标：两个计数均精确的 `tuple exact match`。次要描述指标：每字段 exact、两个字段绝对误差之和、合法 parse 率。12 图只报逐项和计数，不做显著性、CI、泛化或论文就绪判断。
- 总调用上限：`12 images × 2 arms = 24 generate`；总 wall-clock 上限 `1,800 s`。任一臂加载失败、输入 hash 不同、配置漂移、token 截断或 parser 失败均原样记失败；不自动重试。超过上限停止并保留部分结果，不换样本。

### 4.4 三种结果的含义

- 正结果：locked_check 的 adapter `tuple exact >=6/8`、合法 parse=`8/8`，且 paired `adapter-only-correct - base-only-correct >=2`。这是事前可用性停止线，不是假设检验；只说明在该小样本上同时观察到基本任务能力和增量，可进入更严格独立验证与 runtime profiling，不证明缓存、迁移或 RL 收益。
- 无差异：paired 净差在 `[-1, 1]`，或 adapter/base 同时低于可用性线。仍可保留 adapter 作为系统负对照或 task-object，但不能以任务优势解释其缓存价值；是否继续只取决于预先批准的更大独立集，而不是挑正例。
- 负结果：`base-only-correct - adapter-only-correct >=2`，或 adapter 相对 base 引入更多 parse/token-limit 失败。停止将此 adapter 作为正向服务实例；保留完整负结果，不改 prompt、样本或评分补救。
- 混合结果：adapter 达到可用性线但 paired 增量不足，或有 paired 增量但 `tuple exact <6/8`。只能报告两项事实，不能合并成“adapter 有效”；默认不晋级缓存价值实验。

## 5. 对缓存/迁移研究的作用与不能证明的事项

若 Helmet 后续通过任务正确性门禁，它能提供一个共享 `SmolVLM-500M` 底座、RSU 按需驻留小型任务 adapter 的真实实例；可测对象包括 adapter bytes、attach/load 时间、冷/暖服务延迟和跨 RSU 重复需求。只有真实业务流程确实消费计数结果时，才可再建依赖，例如“合规计数 → 规则告警/事件记录”的确定性消费；不得为了 DAG 把 Helmet、ALPR 或停车任务人工串成无关模型链。

即便 24 次调用出现正结果，也不能证明：普遍 adapter 有效、任意道路/天气/人群泛化、车牌/头盔/停车任务共享同一准确率规律、RSU cache 命中收益、handoff/state migration 收益、RL 策略优越、正式统计显著或 TMC-ready。缓存机制可接受某些专用 adapter 不优于 base，但每个被称为“有用服务”的 adapter 必须先有独立、可复算的任务能力证据。

## 6. 公开证据入口

- ALPR 模型卡与配置：https://huggingface.co/Hirai-Labs/FT-SmolVLM-500M-Instruct-ALPR/tree/edccd0e7ae5cbded1f35c21ea108988e1e32b239
- ALPR 关联 demo/repository：https://github.com/Hirai-Labs/alpr-vlm/tree/639f42da29ff2ca64ec9d4c66ffd7949501e8262
- ALPR 当前关联数据：https://huggingface.co/datasets/Hirai-Labs/alpr-vlm-instruct-dataset/tree/dd09bc77ea36460e3434a8241a4ace940f06b806
- Helmet 模型卡与配置：https://huggingface.co/revitotan/FT-SmolVLM-500M-Instruct-Helmet/tree/96794cb1fd9abf1636b8f1541095f6f9229e4387
- Helmet 同作者数据：https://huggingface.co/datasets/revitotan/helmet-threerider-vlm-instruct-dataset/tree/296f05ed159c7dcb494962a2a11313180e1015a1
- Parking adapter：https://huggingface.co/samlucas/smolvlm_500m-parking_occupancy-PKLot-instruct-without-context-without-expert/tree/de341dddb5a082b5c771197e7364a08754e1ee44
- PKLot 官方数据与许可：https://web.inf.ufpr.br/luizoliveira/research-interests/pklot/

## 7. 终态

`NO_SUITABLE_VERIFIABLE_CANDIDATE`

原因不是 adapter 必然无效，而是三项候选均缺少至少一个不可由“成功加载”替代的关键合同：模型—训练数据 revision 绑定、原始输入/prompt、独立标签检查集、上游图像许可或可追溯任务评估。当前不申请下载，也不直接转入自训练。
