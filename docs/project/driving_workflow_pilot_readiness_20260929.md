# 驾驶AI工作流：进入小规模实验前的资源与执行计划

- reviewed_at / literature_cutoff: 2026-09-29
- target_venue: IEEE TMC（目标，不代表资格）
- policy_version: tmc_review_policy_v3_20260621
- git_base: 0eca888
- artifact_run_id: drivelm_pilot_preparation_20260929
- evidence_level: metadata/源码与输入组装直接验证；模型性能 E0_UNAVAILABLE
- status: PLAN_VALIDATED_RESOURCE_SETUP_PENDING；不是实验已完成或运行就绪

## 目标和已经做完的工作

首先验证带上游生成上下文的驾驶QA是否可运行、有质量价值，以及其时间/数据成本，
不是继续寻找让SA胜出的数据。只有这个工作流站得住，后续缓存与迁移比较才有研究对象。

本轮并行核验模型、图像和计划工具，主线程复核代码并运行真实demo输入。
新增 `scripts/prepare_drivelm_pilot.py`：所有scene各选frame ID字典序首帧，四阶段各固定index 0，
不按A/策略结果选题。最多两scene，两臂为独立回答和同帧前序生成上下文，8/16次调用计划。
实际模型调用0；真实demo计划16个task、16次payload检查通过。
stdout为10736 bytes，SHA256 `cd97e9c0ca15f575818c0dd48b459aad44ac9dc03895f5f7ad697a96f0e75322`。
两个确定性场景各一帧不是统计功效充分的测试集；只是接口与质量诊断。

## 资源定位：无需整个图像包

[官方demo入口](https://github.com/OpenDriveLab/DriveLM/blob/1de72a74b257e5373400fa68239e99bd5d20580a/challenge/README.md)
在GitHub提供9帧54张JPEG。Git tree metadata合计6,114,069 bytes，与JSON的54个路径完全相等。
选定两帧12张共1,287,809 bytes，加JSON共1,690,193 bytes；不需要3.48GB训练图像包。
精确路径/大小/Git blob SHA1保存于 `configs/experiment/drivelm_pilot_resources.json`，
源revision为 `1de72a74b257e5373400fa68239e99bd5d20580a`。图像未下载/解码，不伪造content SHA256。

保留六个摄像头及名称，不能只取front：所选预测/规划Q含CAM_FRONT或CAM_BACK坐标标记。
不得拼图后仍当原坐标使用；预处理缩放需保留原图坐标解释及变换记录。
不得将JSON的 `../nuscenes/...` 直接拼接为可读路径；下载与加载须白名单映射、hash校验、拒绝路径逃逸。

[来源许可说明](https://github.com/OpenDriveLab/DriveLM/blob/1de72a74b257e5373400fa68239e99bd5d20580a/README.md#license-and-citation)
区分代码、语言数据与原图来源许可；语言数据为CC BY-NC-SA 4.0，nuScenes原图遵循自身条件。
[nuScenes terms](https://www.nuscenes.org/terms-of-use)完整附加条款本轮文本渠道未核清，需负责人确认适用使用。
HF gated入口未登录或代接受。GitHub存在文件不等于无限制再分发；本轮不发布原图/问答。

## 模型与本机环境

候选 [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
为Apache-2.0多图/文本模型，适合作为小规模可运行性候选，不是驾驶认证。
候选revision `a7da5b986cb59b408707209984f360a5f4ad7e47`；权重约1.015GB，
全仓含额外ONNX格式，不应整仓下载。只取运行所需配置/tokenizer/processor与safetensors。
主线程随后成功读取上述固定revision的LFS pointer：`model.safetensors`为1,015,025,832 bytes，
SHA256 `d05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2`；
这是pointer元数据核验，不是已下载权重的字节复算。最小运行文件集合总量仍待逐项核验。
不引入真实adapter或宣称adapter共享；Qwen2.5-VL先保留为后备，不同时下载两个模型。

本机实查：Python3.9.6、Torch2.8.0、arm64、MPS built/available均true；
transformers、Pillow、safetensors、huggingface_hub、torchvision均不可发现；可用磁盘约354GiB。
[Transformers 4.49.0](https://github.com/huggingface/transformers/blob/v4.49.0/setup.py)
声明范围允许Python>=3.9，包含Idefics3；版本范围匹配不等于本机加载成功。
资源准备应在独立环境完成，保留旧`.venv`。不调用付费服务，不上传项目数据。
Mac不用FlashAttention2；MPS实载待验证，CPU若作为替代必须单列计时，不静默合并。

## 第一次实验的有界规则（待实载后形成可执行包）

- 先最多2次兼容性推理：一图/六图各1次，只验证模型入口与内存；单列不进入比较。
- 随后固定16次比较调用，batch=1、greedy、每次最多64新token；不自动换题或重试找赢家。
- 两臂同图、同问题、同模型、同解码参数；上下文臂只读本臂同帧生成结果，参考A另供评分。
- 不加载key_object_infos作为隐藏提示。原Q可能有标注线索，人工记录并限定任务边界。
- 记录模型/输入身份、相机顺序、原始输出、输入输出token、单调时间、加载时间、截断及异常。
- 当前计划按arm分块，仅用于接口pilot；冷/热及顺序影响未消除，不据16次样本宣称推理速度优势。
- 质量先由盲化两臂标签的人工核验：回答是否响应问题、图像依据是否正确、是否引入上游错误；
  保留原参考与评分理由，不以生成非空视为正确。不给无依据的自动语言相似度冠以驾驶准确率。
- 真实兼容性检查或pilot失败则记录并诊断，不重复调参直到优势出现。长任务只启动一次、写终态回执，
  不用AI持续轮询；当前没有启动任何后台进程。

若任务有足够可用输出，再进入状态序列化/恢复与成本profiling，之后才扩展匹配cache/migration机制实验。
若模型不适用，应按错误类型决定换模型或缩减任务，不把失败藏掉。九帧仍是开发小样，不是新独立holdout。

## 验证及未闭环项

- 计划/预览/审计三组pytest：48 passed。
- smoke：6/6 toy节点完成；新脚本与测试编译、resource metadata结构及git diff检查通过。
- 真demo公共计划CLI：rc0；16次synthetic payload检查通过，不是模型调用。
- 模型、图像、依赖实际安装/实载、原始输出、质量评分、成本测量、实验runner尚未验收。
- 资源下载许可和约1GB模型下载需要确认。未训练、未选模、未运行正式实验或旧holdout。

下一步不是再加授权平台：资源准备→真实模型入口→18次以内小实验→读回质量/成本，
通过后再讨论算法与数据集贡献。
