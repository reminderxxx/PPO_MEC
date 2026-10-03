# ALPR 任务验收数据：来源资格、授权边界与冻结取样方案

## 结论与执行边界

- `reviewed_at`: `2026-10-04T01:56:27+08:00`
- `literature_cutoff`: `2026-10-04`
- `fixed_git_base`: `62e1eeb42758d760684e949deb24727407d40266`
- `plan`: `configs/acceptance/alpr_task_acceptance_data_plan_v1.json`
- `verdict`: `B. 来源已明确，等待精确条款/下载授权`
- `actual_dataset_downloads`: `0`
- `model_loads / generate / training / formal / holdout`: 均为 `0`

本轮只收敛一条线：**完整图像上的 ALPR 车牌文本识别**。首选为 UFPR-ALPR v1.0；它有真实文本标签、
150 个明确车辆/1 秒视频组，能先分组再取帧。其官方条款要求申请人本人从学校邮箱申请并接受非商业、
不再分发、不修改等条件，作者再发放下载链接；本轮没有代用户接受或发信，因此没有下载。CCPD2019 作为唯一下载
备选；DriveLM 只用于核清既有授权，不扩成第三条任务线。

这不是“本地没有数据”的重复结论。当前 blocker 已具体化为：UFPR 数据所有者的人工许可与随后绑定到专属链接、
文件名、字节上限的下载授权。

## 三个候选的限定核验

### 1. 首选：UFPR-ALPR v1.0（2018）

官方[数据页](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/)与
[原论文](https://web.inf.ufpr.br/vri/wp-content/uploads/sites/7/2019/08/laroca2018robust.pdf)说明：

- 4,500 张 `1920×1080` PNG，来自 150 个 1 秒、30 FPS 视频，即每个车辆/视频 30 帧；车辆和采集相机都在运动。
- 官方划分为 40% train、20% validation、40% test；论文按车辆报告 test=60 个 vehicles。
- 每图文本标注含 camera、车辆位置/类型/厂商/车型/年份、车牌 identity/text 与位置、字符位置。
- 因而它不是“只有车牌框”的检测数据；能支持本轮需要的 plate-text recognition，也能以 vehicle/video 为最高合理组。

官方[许可页](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/license-agreement/)不是开放下载许可：只允许教育/研究机构
的非商业学术研究；未经 VRI 明示许可，不得再分发、修改或商业使用；图像只可在学术论文/演示中展示，并有免责声明、
赔偿及引用要求。申请必须由本人使用有效学校邮箱向 `rblsantos@inf.ufpr.br` 发送指定文本，通常 1–3 个工作日后才会
收到链接。Codex 不能替用户接受这些条件、签名或发送邮件。

官方没有公开 archive 文件名、Content-Length、hash 或较小的 per-vehicle 下载单元。一篇公开的
[ALPR 数据集综述](https://thesai.org/Downloads/Volume13No12/Paper_45-Deep_Neural_Network_Training_and_Testing_Datasets.pdf)
给出的平均 PNG 约 2.4 MB，据此仅作磁盘规划估计：4,500×2.4 MB≈10.8 GB（十进制）；它不是官方包大小。未来授权必须先读回作者
链接的实际文件名和大小；超过 12.5 GB 上限或内容并非 v1.0 时停止并重新确认。

隐私与再分发边界：车牌可识别；原图只能保存在本机被忽略的数据目录，不进入 Git，不上传外部 AI 服务。由于许可禁止
未经批准的 modification，本任务不生成或保存裁剪/增强图；模型输入使用原始完整图。检测框仅保留为审计元数据。

### 2. 备选：CCPD2019

原作者仓库固定到 commit `02aaea15137c4d2fe662e57d257c6822356e9304`：

- [README](https://github.com/detectRecog/CCPD/blob/02aaea15137c4d2fe662e57d257c6822356e9304/README.md)
  明确称 dataset 为 MIT，提供原始
  [Google Drive archive](https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc)，无需作者审批。
- `CCPD2019.tar.xz` 超过 300,000 张 JPEG；文件名编码 area、tilt、plate box、四顶点、七字符 plate text、
  brightness 与 blurriness，确实是识别标签。
- 第三方托管页的可追溯 [Git LFS 元数据](https://huggingface.co/datasets/soikit/green_plates/commit/980f71e776e8ff810b90a903c3a3b1d0d86a26e4)
  给出大小 `13,164,924,944` B 与 SHA-256
  `e4fd23c9f289c9e5404d9b25da6b2453714150c71b8804c7e96b9d94debcb44e`；这两个值不是原 Google Drive 页面直接声明，
  若未来下载必须重新计算，任何不符都停止。

缺口是分组：官方七个文件名字段没有 video/track/physical-vehicle ID。规范化 plate text 的 salted hash 可作为 vehicle
proxy group，同一车牌绝不跨组；但它仍不如 UFPR 的官方 vehicle/video 身份，且无法证明相同车辆不会换牌或标签异常。
此外最小官方单元是整个约 13.2 GB archive，没有官方小包。故它只在 UFPR 申请失败、用户明确接受 proxy-group 限制
并单独授权该精确大文件时作为备选。

### 3. 既有授权边界：DriveLM demo / nuScenes

历史记录不能写成“从未授权”。2026-09-29 的真实聊天序列是：

1. 在展示“两帧 12 图约 1.3 MB＋固定模型”的资源范围后，用户回复“允许下载”；助手同时承诺若图像条款需要接受则停止。
2. 在随后展示 [nuScenes terms](https://www.nuscenes.org/terms-of-use)并明确“首个工作流六张图”后，用户回复“确认”；
   助手将范围解释为非商业研究用途、仅这六张图。
3. 之后只读取了本地 workflow metadata，没有图像下载完成回执。

可追溯授权因此只覆盖 DriveLM revision `1de72a74b257e5373400fa68239e99bd5d20580a` 中 scene
`54cdaaae372d421fa4734d66f51a8c48`、frame `0dd14c14cea14dc1b0c0c9b0c9c7c4c3` 的六个相机文件，
总计 `724,191` B；精确路径与 Git blob SHA-1 已保存在 `configs/experiment/drivelm_pilot_resources.json`。
它不覆盖第二帧 6 图，更不覆盖 demo 全部 54 图。

[DriveLM 官方许可](https://github.com/OpenDriveLab/DriveLM/blob/1de72a74b257e5373400fa68239e99bd5d20580a/README.md#license-and-citation)
将语言数据置于 CC BY-NC-SA 4.0，并明确 nuScenes 原图继承上游条款。六个相机文件来自同一 frame，只能算一个
scene/group；它也不是 ALPR plate-text 标签源。因此即使既有授权精确有效，也不能形成本轮需要的开发/检查两组，
本轮不下载它们、不扩权，也不改做驾驶 VQA。

## 冻结取样与组隔离规则

在任何模型执行前固定如下；机器可读原文在 plan JSON 中：

1. 只使用 UFPR 官方 test split；取得 archive 后先验证官方 split 是否把同一 vehicle/video 的 30 帧全部置于同一 split。
   若不能验证，停止，不能用 image ID 冒充分组。
2. seed 固定为 `ppo_mec_alpr_acceptance_v1_20261004`。对合格 vehicle/video group 计算
   `SHA-256(seed + ':' + group_id)` 并升序排列；前 12 组入选，不按清晰度、车牌内容、模型输出或算法预期筛选。
3. 每组只取一帧。帧按 `SHA-256(seed + ':frame:' + relative_path)` 升序，取首个可解码且官方标签完整的帧；所有跳过项
   记录路径和冻结理由。前 4 组为 development，后 8 组为 `locked_check`。
4. 若少于 12 个合格组则如实缩减，不复制、增强或裁剪；少于 2 组时因无法形成跨组隔离而停止。
5. exact duplicate 用原始字节 SHA-256；64-bit perceptual hash 跨组 Hamming distance≤6 只触发本地人工复核，
   不单独证明语义重复。若来源审计确认 group identity 冲突，则合并 group 后按同一冻结顺序补位。
6. `locked_check` 只表示下一轮开发隔离，不称 formal holdout、外部泛化或独立于预训练。

## 任务与标签合同

- 输入：一张原始完整图；不向模型提供 ground-truth vehicle/plate/character box，不保存派生 crop。因此称
  `full-image plate-text recognition with implicit localization`，不称 scored end-to-end detector。
- 输出：唯一可见的巴西车牌文本，不带解释。
- 原始标签与标准化标签同时保存。标准化为 Unicode NFKC、ASCII 字母大写、删除非 `A-Z0-9` 字符；本数据预期为
  `^[A-Z]{3}[0-9]{4}$`。
- 对整段原始模型输出应用同一标准化，禁止在看到标签后挑选有利 substring。exact match 要求标准化字符串完全相等。
- CER=`Levenshtein(prediction,label)/len(label)`，不截断大于 1 的值。
- 官方标签缺失、含不可辨认标记或无法与图像绑定时保留 `uncertain=true` 和理由，不猜标签；不进入主 exact/CER 分母，
  但计数必须报告。真值不得由待测模型产生。
- 本地 ALPR adapter 模型卡明确写 `unknown dataset`。因此即使本包内部跨组无泄漏，也必须记录
  `adapter_training_overlap=UNKNOWN`，不得称独立于 adapter training/pretraining。

未来有合法 payload 时，create-only 包至少写入 `source_manifest.json`、`samples.jsonl`、`split_manifest.json`、
`task_contract.json`、`data_card.md`、`exclusions.jsonl` 与 `integrity_manifest.json`；逐图验证 decode、标签绑定、
bytes/SHA-256、exact/near duplicate 与跨 split group intersection。原图和明文标签不提交 Git。

## 精确的有限授权文本

### 首选：先由用户完成官方申请

以下是官方要求的邮件结构；尖括号字段和签名必须由用户本人填写，Codex 不代填、不代发：

```text
To: rblsantos@inf.ufpr.br
Subject: Application to download the UFPR-ALPR dataset.

Name: <your first and last name>
Affiliation: <university where you work>
Department: <your department>
Current position: <your job title>
E-mail: <must be the e-mail at the above-mentioned institution>

I have read and agreed to follow the terms and conditions specified in the
UFPR-ALPR dataset webpage. This dataset will only be used for research purposes.
I will not make any part of this dataset available to a third party. I’ll not
sell any part of this dataset or make any profit from its use.

<your signature>
```

作者返回专属链接后，可直接批准下一轮的下载文本：

```text
我已本人阅读并接受 UFPR-ALPR 官方许可条款，并已从有效学校邮箱完成申请。
我授权 Codex 仅从作者回复中提供的专属链接下载 UFPR-ALPR v1.0 的一个原始 archive，
下载上限为 12,500,000,000 bytes，只能写入新的本地、Git 忽略目录
/Users/howen/Projects/PPO_MEC/data/raw/ai_task_acceptance/ufpr_alpr_v1_2018_<timestamp>/。
不得登录、使用我的凭证、接受新条款、覆盖旧目录、再分发、上传外部 AI 服务、裁剪/增强原图或提交原图/明文标签到 Git。
下载前先核对最终 URL、文件名和 Content-Length；若版本不是 v1.0、大小超过上限、链接要求新的条款/登录，立即停止。
下载后只执行 alpr_task_acceptance_data_plan_v1.json 已冻结的解包、hash、图像解码、标签绑定、按 vehicle/video 分组取样、
重复检查和 manifest/data card 生成；不得加载模型、generate、训练、运行 formal/holdout 或机制比较。
```

### 备选：只有 UFPR 无法取得时

```text
若 UFPR-ALPR 作者拒绝申请或明确无法提供 v1.0，我授权 Codex 仅下载 CCPD2019.tar.xz：
https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc
预期大小 13,164,924,944 bytes，预期 SHA-256
e4fd23c9f289c9e5404d9b25da6b2453714150c71b8804c7e96b9d94debcb44e。
只写入新的本地、Git 忽略目录
/Users/howen/Projects/PPO_MEC/data/raw/ai_task_acceptance/ccpd2019_<timestamp>/；不下载 CCPD2020 或任何模型。
不得登录、使用凭证、接受新条款、覆盖旧目录、公开原图/明文车牌或上传外部 AI 服务。
若文件名、大小或 SHA-256 不符立即停止。下载后只做冻结计划中的解包、标签解析、以 salted plate-text hash 作为
明确标注为 proxy 的 group、确定性取样和完整性检查；不得调用模型、训练、formal/holdout 或机制比较。
我理解该 proxy group 弱于 UFPR 官方 vehicle/video ID，结果不能声称物理车辆级独立。
```

## 未覆盖风险

- UFPR archive 的真实 URL、文件名、字节、hash 与目录 schema 只有作者批准后才能核验；当前 10.8 GB 只是规划估计。
- UFPR 的官方 split 虽按 150 vehicles/30 frames 设计，仍须在原件上确认 group 不跨 split。
- CCPD plate-text proxy 不是独立的 track/vehicle 标识；大包开销高，不能为了已有直链降低独立性标准。
- ALPR adapter 训练集未知；内部 group isolation 不等于对 adapter training 或 base pretraining 独立。
- 车牌属于潜在可识别信息；本任务不是隐私或法律合规意见，最终使用仍由研究负责人按机构要求审查。

本轮到此停止，不下载、不调用模型。下一轮只有取得上述精确授权与资源后，才创建真实数据包。
