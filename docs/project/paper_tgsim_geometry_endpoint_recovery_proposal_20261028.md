# TGSIM附件入口修正与剩余输入提案（2026-10-28）

状态：`PROPOSED_AWAITING_EXPLICIT_ENDPOINT_CHANGE_AND_COMPLETE_SOURCE_REUSE_ANSWER`。
未执行新数据请求，未实现下一执行器。原有限重试run已终止，不能在原run修复或续写。
私有方案SHA：`9d06abc89b7ef016c8b83df4b07b179c123df3ddb019e3c9185535ef1fe0f2f4`。

## 本轮已经得到的输入与失败

260daca单次恢复取回第一份轨迹全部81页、4023022行；其中20页为本地复用，61页新获取。
原生0.1秒网格、数值与run/vehicle/tick唯一检查通过；独立审计只读核对两个SQLite均4023022行。
六个run中的第五个没有满足8车连续480tick的候选；其余run合计27个带480空闲tick间隔的
时间装箱上界只表示输入容量，不能称27个独立采集或formal/hidden样本。原有缺帧全部保留。

随后第一个几何附件`57eed127-2d00-43c5-8c65-e76ee67552a3`的冻结
`/api/assets/<assetId>`地址返回HTTP404，72字节not_found实体/响应头已保留。
两源前置metadata/count通过，但第二份轨迹、16份几何、两源后置版本/count均未完成。
本轮66次HTTP、768.440340秒、102406316 encoded-body字节、492684515 decoded字节，
额外传输重试0次。单次终端收据/hash/账单审计通过，整体输入任务仍失败；原件全部封存。

## 具体软件差异与尚未验证的部分

保留metadata中的16条几何附件使用`assetId`，没有`blobId`。当前执行器直接构造
`/api/assets/<assetId>`；[sodapy下载实现](https://github.com/afeld/sodapy/blob/main/sodapy/socrata.py)
对`assetId`使用dataset-scoped的`views/files`地址，对`blobId`才使用`api/assets`。
[路径拼接函数](https://github.com/afeld/sodapy/blob/main/sodapy/utils.py)与
[路径常量](https://github.com/afeld/sodapy/blob/main/sodapy/constants.py)共同给出前缀`/api/views`。
据此提出同一官方站点的固定候选模板：

```text
https://data.transportation.gov/api/views/<dataset-id>/files/<attachment-assetId>?download=true&filename=<encoded-original-filename>
```

这是一手客户端实现支持的兼容修正线索，不是已确认的该站点可用性或404唯一根因。
没有真实请求新模板，也没有证明原附件被删除、需要认证或新模板必然成功。
只请求原2个dataset/16个attachment ID，原filename、RSU/坐标/费用与数据源不变。

## 需要明确允许的改变

原[有限重试合同](paper_tgsim_bounded_transport_retry_proposal_20261028.md)规定
“不换源/endpoint/查询/页大小”，且HTTP错误停止。新范围需明确允许：

1. 在全新run中把几何附件地址改为上述固定dataset-scoped候选；不改轨迹API/查询/页大小。
   每个附件仅一次请求；任何HTTP、认证/限流、重定向、schema/完整性或预算错误仍停止。
   不再切换其他地址或来源，不申请账号/token，不通过探测/重试扩预算。
2. 以只读manifest/hash/qualification/索引引用复用已完成的第一份轨迹；不重下、重组、解压
   或重新建立其4023022行原生索引，不复制整个旧raw包，不把旧失败几何实体作为有效几何。
   新消费者须验证来源、所有原件hash、既有两个索引和新前后版本/count绑定；旧run永不改写。

## 剩余资源，不重置预算

- 新HTTP总上限48；两源前后metadata/count共8、第二轨迹21页、原16附件共16，预期45次。
  仅轨迹超时/IncompleteRead/明确断连每页总3次，剩余全局最多3次额外尝试；几何不重试。
- 新encoded body及读取不确定上界最多398676278字节；三轮已用138194634，合计不超过536870912。
- 新执行器最多1523秒，保守扣除前轮整数时限；含等待/诊断的整个历时不称45分钟以内。
- 新decoded最多551200289字节，第一份不再解压/重组；所有旧重复读取继续保留为已消费暴露。
- 新原生索引行最多1976978，实际预期第二份1024281行；加已索引4023022仍不超过600万。
- 累计全部旧新原件、尝试、副本、派生CSV、索引与收据仍最多2GiB；不能删除旧文件回收额度。
- 每份几何响应最多8MiB。所有失败/等待/不确定读取计费；历史缺失HTTP400实体仍使exact wire总量不可核验。

## 获授权后的先行软件资格

当前执行器只允许pre-native-index失败父链，不能直接拿本轮父completion继续。
须先另立实现和合成验收：完整源只读provenance消费、旧4023022行不重新计新索引、三祖先累计扣费、
更严格整数剩余时限、固定16个新附件URL映射、attempt/复用/新索引分账，以及独立审计拒绝篡改和漏账。
仅软件验收通过并冻结clean commit/新plan后，单次启动；真实软件资格不靠重复请求旧数据取得。

本轮仍是输入准备；训练/评价/拟合/hidden、算法/环境/奖励/预测器/规则/基线/RSU/坐标/provider/费用
修改预算均0。成功输入也不能替代独立formal/holdout、公平对照、真实任务质量和物理成本证据。
论文目标保持未完成；不改canonical或老师固定稿位。
