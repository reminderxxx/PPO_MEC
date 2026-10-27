# TGSIM输入准备方案（2026-10-27）

状态：`EXACT_SCOPE_GRANTED_SOFTWARE_QUALIFIED_PENDING_SINGLE_ACQUISITION`。
2026-10-27用户对已展示方案明确回复“允许按该范围准备数据”。原私有方案SHA不改写，
答复单独绑定为grant；范围和硬上限保持如下。执行器17项纯软件检查通过，
实际获取状态以新建执行声明/收据为准。这仍不是训练、评价、formal或holdout合同。

## 具体来源与有限范围

| 数据源 | 官方资产ID | 发布方约计大小 | 作用 |
|---|---|---:|---|
| [TGSIM I-90/I-94 Stationary](https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-i-90-i-94-stationary-trajectories) | `9uas-hf8b` | 304 MB | 固定视野高速轨迹的输入资格候选 |
| [TGSIM I-294 L1](https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-i-294-l1-trajectories) | `7zjf-a4zf` | 70 MB | 较长路段的移动航拍轨迹资格候选 |

两者均由FHWA发布。目录说明采样为0.1秒，坐标/速度为米和米每秒，
分别有6和10个run；API字段将`time`定义为每次run开始后的相对秒数，
`run_index`标识数据采集过程。不能把不同run的相同时间或车辆ID合并。
目录声明public访问；所取API没有单列licenseId，不把public字样扩写为任意再发布许可。
本任务不公开原始轨迹。只核对字段和目录，不据此宣布数据质量或独立性已通过。

仅请求上述两份CSV及目录列出的16份run对应centerline geometry CSV。
不取视频、图片、压缩包、不申请highD账号、不联系数据提供者、不自动换源或扩展范围。
实际下载版本、字段、附件ID必须先与准备清单匹配；改变时停止并保存新状态。

拟议硬上限：总下载512 MiB、新增磁盘2 GiB、总墙钟45分钟、索引最多600万轨迹行。
每份原件只请求一次，失败保存现场，不自动重试或替换另一个数据集。
训练/策略评价/模型调用/hidden开启预算均为0。

通过工作树路径新建：`data/raw/mobility/tgsim/tgsim_input_qualification_20261027_v1/`；
派生检查留在忽略的`artifacts/analysis/paper_tgsim_input_qualification_20261027_v1/`。
目录create-only；路径已存在即拒绝。现有NGSIM、Alibaba、LuST、checkpoint和论文不覆盖。

## 取得数据后只做的检查

1. 冻结source/version、输入用途和有限预算，再请求原件。记录最终URL、响应头、
   SHA、字节、下载前后目录版本和完整失败收据。
2. 校验`id/time/xloc_kf/yloc_kf/speed_kf/run_index`、有限数值及
   `(asset_id, run_index, id, time)`唯一性；按真实时间检查0.1秒网格和缺帧。
   不截断小数秒、不跨run拼接、不平均重复项、不插值或补造车辆/轨迹。
3. 只依据输入记录八车持续覆盖480个真实采样点的可用性、时间间隔、原RSU覆盖
   与坐标/单位边界。当前NGSIM provider保留原始Local_X/Local_Y数值；其官方字典为feet，
   TGSIM为meters，必须先核对单位，不能直接把两种数值坐标当成同一几何。
4. 不为提高覆盖或奖励改变RSU、缩放场景、拉伸时间或调整物理费用。
   若原固定几何不适用，报告失败及所需的新站点几何合同，不现场改变环境。
5. 按采集/run/真实时间记录依赖与输入暴露。16个run标签不等于16次独立采集，
   也不能保证12 formal加12 hidden外层单元。把所有实际扫描用途写进ledger；
   输入准备本身不赋予hidden身份，不释放旧未知隔离。

输出是可用/不适用证据、输入hash与缺口。数据不合格也完整保留，不根据策略奖励
更换窗口。新provider的生产/消费合同、盲法划分与科学计算授权仍需独立任务。
NGSIM + Alibaba保留为原主线；TGSIM目前只作为拟议的实测移动性补充，
不是joint VEC请求、真实RSU成本或任务正确性数据。

## 已排除的快捷替代与验证

现有LuST小文件不足以直接补齐拟议时间单元。纯软件两时刻fixture进一步核实：
当前`export_lust_fcd.py`将0.0与0.1秒都写成整数time_index0，provider也按整数分组。
因此改`--step_length 0.1`不能直接得到正确的细时间网格；未运行SUMO、未修复源码，
也不把旧1秒导出重新判为无效。该问题已独立记录，后续修复须另立实现任务。

[highD](https://levelxdata.com/highd-dataset/)需要人工申请，且本地provider仍需核验采样/坐标；
[pNEUMA官方下载页](https://open-traffic.epfl.ch/index.php/downloads/)同时显示BY-NC与BY许可文字，
并说明23.97/25 FPS、丢失区域与不同CSV组织，暂不作为即插即用来源。
这些取舍来自输入合同和取得条件，没有读取任何策略收益。

公开数据/方法参考已追加`literature_reference_table.md`。完整source快照、软件fixture和
验证收据保存在本地私有输入准备包，不上传。论文目标仍未完成。
