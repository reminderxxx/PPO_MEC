# TGSIM原生输入失败与行账本修正（2026-10-28）

## 原任务终端结论

fe01e53的附件入口恢复单次任务已终止。固定16份附件均HTTP200，独立只读形状检查
确认非空矩形CSV；第二轨迹全部21页、1024281个opaque身份已保留。首份4023022行
继续由旧包只读引用，不重下、解压或索引。保留页面不等于完整输入资格。

第二轨迹原生检查遇到空的`yloc_kf`后停止，后置两源metadata/count未执行。
本轮41次HTTP、334.058548292秒、41625914 encoded-body字节、119192176 decoded字节；
额外传输尝试0，训练/评价/拟合/hidden均0。历史HTTP400实体缺失仍使exact wire总量不可核验。

冻结执行器将19047次尝试误标为`rows_indexed`；数据库实际提交19046行。
原消费者单次审计因此拒绝，错误为`terminal native index row count differs`。
不能宣称本轮完整终端审计通过。该次失败审计、completion、原生空值及所有原件保持封存，
不修改、不重跑、不用新软件补写成功。私有独立诊断先于下面的独立软件实现保存。

## 独立软件实现合同及验收

范围只限失败账本与可观察性，算法、数据数值检查、环境、奖励、基线和取数策略不变。
该软件任务真实数据读取、网络、索引、训练、评价和旧审计重跑均0；只运行合成样例。

新增`native_row_accounting_version=attempts_and_insertions_v1`：

| 字段 | 含义 | 消费方式 |
|---|---|---|
| `rows_examined` | 已尝试的原生行，含被拒绝行 | 硬预算与完整祖先账单扣除此值 |
| `rows_indexed` | 成功INSERT行数 | 与只读SQLite实际行数核对 |
| `native_input_failure` | 数值错误的来源、列、物理行与有界原值 | 诊断信息，不触发修复或传输重试 |

生产端bulk/paged/recovery共同输出该版本；分页审计、完整源引用和祖先账单同步消费。
成功任务的尝试/插入必须相等；失败任务可保持有效前缀并完整收费。未知版本、无版本却带
新尝试字段、非法/倒置计数及越预算均拒绝。旧分页记录不重新解释，原差一行审计继续失败。
旧冻结plan仍绑定旧source SHA，不能拿本次新代码执行旧命令。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_tgsim_input_preparation.py tests/test_tgsim_paged_preparation.py tests/test_tgsim_query_diagnosis.py tests/test_tgsim_recovery.py tests/test_tgsim_bounded_transport.py tests/test_tgsim_geometry_recovery.py tests/test_tgsim_native_failure_accounting.py -q
```

101项合成检查通过，含空坐标/nonfinite、重复键、尝试上限、拒绝行继续收费、新失败可审计、
旧记录严格拒绝、移除版本无法隐藏花费，以及完整父链扣除拒绝尝试。未在真实输入上重跑新消费者。

## 下一任务草案与真实阻塞

当前输入范围以失败收尾，无活动取数或科学job。原合同要求数据错误停止，不能仅因剩余额度
自动重取、重索引或绕过空值。尚未核验缺失坐标在整个第二源的分布与影响，不能断言只有一个坏点。

下一项应先冻结独立的本地数据质量合同：只读现有第二源，记录全部数值缺失/非有限值及原生run归属，
保留所有行与缺失标记；不插值、不置零、不删除、不创建替代轨迹、不扫描奖励或选择科学窗口。
合同须声明本次重复输入暴露、读取行/字节/墙钟和存储上限；不隐式重置原native索引额度。
本文件只是下一任务草案，未启动全量质量扫描、post-version网络请求、provider适配或新科学实验。

即便未来输入质量合同完成，仍需坐标/原RSU/provider/物理成本资格、公平外部对照、独立formal/holdout、
统计与机制支持。此前主方法优势仅是已消费train的warm-start局部开发支持；全部负seed/time保留，
不晋级canonical或改论文主表。目标CSCWD2027，TMC规则为内部审查规范；`paper_goal_complete=false`。
