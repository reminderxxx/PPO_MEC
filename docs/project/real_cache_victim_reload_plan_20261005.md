# 真实缓存 victim→reload 最小实测冻结方案

冻结时间：2026-10-05 21:30 +08:00。解析起点：
`54b5cfb754cf5c3b3191172704f6e74df3f90ec9`。本文件与
`configs/acceptance/real_cache_victim_reload_v1.json` 在读取本轮新模型输出前冻结。

## 成本对称性审计与历史缺陷

历史开发公式把 rerun 的未来准备从原 resident 集 `C` 投影，却只在 recovery 中显式写入当前依赖 `D0` 的准备和
`C→C'` 转移。生产实测 runner 也没有把 native resident ledger 与真实 PEFT 对象连接起来。因此，该公式可保留为
12 点 frozen development method，却不能直接作为一般 victim→reload 全路径公式，也不能据此默认当前模型准备是共同成本。
旧 artifact、动作和分数不改写；本轮独立记录为 frozen scientific defect，并改用逐事件、逐路径全成本账本。

两臂在同一 condition 中共享且只共享：processor、共享 base 与初始 resident adapter 的计时前 setup。该 setup 实际执行、
逐臂记录，但因对象、时点和调用完全相同才从 action comparison 中排除。action window 内的当前 adapter 准入、victim 卸载、
后续 reload、输入/状态、前缀、后缀与未来节点全部分别计入；本地权重读取与模拟网络传输不混称。

## 冻结实例

- base：既有 SmolVLM-500M，权重 `1,015,025,832 B`。
- 当前 `n0/n1`：既有 Helmet LoRA，权重 `9,641,944 B`。
- 后续 `n2`：既有 ALPR LoRA，权重 `154,423,432 B`。
- victim 条件容量：`1,169,449,264 B = base + ALPR`；初始 resident 为 base+ALPR。请求 Helmet 必须合法驱逐
  ALPR，后续 n2 再请求 ALPR 时必须驱逐 Helmet 并从保留的本地权重重新加载。
- 无驱逐对照容量：`1,179,091,208 B = base + ALPR + Helmet`；两 adapter 都在 action 前真实加载。
- 容量全部来自现有权重文件，不填充无用状态，不 sleep，不删除权重，不清 OS 文件缓存。

restart 与 recovery 使用同一个 native typed-cache transaction 和同一个显式 opt-in PEFT lifecycle bridge。restart 执行
`n0,n1,n2`；recovery 消费 source 保存的 `n0`，只执行 `n1,n2`。两臂的当前请求和未来请求完全相同，所以 victim/reload
外部代价若相同就必须保留在两边总成本中，并在差值中自然抵消，不能只向 recovery 收费。

## 调用、顺序与预测

每个 condition 固定：source 1 次、restart 3 次、recovery 2 次，共 6 次；两个 condition 总计 `12` 次 `generate`，等于
硬上限。顺序固定为 victim `recovery→restart`，control `restart→recovery`。无自动重试、无结果后追加实例。

事前估计沿用上一独立检查的 prefix/state/suffix 值；n2 以同一 n1 估计代替。victim 条件另向两臂同时加入旧 ALPR load
`0.817814 s` 与按真实权重字节比例得到的 Helmet load `0.051082 s`。预测为：victim restart/recovery=
`19.553682/8.509755 s`，control=`18.684786/7.640859 s`，两条件均选 recovery。没有事前预测决策翻转；未跨边界不是补跑理由。

## 证据与停止边界

逐事件必须同时记录 logical resident、PEFT 注册 adapter、adapter tensor count/bytes、加载/卸载调用、磁盘文件 hash、
计时和 node calls。RSS 只可作旁证，本轮不以 RSS 证明卸载。PEFT 若不能真实解除 adapter 对象，或 native preview 出现
base victim，本轮立即失败，不改账本冒充成功。真实网络、无线、queue 和任务标签均不可用；网络只保留 100 Mbps + 20 ms
模拟项。该单技术实例不支持交通任务正确性、独立泛化、正确两步前瞻优势或 paper-ready 判断。
