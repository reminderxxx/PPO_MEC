# 公开任务小样本：M元数据门实现（2026-10-30）

用户明确允许下载并继续推进；绑定[原小样本方案](paper_public_task_sample_acquisition_proposal_20261029.md)，
私有proposal SHA `7cf898fcbec44f74e06554f4f4db8691eb99875df897d0bfb06e1611b0b38442`。
先实现M，检查EdgeVision V1与CCPD作者源的小文件资格；未有合格标签/文件清单时不直接取图。
M最多12次请求、5MiB body及中断读取上界、600秒；此花费扣除M+D总32请求/64MiB/1800秒，不能重置。
软件准备另限定5400秒、64MiB、最多20个合成实例，0真实请求。

`acquire_public_task_metadata.py`只支持一次初始化及逐次绑定意图SHA的元数据GET，不实现模型或图像下载。
初始化和每次请求都先核硬cap、固定proposal/source schema及grant；实际入口再核clean commit、source SHA与boot身份。
实际输入放在全新raw目录，原始body、headers、每次意图和收据create-only。
初始种子仅EdgeVision数据页、其已公开API文档页，以及GitHub官方CCPD commit元数据入口。
后续URL必须字面出现在已有成功body且具有父body SHA，资源类型、dataset/版本及endpoint形状也必须匹配固定规则；
不允许直接image/blob endpoint，不猜Mendeley接口、不自动爬取、不安装或执行来源代码，不自动重定向或重试。
GitHub身份冻结后可跟随已有JSON给出的tree URL；可变master不是最终样本身份。

每次GET均独立计数；只接受UTF-8文本/JSON/XML/JavaScript文档，不执行脚本。
非200也受MIME门约束，二进制错误body不读取，保留headers与保守上界；文字错误实体保留后停止。
read1中断记录额外上界，已返回body和磁盘实际保留body分账；未读完不冒充完整输入。
字节口径不是包含HTTP/TLS/header或未消费响应的精确wire总量。
600秒共用系统`CLOCK_MONOTONIC`，每进程按首请求时钟扣除，并绑定Darwin `kern.boottime`，拒绝时钟倒退/跨boot。
SIGALRM覆盖请求/读取总剩余期限；任何HTTP、格式或预算失败封存该run，不恢复、续写或修复重试。

`audit_public_task_metadata.py`是独立终端消费者，从body/intent/receipt核算hash、数量、来源、读取上界、
时钟、失败位置与预算；一次写出audit和terminal，输出存在则复用，不重跑审计。
它仅接受元数据账本，永远`input_qualified=false`，不据producer摘要认定标签、图像或质量。
总耗时含请求间核对和终端消费，存储含整个私有执行包、外部intent/control与raw，另外保守扣1MiB收尾reserve。
导出的D余量只是账面未花余额，不允许续写失败run或自动启动D。

首实现14项合成分支检查通过后，独立只读审查拦下5项执行前缺口：事后cap、GET前grant、二进制错误body、
源/资源类型身份以及时钟/预算收尾。旧SHA、14实例和报告保留；真实请求未启动。
另立核心收尾任务仅6新fixture、单轮1800秒/16MiB，不重跑旧14；6增量检查通过，累计20。
两批测试分别绑定各自源码，不能合称最终版本20项完整链路。最后consumer补齐失败body的MIME/encoding/UTF8门及
固定schema核对，另做AST/顺序检查和独立增量静态接受，新增fixture0；真实CLI初始化还须在clean source冻结后执行。

M有资格发现后，D仍需固定实际标签、文件ID/大小/URL和sample_manifest，另作对应生产/消费验收，继承M全部实际支出。
终端交付实际取得的文件/元数据和账单；只提供超限整包、无单文件标签或CCPD无牌语义仍缺时停止，
不把入口失败当数据集无效，不静默换来源或下载整个训练库。
原12图ALPR负结果、所有旧失败、main及被拒候选不改写；模型/训练/评价/拟合/hidden预算0，论文仍未完成。

初次真实CLI初始化通过，但新raw命名空间未在Git忽略中登记，随后GET被clean-source门在联网前拒绝。
原初始化目录/意图/命令和零请求终端审计保留，不在原目录恢复。独立诊断后仅补`/data/raw/task_quality/`忽略规则，
用`git check-ignore`、`git diff --check`及新clean commit核对，再在fresh目录初始化；HTTP/body实际支出仍为0，不是重取数据。

随后M实际取得7个完整官方元数据body，932037B、264.193802秒，单次账本审计通过但输入未资格。
CCPD demo仅0.jpg–4.jpg，未带牌照文字标签；EdgeVision文件列表由接口加载。
保存的官方OpenAPI明确GET `/datasets/{id}/files`与version query，网页SSR明确`publicApiBaseUrl=/public-api`。
另存独立路由诊断后只适配这一有公开依据的固定候选，匿名可用性尚未验证；不申请账号/token。
新fresh子目录继承原首请求的绝对CLOCK_MONOTONIC和boot，累计原7请求/932037B，保留600秒原deadline，
禁止递归parent或重取旧页面。仅4个内存/现存元数据解析检查和AST检查，新增fixture0；旧20不重跑。
如软件准备已耗尽原deadline即不请求，不能因另立目录暂停或重置计时。

## 本轮终端结果与独立后续范围（2026-10-30）

固定路由子任务在GET前被`metadata total deadline`拦下，新增HTTP为0；其fresh目录没有request目录，
已单次终端审计并封存。累计仍7请求、932037B返回body、中断读取上界0、图像0、模型调用0。
终端审计时距原首请求641.769310秒，明确为`FAILED_METADATA_DEADLINE_LEDGER_RETAINED`，
不能用此前264.193802秒的中间审计掩盖原600秒窗口后来耗尽的事实。
接口适配、来源链补齐及复核留在真实计时段，耗尽了窗口；这是执行安排问题，尚未测试匿名固定路由，
不能推断来源不可用、网络失败或只提供整包。CCPD编号demo也不能补作具备外部文字/无牌GT的ALPR6例。

主目录私有包`quality_public_sample_acquisition_20261030_v1/`保存`acquisition_delivery.json`、
`request_ledger.csv`、三份各单次终端audit及原件；旧目录、原审计及费用不改写。
七个用户修改文件SHA未变，论文稿位未改，`paper_goal_complete=false`。
14旧fixture及6增量fixture各自通过，不声称最终版本20项全链路；路由4纯解析检查及后续AST/静态接受单列。

### 待明确批准：只追加一次公开文件清单查询

原方案M明确规定“最多12个HTTP请求、5MiB响应体、10分钟”；本次已耗尽10分钟，追加时间须另立范围。
具体待答复方案：只查询`https://data.mendeley.com/public-api/datasets/j82bnw7gsr/files?version=1`一次，
新增最多1HTTP、1MiB返回body及中断上界、120秒（含producer/收据/独立consumer）。
原账保留，累计最多8HTTP/1980613B；全部新旧原件/日志/账单继续计入256MiB磁盘上限。
这120秒是明确追加的独立时间，不能把旧641.769310秒重新标为600秒内或重置旧M+D历时。
执行前完成离线软件资格、完整失败父链核对、clean源码与精确计划冻结，之后才启动新绝对deadline。
不重取旧页面、不重定向/重试/Range/换源；HTTP/auth/身份/版本/MIME/encoding/schema或预算失败保存停止。
如果返回只有大归档，报告小样本入口未闭合；此批不取图像、不自动开D或Q。
私有`single_route_extension_proposal.json` SHA为
`258cc36483cb7219073911e6ac6a0083dc961b942f3b7c42a77e817b00905818`；尚未执行，不从旧批准或时间经过推导追加许可。
