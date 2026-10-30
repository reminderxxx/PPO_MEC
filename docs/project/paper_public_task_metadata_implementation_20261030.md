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

### 单次追加已明确批准后的执行合同（2026-10-30）

人工明确答复“允许追加这一次查询”，新grant独立绑定上述追加proposal SHA和本次精确plan SHA。
原M超时事实不改写，新时间不是旧M/D续算：一次固定URL、1MiB含不确定读取上界、120秒含收据及独立消费者，
累计cap精确固定8请求/1980613B，新旧全部存储仍256MiB。不下载图像或归档、不自动开D/Q。
`run_public_task_metadata_extension.py`只启动一次，自动初始化、查询和单次终端消费；任何错误都保存失败prefix。
新的raw为`data/raw/task_quality/public_task_metadata_extension_20261030_v1/`，旧v1/v2/v3均只读封存。

生产与消费分别核v2成功账本→v3零HTTP超时terminal→追加proposal/grant/新plan；
v3继承原7请求，不能与v2再加成14。原641.769310秒已包含旧264.193802秒，两者也不相加。
首轮新增8离线案例3通过/5失败：旧`kern.boottime`包含的微秒字符串发生变化，使旧live比较拦下全部正路径，
没有真实HTTP；此变化既不证明重启，也不证明未重启。原记录和字段全部保留。
新明确授权任务独立采用`kern.bootsessionuuid`（无fallback）和共享`clock_gettime(CLOCK_MONOTONIC)`；
plan/新claim绑定稳定UUID，GET及consumer实时核同UUID。历史boot仅在祖先之间核内部一致，不与新live身份比较。
新120秒从init claim开始，不能从GET重启，也不与旧clock相减或宣称跨boot连续历时。
请求阶段保留10秒给收据与consumer；`D_wall_remaining=null`，不以新clock重置旧M+D账面时间。

独立静态审查另拦下audit落盘前误报成功、坏claim使completion缺失两项；新有限修正仅3新fixture/300秒/8MiB，
实际3通过、首8全部deselect、旧20不重跑。新测试覆盖稳定UUID完整链、terminal写失败不得报audit成功、
部分claim损坏仍须保存未审计prefix。audit和terminal均create-only保存完成后才记成功；失败prefix按新1MiB保守上界，
不重跑consumer。软件原资格、首8与新3分版记录，不能合称最终版本31项全链路。
此前软件准备记录跨进程保存/比较`time.monotonic`值，不能据此认证跨进程时长；原字段保留并另作更正，
约0本身不否定同进程差值。新修正资格和真实任务均采用共享CLOCK_MONOTONIC，不借此暂停或重置真实120秒。

匹配离线资格、独立复核、clean commit、source/plan/grant冻结及一次push尝试完成之后才单次启动新任务。
只有JSON元数据容器返回成功仍不等于图像/标签/版本内容已认证；实际原件留给终端解读，来源字段缺失标UNVERIFIED。

### 单次追加终端结果：HTTP 200，API错误，未取得清单

clean `c810fbe1ed91f39fd33fc3309777e0875761c2a9`冻结三个脚本及精确plan/grant后已成功推送，
本次及此前排队代码均已同步。新入口只启动一次，HTTP返回200、JSON原文`{"error":400}`，13B，
不是文件清单；不能写成HTTP400，也不能推断错误具体参数、认证要求、来源不存在或只有大包。
独立账本audit自动单次完成：6.764155秒；含terminal6.785543秒，外部命令6.847217秒，均在新120秒内。
累计8请求/932050B、中断读取上界0；新图像0、模型/训练/评价0，input_qualified=false、paper_goal_complete=false。
旧641.769310秒FAIL原值保留，不以新成功传输/audit状态将其抹平；跨scope/boot连续总历时不作认证。

producer/audit_success及`AUDITED_METADATA_LEDGER_ONLY`只确认响应传输/JSON容器和账单完整，
不确认API文件清单、单文件可取、标签、图像或真实任务质量。终端独立解读明确为
`INPUT_UNQUALIFIED_API_ERROR_400_OVER_HTTP_200`。原响应/收据/terminal保持不变，不为改变判定改写原件。
私有`extension_result_review.json`、`extension_delivery.json`、`cumulative_request_ledger_after_extension.csv`
及`extension_delivery_verification.json`保存完整结果、全部负记录和7个主目录文件SHA保护核对。
已复用自动audit，不重跑；单次追加授权已消费，无活动job、没有新的循环问题或空闲提醒。
后续真实阻塞是可用且可追溯的发布方单文件/外部标签入口；须有入口证据或另立明确更正范围，
不能根据账面请求余量重试、改endpoint、探测、取大包、自动开D/Q或把图像获取说成完成。

## 最新“继续找办法”：独立、来源支持的参数排查

人工随后要求“你继续找办法”，另立同来源有界技术诊断，旧8请求和原失败目录不改写。
公开代码检索提示文件列表常用`folder_id=root`，此仅是线索，不能当发布方行为证明或直接猜参数下载。
先读取原官方HTML字面列出的`/datasets/bundle.js?8814bfd8bfcb6231e2b5`，仅作为文本，不执行JS。
若已保存发布方脚本片段证实`folder_id`/`root`/`/files`的实际调用约定，再查询同一dataset/version/list路径的
`?folder_id=root&version=1`；没有该证据即不发第二请求，任何错误保留停止。

`diagnose_public_task_entry.py`包含独立capture与单次终端audit操作，先3项离线检查通过，旧suite不重跑。
新最多2HTTP，bundle最多3MiB、列表最多128KiB（均含返回body及不确定读取上界），独立300秒含解读和audit。
原8/932050B只计一次，新满额最多10HTTP/4208850B，仍受累计M12HTTP/5MiB和新旧全部存储256MiB限额。
发布方URL、原HTML hash、原累计audit SHA、源码clean commit及计划先冻结，再启动统一clock和稳定UUID。
body/MIME/编码/HTTP/API/schema/时间/来源证据错误均停；无redirect/retry/Range、账号/token、镜像或模型调用。
本轮不取图像/归档/视频/模型、不自动开D/Q；网页文字检索的工具流量未计量，单列为UNAVAILABLE，
不把HTTP原件body上界冒充所有互联网检索精确wire总账。

新raw为`data/raw/task_quality/public_entry_diagnosis_20261030_v1/`，私有包为主目录
`paper_evidence_strengthening_20261029_v1/quality_public_entry_diagnosis_20261030_v1/`。
固定第0请求仅官方bundle，第1请求必须有第0原件SHA及文字offset/片段证明。终端只运行一次audit，
既有transport完成与真实文件/标签输入资格继续分开；paper_goal_complete=false。
独立审查拦下仅三个token共现不足以证明路由、consumer缺传输字段核验和缺执行中deadline三项。
修正后第1请求须先取得独立语义复核（同一public files调用及root文件夹关系），并在发请求前冻结proof SHA；
仅字段名出现不能授权请求。consumer独立核所有保留文本的MIME/encoding/UTF8，完整响应另核URL/长度，
自身设置300秒剩余时间硬alarm；失败保存未审计上界prefix，禁止重跑。
首3合成检查对应修正前版本；此增量新增fixture0，仅AST/范围检查和独立静态接受，不冒称最终3项全链路重测。
