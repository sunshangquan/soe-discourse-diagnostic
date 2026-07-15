# 外部 validity 信号采集指南

**文件**：`data/external_signals_template.csv`

共 **204 行 = 51 家央企 × 4 年 (2018–2021)**。用 Excel/Numbers/Google Sheets 打开即可（已加 UTF-8 BOM），填完后**保留同样的列名**回存为 CSV 即可。

---

## 一、最关键的优先级

按"必填 / 想要 / 锦上添花"分三档。如果时间紧，**至少填完第一档就能挽救 Channel 3**。

### 🔴 第一档（必须，决定 Channel 3 能否成立）

| 列名 | 含义 | 用途 |
|---|---|---|
| `patent_count_applications` | 当年集团**合并口径专利申请数** | Innovation 维度的核心外部信号 |
| `revenue_bn_cny` | 当年营业收入（亿元 CNY） | 控制变量；patent / revenue 是研发产出率代理 |

仅这两列填完，就能跑出 `LLM Innovation 分数 vs patent_count` 的 Spearman 相关性，构成 Channel 3 主结果。

### 🟡 第二档（应填，能让 Channel 3 多 1-2 个维度）

| 列名 | 含义 |
|---|---|
| `rd_intensity_pct` | 研发投入占营收百分比，年报披露 |
| `rd_absolute_bn_cny` | 研发投入绝对数（亿元） |
| `esg_score_hzcert` | 华证 ESG 评级（AAA-D 或数值） |

### 🟢 第三档（锦上添花）

| 列名 | 含义 |
|---|---|
| `patent_count_granted` | 当年专利授权数（vs 申请数，授权更"硬"） |
| `esg_score_wind` | Wind/商道融绿 ESG 评级 |
| `employees_k` | 员工数（千人，做"人均"指标） |

---

## 二、每个信号去哪找

### 🔵 专利数（最容易拿到，公开免费）

**首选：incoPat（incopat.com）**
- 检索式示例：`SAA=(中国建材集团) OR SAA=(中国建材股份有限公司) AND PD=20180101 TO 20181231`
- SAA 是申请人字段；用 `OR` 拼集团母公司 + 主要上市子公司，避免漏算
- 不用注册账号也能跑基础检索，每天 200 条免费额度足够

**备选：国家知识产权局公开检索系统（cponline.cnipa.gov.cn）**
- 同样支持申请人+年份组合检索
- 免费但 UI 较慢

**备选：WIPO PATENTSCOPE**
- 国际优先级，可能漏算中国国内申请
- 适合做交叉验证

**填写示例**：
- `patent_count_applications = 12847`（建材集团 2018 年合并申请）
- 注意：要的是 **集团合并** 数字，不是某个上市子公司单独的；如果只能拿到子公司数，在 `source_note` 注明

### 🔵 研发投入 / 强度

**首选：上市子公司年报 MD&A 章节**
- 巨潮资讯（cninfo.com.cn）- A 股年报全集
- 大部分有控股 A/H 股的央企在年报"董事会报告"或"经营情况讨论与分析"披露 R&D 投入和占营收比
- 集团整体不公开披露的，用主要上市子公司加总作为代理（在 source_note 说明）

**备选：央企"双百行动"数据集**
- 国资委每年发布央企科技创新榜，含 R&D 投入

**填写示例**：
- 中石化 2018: `rd_intensity_pct = 0.40`，`rd_absolute_bn_cny = 105.7`，`source_url=cninfo.com.cn/.../600028 2018 年报 p.42`

### 🔵 ESG 评分

**首选：华证 ESG（学校 Wind 终端 → "ESG 评级"）**
- 学校图书馆 Wind 账号通常有
- 数值或 AAA/AA/A/BBB/... 字母评级，二选一填进 esg_score_hzcert
- 注意：华证 2018 年开始覆盖大部分央企上市公司；母集团本身可能没评级，用主要上市公司代理

**备选**：
- 商道融绿（syntao.com）
- 中证 ESG（chinabond.com.cn）
- MSCI ESG（仅在港股/海外上市的央企有）

**填写示例**：
- 中国建材 2020: `esg_score_hzcert = "A"`（华证）或转数值如 `4.0`（A=4, AA=5 这种线性映射）

### 🔵 营业收入

**最简单**：年报封面/主要财务摘要
- 央企整体收入：国资委每年公布的"中央企业 2018 年度财务决算情况"
- 备选：每个上市子公司加总

---

## 三、数据质量自查

填完一行（一家公司 × 一年）后：

1. **专利数合理性**：央企 patent_count_applications 一般在 100-30000 之间。50 以下太低（可能漏算控股子公司），50000 以上太高（重复计数）。
2. **R&D 强度合理性**：央企 1-5% 是常见范围；< 0.5% 异常低；> 10% 极少（中电科、航天等极个别军工集团才可能）。
3. **同公司跨年单调性**：专利数通常逐年增长；R&D 强度可上可下；ESG 评级慢变。年度跳变 > 5x 检查是否口径变化。
4. **缺失值用空字符串**：不要填 N/A 或 0；空格代表"未找到"，0 代表"确实为零"。
5. **source_url 必填一项以上信号**：方便后续核验。

---

## 四、覆盖率目标

最低可接受：
- `patent_count_applications`：51 公司 × 4 年 = 204 格，**填满 ≥ 60%**（约 120+ 格）
- `revenue_bn_cny`：填满 ≥ 70%（约 140+ 格）

更佳：
- 前两列 ≥ 85%
- `rd_intensity_pct` ≥ 50%
- `esg_score_hzcert` ≥ 40%

ESG 不全没关系：如果只覆盖大约一半央企，论文里我可以用 sub-sample 分析。

---

## 五、回收后我会做什么

你给我填好的 CSV，我接进 `phase1/src/evaluate.py` 的 Channel 3，跑：

1. 对每个维度，用所有方法（dict/lda/bge/llm/calibrated）的**公司-年 维度分数**对外部信号做 Spearman 相关 + 95% bootstrap CI
2. 报告：
   - LLM 的 `Innovation` 分 vs `patent_count_applications` Spearman → 期望显著正
   - LLM 的 `Innovation` 分 vs `rd_intensity_pct` → 期望显著正
   - LLM 的 `Social Responsibility` 分 vs `esg_score` → 期望显著正
   - 同维度，dict / lda / bge 三个 baseline 相关性显著弱
3. 这一节会成为论文 §6.X (Channel 3 external validity)，main paper 主表加 1 行
4. 论文 Limitations 中的"No external behavioural validity channel"段删除，升级 Discussion 段

---

## 六、特殊情况

**国务院国资委**：监管者，无营业收入/专利。这 4 行 (2018-2021) 全留空，在 source_note 写 "regulator, not a firm"，我会自动从 Channel 3 排除。

**国家开发投资 / 中国国新控股**：纯资本运营公司，无生产经营，专利数和 R&D 一般 < 50。可能要从 Innovation 维度的 Channel 3 排除，但 Social Responsibility 和 ESG 仍适用。

**未上市集团**（如中国中化、中粮、保利等部分时期）：找不到 ESG 评级正常，留空即可；专利和 R&D 一般在国资委或央企财务决算公告中能找到。
