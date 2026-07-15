"""5 entrepreneurial-spirit dimensions: keys, seed words, NL descriptions."""

DIMENSIONS = [
    {
        "key": "innovation",
        "cn_name": "技术创新与商业引领",
        "seed_words": ["创新", "发展", "引领", "改革", "科技",
                       "研发", "专利", "技术", "数字化", "新产品"],
        "nl_description": (
            "技术创新与商业引领：体现在具体的研发投入、专利成果、关键核心技术突破、"
            "新产品/新业态/新模式落地、数字化转型的可观测动作。不包含抽象的'坚持创新驱动'口号。"
        ),
    },
    {
        "key": "competition_cooperation",
        "cn_name": "商业竞争与合作共赢",
        "seed_words": ["竞争", "战略", "开放", "合作", "共赢",
                       "并购", "重组", "走出去", "一带一路", "海外"],
        "nl_description": (
            "商业竞争与合作共赢：体现在具体的市场竞争行为、合资合作项目、并购重组事件、"
            "国际化扩张的可量化动作（项目数/合同额/合作伙伴）。不包含抽象的'合作共赢'口号。"
        ),
    },
    {
        "key": "organization_market",
        "cn_name": "组织激励与商业盈利",
        "seed_words": ["效率", "质量", "一流", "激励", "市场化",
                       "混改", "股权", "上市", "治理", "经营机制"],
        "nl_description": (
            "组织激励与商业盈利：体现在具体的内部市场化改革、混合所有制改革、股权激励、"
            "公司治理结构变更、上市/重组、营收/利润等可量化经营成果。不包含抽象的'做强做优做大'口号。"
        ),
    },
    {
        "key": "social_responsibility",
        "cn_name": "社会责任与可持续发展",
        "seed_words": ["责任", "环保", "可持续", "社会", "生态",
                       "绿色", "减排", "扶贫", "ESG", "安全"],
        "nl_description": (
            "社会责任与可持续发展：体现在具体的环保投入、减排数字、安全生产、扶贫项目、"
            "ESG 披露的可观测动作（项目名/资金/受益人数）。不包含抽象的'绿水青山'口号。"
        ),
    },
    {
        "key": "national_mission",
        "cn_name": "身份认同与使命担当",
        "seed_words": ["国家", "民族", "使命", "建设", "复兴",
                       "战略性", "国之重器", "关键核心", "卡脖子", "强国"],
        "nl_description": (
            "身份认同与使命担当：体现在具体的国家重大工程承接、关键技术攻关、产业报国项目"
            "的可观测落地（工程名/技术指标/承接份额）。不包含抽象的'使命担当''中华民族伟大复兴'口号。"
        ),
    },
]

DIM_KEYS = [d["key"] for d in DIMENSIONS]
DIM_BY_KEY = {d["key"]: d for d in DIMENSIONS}
SEED_WORDS_FLAT = {d["key"]: d["seed_words"] for d in DIMENSIONS}

# L1 stance labels (3-way)
L1_LABELS = ["slogan", "substantive", "irrelevant"]

# L2 sub-labels for substantive (per Phase 0 finding)
L2_SUBSTANTIVE_TYPES = ["firm_action", "policy_history", "system_aggregate"]
