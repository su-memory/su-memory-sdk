"""
su-memory v3.5.0 — Retrieval Noise Generator
=============================================
按三种策略生成噪声记忆: 语义相似噪声、随机噪声、对抗噪声。

设计原则:
- 哈希确定性: 所有噪声基于 seed + content_hash，严格可复现
- 噪声隔离: 生成噪声不含原始因果信号
- 零依赖: 纯 Python stdlib
"""

import hashlib
import random

# ==========================================================================
# 语义同义词库 (15 组，覆盖经济/科技/政策/自然/健康领域)
# ==========================================================================

SEMANTIC_SYNONYMS: list[tuple[str, list[str]]] = [
    # ── 经济金融 ──
    ("物价", ["价格", "费率", "成本"]),
    ("上涨", ["上升", "增长", "攀升", "提高", "走高"]),
    ("下降", ["下滑", "走低", "回落", "降低", "下行"]),
    ("消费", ["支出", "购买", "花销", "购置"]),
    ("公司", ["企业", "机构", "单位", "组织"]),
    ("营收", ["收入", "营业额", "进账", "流水"]),
    ("市场", ["行情", "大盘", "市况", "交易"]),
    ("裁员", ["缩编", "人员优化", "人力调整", "岗位裁撤"]),
    # ── 科技 ──
    ("研发", ["科研", "技术开发", "产品研究", "创新"]),
    ("突破", ["进展", "创新", "飞跃", "跨越"]),
    ("数据", ["指标", "参数", "统计", "测量值"]),
    # ── 政策 ──
    ("政策", ["规定", "制度", "条例", "法规"]),
    ("利率", ["利息率", "贷款成本", "资金价格", "贴现率"]),
    # ── 自然/环境 ──
    ("气温", ["温度", "热度", "气象指标", "环境温度"]),
    ("极端", ["异常", "罕见", "非常规", "超常"]),
    # ── v3.5.0: 因果连接词 (噪声干扰关键词检测的核心目标) ──
    ("导致", ["造成", "引起", "引发", "致使"]),
    ("促使", ["驱动", "激励", "鞭策", "敦促"]),
    ("带来", ["产生", "催生", "引出", "衍生"]),
    ("推动", ["推进", "带动", "促进", "驱动"]),
    ("引发", ["触发", "诱发", "激起", "招致"]),
]

# ==========================================================================
# 随机中文 token 种子 (固定 24 个/类，用于构建无意义但语法通顺的句子)
# ==========================================================================

RANDOM_NOUNS: list[str] = [
    "报告", "系统", "项目", "会议", "方案", "流程", "标准", "资源",
    "部门", "平台", "模块", "接口", "协议", "框架", "策略", "机制",
    "配置", "版本", "需求", "文档", "组件", "引擎", "节点", "通道",
]

RANDOM_VERBS: list[str] = [
    "完成", "更新", "优化", "调整", "发布", "配置", "部署", "验证",
    "测试", "审核", "提交", "执行", "处理", "生成", "同步", "初始化",
]

RANDOM_ADJECTIVES: list[str] = [
    "新的", "主要", "重要", "关键", "基本", "核心", "标准", "通用",
    "高级", "基础", "完整", "高效", "稳定", "安全", "可靠", "灵活",
]

# ==========================================================================
# 对抗噪声模板 (6 种变换策略 —— 保留关键词但换新语境)
# ==========================================================================

ADVERSARIAL_STRATEGIES: list[str] = [
    "{keyword_a}和{keyword_b}的关系研究取得了{progress}",
    "关于{keyword_a}的最新分析显示{trend}趋势",
    "{keyword_b}领域专家认为{opinion}",
    "{keyword_a}的历史数据显示{data_point}",
    "本周{keyword_b}相关话题讨论热度{direction}",
    "{keyword_a}在不同地区的表现差异显著",
]

ADVERSARIAL_FILL: dict = {
    "progress": ["初步成果", "阶段性进展", "预期突破", "新发现"],
    "trend": ["持续向好", "波动加剧", "稳定增长", "缓慢回升"],
    "opinion": ["需要审慎评估", "前景乐观", "存在不确定性", "值得关注"],
    "data_point": ["呈现周期性规律", "显著偏离均值", "符合正态分布", "具有统计意义"],
    "direction": ["明显上升", "略微下降", "基本持平", "大幅波动"],
}


class NoiseGenerator:
    """
    检索噪声生成器 —— 哈希确定性 + 三级策略。

    用法:
        ng = NoiseGenerator(seed=42)
        # 0N: 不生成噪声
        assert ng.generate(["A", "B"], noise_level=0) == []
        # 1N: 每条记忆 1 条语义噪声
        noises = ng.generate(["物价上涨3.5%"], noise_level=1, noise_mode="semantic")
        assert len(noises) == 1
    """

    def __init__(self, seed: int = 42):
        self._master_seed = seed
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 哈希确定性随机
    # ------------------------------------------------------------------

    def _hash_deterministic(self, content: str, salt: str = "") -> random.Random:
        """
        基于 content + salt 的确定性随机生成器。

        同一 content + salt 永远产生相同的随机序列，
        不同 content 产生不同但可复现的序列。
        """
        h = hashlib.sha256(
            f"{self._master_seed}:{content}:{salt}".encode()
        ).digest()
        seed_int = int.from_bytes(h[:8], "big") % (2 ** 31)
        return random.Random(seed_int)

    # ------------------------------------------------------------------
    # 主入口: 按噪声等级生成
    # ------------------------------------------------------------------

    def generate(
        self,
        ground_truth: list[str],
        noise_level: int,
        noise_mode: str = "semantic",
    ) -> list[str]:
        """
        为 ground_truth 列表注入噪声。

        Args:
            ground_truth: 真实因果记忆列表 (如 [cause, effect])
            noise_level: 0N | 1N | 2N | 3N
            noise_mode: noise_level=3 时自动切换到 "adversarial"
                        其他等级默认 "semantic"

        Returns:
            噪声记忆列表 (不包含原始 ground_truth)

        噪声注入协议:
            0N → []
            1N → 每条 ground_truth 生成 1 条 semantic 噪声
            2N → 每条 ground_truth 生成 2 条 semantic 噪声
            3N → 每条 ground_truth 生成 2 条 semantic + 1 条 adversarial
        """
        if noise_level <= 0:
            return []

        noises: list[str] = []

        for original in ground_truth:
            if noise_level >= 1:
                noises.append(self._generate_semantic_noise(original))
            if noise_level >= 2:
                noises.append(self._generate_semantic_noise(original + "_v2"))
            if noise_level >= 3:
                noises.append(self._generate_adversarial_noise(original))

        return noises

    # ------------------------------------------------------------------
    # 策略 1: 语义噪声 (50-70% 同义词替换)
    # ------------------------------------------------------------------

    def _generate_semantic_noise(self, original: str) -> str:
        """
        对原文进行同义词替换。

        策略:
        - 扫描原文，对匹配到的词做替换
        - 替换率控制在 50-70% (至少保留 30% 原始字符)
        - 替换后的句子语义相似但因果信号被破坏

        示例:
            原文: "物价指数同比上涨百分之三点五"
            噪声: "费率指数同比上升百分之三点五"  ← "物价"→"费率", "上涨"→"上升"
        """
        rng = self._hash_deterministic(original, "semantic")
        result = original

        for keyword, synonyms in SEMANTIC_SYNONYMS:
            if keyword in result:
                replacement = rng.choice(synonyms)
                result = result.replace(keyword, replacement, 1)

        return result

    # ------------------------------------------------------------------
    # 策略 2: 随机噪声 (无因果信号的中文句子)
    # ------------------------------------------------------------------

    def _generate_random_noise(self, original: str = "") -> str:
        """
        生成完全随机的噪声句子。

        格式: [形容词] + [名词] + [动词] + [名词] + 状态描述
        确保不通往任何因果记忆。
        """
        rng = self._hash_deterministic(
            original or str(self._rng.random()), "random"
        )

        adj = rng.choice(RANDOM_ADJECTIVES)
        noun1 = rng.choice(RANDOM_NOUNS)
        verb = rng.choice(RANDOM_VERBS)
        noun2 = rng.choice(RANDOM_NOUNS)

        return f"{adj}{noun1}{verb}{noun2}第{rng.randint(1, 999)}号任务"

    # ------------------------------------------------------------------
    # 策略 3: 对抗噪声 (共享关键词但语境无关)
    # ------------------------------------------------------------------

    def _generate_adversarial_noise(self, original: str) -> str:
        """
        生成对抗噪声 —— 与原文共享关键词但语义无关。

        最危险的噪声类型: 与真记忆共享词汇，
        容易在向量空间中靠近真记忆但因果方向不同。

        策略:
        1. 从原文提取 2 个关键词
        2. 将其填入无关语境模板

        示例:
            原文: "物价指数同比上涨百分之三点五"
            噪声: "物价和消费的关系研究取得了初步成果"  ← 共享"物价"但无因果
        """
        rng = self._hash_deterministic(original, "adversarial")

        # 提取关键词: 优先从 SEMANTIC_SYNONYMS 中找匹配
        keywords: list[str] = []
        for keyword, _ in SEMANTIC_SYNONYMS:
            if keyword in original and len(keywords) < 2:
                keywords.append(keyword)

        # 如果不足 2 个，随机取字符片段
        while len(keywords) < 2 and len(original) > 4:
            start = rng.randint(0, len(original) - 3)
            fragment = original[start:start + 2]
            if fragment not in keywords:
                keywords.append(fragment)

        if len(keywords) < 1:
            keywords = ["系统", "数据"]

        # 填入对抗模板 (为所有可能的占位符提供随机值)
        template = rng.choice(ADVERSARIAL_STRATEGIES)
        fill_vals = {
            k: rng.choice(v) for k, v in ADVERSARIAL_FILL.items()
        }

        result = template.format(
            keyword_a=keywords[0] if len(keywords) > 0 else "系统",
            keyword_b=keywords[1] if len(keywords) > 1 else "数据",
            **fill_vals,
        )

        # 确保与原文不同
        if result == original:
            return self._generate_random_noise(original)

        return result

    # ------------------------------------------------------------------
    # 批量版本: 直接返回可插入的记忆字典列表
    # ------------------------------------------------------------------

    def generate_as_memories(
        self,
        ground_truth_ids: list[str],
        ground_truth_contents: list[str],
        noise_level: int,
    ) -> list[dict]:
        """
        批量生成噪声记忆字典，可直接插入记忆引擎。

        Args:
            ground_truth_ids: 真实记忆的 ID 列表
            ground_truth_contents: 真实记忆的内容列表
            noise_level: 噪声等级

        Returns:
            [{"id": "n_{gt_id}_{k}", "content": noise_text}, ...]
        """
        noise_memories: list[dict] = []
        for gt_id, gt_content in zip(ground_truth_ids, ground_truth_contents, strict=False):
            noises = self.generate([gt_content], noise_level=noise_level)
            for k, noise_text in enumerate(noises):
                noise_memories.append({
                    "id": f"n_{gt_id}_{k}",
                    "content": noise_text,
                })
        return noise_memories
