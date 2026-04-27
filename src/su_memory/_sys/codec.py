"""
Semantic encoding system for memory compression

Core concept: Use semantic categories (creative/receptive/light/abyss/thunder/wind/mountain/lake)
for semantic classification, rather than dictionary matching.
Each memory segment receives a semantic category based on its essence, then undergoes high-density encoding.

Semantic mapping:
  creative: strength/positive/rising/active → high-value/important/active memories
  receptive: flexible/bearing/stable → stable/durable/foundation memories
  light: bright/divergent/connected → knowledge/facts/connection networks
  abyss: trapped/dangerous/hidden → problems/risks/hidden items
  thunder: vibration/change/trigger → triggers/events/changes
  wind: entering/permeating/spreading → permeation/spread/influence
  mountain: stopping/blocking/waiting → obstacles/pauses/waiting
  lake: joy/exchange/agreement → agreements/exchange/positive feedback
"""

from typing import Dict, List
import re
from datetime import datetime


# ============================================================
# Semantic pattern library - classification based on imagery and semantics
# ============================================================

SEMANTIC_PATTERNS = {
    # creative - strength, positivity, rising, dominance
    # matches: value growth, positive action, leadership decisions, goal-oriented
    "creative": [
        r'(ROI|收益率|增长|增值|上升|提高|突破|领先|主导|控制)',
        r'(投资|回报|利润|收益|目标|战略|决策|领导|核心|关键)',
        r'(成功|成就|突破|进展|进步|优化|提升|增强|卓越)',
        r'(主动|开创|引领|驱动|推动|促进)',
        r'(重要|首要|优先|紧急|核心|主要)',
    ],

    # receptive - flexible, bearing, stable, foundation
    # matches: stable operation, infrastructure, continuous maintenance, load capacity
    "receptive": [
        r'(稳定|持续|维持|保持|常规|日常|基础|根本|根基)',
        r'(承载|支撑|保障|维护|运营|管理|流程|体系|机制)',
        r'(传统|经典|标准|规范|制度|规则|框架)',
        r'(积累|沉淀|储备|存量|底蕴|根基)',
        r'(服从|配合|协作|支持|辅助|配合)',
    ],

    # light - bright, divergent, connected, wisdom
    # matches: knowledge dissemination, information flow, network connection, understanding
    "light": [
        r'(网络|连接|关联|关系|互联|互通|通信|交互)',
        r'(知识|信息|数据|文档|资料|报告|分析|研究)',
        r'(理解|认知|洞察|判断|识别|分类|定义)',
        r'(传播|分享|传递|交流|讨论|协作)',
        r'(明亮|清晰|透明|公开|可见|曝光)',
        r'(学习|掌握|熟练|精通|专业)',
    ],

    # abyss - trapped, dangerous, hidden, difficult
    # matches: risk problems, crisis hidden dangers, complex dilemmas, unknown threats
    "abyss": [
        r'(风险|危险|威胁|隐患|问题|困难|挑战|危机)',
        r'(不确定|未知|隐匿|潜在|暗中|隐藏|秘密)',
        r'(失败|错误|缺陷|漏洞|故障|崩溃|失效)',
        r'(损失|亏损|负债|压力|紧张|焦虑)',
        r'(复杂|纠缠|陷阱|困境|难处理|难解决)',
        r'(下跌|下降|减少|衰退|恶化|退化)',
    ],

    # thunder - vibration, change, trigger, event
    # matches: sudden events, change dynamics, trigger mechanisms, activation
    "thunder": [
        r'(事件|发生|触发|激活|启动|启动|点燃)',
        r'(变化|改变|转变|转型|演进|进化|迭代)',
        r'(突发|紧急|突然|即时|立即|马上)',
        r'(震动|震荡|波动|起伏|动荡|不稳定)',
        r'(更新|升级|刷新|重置|重启)',
        r'(日出|春天|开始|起步|首发)',
    ],

    # wind - entering, permeating, spreading, influence
    # matches: penetration and expansion, influence propagation, spread, deep involvement
    "wind": [
        r'(渗透|扩散|传播|蔓延|扩展|蔓延|侵入)',
        r'(影响|作用|效果|效力|触动|感染)',
        r'(深入|进入|介入|参与|加入|融入)',
        r'(推广|普及|覆盖|遍布|充斥)',
        r'(风|流动|灵活|适应|变通|顺从)',
        r'(市场|渠道|网络|分支|分布)',
    ],

    # mountain - stopping, blocking, waiting, caution
    # matches: stagnation, obstacles, waiting and observing, conservative caution
    "mountain": [
        r'(停止|暂停|中断|中止|截断|阻碍|阻塞)',
        r'(等待|观望|保守|谨慎|稳健|稳妥)',
        r'(障碍|阻碍|瓶颈|卡点|难点|堵点)',
        r'(限制|约束|规范|规矩|边界|范围)',
        r'(阻止|禁止|不准|不能|不可)',
        r'(静止|不动|稳定|守候|坚持|持续)',
        r'(山|稳重|厚重|踏实|实在)',
    ],

    # lake - joy, exchange, agreement, positive
    # matches: positive feedback, agreement reached, satisfaction, exchange and cooperation
    "lake": [
        r'(满意|喜悦|高兴|快乐|愉快|满足|幸福)',
        r'(协议|合同|约定|承诺|共识|一致|同意)',
        r'(交换|交易|贸易|合作|共赢|互利)',
        r'(正面|积极|乐观|希望|信心|鼓励)',
        r'(评价|反馈|回复|响应|反应)',
        r'(达成|完成|实现|达到|获得|收获)',
        r'(协议|谈判|协商|讨论|协调)',
    ],
}


# ============================================================
# Semantic category to energy type mapping
# ============================================================

SEMANTIC_ENERGY_MAP = {
    "creative": "metal",
    "receptive": "earth",
    "light": "fire",
    "abyss": "water",
    "thunder": "wood",
    "wind": "wood",
    "mountain": "earth",
    "lake": "metal",
}


# ============================================================
# Energy cycle relationships (for energy computation)
# ============================================================

ENERGY_CYCLE = {
    ("wood", "fire"): 1.2,  # wood enhance fire - energy boost
    ("fire", "earth"): 1.2,  # fire enhance earth
    ("earth", "metal"): 1.2,  # earth enhance metal
    ("metal", "water"): 1.2,  # metal enhance water
    ("water", "wood"): 1.2,  # water enhance wood
    ("wood", "earth"): 0.7,  # wood suppress earth - energy reduction
    ("earth", "water"): 0.7,  # earth suppress water
    ("water", "fire"): 0.7,  # water suppress fire
    ("fire", "metal"): 0.7,  # fire suppress metal
    ("metal", "wood"): 0.7,  # metal suppress wood
}

ENERGY_ORDER = ["wood", "fire", "earth", "metal", "water"]


# ============================================================
# Time stem and branch encoding
# ============================================================

STEMS = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
BRANCHES = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]


def get_current_time_code() -> str:
    """Get current time code (simplified annual version)"""
    year = datetime.now().year
    stem_idx = (year - 4) % 10
    branch_idx = (year - 4) % 12
    return STEMS[stem_idx] + BRANCHES[branch_idx]


def extract_time_markers(text: str) -> List[str]:
    """Extract time-related markers from text"""
    results = []
    date_patterns = [
        r'(\d{4})年(\d{1,2})月',
        r'(\d{1,2})/(\d{1,2})',
        r'(\d+日|\d+号)',
        r'(昨天|今天|明天|上周|本周|下周)',
        r'(上午|下午|晚上|早晨|凌晨)',
    ]
    for p in date_patterns:
        if re.search(p, text):
            results.append(f"[{STEMS[hash(text[:5]) % 10]}{BRANCHES[len(text) % 12]}]")
    return results


# ============================================================
# Emotional intensity vocabulary (for energy computation)
# ============================================================

INTENSITY_WORDS = {
    "high_positive": ["非常", "极其", "十分", "显著", "大幅", "剧烈", "突破", "爆发"],
    "medium_positive": ["较好", "明显", "较大", "较好", "持续", "稳步", "较好"],
    "low_positive": ["略", "稍微", "稍有", "轻微", "一点点"],
    "high_negative": ["严重", "危险", "危机", "崩溃", "完全", "彻底", "大量"],
    "medium_negative": ["较大", "明显", "困难", "挑战", "压力"],
    "low_negative": ["轻微", "略", "稍有", "一点点"],
}


# ============================================================
# Causal link detection
# ============================================================

def count_causal_links(text: str) -> int:
    """Count causal link nodes in text"""
    link_patterns = [
        r'因为|由于|所以|因此|导致',
        r'和|与|以及|还有',
        r'如果|则|那么',
        r'既...又...',
        r'既...又...',
        r'一方面...另一方面...',
        r'首先|其次|然后|最后',
        r'包括|包括但不限于',
    ]
    count = 0
    for p in link_patterns:
        count += len(re.findall(p, text))
    return min(count, 9)  # maximum 9


# ============================================================
# Core compression engine
# ============================================================

class SuCompressor:
    """
    Semantic encoding system for memory compression

    Uses semantic categories for classification, not dictionary matching.
    Each memory segment receives a semantic category based on its essence.
    """

    def __init__(self):
        self.mode = "semantic"
        self._compile_patterns()
        self._time_code = get_current_time_code()

    def _compile_patterns(self):
        """Pre-compile all semantic patterns"""
        self._semantic_patterns = {}
        for category, pattern_list in SEMANTIC_PATTERNS.items():
            compiled = [re.compile(p, re.I) for p in pattern_list]
            self._semantic_patterns[category] = compiled

    def compress(self, text: str, mode: str = None) -> Dict:
        """
        Compression entry point (backward compatible)

        Returns: {
            "compressed": str,      # compressed text
            "method": str,          # method used
            "original_size": int,   # original byte size
            "compressed_size": int, # compressed byte size
            "ratio": float,         # compression ratio
            "category": str,        # semantic category (new)
            "energy_type": str,      # energy type (new)
            "energy": float,        # energy level (new)
        }
        """
        mode = mode or self.mode
        orig_sz = len(text.encode("utf-8"))

        if mode == "lossless":
            res = self._lossless(text)
        elif mode == "semantic":
            res = self._semantic(text)
        else:
            res = self._balanced(text)

        res["original_size"] = orig_sz
        res["compressed_size"] = len(res["compressed"].encode("utf-8"))
        # ratio based on BYTES (for compatibility)
        res["ratio"] = round(orig_sz / max(res["compressed_size"], 1), 2)
        # char_ratio based on CHARACTERS (semantically meaningful)
        orig_chars = len(text)
        comp_chars = len(res["compressed"])
        res["char_ratio"] = round(orig_chars / max(comp_chars, 1), 2)
        res["mode"] = mode

        return res

    def _lossless(self, text: str) -> Dict:
        """Lossless compression (using zlib+base64)"""
        import zlib
        import base64
        enc = base64.b64encode(zlib.compress(text.encode(), level=9)).decode()
        return {"compressed": enc, "method": "zlib+base64"}

    def _semantic(self, text: str) -> Dict:
        """Semantic compression"""
        # Short text kept as-is (with semantic tags)
        if len(text) < 15:
            return {
                "compressed": text,
                "method": "direct",
                "category": self._classify_by_semantic(text),
                "energy_type": SEMANTIC_ENERGY_MAP.get(self._classify_by_semantic(text), "earth"),
                "energy": 0.5,
            }

        # Step 1: Semantic classification
        category = self._classify_by_semantic(text)
        energy_type = SEMANTIC_ENERGY_MAP.get(category, "earth")

        # Step 2: Compute energy level
        energy = self._compute_energy(energy_type, text)

        # Step 3: Count causal link nodes
        links = count_causal_links(text)

        # Step 4: Extract time markers
        time_markers = extract_time_markers(text)
        time_code_str = time_markers[0] if time_markers else f"[{self._time_code}]"

        # Step 5: Generate compressed content (summary for long text, compress for short)
        if len(text) > 80:
            summary = self._generate_summary(text, category, max_len=40)
        else:
            summary = self._compress_text(text, category)

        # Step 6: Build high-density template
        output = self._build_template(category, energy_type, energy, links, time_code_str, summary)

        return {
            "compressed": output,
            "method": "semantic-compression",
            "category": category,
            "energy_type": energy_type,
            "energy": energy,
        }

    def _balanced(self, text: str) -> Dict:
        """Balanced mode"""
        if len(text) < 20:
            return self._semantic(text)

        res = self._semantic(text)
        orig_len = len(text)
        comp_len = len(res["compressed"])

        if orig_len / max(comp_len, 1) < 1.5:
            # Further compress summary
            summary = self._generate_summary(text, res["category"], max_len=20)
            output = self._build_template(
                res["category"], res["energy_type"], res["energy"],
                count_causal_links(text), f"[{self._time_code}]", summary
            )
            res["compressed"] = output

        return res

    def _classify_by_semantic(self, text: str) -> str:
        """
        Classify text by semantic content

        Algorithm:
        1. Calculate match score for each category
        2. Return highest scoring category
        3. When multiple high scores, use ambiguity resolution
        """
        scores = {}

        for category, patterns in self._semantic_patterns.items():
            score = 0
            matched_patterns = 0
            for pattern in patterns:
                if pattern.search(text):
                    score += 1
                    matched_patterns += 1
            if matched_patterns > 0:
                scores[category] = score

        if not scores:
            # No match - use fallback inference
            return self._infer_semantic_fallback(text)

        # Get highest score
        best_category = max(scores, key=scores.get)
        scores[best_category]

        # Ambiguity detection: check for close scores
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            if sorted_scores[0] - sorted_scores[1] <= 1:
                # Scores are close, use ambiguity resolution
                return self._resolve_ambiguity(text, scores)

        return best_category

    def _infer_semantic_fallback(self, text: str) -> str:
        """Fallback inference when no patterns match"""
        # Infer based on basic text features
        if any(c in text for c in "成功完成实现达成"):
            return "lake"  # joy/achievement
        elif any(c in text for c in "问题困难挑战风险"):
            return "abyss"  # danger/problems
        elif any(c in text for c in "变化更新改变转型"):
            return "thunder"  # change/trigger
        elif any(c in text for c in "稳定持续保持维持"):
            return "receptive"  # stable/continuous
        elif any(c in text for c in "重要关键核心首要"):
            return "creative"  # strength/importance
        elif any(c in text for c in "知识信息数据网络"):
            return "light"  # brightness/knowledge
        elif any(c in text for c in "停止等待阻碍限制"):
            return "mountain"  # stopping/blocking
        elif any(c in text for c in "影响传播扩散深入"):
            return "wind"  # permeation/influence
        else:
            return "receptive"  # default: stable/bearing

    def _resolve_ambiguity(self, text: str, scores: Dict[str, int]) -> str:
        """Ambiguity resolution when multiple categories have close scores"""
        # Priority ranking
        priority = {
            "creative": 1,   # strength/importance priority
            "abyss": 2,       # risk/problems priority
            "lake": 3,        # positive/achievement priority
            "thunder": 4,     # change/trigger
            "light": 5,       # knowledge/connection
            "receptive": 6,   # stable/bearing
            "wind": 7,         # permeation/influence
            "mountain": 8,     # stopping/blocking
        }

        # Sort by priority
        sorted_categories = sorted(scores.keys(), key=lambda c: priority.get(c, 9))

        return sorted_categories[0]

    def _compute_energy(self, energy_type: str, text: str) -> float:
        """
        Compute memory energy level (0.0-2.0)

        Based on:
        1. Energy type base energy
        2. Emotional intensity adjustment
        3. Text length adjustment
        """
        # Energy type base energy
        base_energy = {
            "wood": 1.0,
            "fire": 1.1,
            "earth": 0.9,
            "metal": 1.0,
            "water": 0.95,
        }
        energy = base_energy.get(energy_type, 1.0)

        # Emotional intensity adjustment
        text.lower()

        for cat, words in INTENSITY_WORDS.items():
            for word in words:
                if word in text:
                    if "positive" in cat:
                        if "high" in cat:
                            energy *= 1.3
                        elif "medium" in cat:
                            energy *= 1.15
                        else:
                            energy *= 1.05
                    elif "negative" in cat:
                        if "high" in cat:
                            energy *= 0.7
                        elif "medium" in cat:
                            energy *= 0.85
                        else:
                            energy *= 0.95

        # Length adjustment (too short or too long both reduce energy)
        text_len = len(text)
        if text_len < 15:
            energy *= 0.8
        elif text_len > 200:
            energy *= 0.9

        # Clamp to 0.0-2.0 range
        energy = max(0.0, min(2.0, energy))

        # Keep one decimal place
        return round(energy, 1)

    def _generate_summary(self, text: str, category: str, max_len: int = 35) -> str:
        """
        Generate semantic summary (20-50 characters)

        Strategy:
        1. Remove stop words
        2. Extract core keywords
        3. Adjust summary style based on category
        """
        # Stop words
        stop_words = [
            "的", "了", "是", "在", "有", "和", "与", "以及",
            "也", "都", "而", "但", "或", "等", "这", "那",
            "一个", "一些", "这个", "那个", "我们", "你们",
        ]

        # Tokenize (simple split by punctuation and whitespace)
        words = re.split(r'[\s,,。;、:""''()()]+', text)

        # Filter stop words, keep meaningful words
        meaningful = [w for w in words if w and w not in stop_words and len(w) >= 2]

        # Adjust summary based on category
        if category == "creative":
            # Strength: highlight goals and achievements
            keywords = [w for w in meaningful if any(c in w for c in "增长成功目标关键")]
        elif category == "abyss":
            # Danger: highlight problems and risks
            keywords = [w for w in meaningful if any(c in w for c in "问题风险困难挑战")]
        elif category == "thunder":
            # Change: highlight events and changes
            keywords = [w for w in meaningful if any(c in w for c in "变化更新触发事件")]
        elif category == "light":
            # Connection: highlight knowledge and associations
            keywords = [w for w in meaningful if any(c in w for c in "知识网络连接信息")]
        elif category == "lake":
            # Joy: highlight achievement and satisfaction
            keywords = [w for w in meaningful if any(c in w for c in "完成达成满意协议")]
        elif category == "receptive":
            # Stable: highlight foundation and continuity
            keywords = [w for w in meaningful if any(c in w for c in "稳定持续基础维护")]
        elif category == "wind":
            # Permeation: highlight spread and influence
            keywords = [w for w in meaningful if any(c in w for c in "影响传播渗透扩散")]
        elif category == "mountain":
            # Blocking: highlight constraints and waiting
            keywords = [w for w in meaningful if any(c in w for c in "停止阻碍等待限制")]
        else:
            keywords = meaningful[:5]

        # If not enough keywords, use original words
        if len(keywords) < 2:
            keywords = meaningful[:4]

        # Concatenate summary
        summary = "".join(keywords[:6])

        # Truncate to max length
        if len(summary) > max_len:
            summary = summary[:max_len-2] + ".."

        return summary if summary else text[:max_len]

    def _compress_text(self, text: str, category: str) -> str:
        """
        Text compression algorithm - structured compression based on semantics

        Strategy:
        1. Remove stop words and redundant modifiers
        2. Merge similar concepts
        3. Preserve core semantics
        """
        # Stop word patterns (can be deleted during compression)
        stop_word_patterns = [
            r'的', r'了', r'是', r'在', r'有', r'和', r'与', r'以及',
            r'也', r'都', r'而', r'但', r'或', r'等', r'这', r'那',
            r'一个', r'一些', r'这个', r'那个', r'我们', r'你们',
            r'已经', r'正在', r'将要', r'可以', r'能够', r'应该',
            r'非常', r'十分', r'特别', r'相当', r'比较',
        ]

        result = text

        # Merge consecutive spaces and punctuation
        result = re.sub(r'\s+', '', result)
        result = re.sub(r'[,，;；、、]+', ' ', result)

        # Number compression: keep key numbers
        # Keep percentages, simplify large numbers
        result = re.sub(r'(\d{4,})年', r'\1y', result)
        result = re.sub(r'(\d+)%', r'\1%', result)  # Keep percentage

        # Common word compression mapping
        abbr_map = {
            '这个': '', '那个': '', '因此': '故', '因为': '因',
            '所以': '故', '但是': '但', '而且': '且',
            '可能': '或', '能够': '能', '已经': '已',
            '并且': '且', '或者': '或', '可以': '可',
            '开始': '始', '继续': '续', '完成': '成',
            '成功': '成', '失败': '败', '增长': '增',
            '下降': '降', '提高': '升', '降低': '降',
            '增加': '增', '减少': '减', '实现': '达',
            '达到': '达', '通过': '经', '根据': '依',
            '按照': '依', '由于': '因', '对于': '对',
            '关于': '关', '经过': '经',
            '之前': '前', '之后': '后', '目前': '今',
            '现在': '今', '当前': '今', '今天': '今',
        }

        for old, new in abbr_map.items():
            result = result.replace(old, new)

        # Remove stop words
        for pattern in stop_word_patterns:
            result = re.sub(pattern, '', result)

        # Merge consecutive spaces
        result = re.sub(r'\s+', '', result)

        return result if result else text[:40]

    def _build_template(
        self,
        category: str,
        energy_type: str,
        energy: float,
        links: int,
        time_code: str,
        summary: str
    ) -> str:
        """
        Build high-density semantic template

        Format: [category][energy_type][energy][links][time_code]summary

        Examples:
        - creative-metal1.2-fire3[甲午]核心资产持续增值
        - light-fire0.9-water1[丙子]知识网络互联互通
        - abyss-water0.6-wood2[戊寅]风险节点待观察
        """
        # Energy formatting (keep one decimal place)
        energy_str = f"{energy}"

        # Links (single digit)
        links_str = str(min(links, 9))

        # Assemble template: [category][energy_type][energy][links][time_code]summary
        template = f"{category}{energy_type}{energy_str}{links_str}{time_code}{summary}"

        return template

    def decompress(self, comp: str, mode: str) -> str:
        """Decompress (semantic mode is not reversible, only lossless mode can restore)"""
        if mode == "lossless":
            try:
                import zlib
                import base64
                return zlib.decompress(base64.b64decode(comp)).decode()
            except Exception:
                return comp
        # Semantic compression cannot be fully restored
        return comp


# ============================================================
# Tests
# ============================================================

if __name__ == "__main__":
    c = SuCompressor()
    tests = [
        "这个项目的ROI持续增长,预计明年收益率达到25%",
        "人与人之间的信息传递形成了复杂的知识网络",
        "当前市场存在较大的不确定性和潜在风险",
        "团队的协作效率显著提升,沟通成本下降30%",
        "技术架构需要升级以应对更高的并发需求",
    ]

    print("=" * 60)
    print("Semantic encoding system test")
    print("=" * 60)

    for t in tests:
        r = c.compress(t)
        print(f"IN: {t}")
        print(f"OUT: {r['compressed']}")
        print(f"CATEGORY: {r['category']} | ENERGY_TYPE: {r['energy_type']} | ENERGY: {r['energy']}")
        print(f"RATIO: {r['ratio']}x")
        print("-" * 40)
