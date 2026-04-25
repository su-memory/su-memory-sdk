"""
八卦语义编码器 - 基于易学象数的记忆压缩引擎

核心概念:用八卦(乾/坤/离/坎/震/巽/艮/兑)进行语义分类,而非医学词典匹配。
每段记忆根据其语义本质获得八卦分类,再进行高密度编码。

八卦语义映射:
  乾 (☰): 刚健/积极/上升/主动 → 高价值/重要/活跃的记忆
  坤 (☷): 柔顺/承载/稳定 → 稳定/持久/根基性记忆
  离 (☲): 明亮/发散/连接 → 知识/事实/关联网络
  坎 (☵): 陷入/危险/隐匿 → 问题/风险/隐藏事项
  震 (☳): 震动/变化/触发 → 触发器/事件/变化
  巽 (☴): 进入/渗透/扩散 → 渗透/扩散/影响力
  艮 (☶): 停止/阻碍/守候 → 障碍/暂停/等待
  兑 (☱): 喜悦/交换/协议 → 协议/交换/正向反馈
"""

from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime


# ============================================================
# 八卦语义模式库 - 基于意象和语义的分类
# ============================================================

BAGUA_SEMANTIC_PATTERNS = {
    # 乾 (☰) - 刚健、积极、上升、主导
    # 匹配:价值增长、积极行动、领导决策、目标导向
    "乾": [
        r'(ROI|收益率|增长|增值|上升|提高|突破|领先|主导|控制)',
        r'(投资|回报|利润|收益|目标|战略|决策|领导|核心|关键)',
        r'(成功|成就|突破|进展|进步|优化|提升|增强|卓越)',
        r'(主动|开创|引领|驱动|推动|促进)',
        r'(重要|首要|优先|紧急|核心|主要)',
    ],

    # 坤 (☷) - 柔顺、承载、稳定、根基
    # 匹配:稳定运行、基础建设、持续维护、承载能力
    "坤": [
        r'(稳定|持续|维持|保持|常规|日常|基础|根本|根基)',
        r'(承载|支撑|保障|维护|运营|管理|流程|体系|机制)',
        r'(传统|经典|标准|规范|制度|规则|框架)',
        r'(积累|沉淀|储备|存量|底蕴|根基)',
        r'(服从|配合|协作|支持|辅助|配合)',
    ],

    # 离 (☲) - 明亮、发散、连接、智慧
    # 匹配:知识传播、信息流动、网络连接、理解洞察
    "离": [
        r'(网络|连接|关联|关系|互联|互通|通信|交互)',
        r'(知识|信息|数据|文档|资料|报告|分析|研究)',
        r'(理解|认知|洞察|判断|识别|分类|定义)',
        r'(传播|分享|传递|交流|讨论|协作)',
        r'(明亮|清晰|透明|公开|可见|曝光)',
        r'(学习|掌握|熟练|精通|专业)',
    ],

    # 坎 (☵) - 陷入、危险、隐匿、困难
    # 匹配:风险问题、危机隐患、复杂困境、未知威胁
    "坎": [
        r'(风险|危险|威胁|隐患|问题|困难|挑战|危机)',
        r'(不确定|未知|隐匿|潜在|暗中|隐藏|秘密)',
        r'(失败|错误|缺陷|漏洞|故障|崩溃|失效)',
        r'(损失|亏损|负债|压力|紧张|焦虑)',
        r'(复杂|纠缠|陷阱|困境|难处理|难解决)',
        r'(下跌|下降|减少|衰退|恶化|退化)',
    ],

    # 震 (☳) - 震动、变化、触发、事件
    # 匹配:突发事件、变化动态、触发机制、激活启动
    "震": [
        r'(事件|发生|触发|激活|启动|启动|点燃)',
        r'(变化|改变|转变|转型|演进|进化|迭代)',
        r'(突发|紧急|突然|即时|立即|马上)',
        r'(震动|震荡|波动|起伏|动荡|不稳定)',
        r'(更新|升级|刷新|重置|重启)',
        r'(日出|春天|开始|起步|首发)',
    ],

    # 巽 (☴) - 进入、渗透、扩散、影响
    # 匹配:渗透扩展、影响传播、蔓延扩散、深入介入
    "巽": [
        r'(渗透|扩散|传播|蔓延|扩展|蔓延|侵入)',
        r'(影响|作用|效果|效力|触动|感染)',
        r'(深入|进入|介入|参与|加入|融入)',
        r'(推广|普及|覆盖|遍布|充斥)',
        r'(风|流动|灵活|适应|变通|顺从)',
        r'(市场|渠道|网络|分支|分布)',
    ],

    # 艮 (☶) - 停止、阻碍、守候、等待
    # 匹配:停滞阻塞、障碍阻挡、等待观望、保守谨慎
    "艮": [
        r'(停止|暂停|中断|中止|截断|阻碍|阻塞)',
        r'(等待|观望|保守|谨慎|稳健|稳妥)',
        r'(障碍|阻碍|瓶颈|卡点|难点|堵点)',
        r'(限制|约束|规范|规矩|边界|范围)',
        r'(阻止|禁止|不准|不能|不可)',
        r'(静止|不动|稳定|守候|坚持|持续)',
        r'(山|稳重|厚重|踏实|实在)',
    ],

    # 兑 (☱) - 喜悦、交换、协议、正向
    # 匹配:正向反馈、协议达成、喜悦满足、交换合作
    "兑": [
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
# 八卦与五行对应关系
# ============================================================

BAGUA_WUXING = {
    "乾": "金",
    "坤": "土",
    "离": "火",
    "坎": "水",
    "震": "木",
    "巽": "木",
    "艮": "土",
    "兑": "金",
}


# ============================================================
# 五行生克关系(用于能量计算)
# ============================================================

WUXING_CYCLE = {
    ("木", "火"): 1.2,  # 木生火 - 能量增强
    ("火", "土"): 1.2,  # 火生土
    ("土", "金"): 1.2,  # 土生金
    ("金", "水"): 1.2,  # 金生水
    ("水", "木"): 1.2,  # 水生木
    ("木", "土"): 0.7,  # 木克土 - 能量削弱
    ("土", "水"): 0.7,  # 土克水
    ("水", "火"): 0.7,  # 水克火
    ("火", "金"): 0.7,  # 火克金
    ("金", "木"): 0.7,  # 金克木
}

WUXING_ORDER = ["木", "火", "土", "金", "水"]


# ============================================================
# 天干地支时间编码
# ============================================================

GAN = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
ZHI = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]


def get_current_gangzhi() -> str:
    """获取当前干支(简化版,以年为单位)"""
    year = datetime.now().year
    gan_idx = (year - 4) % 10
    zhi_idx = (year - 4) % 12
    return GAN[gan_idx] + ZHI[zhi_idx]


def extract_time_ganzhis(text: str) -> List[str]:
    """从文本中提取时间相关的干支标记"""
    results = []
    # 匹配日期模式
    date_patterns = [
        r'(\d{4})年(\d{1,2})月',
        r'(\d{1,2})/(\d{1,2})',
        r'(\d+日|\d+号)',
        r'(昨天|今天|明天|上周|本周|下周)',
        r'(上午|下午|晚上|早晨|凌晨)',
    ]
    for p in date_patterns:
        if re.search(p, text):
            # 生成简化干支标记
            results.append(f"[{GAN[hash(text[:5]) % 10]}{ZHI[len(text) % 12]}]")
    return results


# ============================================================
# 情感强度词汇库(用于能量计算)
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
# 关联节点检测
# ============================================================

def count_causal_links(text: str) -> int:
    """计算文本中的关联节点数量"""
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
    return min(count, 9)  # 最多9个


# ============================================================
# 核心压缩引擎
# ============================================================

class SuCompressor:
    """
    八卦语义编码器 - 基于易学象数的记忆压缩引擎

    用八卦进行语义分类,而非医学词典匹配。
    每段记忆根据其语义本质获得八卦分类。
    """

    def __init__(self):
        self.mode = "semantic"
        self._compile_patterns()
        self._gangzhi = get_current_gangzhi()

    def _compile_patterns(self):
        """预编译所有八卦模式"""
        self._bagua_patterns = {}
        for bagua, pattern_list in BAGUA_SEMANTIC_PATTERNS.items():
            compiled = [re.compile(p, re.I) for p in pattern_list]
            self._bagua_patterns[bagua] = compiled

    def compress(self, text: str, mode: str = None) -> Dict:
        """
        压缩入口(兼容旧API)

        Returns: {
            "compressed": str,      # 压缩后的文本
            "method": str,          # 使用的方法
            "original_size": int,   # 原始字节数
            "compressed_size": int, # 压缩后字节数
            "ratio": float,         # 压缩率
            "bagua": str,           # 八卦分类(新增)
            "wuxing": str,          # 五行属性(新增)
            "energy": float,        # 能量等级(新增)
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
        """无损压缩(使用zlib+base64)"""
        import zlib, base64
        enc = base64.b64encode(zlib.compress(text.encode(), level=9)).decode()
        return {"compressed": enc, "method": "zlib+base64"}

    def _semantic(self, text: str) -> Dict:
        """八卦语义压缩"""
        # 短文本直接保留（带八卦标签）
        if len(text) < 15:
            return {
                "compressed": text,
                "method": "direct",
                "bagua": self._classify_by_bagua(text),
                "wuxing": BAGUA_WUXING.get(self._classify_by_bagua(text), "土"),
                "energy": 0.5,
            }

        # 步骤1: 八卦分类
        bagua = self._classify_by_bagua(text)
        wuxing = BAGUA_WUXING.get(bagua, "土")

        # 步骤2: 计算能量等级
        energy = self._compute_energy(wuxing, text)

        # 步骤3: 计算关联节点数
        links = count_causal_links(text)

        # 步骤4: 提取时间干支
        ganzhi_list = extract_time_ganzhis(text)
        ganzhi = ganzhi_list[0] if ganzhi_list else f"[{self._gangzhi}]"

        # 步骤5: 生成压缩内容（长文本用摘要，短文本用压缩）
        if len(text) > 80:
            summary = self._generate_summary(text, bagua, max_len=40)
        else:
            summary = self._compress_text(text, bagua)

        # 步骤6: 构建高密度模板
        output = self._build_template(bagua, wuxing, energy, links, ganzhi, summary)

        return {
            "compressed": output,
            "method": "bagua-semantic",
            "bagua": bagua,
            "wuxing": wuxing,
            "energy": energy,
        }

    def _balanced(self, text: str) -> Dict:
        """均衡模式"""
        if len(text) < 20:
            return self._semantic(text)

        res = self._semantic(text)
        orig_len = len(text)
        comp_len = len(res["compressed"])

        if orig_len / max(comp_len, 1) < 1.5:
            # 进一步压缩摘要
            summary = self._generate_summary(text, res["bagua"], max_len=20)
            output = self._build_template(
                res["bagua"], res["wuxing"], res["energy"],
                count_causal_links(text), f"[{self._gangzhi}]", summary
            )
            res["compressed"] = output

        return res

    def _classify_by_bagua(self, text: str) -> str:
        """
        根据语义内容分类到八卦

        算法:
        1. 对每个八卦,计算匹配分数
        2. 返回最高分的八卦
        3. 有多个高分区时,使用歧义解决
        """
        scores = {}

        for bagua, patterns in self._bagua_patterns.items():
            score = 0
            matched_patterns = 0
            for pattern in patterns:
                if pattern.search(text):
                    score += 1
                    matched_patterns += 1
            if matched_patterns > 0:
                scores[bagua] = score

        if not scores:
            # 无匹配时,根据文本特征推断
            return self._infer_bagua_fallback(text)

        # 获取最高分
        best_bagua = max(scores, key=scores.get)
        best_score = scores[best_bagua]

        # 歧义检测:检查是否有接近的分数
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            if sorted_scores[0] - sorted_scores[1] <= 1:
                # 分数接近,使用歧义解决
                return self._resolve_ambiguity(text, scores)

        return best_bagua

    def _infer_bagua_fallback(self, text: str) -> str:
        """无匹配时的回退推断"""
        # 根据文本基本特征推断八卦
        if any(c in text for c in "成功完成实现达成"):
            return "兑"  # 喜悦/达成
        elif any(c in text for c in "问题困难挑战风险"):
            return "坎"  # 危险/问题
        elif any(c in text for c in "变化更新改变转型"):
            return "震"  # 变化/触发
        elif any(c in text for c in "稳定持续保持维持"):
            return "坤"  # 稳定/持续
        elif any(c in text for c in "重要关键核心首要"):
            return "乾"  # 刚健/重要
        elif any(c in text for c in "知识信息数据网络"):
            return "离"  # 明亮/知识
        elif any(c in text for c in "停止等待阻碍限制"):
            return "艮"  # 停止/阻碍
        elif any(c in text for c in "影响传播扩散深入"):
            return "巽"  # 渗透/影响
        else:
            return "坤"  # 默认:稳定/承载

    def _resolve_ambiguity(self, text: str, scores: Dict[str, int]) -> str:
        """歧义解决:当多个八卦分数接近时"""
        # 优先级排序
        priority = {
            "乾": 1,   # 刚健/重要优先
            "坎": 2,   # 风险问题优先
            "兑": 3,   # 正向达成优先
            "震": 4,   # 变化触发
            "离": 5,   # 知识连接
            "坤": 6,   # 稳定承载
            "巽": 7,   # 渗透影响
            "艮": 8,   # 停止阻碍
        }

        # 按优先级排序
        sorted_bagua = sorted(scores.keys(), key=lambda b: priority.get(b, 9))

        return sorted_bagua[0]

    def _compute_energy(self, wuxing: str, text: str) -> float:
        """
        计算记忆能量等级(0.0-2.0)

        基于:
        1. 五行基础能量
        2. 情感强度调节
        3. 文本长度调节
        """
        # 五行基础能量
        base_energy = {
            "木": 1.0,
            "火": 1.1,
            "土": 0.9,
            "金": 1.0,
            "水": 0.95,
        }
        energy = base_energy.get(wuxing, 1.0)

        # 情感强度调节
        text_lower = text.lower()

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

        # 长度调节(过短或过长都降低能量)
        text_len = len(text)
        if text_len < 15:
            energy *= 0.8
        elif text_len > 200:
            energy *= 0.9

        # 限制在0.0-2.0范围内
        energy = max(0.0, min(2.0, energy))

        # 保留一位小数
        return round(energy, 1)

    def _generate_summary(self, text: str, bagua: str, max_len: int = 35) -> str:
        """
        生成语义摘要(20-50字符)

        策略:
        1. 去除停用词
        2. 提取核心关键词
        3. 根据八卦调整摘要风格
        """
        # 停用词
        stop_words = [
            "的", "了", "是", "在", "有", "和", "与", "以及",
            "也", "都", "而", "但", "或", "等", "这", "那",
            "一个", "一些", "这个", "那个", "我们", "你们",
        ]

        # 分词(简单按标点和空格分)
        words = re.split(r'[\s,,。;、:""''()()]+', text)

        # 过滤停用词,保留有意义的词
        meaningful = [w for w in words if w and w not in stop_words and len(w) >= 2]

        # 根据八卦调整摘要
        if bagua == "乾":
            # 刚健:突出目标和成就
            keywords = [w for w in meaningful if any(c in w for c in "增长成功目标关键")]
        elif bagua == "坎":
            # 危险:突出问题和风险
            keywords = [w for w in meaningful if any(c in w for c in "问题风险困难挑战")]
        elif bagua == "震":
            # 变化:突出事件和变化
            keywords = [w for w in meaningful if any(c in w for c in "变化更新触发事件")]
        elif bagua == "离":
            # 连接:突出知识和关联
            keywords = [w for w in meaningful if any(c in w for c in "知识网络连接信息")]
        elif bagua == "兑":
            # 喜悦:突出达成和满意
            keywords = [w for w in meaningful if any(c in w for c in "完成达成满意协议")]
        elif bagua == "坤":
            # 稳定:突出基础和持续
            keywords = [w for w in meaningful if any(c in w for c in "稳定持续基础维护")]
        elif bagua == "巽":
            # 渗透:突出传播和影响
            keywords = [w for w in meaningful if any(c in w for c in "影响传播渗透扩散")]
        elif bagua == "艮":
            # 阻碍:突出限制和等待
            keywords = [w for w in meaningful if any(c in w for c in "停止阻碍等待限制")]
        else:
            keywords = meaningful[:5]

        # 如果关键词不够,用原始词
        if len(keywords) < 2:
            keywords = meaningful[:4]

        # 拼接摘要
        summary = "".join(keywords[:6])

        # 截断到最大长度
        if len(summary) > max_len:
            summary = summary[:max_len-2] + ".."

        return summary if summary else text[:max_len]

    def _compress_text(self, text: str, bagua: str) -> str:
        """
        文本压缩算法 - 基于语义的结构化压缩

        策略：
        1. 删除停用词和冗余修饰
        2. 合并相似概念
        3. 保留核心语义
        """
        # 停用词模式（这些词在压缩时可删除）
        stop_word_patterns = [
            r'的', r'了', r'是', r'在', r'有', r'和', r'与', r'以及',
            r'也', r'都', r'而', r'但', r'或', r'等', r'这', r'那',
            r'一个', r'一些', r'这个', r'那个', r'我们', r'你们',
            r'已经', r'正在', r'将要', r'可以', r'能够', r'应该',
            r'非常', r'十分', r'特别', r'相当', r'比较',
        ]

        result = text

        # 合并连续空格和标点
        result = re.sub(r'\s+', '', result)
        result = re.sub(r'[,，;；、、]+', ' ', result)

        # 数字压缩：保留关键数字
        # 百分比保留，较大数字简化
        result = re.sub(r'(\d{4,})年', r'\1y', result)
        result = re.sub(r'(\d+)%', r'\1%', result)  # 保留百分比

        # 常见词汇压缩映射
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
            '关于': '关', '通过': '经', '经过': '经',
            '之前': '前', '之后': '后', '目前': '今',
            '现在': '今', '当前': '今', '今天': '今',
        }

        for old, new in abbr_map.items():
            result = result.replace(old, new)

        # 删除停用词
        for pattern in stop_word_patterns:
            result = re.sub(pattern, '', result)

        # 合并连续空格
        result = re.sub(r'\s+', '', result)

        return result if result else text[:40]

    def _build_template(
        self, 
        bagua: str, 
        wuxing: str, 
        energy: float,
        links: int,
        ganzhi: str,
        summary: str
    ) -> str:
        """
        构建高密度八卦模板

        格式：[卦][五行][能量][关联节点数][时间干支]语义摘要

        示例：
        - 乾金1.2火3[甲午]核心资产持续增值
        - 离火0.9水1[丙子]知识网络互联互通
        - 坎水0.6木2[戊寅]风险节点待观察
        """
        # 能量格式化（保留一位小数）
        energy_str = f"{energy}"

        # 关联节点数（单个数字）
        links_str = str(min(links, 9))

        # 组装模板: [卦][五行][能量][关联节点数][干支]摘要
        template = f"{bagua}{wuxing}{energy_str}{links_str}{ganzhi}{summary}"

        return template

    def decompress(self, comp: str, mode: str) -> str:
        """解压(语义模式不可逆,仅无损模式可还原)"""
        if mode == "lossless":
            try:
                import zlib, base64
                return zlib.decompress(base64.b64decode(comp)).decode()
            except:
                return comp
        # 语义压缩不可完全还原
        return comp


# ============================================================
# 测试
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
    print("八卦语义编码器测试")
    print("=" * 60)

    for t in tests:
        r = c.compress(t)
        print(f"IN: {t}")
        print(f"OUT: {r['compressed']}")
        print(f"BAGUA: {r['bagua']} | WUXING: {r['wuxing']} | ENERGY: {r['energy']}")
        print(f"RATIO: {r['ratio']}x")
        print("-" * 40)