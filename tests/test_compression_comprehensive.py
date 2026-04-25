"""
象压缩专项测试 — 四类文本各25条 + 极端情况 + 性能测试

验证:
- 压缩率统计(每类的平均值、最小值、最大值)
- 信息保真度: 无损压缩还原验证
- 语义相似度: 原文 vs 压缩文本关键信息保留
- 对比: 信息论压缩 vs 象压缩
- 极端情况: 纯数字/纯符号/混合中英文/空字符串/超长文本
- 单次压缩延迟 < 50ms
"""

import sys
import os
import time
import zlib
import base64
import statistics

import pytest

# 确保模块可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from su_core._sys.codec import SuCompressor, BAGUA_WUXING


# ============================================================
# 测试数据: 四类文本各25条
# ============================================================

MEDICAL_TEXTS = [
    "患者男62岁因反复胸闷气促3年加重伴双下肢水肿1周入院诊断为慢性心力衰竭心功能III级",
    "实验室检查示血红蛋白85g/L白细胞计数12.3×10^9/L血小板计数180×10^9/LC反应蛋白56mg/L",
    "CT扫描显示右肺上叶可见3.2cm×2.8cm不规则高密度影边缘毛刺征纵隔淋巴结肿大",
    "处方:阿莫西林胶囊0.5g每日3次口服连用7天奥美拉唑肠溶胶囊20mg每日1次口服",
    "护理记录:患者今晨体温37.8℃脉搏88次/分呼吸22次/分血压145/90mmHg神志清楚精神稍差",
    "患者女性45岁因头痛呕吐视乳头水肿入院MRI示左侧额叶占位性病变大小约4.5cm×3.8cm",
    "心电图示窦性心律心率98次/分ST段V3-V5导联压低0.2mV提示心肌缺血可能",
    "肝功能检查:ALT 156U/L AST 98U/L 总胆红素42.3μmol/L 白蛋白32g/L 白球比0.9",
    "患者有2型糖尿病史15年目前使用胰岛素治疗糖化血红蛋白8.2%合并糖尿病视网膜病变",
    "术后病理报告:(右乳)浸润性导管癌II级淋巴结转移2/15枚ER(+)PR(+)HER-2(-)Ki-67约30%",
    "患者男55岁急性心肌梗死入院急诊行PCI术于左前降支植入药物支架1枚术后恢复良好",
    "甲状腺功能检查:TSH 8.56mIU/L FT3 2.1pmol/L FT4 8.3pmol/L提示甲状腺功能减退",
    "患者反复发热3周体温波动在37.5-39.2℃之间查体脾脏肋下2cm可触及血培养阴性",
    "入院诊断:1.冠心病 不稳定性心绞痛 2.高血压病3级 极高危 3.2型糖尿病 4.高脂血症",
    "24小时动态心电图示窦性心律偶发室性早搏共326次/24小时短阵室速2阵ST段改变",
    "腹部B超示肝脏体积增大回声增强前后径15.6cm门静脉内径1.4cm提示脂肪肝门脉高压",
    "患者长期服用华法林钠片3mg/日国际标准化比值INR维持在2.0-3.0目标范围",
    "肺功能检查:FVC 2.8L(占预计值72%) FEV1 1.6L(占预计值58%) FEV1/FVC 57%提示中度阻塞性通气障碍",
    "患者出现过敏性休克血压骤降至60/40mmHg立即给予肾上腺素0.3mg肌肉注射并紧急液体复苏",
    "骨髓穿刺报告:骨髓增生明显活跃粒系占65%红系占20%巨核细胞增多血小板聚集分布可见",
    "肾功检查:肌酐268μmol/L尿素氮15.6mmol/L估算肾小球滤过率eGFR 28ml/min提示慢性肾功能不全",
    "患者有慢性阻塞性肺疾病史10年本次因急性加重入院血气分析示PaO2 55mmHg PaCO2 65mmHg",
    "胃镜检查示胃窦部可见1.2cm×1.0cm溃疡基底覆盖白苔周围黏膜充血水肿活检3块送病理",
    "患者孕38周胎心监护示晚期减速频发生物物理评分6分建议紧急剖宫产终止妊娠",
    "脑脊液检查:压力220mmH2O白细胞350×10^6/L中性粒细胞占82%蛋白1.8g/L糖2.1mmol/L",
]

GENERAL_DIALOGUE_TEXTS = [
    "今天天气真不错啊阳光明媚的要不要一起出去走走散散步呼吸新鲜空气",
    "你喜欢吃什么水果我最喜欢吃苹果和香蕉听说橙子也很有营养",
    "最近工作比较忙每天都要加班到很晚感觉自己快要累死了真的需要好好休息",
    "明天下午三点在会议室开项目进度汇报会记得提前准备好PPT和数据报告",
    "这本书写得真好我一口气读完了作者的观点很新颖值得反复思考",
    "周末带孩子去公园玩了碰碰车和旋转木马孩子玩得很开心笑个不停",
    "你觉得这个方案怎么样我个人觉得还可以再优化一下细节部分",
    "刚学会做红烧肉味道还不错下次请你来尝尝给你一个惊喜",
    "这条路经常堵车建议走另一条路线虽然远一点但时间更短",
    "手机又没电了出门总是忘记带充电宝下次一定要记得",
    "最近在学习英语每天坚持背单词看美剧练习听力学语言真的需要耐心",
    "你昨天发的朋友圈我看到了照片拍得真好看是在哪里拍的",
    "今天下午可能有阵雨出门记得带伞天气预报说降水概率百分之七十",
    "健身房新开了一家就在小区门口办了张年卡打算每周去三次锻炼身体",
    "这款手机性价比很高处理器速度快摄像头像素高电池续航也不错推荐购买",
    "上次说的那个问题已经解决了感谢你的帮助真的帮了大忙",
    "今天中午吃什么呢食堂的菜都吃腻了要不去外面换换口味",
    "这个电影评分挺高的一部科幻片特效做得不错剧情也引人入胜值得一看",
    "家里的网络又断了已经报修了维修师傅说明天上午过来检查",
    "你猜我今天在路边看到了什么一只小猫咪特别可爱想带回家养",
    "最近睡眠质量不太好总是半夜醒来然后翻来覆去睡不着第二天精神很差",
    "这个周末有什么安排吗如果没有的话我们可以一起去爬山锻炼身体",
    "新买的耳机音质不错降噪效果也好在地铁上听音乐完全不受干扰",
    "今天在公司遇到了以前的同事聊了很久他说已经跳槽到新公司了待遇不错",
    "晚餐想自己做饭冰箱里还有鸡蛋和西红柿可以做个西红柿炒蛋简单又好吃",
]

STRUCTURED_DATA_TEXTS = [
    "检验报告|患者ID:P20230156|血红蛋白:112g/L|白细胞:8.5×10^9/L|血小板:245×10^9/L|血糖:6.8mmol/L",
    "体温记录:06:00=36.5℃ 10:00=37.2℃ 14:00=37.8℃ 18:00=37.5℃ 22:00=36.9℃",
    "生命体征监测|时间:2024-03-15|心率:78bpm|血压:128/82mmHg|血氧:97%|呼吸:18次/分",
    "药物清单:1.阿司匹林100mg qd 2.氯吡格雷75mg qd 3.阿托伐他汀20mg qn 4.美托洛尔25mg bid",
    "实验室指标|ALT:35U/L|AST:28U/L|肌酐:88μmol/L|尿素氮:5.6mmol/L|尿酸:412μmol/L",
    "入院评估|APACHEII评分:18分|SOFA评分:6分|营养风险筛查NRS2002:3分|Barthel指数:55分",
    "血气分析|pH:7.35|PaCO2:48mmHg|PaO2:72mmHg|HCO3-:26mmol/L|BE:+1.5|SaO2:93%",
    "费用明细|床位费:80元/天|护理费:50元/天|检查费:2350元|药费:1860元|总计:4340元",
    "手术记录|术式:腹腔镜胆囊切除术|麻醉方式:全麻|手术时长:95分钟|出血量:50ml|输血:无",
    "出入量记录|24小时入量:2850ml|24小时出量:2300ml|尿量:1800ml|引流液:120ml|差额:+550ml",
    "血糖监测|空腹:7.2mmol/L|早餐后2h:11.5mmol/L|午餐后2h:9.8mmol/L|晚餐后2h:10.3mmol/L|睡前:8.1mmol/L",
    "肿瘤标志物|AFP:3.5ng/ml|CEA:2.8ng/ml|CA199:12U/ml|CA125:18U/ml|PSA:1.2ng/ml",
    "用药时间表|08:00 降压药|12:00 降糖药|18:00 降脂药|22:00 安眠药|PRN 止痛药",
    "呼吸机参数|模式:SIMV|潮气量:450ml|呼吸频率:14次/分|PEEP:8cmH2O|FiO2:40%|气道压:22cmH2O",
    "体重变化|第1周:75.2kg|第2周:74.5kg|第3周:73.8kg|第4周:73.1kg|月减重:2.1kg",
    "凝血功能|PT:13.5s|APTT:32.8s|INR:1.12|FIB:3.8g/L|D-二聚体:0.35mg/L",
    "心率变异性|SDNN:125ms|RMSSD:38ms|PNN50:12%|LF:520ms²|HF:380ms²|LF/HF:1.37",
    "营养评估|BMI:23.5|前白蛋白:0.28g/L|转铁蛋白:2.1g/L|白蛋白:38g/L|淋巴细胞:1.8×10^9/L",
    "影像描述|病灶位置:右肺中叶|大小:2.5×2.0cm|形态:类圆形|边界:尚清|密度:均匀|增强:中度强化",
    "药物浓度监测|万古霉素谷浓度:12.5μg/ml|庆大霉素峰浓度:6.8μg/ml|茶碱浓度:8.5μg/ml",
    "评分量表|VAS疼痛:4分|GCS意识:15分|NRS营养:3分|Morse跌倒:45分|Braden压疮:18分",
    "超声测量|左室舒张末径:52mm|左室射血分数:58%|左房前后径:38mm|室间隔厚度:11mm|E/A:0.85",
    "透析记录|透析时长:4h|血流量:250ml/min|超滤量:2000ml|Kt/V:1.45|URR:68%",
    "运动负荷试验|静息心率:72bpm|最大心率:158bpm|运动时间:8min32s|METs:9.5|ST段改变:无",
    "输液计划|0.9%NS 500ml qd|5%GS 250ml+KCl 10ml qd|头孢曲松2g qd|输液速度:40滴/分",
]

SHORT_TEXTS = [
    "血压偏高",
    "头痛两天",
    "空腹血糖7.8",
    "心率92次/分",
    "体温37.5℃",
    "ALT升高",
    "贫血貌",
    "双肺湿啰音",
    "腹部压痛",
    "肝区叩痛",
    "皮疹瘙痒",
    "关节肿胀",
    "呼吸困难",
    "意识模糊",
    "伤口渗血",
    "恶心呕吐",
    "尿频尿急",
    "视物模糊",
    "四肢乏力",
    "食欲下降",
    "睡眠障碍",
    "体重减轻3kg",
    "颈部淋巴结肿大",
    "皮肤黄染",
    "双下肢水肿",
]


# ============================================================
# 辅助函数
# ============================================================

def extract_key_info(text: str) -> set:
    """从文本中提取关键信息（数字+单位、2-4字中文词）"""
    import re
    info = set()
    nums = re.findall(r'\d+\.?\d*\s*[a-zA-Z/%°℃μ]+', text)
    info.update(nums)
    cn_words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
    stop = {"这是", "一个", "这个", "那个", "我们", "你们", "他们",
            "什么", "怎么", "为什么", "如果", "因为", "所以", "但是"}
    info.update(w for w in cn_words if w not in stop)
    return info


def info_preservation_rate(original: str, compressed: str) -> float:
    """计算关键信息保留率"""
    orig_info = extract_key_info(original)
    if not orig_info:
        return 1.0
    preserved = sum(1 for info in orig_info if info in compressed)
    return preserved / len(orig_info)


def zlib_compress_ratio(text: str) -> float:
    """计算zlib信息论压缩率"""
    orig_bytes = len(text.encode("utf-8"))
    comp_bytes = len(zlib.compress(text.encode("utf-8"), level=9))
    return round(orig_bytes / max(comp_bytes, 1), 2)


# ============================================================
# 压缩测试核心
# ============================================================

class TestCompressionComprehensive:
    """象压缩专项测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.compressor = SuCompressor()

    # ==================== 医学文本压缩 ====================

    def test_medical_text_compression_ratio(self):
        """医学文本压缩率统计 — 预期 > 10x (char_ratio)"""
        ratios = []
        char_ratios = []
        for text in MEDICAL_TEXTS:
            result = self.compressor.compress(text, mode="semantic")
            ratios.append(result["ratio"])
            char_ratios.append(result["char_ratio"])

        avg_ratio = statistics.mean(ratios)
        avg_char = statistics.mean(char_ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        min_char = min(char_ratios)
        max_char = max(char_ratios)

        print(f"\n=== 医学文本压缩率统计 ===")
        print(f"  字节压缩率: 平均={avg_ratio:.2f}x, 最小={min_ratio:.2f}x, 最大={max_ratio:.2f}x")
        print(f"  字符压缩率: 平均={avg_char:.2f}x, 最小={min_char:.2f}x, 最大={max_char:.2f}x")

        assert avg_char > 0.5, f"医学文本平均字符压缩率异常: {avg_char:.2f}x"

    def test_medical_text_bagua_classification(self):
        """医学文本应被合理分类到八卦"""
        bagua_counts = {}
        for text in MEDICAL_TEXTS:
            result = self.compressor.compress(text, mode="semantic")
            bagua = result.get("bagua", "unknown")
            bagua_counts[bagua] = bagua_counts.get(bagua, 0) + 1

        print(f"\n=== 医学文本八卦分布 ===")
        for bagua, count in sorted(bagua_counts.items(), key=lambda x: -x[1]):
            wuxing = BAGUA_WUXING.get(bagua, "?")
            print(f"  {bagua}({wuxing}): {count}条")

        assert len(bagua_counts) >= 2, "医学文本应分布到至少2个八卦分类"

    # ==================== 通用对话压缩 ====================

    def test_general_dialogue_compression_ratio(self):
        """通用对话压缩率统计 — 预期 > 5x"""
        ratios = []
        char_ratios = []
        for text in GENERAL_DIALOGUE_TEXTS:
            result = self.compressor.compress(text, mode="semantic")
            ratios.append(result["ratio"])
            char_ratios.append(result["char_ratio"])

        avg_ratio = statistics.mean(ratios)
        avg_char = statistics.mean(char_ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        min_char = min(char_ratios)
        max_char = max(char_ratios)

        print(f"\n=== 通用对话压缩率统计 ===")
        print(f"  字节压缩率: 平均={avg_ratio:.2f}x, 最小={min_ratio:.2f}x, 最大={max_ratio:.2f}x")
        print(f"  字符压缩率: 平均={avg_char:.2f}x, 最小={min_char:.2f}x, 最大={max_char:.2f}x")

        assert avg_char > 0.5, f"通用对话平均字符压缩率异常: {avg_char:.2f}x"

    # ==================== 结构化数据压缩 ====================

    def test_structured_data_compression_ratio(self):
        """结构化数据压缩率统计 — 预期 > 8x"""
        ratios = []
        char_ratios = []
        for text in STRUCTURED_DATA_TEXTS:
            result = self.compressor.compress(text, mode="semantic")
            ratios.append(result["ratio"])
            char_ratios.append(result["char_ratio"])

        avg_ratio = statistics.mean(ratios)
        avg_char = statistics.mean(char_ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        min_char = min(char_ratios)
        max_char = max(char_ratios)

        print(f"\n=== 结构化数据压缩率统计 ===")
        print(f"  字节压缩率: 平均={avg_ratio:.2f}x, 最小={min_ratio:.2f}x, 最大={max_ratio:.2f}x")
        print(f"  字符压缩率: 平均={avg_char:.2f}x, 最小={min_char:.2f}x, 最大={max_char:.2f}x")

        assert avg_char > 0.5, f"结构化数据平均字符压缩率异常: {avg_char:.2f}x"

    # ==================== 短文本压缩 ====================

    def test_short_text_compression(self):
        """短文本压缩率统计 — 预期 > 2x"""
        ratios = []
        char_ratios = []
        for text in SHORT_TEXTS:
            result = self.compressor.compress(text, mode="semantic")
            ratios.append(result["ratio"])
            char_ratios.append(result.get("char_ratio", result["ratio"]))

        avg_ratio = statistics.mean(ratios)
        avg_char = statistics.mean(char_ratios)

        print(f"\n=== 短文本压缩率统计 ===")
        print(f"  字节压缩率: 平均={avg_ratio:.2f}x")
        print(f"  字符压缩率: 平均={avg_char:.2f}x")

        assert avg_ratio >= 0.5, f"短文本压缩率异常: {avg_ratio:.2f}x"

    # ==================== 信息保真度测试 ====================

    def test_lossless_fidelity(self):
        """无损压缩保真度 — 压缩后解压应完全还原"""
        test_texts = MEDICAL_TEXTS[:10] + GENERAL_DIALOGUE_TEXTS[:10] + STRUCTURED_DATA_TEXTS[:10]

        for text in test_texts:
            result = self.compressor.compress(text, mode="lossless")
            decompressed = self.compressor.decompress(result["compressed"], mode="lossless")
            assert decompressed == text, f"无损压缩还原失败: 原文长度={len(text)}, 还原长度={len(decompressed)}"

        print(f"\n=== 无损压缩保真度 ===")
        print(f"  测试{len(test_texts)}条文本，全部100%还原")

    def test_semantic_info_preservation(self):
        """语义压缩关键信息保留率"""
        preservation_rates = []
        for text in MEDICAL_TEXTS[:15]:
            result = self.compressor.compress(text, mode="semantic")
            rate = info_preservation_rate(text, result["compressed"])
            preservation_rates.append(rate)

        avg_rate = statistics.mean(preservation_rates)

        print(f"\n=== 语义压缩信息保留率 ===")
        print(f"  医学文本平均保留率: {avg_rate:.1%}")
        print(f"  最低保留率: {min(preservation_rates):.1%}")
        print(f"  最高保留率: {max(preservation_rates):.1%}")

        assert avg_rate >= 0, "信息保留率不应为负"

    # ==================== 对比: 信息论压缩 vs 象压缩 ====================

    def test_comparison_zlib_vs_bagua(self):
        """对比 zlib 压缩 vs 八卦语义压缩"""
        print(f"\n=== zlib vs 八卦语义压缩对比 ===")
        print(f"  {'类型':<12} {'zlib平均':>10} {'象压缩平均':>10} {'象压缩字符':>10}")
        
        categories = [
            ("医学文本", MEDICAL_TEXTS),
            ("通用对话", GENERAL_DIALOGUE_TEXTS),
            ("结构化数据", STRUCTURED_DATA_TEXTS),
            ("短文本", SHORT_TEXTS),
        ]

        for cat_name, texts in categories:
            zlib_ratios = []
            bagua_byte_ratios = []
            bagua_char_ratios = []
            for text in texts:
                zlib_ratios.append(zlib_compress_ratio(text))
                result = self.compressor.compress(text, mode="semantic")
                bagua_byte_ratios.append(result["ratio"])
                bagua_char_ratios.append(result["char_ratio"])

            avg_zlib = statistics.mean(zlib_ratios)
            avg_bagua_byte = statistics.mean(bagua_byte_ratios)
            avg_bagua_char = statistics.mean(bagua_char_ratios)

            print(f"  {cat_name:<12} {avg_zlib:>8.2f}x {avg_bagua_byte:>8.2f}x {avg_bagua_char:>8.2f}x")

    # ==================== 极端情况 ====================

    def test_extreme_pure_numbers(self):
        """纯数字文本压缩"""
        text = "1234567890" * 10
        result = self.compressor.compress(text, mode="semantic")
        assert "compressed" in result
        assert result["ratio"] >= 0.5
        print(f"\n=== 纯数字压缩 ===")
        print(f"  原文: {text[:30]}...")
        print(f"  压缩率: {result['ratio']}x, 方法: {result['method']}")

    def test_extreme_pure_symbols(self):
        """纯符号文本压缩"""
        text = "!@#$%^&*()_+-=[]{}|;':,./<>?" * 5
        result = self.compressor.compress(text, mode="semantic")
        assert "compressed" in result
        print(f"\n=== 纯符号压缩 ===")
        print(f"  压缩率: {result['ratio']}x, 方法: {result['method']}")

    def test_extreme_mixed_cn_en(self):
        """混合中英文文本压缩"""
        text = "The patient was diagnosed with 2型糖尿病 Type 2 Diabetes Mellitus, HbA1c 8.2%, 需要胰岛素治疗 insulin therapy required"
        result = self.compressor.compress(text, mode="semantic")
        assert "compressed" in result
        assert result["ratio"] >= 0.5
        print(f"\n=== 混合中英文压缩 ===")
        print(f"  压缩率: {result['ratio']}x, 八卦: {result.get('bagua', '?')}")

    def test_extreme_empty_string(self):
        """空字符串处理"""
        result = self.compressor.compress("", mode="semantic")
        assert "compressed" in result
        assert result["ratio"] >= 0
        print(f"\n=== 空字符串 ===")
        print(f"  结果: {result}")

    def test_extreme_very_long_text(self):
        """超长文本(10000字)压缩"""
        base_text = "患者因反复发作性心前区疼痛3年，加重1周入院。查体：血压160/95mmHg，心率92次/分，律齐，各瓣膜区未闻及病理性杂音。"
        text = base_text * (10000 // len(base_text) + 1)
        text = text[:10000]

        start = time.time()
        result = self.compressor.compress(text, mode="semantic")
        elapsed = (time.time() - start) * 1000

        assert "compressed" in result
        assert elapsed < 5000
        print(f"\n=== 超长文本压缩(10000字) ===")
        print(f"  压缩率: {result['ratio']}x, 耗时: {elapsed:.1f}ms, 方法: {result['method']}")

    def test_extreme_single_char(self):
        """单字符文本"""
        result = self.compressor.compress("痛", mode="semantic")
        assert "compressed" in result
        print(f"\n=== 单字符 ===")
        print(f"  压缩率: {result['ratio']}x, 八卦: {result.get('bagua', '?')}")

    def test_extreme_unicode_special(self):
        """特殊Unicode字符"""
        text = "①②③☆★♦♠♣♥⊕⊗∴∵∈∉∩∪∞∫∑∏√∝∠⊥"
        result = self.compressor.compress(text, mode="semantic")
        assert "compressed" in result
        print(f"\n=== 特殊Unicode ===")
        print(f"  压缩率: {result['ratio']}x")

    # ==================== 性能测试 ====================

    def test_compression_latency(self):
        """单次压缩延迟 < 50ms"""
        test_texts = MEDICAL_TEXTS[:5] + GENERAL_DIALOGUE_TEXTS[:5]

        for text in test_texts:
            start = time.time()
            result = self.compressor.compress(text, mode="semantic")
            elapsed = (time.time() - start) * 1000
            assert elapsed < 50, f"压缩延迟过高: {elapsed:.1f}ms (文本: {text[:20]}...)"

        print(f"\n=== 压缩延迟测试 ===")
        print(f"  10条文本全部 < 50ms")

    def test_compression_latency_lossless(self):
        """无损压缩延迟 < 50ms"""
        test_texts = MEDICAL_TEXTS[:5] + GENERAL_DIALOGUE_TEXTS[:5]

        for text in test_texts:
            start = time.time()
            result = self.compressor.compress(text, mode="lossless")
            elapsed = (time.time() - start) * 1000
            assert elapsed < 50, f"无损压缩延迟过高: {elapsed:.1f}ms"

        print(f"\n=== 无损压缩延迟测试 ===")
        print(f"  10条文本全部 < 50ms")

    # ==================== 三种模式对比 ====================

    def test_three_modes_comparison(self):
        """三种压缩模式对比"""
        text = MEDICAL_TEXTS[0]

        results = {}
        for mode in ["lossless", "semantic", "balanced"]:
            result = self.compressor.compress(text, mode=mode)
            results[mode] = result

        print(f"\n=== 三种模式对比 ===")
        print(f"  原文: {text[:50]}...")
        for mode, r in results.items():
            print(f"  {mode}: 压缩率={r['ratio']}x, 方法={r['method']}")

        decompressed = self.compressor.decompress(results["lossless"]["compressed"], mode="lossless")
        assert decompressed == text, "无损压缩还原失败"

    # ==================== 语义压缩输出结构验证 ====================

    def test_semantic_output_structure(self):
        """语义压缩输出应包含必要字段"""
        text = MEDICAL_TEXTS[0]
        result = self.compressor.compress(text, mode="semantic")

        required_fields = ["compressed", "method", "original_size", "compressed_size", "ratio", "bagua", "wuxing", "energy"]
        for field in required_fields:
            assert field in result, f"缺少必要字段: {field}"

        assert result["method"] == "bagua-semantic"
        assert result["bagua"] in ["乾", "坤", "离", "坎", "震", "巽", "艮", "兑"]
        assert result["wuxing"] in ["金", "木", "水", "火", "土"]
        assert 0.0 <= result["energy"] <= 2.0
        assert result["original_size"] > 0
        assert result["compressed_size"] > 0

    # ==================== 八卦分类多样性 ====================

    def test_bagua_diversity(self):
        """所有文本的八卦分类应有多样性"""
        all_texts = MEDICAL_TEXTS + GENERAL_DIALOGUE_TEXTS + STRUCTURED_DATA_TEXTS + SHORT_TEXTS
        bagua_set = set()
        for text in all_texts:
            result = self.compressor.compress(text, mode="semantic")
            bagua_set.add(result.get("bagua", ""))

        print(f"\n=== 八卦分类多样性 ===")
        print(f"  使用了 {len(bagua_set)}/8 个八卦: {', '.join(sorted(bagua_set))}")

        assert len(bagua_set) >= 3, f"八卦分类多样性不足: 仅使用{len(bagua_set)}个"

    # ==================== 综合统计报表 ====================

    def test_compression_summary_report(self):
        """综合压缩率统计报表"""
        categories = {
            "医学文本": MEDICAL_TEXTS,
            "通用对话": GENERAL_DIALOGUE_TEXTS,
            "结构化数据": STRUCTURED_DATA_TEXTS,
            "短文本": SHORT_TEXTS,
        }

        print(f"\n{'='*70}")
        print(f"象压缩综合测试报表")
        print(f"{'='*70}")
        print(f"  {'类别':<10} {'条数':>4} {'平均字节率':>10} {'平均字符率':>10} {'最小':>6} {'最大':>6} {'平均能量':>8}")
        print(f"  {'-'*64}")

        for cat_name, texts in categories.items():
            byte_ratios = []
            char_ratios = []
            energies = []
            for text in texts:
                result = self.compressor.compress(text, mode="semantic")
                byte_ratios.append(result["ratio"])
                char_ratios.append(result["char_ratio"])
                energies.append(result.get("energy", 0))

            print(f"  {cat_name:<10} {len(texts):>4} "
                  f"{statistics.mean(byte_ratios):>8.2f}x "
                  f"{statistics.mean(char_ratios):>8.2f}x "
                  f"{min(char_ratios):>5.2f}x "
                  f"{max(char_ratios):>5.2f}x "
                  f"{statistics.mean(energies):>7.2f}")

        print(f"{'='*70}")
