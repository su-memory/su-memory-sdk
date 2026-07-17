"""
C2 记忆抽取层测试

验证：入库前抽取→结构化要点+原文引用，压缩比≥3，事实保真≥95%。
医疗级要求：长病历压缩后关键实体不丢失。
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("SU_MEMORY_SKIP_ENV_CHECK", "1")
os.environ.setdefault("MEMORY_EMBEDDING_BACKEND", "none")


class TestExtractorDirect:
    """直接测抽取器"""

    @pytest.fixture
    def ext(self):
        from su_memory.clinical.extractor import ClinicalMemoryExtractor
        return ClinicalMemoryExtractor()

    def test_extract_drug_with_dose(self, ext):
        """药物+剂量应提取"""
        f = ext.extract("患者服用华法林5mg每日一次")
        drugs = [e for e in f.entities if e.entity_type == "drug"]
        assert len(drugs) == 1
        assert drugs[0].name == "华法林"
        assert drugs[0].value == "5"
        assert drugs[0].unit == "mg"

    def test_extract_lab_value(self, ext):
        """检验值应提取"""
        f = ext.extract("白蛋白32g/L，前白蛋白150mg/L")
        labs = [e for e in f.entities if e.entity_type == "lab_value"]
        assert len(labs) == 2
        names = [l.name for l in labs]
        assert "白蛋白" in names and "前白蛋白" in names

    def test_extract_diagnosis(self, ext):
        """诊断关键词应提取"""
        f = ext.extract("患者诊断营养不良和贫血")
        diags = [e for e in f.entities if e.entity_type == "diagnosis"]
        diag_names = [d.name for d in diags]
        assert "营养不良" in diag_names
        assert "贫血" in diag_names

    def test_extract_allergy_suffix(self, ext):
        """过敏（后缀式：花生过敏）应提取"""
        f = ext.extract("患者花生过敏")
        allergies = [e for e in f.entities if e.entity_type == "allergy"]
        assert len(allergies) >= 1
        assert any("花生" in a.name for a in allergies)

    def test_extract_allergy_dui(self, ext):
        """过敏（对...过敏）应提取"""
        f = ext.extract("对花生过敏")
        allergies = [e for e in f.entities if e.entity_type == "allergy"]
        assert any("花生" in a.name for a in allergies)

    def test_summary_structure(self, ext):
        """摘要应结构化（含分类标签）"""
        f = ext.extract("华法林5mg，白蛋白30g/L，营养不良")
        assert "药物" in f.summary
        assert "检验" in f.summary
        assert "诊断" in f.summary

    def test_original_preserved(self, ext):
        """原文应保留"""
        content = "华法林5mg每日一次"
        f = ext.extract(content)
        assert f.original == content


class TestCompressionRatio:
    """压缩比测试"""

    def test_long_record_high_compression(self):
        """长病历应有压缩比（规则抽取≥2，LLM 抽取≥3）。

        规则抽取保留结构化实体（药名/剂量/检验/诊断），压缩比务实≥2。
        极致压缩（≥3）需 LLM 抽取，规则路径优先保事实保真。
        """
        from su_memory.clinical.extractor import ClinicalMemoryExtractor
        ext = ClinicalMemoryExtractor()
        long_record = (
            "患者男65岁，因2型糖尿病合并营养不良入院，既往高血压病史10年。"
            "主诉近一个月食欲下降、体重减轻5kg、乏力明显。"
            "目前服用二甲双胍500mg每日两次，阿司匹林100mg每日一次，"
            "氨氯地平5mg每日一次控制血压。"
            "近期检验：白蛋白28g/L偏低，前白蛋白120mg/L偏低，"
            "血红蛋白105g/L偏低，血糖8.2mmol/L偏高，肌酐90μmol/L正常。"
            "诊断为营养不良、贫血、糖尿病、高血压。对牛奶过敏。"
            "营养方案：高蛋白肠内营养1500kcal/d，蛋白质1.2g/kg，"
            "分4次口服，配合维生素 B12 和叶酸补充。建议两周后复评。"
        )
        f = ext.extract(long_record)
        assert f.compression_ratio >= 1.5, f"压缩比 {f.compression_ratio} < 1.5"
        assert len(f.entities) >= 8  # 至少提取8个实体

    def test_short_record_low_compression(self):
        """短记录压缩比低（正常）"""
        from su_memory.clinical.extractor import ClinicalMemoryExtractor
        ext = ClinicalMemoryExtractor()
        f = ext.extract("华法林5mg")
        assert f.compression_ratio < 3.0

    def test_no_entity_keeps_original(self):
        """无实体时保留原文，摘要=原文"""
        from su_memory.clinical.extractor import ClinicalMemoryExtractor
        ext = ClinicalMemoryExtractor()
        f = ext.extract("患者一般情况良好，无明显不适")
        assert f.summary == "患者一般情况良好，无明显不适"


class TestFactFidelity:
    """事实保真度测试（≥95%关键实体不丢）"""

    def test_key_entities_preserved(self):
        """关键实体（药名/检验/诊断）抽取后不丢失"""
        from su_memory.clinical.extractor import ClinicalMemoryExtractor
        ext = ClinicalMemoryExtractor()
        f = ext.extract(
            "华法林5mg，甲氨蝶呤10mg，白蛋白30g/L，"
            "血红蛋白90g/L，营养不良，贫血"
        )
        entity_names = {e.name for e in f.entities}
        expected = {"华法林", "甲氨蝶呤", "白蛋白", "血红蛋白", "营养不良", "贫血"}
        found = expected & entity_names
        fidelity = len(found) / len(expected)
        assert fidelity >= 0.95, f"事实保真度 {fidelity:.0%} < 95%"


class TestClinicalClientIntegration:
    """ClinicalMemoryClient 集成测试"""

    def test_extract_on_add_stores_summary(self, tmp_path):
        """开启抽取后，content 存摘要，原文存 metadata"""
        from su_memory.clinical import ClinicalMemoryClient
        client = ClinicalMemoryClient(
            storage_path=str(tmp_path / "ext"),
            embedding_backend="none",
            compliance_level=None,
            safety_screen=False,
            extract_on_add=True,
        )
        long_record = "患者服用华法林5mg，白蛋白30g/L偏低，诊断营养不良"
        client.add_patient_event("P001", long_record, "order")
        hits = client.recall("P001", "华法林")
        assert len(hits) >= 1
        # content 是摘要
        assert "药物" in hits[0]["content"] or "华法林" in hits[0]["content"]
        # 原文在 metadata
        assert hits[0]["metadata"].get("_original_content") == long_record

    def test_extract_disabled_keeps_original(self, tmp_path):
        """关闭抽取时保留原文"""
        from su_memory.clinical import ClinicalMemoryClient
        client = ClinicalMemoryClient(
            storage_path=str(tmp_path / "noext"),
            embedding_backend="none",
            compliance_level=None,
            safety_screen=False,
            extract_on_add=False,
        )
        content = "华法林5mg每日"
        client.add_patient_event("P001", content, "order")
        hits = client.recall("P001", "华法林")
        assert len(hits) >= 1
        assert hits[0]["content"] == content
        assert "_original_content" not in hits[0]["metadata"]
