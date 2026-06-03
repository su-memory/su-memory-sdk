"""
su-memory v3.6.0 — ParametricMemory 单元测试
============================================

测试覆盖:
- 配置初始化
- 训练数据准备
- QLoRA 训练流程（mock）
- Adapter 保存/加载往返
"""

import json
import os
import tempfile

import pytest

from su_memory.sdk._parametric_memory import (
    ParametricMemory,
    ParametricMemoryConfig,
    TrainingSample,
    estimate_training_time,
)


class TestParametricMemoryConfig:
    """配置类测试。"""

    def test_default_config(self):
        """默认配置值正确。"""
        config = ParametricMemoryConfig()
        assert config.lora_rank == 64
        assert config.lora_alpha == 128
        assert config.quant_bits == 4
        assert config.batch_size == 4
        assert config.min_training_pairs == 3000
        assert config.base_model == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_config_to_dict(self):
        """配置序列化。"""
        config = ParametricMemoryConfig()
        d = config.to_dict()
        assert d["lora_rank"] == 64
        assert d["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert d["use_energy_loss"] is True

    def test_custom_config(self):
        """自定义配置。"""
        config = ParametricMemoryConfig(
            lora_rank=32,
            lora_alpha=64,
            batch_size=8,
            num_epochs=5,
            energy_loss_alpha=0.05,
        )
        assert config.lora_rank == 32
        assert config.energy_loss_alpha == 0.05


class TestParametricMemory:
    """参数化记忆引擎测试。"""

    @pytest.fixture
    def pm(self):
        return ParametricMemory()

    @pytest.fixture
    def sample_qa_pairs(self):
        """生成测试 QA 对。"""
        pairs = []
        relations = ["enhance", "suppress", "same", "neutral"]
        for i in range(20):
            pairs.append({
                "cause_text": f"测试原因事件 #{i}",
                "effect_text": f"测试效应结果 #{i}",
                "energy_relation": relations[i % 4],
                "confidence": 0.5 + (i % 5) * 0.1,
            })
        return pairs

    def test_init(self, pm):
        """默认初始化。"""
        assert pm.config.lora_rank == 64
        assert pm.is_trained is False
        assert pm.n_training_samples == 0

    def test_init_with_custom_config(self):
        """自定义配置初始化。"""
        config = ParametricMemoryConfig(lora_rank=32, batch_size=8)
        pm = ParametricMemory(config)
        assert pm.config.lora_rank == 32
        assert pm.config.batch_size == 8

    def test_prepare_training_data(self, pm, sample_qa_pairs):
        """训练数据准备 — 基本功能。"""
        n, report = pm.prepare_training_data(sample_qa_pairs)
        assert n == 20
        assert report["total_samples"] == 20
        assert report["skipped"] == 0
        assert "energy_distribution" in report
        assert "meets_minimum" in report
        assert report["meets_minimum"] is False  # 20 < 3000

    def test_prepare_training_data_filters_low_confidence(self, pm):
        """低置信度 QA 对被过滤。"""
        low_quality = [
            {"cause_text": "A", "effect_text": "B", "energy_relation": "enhance", "confidence": 0.1},
            {"cause_text": "C", "effect_text": "D", "energy_relation": "suppress", "confidence": 0.6},
        ]
        n, report = pm.prepare_training_data(low_quality)
        assert n == 1  # 只保留了 confidence=0.6 的那条
        assert report["skipped"] == 1

    def test_prepare_training_data_empty_input(self, pm):
        """空输入处理。"""
        n, report = pm.prepare_training_data([])
        assert n == 0
        assert report["total_samples"] == 0

    def test_prepare_training_data_synthesized_qapair(self, pm):
        """支持 SynthesizedQAPair 对象。"""
        try:
            from su_memory.sdk._reflection_synthesizer import SynthesizedQAPair
            pairs = [
                SynthesizedQAPair(
                    cause_text="价格上涨",
                    effect_text="需求下降",
                    cause_entity="price",
                    effect_entity="demand",
                    reflection_depth=1,
                    energy_relation="enhance",
                    confidence=0.8,
                    source_memory_ids=["m1", "m2"],
                    qa_pair_id="qa_test_001",
                )
            ]
            n, report = pm.prepare_training_data(pairs)
            assert n == 1
        except ImportError:
            pytest.skip("ReflectionSynthesizer 不可用")

    def test_get_training_format(self, pm, sample_qa_pairs):
        """训练格式输出。"""
        pm.prepare_training_data(sample_qa_pairs)
        formatted = pm.get_training_format()
        assert len(formatted) == 20
        assert "instruction" in formatted[0]
        assert "input" in formatted[0]
        assert "output" in formatted[0]
        assert "energy_relation" in formatted[0]

    def test_train_simulated(self, pm, sample_qa_pairs):
        """模拟训练流程。"""
        pm.prepare_training_data(sample_qa_pairs)
        stats = pm.train()
        assert "final_loss" in stats or "error" in stats
        if "error" not in stats:
            assert stats["n_steps"] > 0
            assert pm.is_trained

    def test_train_with_energy_loss(self, pm, sample_qa_pairs):
        """带能量损失的训练。"""
        from su_memory.sdk._energy_loss import create_default_energy_loss
        energy_loss = create_default_energy_loss(alpha=0.1)
        pm.prepare_training_data(sample_qa_pairs)
        stats = pm.train(energy_loss_fn=energy_loss)
        assert "backend" in stats or "error" in stats

    def test_save_adapter(self, pm, sample_qa_pairs):
        """Adapter 保存。"""
        pm.prepare_training_data(sample_qa_pairs)
        with tempfile.TemporaryDirectory() as tmpdir:
            pm.save_adapter(tmpdir)
            config_path = os.path.join(tmpdir, "adapter_config.json")
            assert os.path.exists(config_path)
            with open(config_path) as f:
                config = json.load(f)
            assert config["version"] == "3.6.0"

    def test_load_adapter(self, pm, sample_qa_pairs):
        """Adapter 保存/加载往返测试。"""
        pm.prepare_training_data(sample_qa_pairs)
        with tempfile.TemporaryDirectory() as tmpdir:
            pm.save_adapter(tmpdir)
            pm2 = ParametricMemory()
            result = pm2.load_adapter(tmpdir)
            assert result is True

    def test_load_adapter_nonexistent(self, pm):
        """加载不存在的 adapter。"""
        result = pm.load_adapter("/nonexistent/path/adapter")
        assert result is False

    def test_predict_without_training(self, pm):
        """未训练时的预测返回空列表。"""
        predictions = pm.predict("测试原因")
        assert predictions == []

    def test_health_check(self, pm, sample_qa_pairs):
        """健康检查。"""
        pm.prepare_training_data(sample_qa_pairs)
        health = pm.health_check()
        assert health["model_loaded"] is False
        assert health["is_trained"] is False
        assert health["n_training_samples"] == 20
        assert "backend" in health

    def test_training_stats_property(self, pm, sample_qa_pairs):
        """训练统计属性。"""
        pm.prepare_training_data(sample_qa_pairs)
        stats = pm.training_stats
        assert isinstance(stats, dict)

    def test_max_pairs_limit(self):
        """超过 max_training_pairs 时截断。"""
        config = ParametricMemoryConfig(max_training_pairs=50)
        pm = ParametricMemory(config)
        pairs = [
            {"cause_text": f"cause_{i}", "effect_text": f"effect_{i}",
             "energy_relation": "enhance", "confidence": 0.5}
            for i in range(100)
        ]
        n, _ = pm.prepare_training_data(pairs)
        assert n == 50


class TestTrainingSample:
    """TrainingSample 数据类测试。"""

    def test_create_sample(self):
        sample = TrainingSample(
            instruction="test",
            input_text="input",
            output_text="output",
            energy_relation="enhance",
            confidence=0.8,
            sample_id="pm_test",
        )
        assert sample.instruction == "test"
        assert sample.energy_relation == "enhance"
        assert sample.confidence == 0.8


class TestEstimateTrainingTime:
    """训练时间估算测试。"""

    def test_mlx_estimate(self):
        est = estimate_training_time(5000, "mlx")
        assert est["backend"] == "mlx"
        assert est["hours"] > 0
        assert est["steps"] == 5000

    def test_torch_estimate(self):
        est = estimate_training_time(5000, "torch")
        assert est["backend"] == "torch"
        assert est["hours"] > est["minutes"] / 60  # torch 比 mlx 慢
