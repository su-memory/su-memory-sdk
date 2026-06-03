"""
MLX GGUF 模型转换工具 (v3.5.2)

将 HuggingFace embedding 模型转换为 GGUF 格式，供 LlamaCppEmbedding 使用。

依赖: pip install su-memory[mlx]
用法: python -m su_memory.tools.convert_mlx --model BAAI/bge-m3 --quantize Q4_K_M

转换流程:
  1. 从 HuggingFace Hub 下载模型
  2. 使用 mlx-lm 转换为 MLX 格式
  3. 量化并输出为 GGUF 格式到 ~/.cache/su-memory/models/
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path.home() / ".cache" / "su-memory" / "models"

# 常用 embedding 模型映射
KNOWN_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "bge-large-zh": "BAAI/bge-large-zh-v1.5",
    "e5-large": "intfloat/multilingual-e5-large",
    "gte-large": "thenlper/gte-large",
    "nomic-embed": "nomic-ai/nomic-embed-text-v1.5",
}

QUANTIZE_OPTIONS = [
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S",
    "Q8_0", "F16", "F32",
]


def resolve_model_name(model: str) -> str:
    """解析模型名称 (支持别名)"""
    return KNOWN_MODELS.get(model, model)


def download_model(model_id: str, cache_dir: Path | None = None) -> Path:
    """从 HuggingFace Hub 下载模型"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub 未安装。请执行: pip install huggingface-hub"
        ) from None

    dl_dir = cache_dir or (Path.home() / ".cache" / "huggingface" / "hub")
    logger.info(f"下载模型: {model_id}")
    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(dl_dir),
    )
    return Path(local_path)


def convert_to_gguf(
    model_path: Path,
    output_path: Path,
    quantize: str = "Q4_K_M",
) -> Path:
    """
    使用 llama.cpp 的 convert 工具转换为 GGUF 格式

    优先尝试:
    1. llama-cpp-python 内置 convert (如果可用)
    2. 系统 convert-hf-to-gguf.py
    3. mlx-lm convert + quantize 输出兼容格式
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 方法 1: 尝试 llama-quantize CLI
    llama_quantize = shutil.which("llama-quantize")
    convert_script = shutil.which("convert-hf-to-gguf.py")

    if convert_script:
        logger.info("使用 convert-hf-to-gguf.py 转换...")
        fp16_path = output_path.with_suffix(".fp16.gguf")
        cmd = [
            sys.executable, convert_script,
            str(model_path),
            "--outfile", str(fp16_path),
            "--outtype", "f16",
        ]
        subprocess.run(cmd, check=True)

        if quantize != "F16" and llama_quantize:
            logger.info(f"量化为 {quantize}...")
            subprocess.run(
                [llama_quantize, str(fp16_path), str(output_path), quantize],
                check=True,
            )
            fp16_path.unlink(missing_ok=True)
        else:
            fp16_path.rename(output_path)
        return output_path

    # 方法 2: 使用 mlx-lm 转换 (Apple Silicon 优化)
    try:
        from mlx_lm import convert as mlx_convert
        logger.info("使用 mlx-lm 转换...")

        mlx_output = output_path.parent / f"{output_path.stem}_mlx"
        mlx_convert(
            model_path=str(model_path),
            mlx_path=str(mlx_output),
            quantize=(quantize != "F16" and quantize != "F32"),
        )

        # mlx-lm 原生输出 safetensors, 如果有 gguf 导出就用
        gguf_files = list(mlx_output.glob("*.gguf"))
        if gguf_files:
            shutil.move(str(gguf_files[0]), str(output_path))
        else:
            logger.warning(
                f"mlx-lm 未生成 GGUF 文件，MLX 格式已保存到: {mlx_output}\n"
                f"请手动使用 convert-hf-to-gguf.py 转换。"
            )
            return mlx_output

        # 清理中间文件
        shutil.rmtree(mlx_output, ignore_errors=True)
        return output_path

    except ImportError:
        pass

    raise RuntimeError(
        "未找到可用的转换工具。请安装以下之一:\n"
        "  1. llama.cpp: 提供 convert-hf-to-gguf.py + llama-quantize\n"
        "  2. mlx-lm: pip install mlx-lm (Apple Silicon)\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="将 HuggingFace embedding 模型转换为 GGUF 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m su_memory.tools.convert_mlx --model BAAI/bge-m3 --quantize Q4_K_M
  python -m su_memory.tools.convert_mlx --model bge-m3  # 使用别名
  python -m su_memory.tools.convert_mlx --model nomic-embed --output ./my_model.gguf

已知别名:
  bge-m3         → BAAI/bge-m3
  bge-large-zh   → BAAI/bge-large-zh-v1.5
  e5-large       → intfloat/multilingual-e5-large
  gte-large      → thenlper/gte-large
  nomic-embed    → nomic-ai/nomic-embed-text-v1.5
        """,
    )
    parser.add_argument(
        "--model", "-m", required=True,
        help="HuggingFace 模型 ID 或别名",
    )
    parser.add_argument(
        "--quantize", "-q", default="Q4_K_M",
        choices=QUANTIZE_OPTIONS,
        help="量化类型 (默认: Q4_K_M)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help=f"输出路径 (默认: {DEFAULT_OUTPUT_DIR}/{{model}}-{{quantize}}.gguf)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="输出详细日志",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    model_id = resolve_model_name(args.model)
    logger.info(f"模型: {model_id}")
    logger.info(f"量化: {args.quantize}")

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        model_short = model_id.split("/")[-1].lower().replace("_", "-")
        output_path = DEFAULT_OUTPUT_DIR / f"{model_short}-{args.quantize.lower()}.gguf"

    logger.info(f"输出: {output_path}")

    if output_path.exists():
        logger.warning(f"文件已存在: {output_path}")
        resp = input("覆盖? [y/N] ").strip().lower()
        if resp != "y":
            logger.info("取消")
            return

    # 下载
    model_path = download_model(model_id)

    # 转换
    result = convert_to_gguf(model_path, output_path, args.quantize)

    logger.info(f"✅ 转换完成: {result}")
    logger.info(f"   文件大小: {os.path.getsize(result) / 1024 / 1024:.1f} MB")
    logger.info("")
    logger.info("LlamaCppEmbedding 将自动检测此模型。")
    logger.info("或手动指定: SU_MEMORY_GGUF_MODEL_PATH=" + str(result))


if __name__ == "__main__":
    main()
