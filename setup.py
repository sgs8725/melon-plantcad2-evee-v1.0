from setuptools import setup, find_packages

setup(
    name="melon-plantcad2-evee",
    version="0.2.1",
    description="甜瓜（Cucumis melo）专有基因组大模型与智能育种模型："
                "PlantCAD2 × EVEE（变异效应预测 / 零样本 LLR / 注释扰动谱 / "
                "Gibbs 元件设计 / 多性状育种值）。",
    packages=find_packages(include=["meloncad*", "probes*", "breeding*", "data*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "transformers>=4.44",
        "numpy>=1.24",
        "polars>=1.0",
        "pyarrow>=15.0",
        "safetensors>=0.4.0",
        "intervaltree>=3.1",
        "biopython>=1.83",
        "huggingface_hub>=0.23.0",
    ],
    extras_require={
        # 真实 PlantCAD2 (Caduceus + Mamba2) 推理需 CUDA 扩展
        "gpu": ["mamba-ssm>=2.2", "causal-conv1d>=1.2"],
        "llm": ["anthropic>=0.40.0"],
        "vcf": ["pysam>=0.22.0"],
        "tracks": ["pyBigWig>=0.3.18"],  # bigWig 实验轨道（BED 无需）
        "liftover": ["pyliftover>=0.4"],  # 跨组装 liftOver（需 UCSC chain 文件）
    },
)
