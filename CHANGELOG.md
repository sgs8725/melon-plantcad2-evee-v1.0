# 更新日志

## v0.2.1 — 连续信号回归 + 跨组装 liftOver

- `scripts/train_annotation_tracks.py` 新增**回归模式** `--mode regression`
  （配 `--log1p`）：把峰强 / 染色质可及性 / 保守性等**连续信号**作为回归目标
  （BED 第 4 列信号，或 bigWig 原始信号），用 MSE 训练、不过 sigmoid；二值模式
  仍为默认。探针 sidecar 记录 `task`/`transform`。
- `scripts/tracks_to_bed.py` 新增：
  - `--keep-signal`（配 `--signal-col`）：输出 **BED4**（第 4 列=信号值，
    narrowPeak 默认取 col7 signalValue），直接喂回归训练；
  - `--liftover-chain CHAIN`：用 UCSC chain 文件做**跨组装 liftOver**
    （如把 v3.5.1 坐标轨道转到 melonv4.0），区间两端分别映射、要求落同一目标
    染色体、不可映射者丢弃计数；需可选 `pyliftover`（setup extras `[liftover]`）。
- `probes/annotation_probe.py` 的 `disruption_profile` 新增 `regression` 开关；
  `predict.py` 据注释探针 sidecar 的 `task` 自动选择：回归探针用**原始信号差**、
  二值探针用 sigmoid 概率差作为扰动。

## v0.2.0 — 流水线工程化升级（参照 tomato-gfm-evee2 实战经验）

本版本在 v0.1（PlantCAD2 × EVEE 基础流水线）之上，吸收上游番茄
tomato-gfm-evee2 在真实 GPU 服务器（NVIDIA L20）跑通全流程后沉淀的工程经验，
新增编排、评估与注释探针训练能力。**基础模型仍为 PlantCAD2**（非回退到
PlantGFM），所有新增脚本均适配 d_model=768、反向互补不变嵌入与离线 mock。

### 新增
- `scripts/train_annotation_probe.py`：从甜瓜 GFF 自动生成逐位置功能标签，
  训练注释探针（GFF 可生成的 17 类结构注释：region/splice/codon/起止密码子/
  启动子/终止子）。用 `MelonCADEncoder.encode_reference` 取 RC 不变嵌入；
  正负样本不平衡用 `BCEWithLogitsLoss(pos_weight)` + 负样本下采样；
  保存 `.safetensors` + `.json` 配置 + `metadata/annotation_probe.json` 面板。
  支持 `--smoke-test`（纯 CPU、合成数据）。
- `scripts/train_annotation_tracks.py`：从**实验轨道**训练扩展注释类的入口
  ——BED 峰（DAP-seq/ChIP-seq TFBS、ATAC/DNase 染色质可及性、motif/CRE 峰区）
  逐位置取「在峰内=1」；bigWig 信号（phyloP/phastCons 保守性、ATAC 信号）按
  阈值或分位数二值化。用轨道配置 JSON（class/path/type[/threshold/quantile]）
  指定任意类集，训练独立的扩展注释探针（含 `.json` sidecar + `metadata/`）。
  BED 无额外依赖；bigWig 需可选的 `pyBigWig`（setup extras `[tracks]`）。
  支持 `--smoke-test`（合成 BED，纯 CPU）。
- `scripts/tracks_to_bed.py`：轨道格式归一化入口，把 narrowPeak / broadPeak /
  bigBed / bedGraph 统一转成 `train_annotation_tracks.py` 所需的纯 BED3。
  narrowPeak/broadPeak 可按信号值(col7)或 -log10(q)(col9)过滤；bedGraph 按
  绝对阈值或分位数选阳性 bin 并合并相邻峰(`--merge-gap`)；bigBed 需 pyBigWig。
  内置染色体改名（`--strip-chr` / `--chrom-map`）与单染色体筛选（`--chrom`），
  直接解决「轨道 chrom 命名需与基因组 FASTA 一致」的对齐问题；支持单文件与
  `--input-dir/--out-dir` 批量，及 `--smoke-test`。
  **同时修复** `predict.py`：改为从注释探针自带的 sidecar 读取类别列表与
  d_model 重建探针，兼容结构注释(17 类)与轨道扩展(任意类集)两种探针——
  此前固定用默认全 panel 会导致自定义类数的探针加载形状不匹配。
- `scripts/evaluate.py`：独立评估器。分类输出 AUROC / AUPRC / 准确率 /
  精确率 / 召回 / F1 / 混淆矩阵并给判读；回归输出每性状 MSE 与 Pearson r。
  纯 Python 统计，无 sklearn 依赖。支持 `--smoke-test`。
- `runners/run_full_pipeline.sh`：可移植全流程编排（build → activations →
  labels → train_probe → train_annotation_probe → predict → evaluate），
  路径全用变量；`smoke` 子命令一键跑通离线全链路。
- `setup.py`：包可 `pip install -e .`；GPU/LLM/VCF 依赖以 extras 分组。
- `metadata/`：探针配置/面板 JSON（注释面板、协方差探针配置模板）。

### 内化的运维教训（来自上游报告）
1. **标签质量 > 基因组覆盖**：GWAS p 值标注的 AUROC 明显高于文献 meta-QTL
   区间标注。evaluate.py 的判读直接据此提示。优先用 `build_phenotype_labels.py`
   的 GWAS 模式回填高质量标签。
2. **防 OOM**：全量 VCF 可达千万级行，`build_variant_dataset.py` 用
   `--max-variants` 限流；runner 默认 5 万条。
3. **防显存碎片**：runner 设 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`。
4. **config 对齐**：d_model 必须与所用 PlantCAD2 权重 hidden_size 一致；
   探针权重旁的 `.json` 固化该值，predict.py 据此重建。
5. **smoke-test 陷阱**：`--smoke-test` 用随机/合成标签，仅验证流程，
   不代表真实精度；正式训练务必去掉该标志并回填真实标签。

### 不变
- PlantCAD2（Caduceus + Mamba2，双向 / RC 等变 / MLM）为冻结基础模型；
- EVEE 协方差探针、零样本 LLR、Gibbs 元件设计、多性状育种值聚合一仍其旧；
- 甜瓜生物学（糖度 CmTST2 / 香气 CmAAT / 乙烯成熟 CmNAC-NOR / 果肉色 CmOr /
  性别决定 CmACS7·CmWIP1 / 抗病 Fom·Vat）与数据源（DHL92 melonv4.0、CuGenDB、
  Zhao 2019、Liu 2020）不变。
