# Melon-PlantCAD2-EVEE 模型验证数据包
# 生成日期: 2026-06-22

## 目录说明

| 文件 | 内容 |
|---|---|
| `melon_GWAS_published_loci.csv` | 18个已发表甜瓜GWAS/QTL显著位点（含完整文献引用） |
| `melon_GWAS_dataset.xlsx` | GWAS数据集（真实位点_REAL + 模拟数据_SIMULATED + 列说明） |
| `validation_AB.py` | 验证脚本（方案A：基因坐标验证 + 方案B：染色体富集验证） |
| `validation_output.log` | 方案A+B原始输出日志 |
| `validation_results.csv` | 验证结果汇总表 |
| `candidate_gene_results.csv` | 候选基因区域效应分数详细数据 |
| `chromosome_enrichment.csv` | 染色体富集统计表 |

## 参考文献（验证位点来源）

| 编号 | 位点 | 文献 | 年份 |
|---|---|---|---|
| LOC01 | CmOr (果肉色) | Tzuri et al. / Yousef et al. IJMS 24(20):15490 | 2015/2023 |
| LOC02 | CmPPR1 (果肉色) | Galpaz et al. | 2018 |
| LOC03 | CmACS7 (果长/形) | Monforte et al. / Zhao 2021 BMC Plant Biol | 2014/2021 |
| LOC04 | SSC QTL (糖度) | Argyris et al. | 2017 |
| LOC05 | CmTST2 (糖积累) | Cheng et al. | 2018 |
| LOC06 | SSC GWAS (糖度) | Liu et al. (SSC GWAS), PMC10530127 | 2023 |
| LOC07-09 | 果面沟纹 | Hu et al. / Zhang et al. Frontiers | 2019/2022 |
| LOC10-12 | 果实硬度 | Nimmakayala et al. Front Plant Sci 7:1436 | 2016 |
| LOC13-16 | 果实大小/肉厚/心皮/香气 | Liu et al. (297 accessions), PMC7680547 | 2020 |
| LOC17 | 果肉苦味 | Zhao et al. Nature Genetics | 2019 |
| LOC18 | 16个农艺性状 | Zhao et al. Nature Genetics (1175 accessions) | 2019 |

## 验证核心结果

### 候选基因区域效应分数

| 基因 | 性状 | 变异数 | 平均效应 | 高效应占比 | 结论 |
|---|---|---|---|---|---|
| CmTST2 | 糖积累（液泡膜转运） | 5 | 0.7922 | 80% | ✅ 极高效应 |
| CmBt 附近 | 果肉苦味 | 103 | 0.5505 | — | ✅ 高效应 |
| SSC QTL | 可溶性固形物/蔗糖 | 104 | 0.4366 | 33.7% | ✅ 中高效应 |
| CmOr | 果肉颜色 | 区域无SNP | 附近0.4459 | — | ⚠️ GBS密度不足 |
| CmACS7 | 果长/果形 | 区域无SNP | 附近0.3768 | — | ⚠️ GBS密度不足 |
| CmCLV3 | 心皮数/果形 | 区域无SNP | 附近0.4025 | — | ⚠️ GBS密度不足 |
| CmAAT1/2 | 香气 | 区域无SNP | 附近0.3578 | — | ⚠️ GBS密度不足 |

### 染色体富集排名（由高到低）

| 排名 | 染色体 | 不利% | 平均效应 |
|---|---|---|---|
| 1 | contig7 (chr01) | 34.9% | 0.4436 |
| 1 | contig8 (chr09) | 34.9% | 0.4442 |
| 3 | contig3 (chr08) | 33.3% | 0.4204 |
| 4 | contig11 (chr11) | 32.7% | 0.4200 |
| 5 | contig5 (chr04) | 32.2% | 0.3263 |
| 6 | contig6 (chr07) | 32.1% | 0.4162 |
| 7-12 | 其余 | 28.8-31.2% | 0.394-0.416 |
| 13 | contig1 (chr01) | 22.2% | 0.2072 |

## 最终结论
模型对已知功能位点（特别是糖代谢基因CmTST2）具有合理的效应区分能力。
