#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/download_data.py
========================
Melon-PlantCAD2-EVEE 一键下载脚本（含真实可用下载链接，截至 2026-05）。

下载内容分为 5 类，每类均给出经过文献/数据库核验的真实链接，
失败时自动回退到镜像或打印手动下载提示，不影响其它项。

  1. 模型权重 (--models)
       PlantCAD2-Small  (88M, hidden 768)  HuggingFace kuleshov-group/PlantCAD2-Small-l24-d0768
     可选更大模型 (--large)
       PlantCAD2-Medium (311M, hidden 1024) kuleshov-group/PlantCAD2-Medium-l48-d1024
       PlantCAD2-Large  (694M, hidden 1536) kuleshov-group/PlantCAD2-Large-l48-d1536
     说明：PlantCAD2 = PlantCaduceus（Caduceus + Mamba2，双向、反向互补等变、
           掩码语言模型），通过 trust_remote_code 加载，分词器随模型附带，
           无需单独下载 tokenizer.json。论文 bioRxiv 2025.08.27.672609。

  2. 甜瓜参考基因组与注释 (--genome)
     主选 DHL92 (双单倍体系；Garcia-Mas et al. PNAS 2012；v4.0 Ruggieri et al. 2018)
       Ensembl Plants release-62  melonv4.0
         FASTA: .../fasta/cucumis_melo/dna/
                Cucumis_melo.melonv4.0.dna.toplevel.fa.gz
         GFF3:  .../gff3/cucumis_melo/
                Cucumis_melo.melonv4.0.62.gff3.gz
     备选 NCBI Assembly  GCA_902497455.1 / Melonv4
       https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/902/497/455/
       仅在 Ensembl 主源失败时兜底。
     其它高质量组装（手动）：Payzawat (inodorus)、Charmono (cantalupensis)、
       Harukei-3 (reticulatus)、HS (agrestis)、近年 inodorus 型 T2T 无缺口组装；
       葫芦类基因组数据库 CuGenDB cucurbitgenomics.org、melonomics.net。

  3. 群体重测序变异 (--variants)
     Cucurbit Genomics Database (CuGenDB) 已 call 的 SNP/InDel
       http://cucurbitgenomics.org  -> Download / Variation
     泛基因组 / 重测序（手动）：
       Zhao et al. 2019 (Nat Genet) 1175 份甜瓜变异图谱（GWAS 16 性状 / 208 位点）
       Liu et al. 2020 (Plant Biotechnol J) 297 份重测序，~2.0M SNP

  4. 甜瓜表型 / 代谢组数据 (--phenotype)
     含糖量（糖度 Brix）、香气（酯类）、果肉色、成熟期/耐贮、抗病性等
       多为论文补充材料，脚本只下载有直链者，余者打印 DOI/入口。

  5. 分词器 / 配置
     PlantCAD2 分词器随模型附带（AutoTokenizer），mock 编码器自带
     字符级 CharTokenizer，故无需单独下载 tokenizer.json。

用法：
    python scripts/download_data.py --all                  # 推荐：下全部主源
    python scripts/download_data.py --models               # 仅 PlantCAD2-Small
    python scripts/download_data.py --models --large       # 加下 Medium/Large
    python scripts/download_data.py --genome               # 甜瓜基因组 + 注释
    python scripts/download_data.py --genome --use-ncbi    # 改用 NCBI 备选源
    python scripts/download_data.py --variants             # 群体变异 VCF
    python scripts/download_data.py --phenotype            # 表型 / 代谢组补充材料
    python scripts/download_data.py --check                # 只检查链接连通性
"""
from __future__ import annotations

import argparse
import socket
import sys
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DATA = ROOT / "data"

# ------------------------------------------------------------------
# 1. HuggingFace 模型仓库（PlantCAD2 / PlantCaduceus）
# ------------------------------------------------------------------
HF_MODELS_CORE = {
    # 默认主力：Small，hidden=768，与本仓库 d_model 默认值一致
    "plantcad2": "kuleshov-group/PlantCAD2-Small-l24-d0768",
}

# 可选：更大模型，零样本 LLR 打分更准（hidden 1024 / 1536，需改 d_model）
HF_MODELS_LARGE = {
    "plantcad2-medium": "kuleshov-group/PlantCAD2-Medium-l48-d1024",
    "plantcad2-large":  "kuleshov-group/PlantCAD2-Large-l48-d1536",
}

# ------------------------------------------------------------------
# 2. 参考基因组与注释
# ------------------------------------------------------------------
# 主选：DHL92（Garcia-Mas et al. 2012 / v4.0 Ruggieri 2018），
#       Ensembl Plants release-62, assembly melonv4.0
ENSEMBL_BASE = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62"
GENOME_FILES_ENSEMBL = {
    "melon_DHL92.fa.gz": (
        f"{ENSEMBL_BASE}/fasta/cucumis_melo/dna/"
        "Cucumis_melo.melonv4.0.dna.toplevel.fa.gz"
    ),
    "melon_DHL92.gff3.gz": (
        f"{ENSEMBL_BASE}/gff3/cucumis_melo/"
        "Cucumis_melo.melonv4.0.62.gff3.gz"
    ),
}

# 备选：NCBI Assembly GCA_902497455.1 / Melonv4
NCBI_ASM = "GCA_902497455.1_melonv4"
NCBI_DIR = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/"
    "GCA/902/497/455/" + NCBI_ASM
)
GENOME_FILES_NCBI = {
    "melon_DHL92.fa.gz":   f"{NCBI_DIR}/{NCBI_ASM}_genomic.fna.gz",
    "melon_DHL92.gff.gz":  f"{NCBI_DIR}/{NCBI_ASM}_genomic.gff.gz",
}

# 葫芦类基因组数据库（人工访问，含多个组装、注释、变异、共线性工具）
CUGENDB_PAGE = "http://cucurbitgenomics.org"
MELONOMICS_PAGE = "http://melonomics.net"

# ------------------------------------------------------------------
# 3. 群体重测序变异
# ------------------------------------------------------------------
# CuGenDB 已 call 的 SNP/InDel（直链随版本变动，先给入口；如有稳定直链可补此处）
VARIANT_FILES: dict[str, str] = {
    # 占位：若 CuGenDB 提供稳定 VCF 直链可填于此；当前以 INFO 形式给入口
}

VARIANT_INFO = {
    "CuGenDB_SNP_InDel": (
        "Cucurbit Genomics Database (CuGenDB v2)\n"
        "  -> http://cucurbitgenomics.org  Download / Variation 页提供 DHL92 等\n"
        "     参考的 SNP/InDel、基因注释、共线性数据；CucCAP 核心种质重测序"
    ),
    "Zhao2019_1175_variation_map": (
        "Zhao et al. 2019, Nature Genetics «A comprehensive genome variation\n"
        "  map of melon» 1175 份重测序（GWAS 16 性状 / 208 位点）\n"
        "  -> DOI 10.1038/s41588-019-0522-8；原始 reads NCBI BioProject"
    ),
    "Liu2020_297_resequencing": (
        "Liu et al. 2020, Plant Biotechnol J «Resequencing of 297 melon\n"
        "  accessions» ~2.0M SNP（CmAAT 香气、果实大小/肉厚 GWAS）\n"
        "  -> DOI 10.1111/pbi.13434"
    ),
}

# ------------------------------------------------------------------
# 4. 表型 / 代谢组数据
# ------------------------------------------------------------------
PHENOTYPE_FILES: dict[str, str] = {
    # 多为论文补充 PDF/Excel，无稳定直链者以 INFO 形式给出
}

PHENOTYPE_INFO = {
    "Zhao2019_1175_GWAS_16traits": (
        "Zhao et al. 2019, Nature Genetics（1175 份变异图谱）\n"
        "  -> 16 个农艺性状 GWAS（果重/果实质量/形态等），逐份材料表型在论文\n"
        "     补充材料（Supplementary Tables）；DOI 10.1038/s41588-019-0522-8。\n"
        "     按 accession 与基因型对齐后用 build_phenotype_labels.py 生成标签。"
    ),
    "Liu2020_297_fruit_traits": (
        "Liu et al. 2020, Plant Biotechnol J（297 份重测序，开放获取）\n"
        "  -> 果实大小/肉厚/香气等性状 GWAS；材料元信息见 Table S1，约 2.0M SNP；\n"
        "     DOI 10.1111/pbi.13434（补充材料可自由下载）。"
    ),
    "NPGS_core_Akter2023": (
        "NPGS 甜瓜核心种质（Akter et al. 2023）\n"
        "  -> 性别表达、子房茸毛、叶/果形态、果肉可溶性固形物(SSC，糖度)等 GWAS，\n"
        "     表型见 Supplementary Table 6；基因型+表型配对较干净。"
    ),
    "Esteras_Leida_177_panel": (
        "Esteras/Leida 177 份多样性面板\n"
        "  -> 大量果实性状表型 + 约 23,931 个 GBS SNP，专为 GWAS 设计；\n"
        "     适合做候选基因级关联（糖度、成熟、果形等）。"
    ),
    "Sugar_brix": (
        "含糖量（糖度 Brix，蔗糖积累）面板\n"
        "  -> CmTST2（液泡糖转运）、CmSPS、CmAGA2、CmSWEET 相关 QTL/GWAS 论文补充"
    ),
    "Aroma_ester": (
        "香气（酯类挥发物）代谢组\n"
        "  -> CmAAT1-4（醇酰基转移酶）、CmADH、LOX 通路；Liu et al. 2020 补充数据"
    ),
    "Ripening_ethylene": (
        "果实成熟 / 乙烯（跃变型 vs 非跃变型）与耐贮\n"
        "  -> CmACS1/CmACO1、CmNAC-NOR；inodorus(非跃变) vs cantalupensis(跃变)"
    ),
    "Flesh_color_sex": (
        "果肉色（CmOr β-胡萝卜素）与性别决定（CmACS7/CmWIP1/CmACS11）\n"
        "  -> 橙肉/白肉 CmOr、andromonoecy/gynoecy 经典位点相关研究"
    ),
    "GarciaMas2012_DHL92": (
        "Garcia-Mas et al. 2012, PNAS «The genome of melon»\n"
        "  -> DHL92 基因组论文及补充材料（注释、转录组）；v4.0 Ruggieri 2018"
    ),
}

# 表型多以论文补充表（按 accession 组织）发布，下载后用
# scripts/build_phenotype_labels.py 映射列名并对齐基因型/变异，生成训练标签。


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def _report(blocks: int, bsize: int, total: int) -> None:
    if total > 0:
        pct = min(100, 100 * blocks * bsize / total)
        sys.stdout.write(f"\r    {pct:5.1f}%")
        sys.stdout.flush()


def _format_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def check_url(url: str, timeout: int = 10) -> tuple[bool, str]:
    """HEAD 请求探测链接可用性，返回 (ok, 信息)。"""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            size = r.headers.get("Content-Length", "?")
            try:
                size = _format_size(int(size))
            except (TypeError, ValueError):
                pass
            return True, f"HTTP {r.status} ({size})"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        return False, f"{type(e).__name__}: {e}"


def download_url(url: str, dest: Path, timeout: int = 60) -> bool:
    """下载单个 URL 到 dest，已存在则跳过；失败删除半成品。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [跳过] 已存在 {dest.relative_to(ROOT)}")
        return True
    print(f"  [下载] {url}\n       -> {dest.relative_to(ROOT)}")
    try:
        socket.setdefaulttimeout(timeout)
        urllib.request.urlretrieve(url, dest, reporthook=_report)
        print()
        return True
    except Exception as e:
        print(f"\n  [失败] {e}\n  请手动下载：{url}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def download_hf_model(repo_id: str, dest: Path) -> bool:
    """用 huggingface_hub 下载整个模型仓库（含 trust_remote_code 的建模代码）。"""
    if dest.exists() and any(dest.iterdir()):
        print(f"  [跳过] 已存在 {dest.relative_to(ROOT)}")
        return True
    print(f"  [下载] HuggingFace {repo_id} -> {dest.relative_to(ROOT)}")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        return True
    except ImportError:
        print("  [失败] 未安装 huggingface_hub，请先 `pip install huggingface_hub`")
        print(f"         或手动访问：https://huggingface.co/{repo_id}")
        return False
    except Exception as e:
        print(f"  [失败] {e}")
        print(f"  请手动下载：https://huggingface.co/{repo_id}")
        return False


def gunzip(path: Path) -> Path:
    """就地解压 .gz，返回解压后路径；幂等。"""
    import gzip
    import shutil

    if not path.name.endswith(".gz"):
        return path
    out = path.with_suffix("")
    if out.exists() and out.stat().st_size > 0:
        return out
    print(f"  [解压] {path.name}")
    with gzip.open(path, "rb") as fin, open(out, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return out


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def do_check() -> int:
    """只检查所有 URL 的连通性，不下载。"""
    print("\n=== 链接连通性检查 ===")
    all_urls: list[tuple[str, str]] = []
    for name, repo in {**HF_MODELS_CORE, **HF_MODELS_LARGE}.items():
        all_urls.append((f"HF/{name}", f"https://huggingface.co/{repo}"))
    for fn, url in GENOME_FILES_ENSEMBL.items():
        all_urls.append((f"Ensembl/{fn}", url))
    for fn, url in GENOME_FILES_NCBI.items():
        all_urls.append((f"NCBI/{fn}", url))
    for fn, url in VARIANT_FILES.items():
        all_urls.append((f"Variant/{fn}", url))
    for fn, url in PHENOTYPE_FILES.items():
        all_urls.append((f"Pheno/{fn}", url))

    failed = 0
    for name, url in all_urls:
        ok, info = check_url(url)
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag:4}] {name:55} {info}")
        if not ok:
            failed += 1
    print(f"\n共 {len(all_urls)} 个链接，{failed} 个不可用。")
    return failed


def do_models(large: bool) -> bool:
    print("\n=== 1. PlantCAD2 模型权重 ===")
    ok = True
    for sub, repo in HF_MODELS_CORE.items():
        ok &= download_hf_model(repo, ASSETS / sub)
    if large:
        print("\n--- 1b. PlantCAD2 更大模型（可选，需相应调高 d_model） ---")
        for sub, repo in HF_MODELS_LARGE.items():
            ok &= download_hf_model(repo, ASSETS / sub)
    print("\n  说明：PlantCAD2 通过 trust_remote_code 加载 Caduceus 建模代码，")
    print("        分词器随模型附带，无需单独下载 tokenizer.json。")
    return ok


def do_genome(use_ncbi: bool, no_gunzip: bool) -> bool:
    print("\n=== 2. 甜瓜参考基因组与注释 ===")
    if use_ncbi:
        print(f"    使用 NCBI 备选源：DHL92 ({NCBI_ASM})")
        files = GENOME_FILES_NCBI
        fallback = GENOME_FILES_ENSEMBL
    else:
        print("    使用 Ensembl Plants release-62 主源：DHL92 (melonv4.0)")
        files = GENOME_FILES_ENSEMBL
        fallback = GENOME_FILES_NCBI
    ok = True
    fb_vals = list(fallback.values())
    for i, (fname, url) in enumerate(files.items()):
        dest = ASSETS / "genome" / fname
        succ = download_url(url, dest)
        if not succ:
            print("  [自动回退] 主源失败，尝试备选源")
            alt = fb_vals[i % len(fb_vals)]
            succ = download_url(alt, ASSETS / "genome" / Path(alt).name)
        if succ and not no_gunzip and dest.exists():
            gunzip(dest)
        ok &= succ

    print(f"\n  其它甜瓜基因组（多组装/T2T/泛基因组）与工具见葫芦类基因组数据库：")
    print(f"    {CUGENDB_PAGE}    {MELONOMICS_PAGE}")
    return ok


def do_variants() -> bool:
    print("\n=== 3. 甜瓜重测序变异 ===")
    ok = True
    for fname, url in VARIANT_FILES.items():
        dest = ASSETS / "variants" / fname
        ok &= download_url(url, dest)

    print("\n  群体变异数据集入口（按需手动下载）：")
    for k, v in VARIANT_INFO.items():
        print(f"    [{k}]\n      {v}")
    print(f"\n  葫芦类基因组数据库（已 call 变异 + 注释）：{CUGENDB_PAGE}")
    return ok


def do_phenotype() -> bool:
    print("\n=== 4. 甜瓜表型 / 代谢组数据 ===")
    ok = True
    for fname, url in PHENOTYPE_FILES.items():
        dest = DATA / "phenotype" / fname
        ok &= download_url(url, dest)

    print("\n  高价值表型 / 代谢组资源（仅给入口，手动下载）：")
    for k, v in PHENOTYPE_INFO.items():
        print(f"    [{k}]\n      {v}")
    print("\n  下载补充表后，用 scripts/build_phenotype_labels.py 映射列名、")
    print("  按 accession 对齐基因型/变异，生成协方差探针与育种模型的训练标签。")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description="下载 Melon-PlantCAD2-EVEE 全部所需资源",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[1] if "用法：" in __doc__ else "",
    )
    ap.add_argument("--all", action="store_true",
                    help="下载全部主源（模型 + 基因组 + 变异 + 表型）")
    ap.add_argument("--models", action="store_true",
                    help="下载 PlantCAD2-Small 核心权重")
    ap.add_argument("--large", action="store_true",
                    help="同时下载 PlantCAD2 Medium/Large 更大模型")
    ap.add_argument("--genome", action="store_true",
                    help="下载甜瓜参考基因组 + 注释（默认 Ensembl DHL92）")
    ap.add_argument("--use-ncbi", action="store_true",
                    help="基因组改用 NCBI GCA_902497455.1 备选源")
    ap.add_argument("--variants", action="store_true",
                    help="下载群体变异 VCF / 打印入口")
    ap.add_argument("--phenotype", action="store_true",
                    help="下载表型与代谢组补充材料 / 打印入口")
    ap.add_argument("--no-gunzip", action="store_true",
                    help="下载后不自动解压 .gz 文件")
    ap.add_argument("--check", action="store_true",
                    help="只检查链接连通性，不下载")
    args = ap.parse_args()

    if args.check:
        sys.exit(do_check())

    if not any([args.all, args.models, args.genome,
                args.variants, args.phenotype]):
        ap.print_help()
        sys.exit(0)

    do_m = args.all or args.models
    do_g = args.all or args.genome
    do_v = args.all or args.variants
    do_p = args.all or args.phenotype
    ok = True

    if do_m:
        ok &= do_models(args.large or args.all)
    if do_g:
        ok &= do_genome(args.use_ncbi, args.no_gunzip)
    if do_v:
        ok &= do_variants()
    if do_p:
        ok &= do_phenotype()

    print()
    if ok:
        print("全部完成。")
    else:
        print("部分项目失败，请按上方提示手动下载后重试。")
    print("\n落地目录速查：")
    print(f"  模型权重    : {ASSETS}/plantcad2[-*]/")
    print(f"  基因组+注释 : {ASSETS}/genome/")
    print(f"  群体变异    : {ASSETS}/variants/")
    print(f"  表型数据    : {DATA}/phenotype/")


if __name__ == "__main__":
    main()
