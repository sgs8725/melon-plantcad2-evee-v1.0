#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/tracks_to_bed.py
========================
把常见实验轨道格式统一转成 scripts/train_annotation_tracks.py 所需的 BED
（默认 BED3：chrom start end，0-based 半开；--keep-signal 时输出 BED4 并附信号值），
并顺带做：跨组装版本 liftOver、染色体改名 / 单染色体筛选、显著性/信号阈值过滤、
bedGraph 阈值选峰 + 合并。

支持输入格式（--format auto 按扩展名自动识别，支持 .gz）：
  - narrowPeak / broadPeak (ENCODE)：前 3 列即 chrom/start/end；可按
      信号值(col7) 或 -log10(q)(col9, narrowPeak) 过滤；--keep-signal 取 col7
      （或 --signal-col N 指定列）作为信号值写第 4 列。
  - bigBed (.bb/.bigBed)：需可选 pyBigWig；逐染色体取区间。
  - bedGraph (.bedgraph/.bg)：chrom/start/end/value 四列；按绝对阈值
      (--threshold) 或分位数(--quantile) 选阳性 bin 并可合并相邻区间
      (--merge-gap)；--keep-signal 时第 4 列取合并区间内的最大信号。
  - bed / bed3+（.bed）：取前 3 列透传；--keep-signal 时用 --signal-col 指定值列。

跨组装 liftOver（处理旧坐标轨道，如把 melon v3.5.1 坐标转到 melonv4.0）：
  --liftover-chain CHAIN   UCSC 链文件（源→目标）；需可选 pyliftover。
                           区间两端分别 lift，要求落到同一目标染色体；不可
                           映射的区间被丢弃并计数。lift 在改名/筛选之前进行。
  说明：甜瓜不同组装版本间的 chain 文件多需自行生成（minimap2 + 链化工具 /
        nf-LO 等）或向数据发布方索取；本入口只负责“有 chain 就用”。

染色体命名对齐（必须与基因组 FASTA 一致）：
  --strip-chr           去掉前缀 "chr"（chr1 -> 1）
  --chrom-map "chr1:1,chr2:2" 或 map.json   显式改名
  --chrom 1             只保留某条染色体

用法：
    # narrowPeak -> BED3（按信号过滤 + 去 chr + 只留 1 号）
    python scripts/tracks_to_bed.py --input MYB_dapseq.narrowPeak \
        --out assets/tracks/MYB_dapseq.bed --min-signal 2.0 --strip-chr --chrom 1

    # narrowPeak -> BED4（保留 col7 信号值，供回归训练）
    python scripts/tracks_to_bed.py --input MYB_dapseq.narrowPeak \
        --out assets/tracks/MYB_dapseq.signal.bed --keep-signal

    # bedGraph -> BED（top-10% 且合并 50bp 内相邻峰）
    python scripts/tracks_to_bed.py --input flesh_atac.bedgraph \
        --out assets/tracks/flesh_atac_peaks.bed --quantile 0.90 --merge-gap 50

    # 旧组装坐标 liftOver 到 melonv4.0
    python scripts/tracks_to_bed.py --input old_v351.narrowPeak \
        --out assets/tracks/lifted.bed --liftover-chain v351_to_v4.over.chain \
        --min-signal 2.0 --chrom 1

    # 批量目录
    python scripts/tracks_to_bed.py --input-dir raw_tracks/ --out-dir assets/tracks/ \
        --strip-chr --chrom 1

    # 离线冒烟测试
    python scripts/tracks_to_bed.py --smoke-test
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


# ------------------------------------------------------------------ #
def _open(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def detect_format(path: str) -> str:
    p = str(path).lower()
    if p.endswith(".gz"):
        p = p[:-3]
    if p.endswith(".narrowpeak"):
        return "narrowpeak"
    if p.endswith(".broadpeak"):
        return "broadpeak"
    if p.endswith((".bb", ".bigbed")):
        return "bigbed"
    if p.endswith((".bedgraph", ".bg", ".bdg")):
        return "bedgraph"
    if p.endswith(".bed"):
        return "bed"
    raise SystemExit(f"[错误] 无法从扩展名识别格式：{path}（请用 --format 指定）")


def parse_chrom_map(spec: str | None) -> dict:
    if not spec:
        return {}
    if Path(spec).exists():
        return json.loads(Path(spec).read_text())
    out = {}
    for pair in spec.split(","):
        if ":" in pair:
            a, b = pair.split(":", 1)
            out[a.strip()] = b.strip()
    return out


def remap_chrom(chrom: str, chrom_map: dict, strip_chr: bool) -> str:
    if chrom in chrom_map:
        return chrom_map[chrom]
    if strip_chr and chrom.lower().startswith("chr"):
        return chrom[3:]
    return chrom


# ------------------------------------------------------------------ #
# 各格式 -> 区间列表 [(chrom, start, end, val_or_None), ...]（0-based 半开）
# ------------------------------------------------------------------ #
def from_peak(path, is_narrow, min_signal, min_neglogq, keep_signal, signal_col):
    out = []
    vcol = (signal_col - 1) if signal_col else 6      # 默认 col7=signalValue
    with _open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            try:
                chrom, s, e = p[0], int(p[1]), int(p[2])
            except ValueError:
                continue
            if min_signal is not None and len(p) >= 7:
                try:
                    if float(p[6]) < min_signal:
                        continue
                except ValueError:
                    pass
            if is_narrow and min_neglogq is not None and len(p) >= 9:
                try:
                    if float(p[8]) < min_neglogq:
                        continue
                except ValueError:
                    pass
            val = None
            if keep_signal and len(p) > vcol:
                try:
                    val = float(p[vcol])
                except ValueError:
                    val = None
            if s < e:
                out.append((chrom, s, e, val))
    return out


def from_bed(path, keep_signal, signal_col):
    out = []
    vcol = (signal_col - 1) if signal_col else 4      # 默认 col5=score
    with _open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            p = line.split()
            if len(p) < 3:
                continue
            try:
                chrom, s, e = p[0], int(p[1]), int(p[2])
            except ValueError:
                continue
            val = None
            if keep_signal and len(p) > vcol:
                try:
                    val = float(p[vcol])
                except ValueError:
                    val = None
            if s < e:
                out.append((chrom, s, e, val))
    return out


def from_bigbed(path, keep_signal):
    try:
        import pyBigWig
    except ImportError:
        raise SystemExit("[错误] 读取 bigBed 需要 pyBigWig：pip install pyBigWig")
    bb = pyBigWig.open(path)
    out = []
    for chrom, clen in bb.chroms().items():
        for s, e, *rest in (bb.entries(chrom, 0, int(clen)) or []):
            val = None
            if keep_signal and rest:
                tok = str(rest[0]).split("\t")
                for t in tok:
                    try:
                        val = float(t)
                        break
                    except ValueError:
                        continue
            if s < e:
                out.append((chrom, int(s), int(e), val))
    bb.close()
    return out


def from_bedgraph(path, threshold, quantile, merge_gap, keep_signal):
    rows = []
    with _open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            p = line.split()
            if len(p) < 4:
                continue
            try:
                rows.append((p[0], int(p[1]), int(p[2]), float(p[3])))
            except ValueError:
                continue
    if not rows:
        return []
    if threshold is None:
        q = quantile if quantile is not None else 0.90
        vals = sorted(r[3] for r in rows)
        threshold = vals[min(len(vals) - 1, int(q * len(vals)))]
        print(f"  [bedGraph] 分位数 {q} -> 阈值 {threshold:.4g}")
    passed = [(c, s, e, v) for c, s, e, v in rows if v > threshold]
    passed.sort()
    merged = []
    for c, s, e, v in passed:
        if merged and merged[-1][0] == c and s - merged[-1][2] <= merge_gap:
            pc, ps, pe, pv = merged[-1]
            merged[-1] = (pc, ps, max(pe, e), max(pv, v))      # 合并区间取最大信号
        else:
            merged.append((c, s, e, v))
    return [(c, s, e, (v if keep_signal else None)) for c, s, e, v in merged]


# ------------------------------------------------------------------ #
# liftOver（跨组装）
# ------------------------------------------------------------------ #
def liftover_intervals(intervals, chain_path: str):
    try:
        from pyliftover import LiftOver
    except ImportError:
        raise SystemExit(
            "[错误] liftOver 需要 pyliftover：pip install pyliftover\n"
            "        并提供 UCSC chain 文件（源组装->目标组装）。")
    lo = LiftOver(chain_path)
    out, dropped = [], 0
    for chrom, s, e, val in intervals:
        a = lo.convert_coordinate(chrom, s)
        b = lo.convert_coordinate(chrom, max(s, e - 1))
        if not a or not b:
            dropped += 1
            continue
        ca, pa = a[0][0], a[0][1]
        cb, pb = b[0][0], b[0][1]
        if ca != cb:
            dropped += 1
            continue
        ns, ne = min(pa, pb), max(pa, pb) + 1
        if ns < ne:
            out.append((ca, ns, ne, val))
        else:
            dropped += 1
    print(f"  [liftOver] 映射成功 {len(out)} / 丢弃 {dropped}（链文件 {Path(chain_path).name}）")
    return out


# ------------------------------------------------------------------ #
def write_bed(intervals, out, chrom_map, strip_chr, keep_chrom, sort, keep_signal):
    rows = []
    for chrom, s, e, val in intervals:
        c = remap_chrom(chrom, chrom_map, strip_chr)
        if keep_chrom is not None and c != keep_chrom:
            continue
        rows.append((c, s, e, val))
    if sort:
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
    op = Path(out)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w") as f:
        for c, s, e, val in rows:
            if keep_signal:
                f.write(f"{c}\t{s}\t{e}\t{val if val is not None else 0.0}\n")
            else:
                f.write(f"{c}\t{s}\t{e}\n")
    return len(rows)


def convert_one(inp, out, fmt, args) -> int:
    fmt = fmt if fmt != "auto" else detect_format(inp)
    if fmt == "narrowpeak":
        ivs = from_peak(inp, True, args.min_signal, args.min_neglogq,
                        args.keep_signal, args.signal_col)
    elif fmt == "broadpeak":
        ivs = from_peak(inp, False, args.min_signal, None,
                        args.keep_signal, args.signal_col)
    elif fmt == "bigbed":
        ivs = from_bigbed(inp, args.keep_signal)
    elif fmt == "bedgraph":
        ivs = from_bedgraph(inp, args.threshold, args.quantile, args.merge_gap,
                            args.keep_signal)
    elif fmt == "bed":
        ivs = from_bed(inp, args.keep_signal, args.signal_col)
    else:
        raise SystemExit(f"[错误] 未知格式：{fmt}")
    if args.liftover_chain:
        ivs = liftover_intervals(ivs, args.liftover_chain)
    chrom_map = parse_chrom_map(args.chrom_map)
    n = write_bed(ivs, out, chrom_map, args.strip_chr, args.chrom,
                  not args.no_sort, args.keep_signal)
    tag = "BED4(含信号)" if args.keep_signal else "BED3"
    print(f"  {Path(inp).name}  [{fmt}]  -> {out}  （{n} 区间，{tag}）")
    return n


# ------------------------------------------------------------------ #
def run_smoke() -> None:
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="trk2bed_"))
    print(f"[冒烟测试] {tmp}")

    class A:
        min_signal = 2.0; min_neglogq = None; threshold = None
        quantile = None; merge_gap = 0; strip_chr = True
        chrom_map = None; chrom = "1"; no_sort = False
        keep_signal = False; signal_col = None; liftover_chain = None

    # 1) narrowPeak -> BED3（信号过滤 + 去 chr + 只留 1）
    npk = tmp / "demo.narrowPeak"
    with open(npk, "w") as f:
        f.write("chr1\t100\t250\tp1\t800\t.\t3.5\t12.0\t8.0\t75\n")
        f.write("chr1\t400\t460\tp2\t200\t.\t1.0\t2.0\t1.0\t30\n")
        f.write("chr2\t500\t600\tp3\t900\t.\t5.0\t20.0\t15.0\t50\n")
    a = A(); out1 = tmp / "demo.bed"
    print("--- narrowPeak -> BED3 ---")
    convert_one(str(npk), str(out1), "auto", a)
    assert out1.read_text().strip() == "1\t100\t250"

    # 2) narrowPeak -> BED4（保留信号，供回归）
    b = A(); b.min_signal = None; b.keep_signal = True; b.chrom = "1"
    out2 = tmp / "demo.signal.bed"
    print("--- narrowPeak -> BED4（keep-signal）---")
    convert_one(str(npk), str(out2), "auto", b)
    print("  内容:\n" + out2.read_text().rstrip())
    assert "\t3.5" in out2.read_text(), "BED4 应保留 col7 信号 3.5"

    # 3) bedGraph -> BED（分位数 + 合并）
    bg = tmp / "demo.bedgraph"
    with open(bg, "w") as f:
        for i, v in enumerate([0.1, 0.2, 5.0, 5.2, 0.1, 6.0, 0.0]):
            f.write(f"chr1\t{i*10}\t{i*10+10}\t{v}\n")
    c = A(); c.min_signal = None; c.quantile = 0.6; c.merge_gap = 10
    out3 = tmp / "demo_bg.bed"
    print("--- bedGraph -> BED（quantile + merge）---")
    n3 = convert_one(str(bg), str(out3), "auto", c)
    assert n3 >= 1

    # 4) liftOver（若装了 pyliftover）：构造最小 chain，把 chr1 偏移到 "1"
    try:
        import pyliftover  # noqa
        chain = tmp / "demo.over.chain"
        # chr1:0-1000(+) -> 1:0-1000(+)，单块无 gap
        chain.write_text("chain 1000 chr1 1000 + 0 1000 1 1000 + 0 1000 1\n1000\n\n")
        d = A(); d.min_signal = None; d.strip_chr = False; d.chrom = None
        d.liftover_chain = str(chain)
        out4 = tmp / "demo_lifted.bed"
        print("--- liftOver（chr1 -> 1）---")
        convert_one(str(npk), str(out4), "auto", d)
        txt = out4.read_text()
        print("  内容:\n" + txt.rstrip())
        assert txt.startswith("1\t100\t250"), "liftOver 后 chrom 应为 '1'"
        print("  liftOver 路径 OK")
    except ImportError:
        print("--- liftOver 跳过（未安装 pyliftover）---")

    print("\n[冒烟测试] 通过：narrowPeak/bedGraph -> BED3/BED4，含改名/筛选/合并/liftOver。")
    print("  BED3 -> tracks.json type=bed（二值）；BED4 -> 回归训练 (--mode regression)。")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input")
    ap.add_argument("--out")
    ap.add_argument("--input-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--format", default="auto",
                    choices=["auto", "narrowpeak", "broadpeak", "bigbed", "bedgraph", "bed"])
    ap.add_argument("--min-signal", type=float, default=None,
                    help="narrow/broadPeak 信号值(col7)下限")
    ap.add_argument("--min-neglogq", type=float, default=None,
                    help="narrowPeak -log10(q)(col9)下限")
    ap.add_argument("--threshold", type=float, default=None, help="bedGraph 绝对阈值")
    ap.add_argument("--quantile", type=float, default=None, help="bedGraph 分位数阈值(默认0.90)")
    ap.add_argument("--merge-gap", type=int, default=0, help="bedGraph 合并间隔<=该值(bp)")
    ap.add_argument("--keep-signal", action="store_true",
                    help="输出 BED4（第4列=信号值），供回归训练")
    ap.add_argument("--signal-col", type=int, default=None,
                    help="信号值取第几列(1-based)；narrowPeak 默认7，bed 默认5")
    ap.add_argument("--liftover-chain", default=None,
                    help="UCSC chain 文件，跨组装 liftOver（需 pyliftover）")
    ap.add_argument("--strip-chr", action="store_true")
    ap.add_argument("--chrom-map", default=None)
    ap.add_argument("--chrom", default=None)
    ap.add_argument("--no-sort", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        run_smoke()
        return

    if args.input_dir:
        if not args.out_dir:
            raise SystemExit("[错误] --input-dir 需配 --out-dir。")
        exts = (".narrowpeak", ".broadpeak", ".bb", ".bigbed",
                ".bedgraph", ".bg", ".bdg", ".bed")
        files = [p for p in Path(args.input_dir).iterdir()
                 if p.name.lower().replace(".gz", "").endswith(exts)]
        if not files:
            raise SystemExit(f"[错误] {args.input_dir} 下无可识别轨道文件。")
        print(f"[批量] {len(files)} 个文件 -> {args.out_dir}")
        total = 0
        for p in sorted(files):
            stem = p.name
            for suf in (".gz", ".narrowPeak", ".broadPeak", ".bb", ".bigBed",
                        ".bedgraph", ".bg", ".bdg", ".bed"):
                if stem.endswith(suf):
                    stem = stem[:-len(suf)]
            total += convert_one(str(p), str(Path(args.out_dir) / f"{stem}.bed"),
                                 args.format, args)
        print(f"[批量完成] 共 {total} 区间。")
        return

    if not (args.input and args.out):
        raise SystemExit("[错误] 需 --input 与 --out（或 --input-dir/--out-dir，或 --smoke-test）。")
    convert_one(args.input, args.out, args.format, args)
    if args.keep_signal:
        print("[完成] BED4 可作 tracks.json 的 path（配 train_annotation_tracks --mode regression）。")
    else:
        print("[完成] BED3 可直接填进 tracks.json 的 path（type=bed，二值）。")


if __name__ == "__main__":
    main()
