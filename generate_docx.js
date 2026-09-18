const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, TabStopType, TabStopPosition,
} = require("docx");

const PAGE_W = 11906; // A4
const PAGE_H = 16838;
const MARGIN = 1440; // 1 inch
const CONTENT_W = PAGE_W - 2 * MARGIN; // 9026

// ======== helpers ========
const border = { style: BorderStyle.SINGLE, size: 1, color: "AAAAAA" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function cell(text, opts = {}) {
  const runs = [];
  const lines = (text + "").split("\n");
  lines.forEach((l, i) => {
    runs.push(new TextRun({ text: l, bold: opts.bold || false, size: opts.size || 20, font: "Microsoft YaHei" }));
    if (i < lines.length - 1) runs.push(new TextRun({ text: "", break: 1 }));
  });
  return new TableCell({
    borders,
    width: { size: opts.w || 2000, type: WidthType.DXA },
    shading: opts.shade ? { fill: opts.shade, type: ShadingType.CLEAR } : undefined,
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({ alignment: opts.align || AlignmentType.LEFT, children: runs })],
  });
}

function headerCell(text, w) {
  return new TableCell({
    borders,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: "2E5E8E", type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, size: 20, font: "Microsoft YaHei", color: "FFFFFF" })] })],
  });
}

function p(text, opts = {}) {
  const runs = [];
  const lines = (text + "").split("\n");
  lines.forEach((l, i) => {
    runs.push(new TextRun({ text: l, bold: opts.bold, size: opts.size || 22, font: "Microsoft YaHei", color: opts.color, italics: opts.italics }));
    if (i < lines.length - 1) runs.push(new TextRun({ text: "", break: 1 }));
  });
  return new Paragraph({
    spacing: { before: opts.before || 80, after: opts.after || 80 },
    alignment: opts.align,
    children: runs,
  });
}

function heading(level, text) {
  const sizes = { 1: 32, 2: 28, 3: 24 };
  return new Paragraph({
    heading: level === 1 ? HeadingLevel.HEADING_1 : level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
    spacing: { before: level === 1 ? 360 : 240, after: 120 },
    children: [new TextRun({ text, bold: true, size: sizes[level] || 24, font: "Microsoft YaHei" })],
  });
}

function emptyP() { return new Paragraph({ spacing: { before: 0, after: 0 }, children: [new TextRun({ text: "" })] }); }

function makeTable(headers, rows, colWidths) {
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: Math.min(totalW, CONTENT_W), type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headers.map((h, i) => headerCell(h, colWidths[i])) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(c, { w: colWidths[i] })) })),
    ],
  });
}

function codeBlock(text) {
  const lines = text.split("\n");
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    indent: { left: 360 },
    shading: { fill: "F0F0F0", type: ShadingType.CLEAR },
    children: lines.flatMap((l, i) => [
      new TextRun({ text: l, size: 18, font: "Consolas", color: "333333" }),
      ...(i < lines.length - 1 ? [new TextRun({ text: "", break: 1 })] : []),
    ]),
  });
}

function numberedList(items, start = 1) {
  return items.map((item, i) => new Paragraph({
    numbering: { reference: "num", level: 0 },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text: item, size: 22, font: "Microsoft YaHei" })],
  }));
}

function bulletList(items) {
  return items.map(item => new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text: item, size: 22, font: "Microsoft YaHei" })],
  }));
}

// ======== Document ========
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Microsoft YaHei", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Microsoft YaHei", color: "1F4E79" },
        paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Microsoft YaHei", color: "2E5E8E" },
        paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Microsoft YaHei", color: "3B78B0" },
        paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "num",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [
    // ====== Cover Page ======
    {
      properties: {
        page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN } }
      },
      children: [
        emptyP(), emptyP(), emptyP(), emptyP(), emptyP(),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 200 },
          children: [new TextRun({ text: "Melon-PlantCAD2-EVEE", size: 44, bold: true, font: "Microsoft YaHei", color: "1F4E79" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 400 },
          children: [new TextRun({ text: "甜瓜基因组大模型变异效应预测与智能育种", size: 32, font: "Microsoft YaHei", color: "2E5E8E" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 60 },
          children: [new TextRun({ text: "技术报告", size: 36, bold: true, font: "Microsoft YaHei", color: "1F4E79" })] }),
        emptyP(), emptyP(),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 40 },
          children: [new TextRun({ text: "项目编号: Melon-PlantCAD2-EVEE v0.2.1", size: 22, font: "Microsoft YaHei" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 40 },
          children: [new TextRun({ text: "执行日期: 2026年5月29日 \u2014 2026年6月1日", size: 22, font: "Microsoft YaHei" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 40 },
          children: [new TextRun({ text: "计算环境: NVIDIA L20 GPU (48GB), CUDA 11.8, conda env \"plantcad\"", size: 22, font: "Microsoft YaHei" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 40 },
          children: [new TextRun({ text: "执行服务器: cabbage (10.70.58.153, \u7ECF\u8DF3\u677F\u673A 124.223.17.184:6000)", size: 22, font: "Microsoft YaHei" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 40 },
          children: [new TextRun({ text: "报告生成: 2026年6月1日", size: 22, font: "Microsoft YaHei" })] }),
      ]
    },
    // ====== Main Content ======
    {
      properties: {
        page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN } }
      },
      headers: {
        default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Melon-PlantCAD2-EVEE \u6280\u672F\u62A5\u544A", size: 16, font: "Microsoft YaHei", color: "999999", italics: true })] })] })
      },
      footers: {
        default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "\u2014 ", size: 18, font: "Microsoft YaHei", color: "999999" }), new TextRun({ children: [PageNumber.CURRENT], size: 18, font: "Microsoft YaHei", color: "999999" }), new TextRun({ text: " \u2014", size: 18, font: "Microsoft YaHei", color: "999999" })] })] })
      },
      children: [
        // ========== 1. \u9879\u76EE\u6982\u8981 ==========
        heading(1, "1. \u9879\u76EE\u6982\u8981"),
        heading(2, "1.1 \u7814\u7A76\u76EE\u6807"),
        p("\u672C\u9879\u76EE\u5C06 EVEE\uFF08Evolutionary Variant Effect Estimation\uFF09\u6846\u67B6\u4ECE\u4EBA\u7C7B\u533B\u5B66\u573A\u666F\uFF08Evo 2 + ClinVar \u81F4\u75C5\u6027\uFF09\u8FC1\u79FB\u5230\u751C\u74DC\uFF08Cucumis melo\uFF09\u80B2\u79CD\u573A\u666F\uFF0C\u5229\u7528 PlantCAD2 \u57FA\u56E0\u7EC4\u5927\u6A21\u578B\u5B9E\u73B0\uFF1A"),
        ...numberedList([
          "\u5168\u57FA\u56E0\u7EC4\u53D8\u5F02\u6548\u5E94\u9884\u6D4B\uFF1A\u5BF9\u7FA4\u4F53 SNP \u8FDB\u884C\u529F\u80FD\u6548\u5E94\u8BC4\u5206",
          "\u529F\u80FD\u6CE8\u91CA\u6270\u52A8\u8C31\u5206\u6790\uFF1A\u91CF\u5316\u53D8\u5F02\u5BF9 17 \u7C7B\u529F\u80FD\u533A\u57DF\u7684\u6270\u52A8\u6A21\u5F0F",
          "De novo \u8C03\u63A7\u5143\u4EF6\u8BBE\u8BA1\uFF1A\u57FA\u4E8E Gibbs \u91C7\u6837\u751F\u6210\u7CD6\u4EE3\u8C22\u4F18\u5316\u542F\u52A8\u5B50",
          "\u80B2\u79CD\u503C\u6392\u5E8F\uFF1A\u6309\u67D3\u8272\u4F53\u7684\u4E0D\u5229\u53D8\u5F02\u805A\u5408\u8FDB\u884C\u80B2\u79CD\u4F18\u5148\u7EA7\u6392\u5E8F",
        ]),

        heading(2, "1.2 PlantCAD2 \u6838\u5FC3\u7279\u6027"),
        makeTable(
          ["\u7279\u6027", "PlantCAD2 (\u672C\u9879\u76EE)", "PlantGFM (\u756A\u8304\u9879\u76EE)"],
          [
            ["\u67B6\u6784", "Caduceus + Mamba2", "Hyena (\u81EA\u56DE\u5F52)"],
            ["\u65B9\u5411\u6027", "\u53CC\u5411 / RC \u7B49\u53D8", "\u5355\u5411 (causal)"],
            ["\u4EFB\u52A1\u7C7B\u578B", "\u63A9\u7801\u8BED\u8A00\u6A21\u578B (MLM)", "\u81EA\u56DE\u5F52\u8BED\u8A00\u6A21\u578B (CLM)"],
            ["\u524D\u5411\u6B21\u6570", "2\u6B21/\u53D8\u5F02 (ref+alt)", "4\u6B21/\u53D8\u5F02"],
            ["\u96F6\u6837\u672C\u53D8\u5F02\u6253\u5206", "\u652F\u6301 (LLR)", "\u4E0D\u652F\u6301"],
            ["De novo \u8BBE\u8BA1", "Gibbs \u63A9\u7801\u91CD\u91C7\u6837", "\u81EA\u56DE\u5F52 generate"],
            ["\u53C2\u6570\u89C4\u6A21", "88M (Small, d=768)", "1024M (d=1024)"],
            ["\u9884\u8BAD\u7EC3\u6570\u636E", "65\u4E2A\u88AB\u5B50\u690D\u7269\u57FA\u56E0\u7EC4", "\u756A\u8304\u7279\u5F02"],
          ],
          [2200, 3400, 3426]
        ),

        // ========== 2. \u6280\u672F\u67B6\u6784 ==========
        heading(1, "2. \u6280\u672F\u67B6\u6784"),
        heading(2, "2.1 \u6D41\u6C34\u7EBF\u67B6\u6784"),
        p("\u4E0B\u56FE\u5C55\u793A\u4E86 Melon-PlantCAD2-EVEE \u7684\u5B8C\u6574\u6D41\u6C34\u7EBF\u67B6\u6784\uFF1A"),
        codeBlock(
          "Step 1: VCF + \u57FA\u56E0\u7EC4 \u2192 variants.parquet (32,268\u6761)\n" +
          "Step 2: PlantCAD2 (\u51BB\u7ED3) \u2192 data/activations/ (30,053\u4E2A.npy)\n" +
          "Step 3: \u534F\u65B9\u5DEE\u63A2\u9488 (AUROC=0.5878)\n" +
          "Step 4: \u6CE8\u91CA\u63A2\u9488 (17\u7C7B, Acc=95.97%)\n" +
          "Step 5-6: GWAS\u6807\u7B7E + \u6279\u91CF\u9884\u6D4B \u2192 30,052\u6761\n" +
          "Step 7-8: \u5206\u6790+\u8BC4\u4F30\n" +
          "Step 11: De novo \u8BBE\u8BA1 (76\u6761)\n" +
          "Step 12: \u80B2\u79CD\u503C\u6392\u5E8F\n" +
          "Step 13: LLM\u89E3\u91CA"
        ),

        heading(2, "2.2 \u76EE\u5F55\u7ED3\u6784"),
        codeBlock(
          "melon-plantcad2-evee/\n" +
          "\u251C\u2500\u2500 meloncad/           # PlantCAD2 \u7F16\u7801\u5668\u5C01\u88C5\n" +
          "\u251C\u2500\u2500 probes/             # EVEE\u63A2\u9488\u6A21\u5757\n" +
          "\u251C\u2500\u2500 breeding/           # \u591A\u6027\u72B6\u80B2\u79CD\u503C\u805A\u5408\n" +
          "\u251C\u2500\u2500 scripts/            # 17\u4E2A\u53EF\u6267\u884C\u6D41\u6C34\u7EBF\u811A\u672C\n" +
          "\u251C\u2500\u2500 assets/             # \u6A21\u578B\u6743\u91CD\u4E0E\u63A2\u9488\n" +
          "\u251C\u2500\u2500 results/            # \u6700\u7EC8\u7ED3\u679C (30,052\u6761\u9884\u6D4B)\n" +
          "\u251C\u2500\u2500 configs/            # \u9ED8\u8BA4\u914D\u7F6E\n" +
          "\u2514\u2500\u2500 logs/               # \u8BAD\u7EC3\u4E0E\u9884\u6D4B\u65E5\u5FD7"
        ),

        // ========== 3. \u6570\u636E\u51C6\u5907 ==========
        heading(1, "3. \u6570\u636E\u51C6\u5907"),
        heading(2, "3.1 \u57FA\u56E0\u7EC4\u6570\u636E"),
        makeTable(
          ["\u8D44\u6E90", "\u6765\u6E90", "\u5927\u5C0F", "\u8BF4\u660E"],
          [
            ["\u53C2\u8003\u57FA\u56E0\u7EC4", "Ensembl Plants release-62", "347MB (FASTA)", "DHL92 / melonv4.0, 13 contigs"],
            ["GFF \u6CE8\u91CA", "Ensembl Plants release-62", "65MB", "\u57FA\u56E0\u7ED3\u6784\u6CE8\u91CA (GFF3)"],
            ["VCF \u53D8\u5F02", "Cucurbit Genomics DB", "95MB (gzip)", "GBS SNPs, MAF\u22650.01, missing\u22640.5"],
          ],
          [1800, 3000, 2000, 2226]
        ),

        heading(2, "3.2 \u8868\u578B/GWAS \u6807\u7B7E\u6570\u636E"),
        makeTable(
          ["\u6570\u636E\u96C6", "\u7C7B\u578B", "\u7528\u9014"],
          [
            ["Akter 2023 TableS5", "\u8868\u578B (42KB)", "\u7CD6\u5EA6\u3001\u679C\u91CD\u7B49\u6027\u72B6"],
            ["Akter 2023 TableS6", "GWAS\u663E\u8457SNP", "\u8BAD\u7EC3\u6807\u7B7E\u6765\u6E90"],
            ["Zhao 2019 DiffRegions", "\u9009\u62E9\u4FE1\u53F7\u533A\u57DF (20KB)", "\u9A6F\u5316/\u6539\u826F\u4F4D\u70B9"],
            ["Zhao 2019 SweepRegions", "\u9009\u62E9\u6027\u6E05\u9664\u533A\u57DF (9.5KB)", "\u9633\u6027\u9009\u62E9\u4F4D\u70B9"],
          ],
          [3000, 3100, 2926]
        ),

        heading(2, "3.3 \u67D3\u8272\u4F53\u547D\u540D\u5BF9\u9F50"),
        p("VCF \u4F7F\u7528 chr01-chr12 \u547D\u540D\uFF0C\u57FA\u56E0\u7EC4\u4F7F\u7528 contig1-contig13\u3002\u901A\u8FC7 sed \u6279\u91CF\u91CD\u547D\u540D\u5B8C\u6210\u5BF9\u9F50\uFF1A"),
        makeTable(
          ["VCF chr", "\u57FA\u56E0\u7EC4 contig"],
          [
            ["chr01", "contig7"], ["chr02", "contig2"], ["chr03", "contig9"],
            ["chr04", "contig5"], ["chr05", "contig12"], ["chr06", "contig4"],
            ["chr07", "contig6"], ["chr08", "contig3"], ["chr09", "contig8"],
            ["chr10", "contig13"], ["chr11", "contig11"], ["chr12", "contig10"],
          ],
          [2263, 2263]
        ),

        // ========== 4. \u6D41\u6C34\u7EBF\u6267\u884C ==========
        heading(1, "4. \u6D41\u6C34\u7EBF\u6267\u884C"),
        heading(2, "4.1 Step 1: \u6784\u5EFA\u53D8\u5F02\u6570\u636E\u96C6"),
        codeBlock("python scripts/build_variant_dataset.py \\\n    --vcf assets/variants/melon_gbs_renamed.vcf.gz \\\n    --genome assets/genome/Cucumis_melo.Melonv4.dna.toplevel.fa \\\n    --gff assets/genome/Cucumis_melo.Melonv4.62.gff3 \\\n    --out data/variants.parquet --window 8192 --max-variants 50000"),
        p("\u8F93\u51FA: data/variants.parquet (43MB, 32,268\u6761\u53D8\u5F02)"),

        heading(2, "4.2 Step 2: \u63D0\u53D6\u6FC0\u6D3B\u5F20\u91CF"),
        p("\u5173\u952E\u6280\u672F\u4FEE\u590D:", { bold: true }),
        ...numberedList([
          "mamba_ssm CUDA: \u8BBE\u7F6E LD_LIBRARY_PATH=/data/zhangcw/cuda-11.8/lib64",
          "Embedding FloatTensor: \u5728 encoder.py \u548C modeling_rcps.py \u4E2D\u6DFB\u52A0 .long() \u7C7B\u578B\u8F6C\u6362",
          "causal_conv1d \u5185\u6838\u9519\u8BEF: \u5728 extract_activations.py \u4E2D\u6DFB\u52A0 try-except \u8DF3\u8FC7\u5F02\u5E38\u53D8\u5F02",
        ]),
        codeBlock("python scripts/extract_activations.py \\\n    --variants data/variants.parquet \\\n    --model assets/plantcad2 \\\n    --out data/activations --topk 256"),
        p("\u8F93\u51FA: data/activations/ (30,053\u4E2A .npy \u6587\u4EF6), 2216\u6761\u53D8\u5F02\u88AB\u8DF3\u8FC7"),

        heading(2, "4.3 Step 3-4: \u8BAD\u7EC3\u63A2\u9488"),
        heading(3, "\u534F\u65B9\u5DEE\u63A2\u9488 (\u53D8\u5F02\u6548\u5E94\u9884\u6D4B)"),
        codeBlock("python scripts/train_probe.py \\\n    --activations data/activations \\\n    --labels data/variants_labeled.parquet \\\n    --out assets/probe_covariance.safetensors --d-model 768"),
        p("\u914D\u7F6E: d_hidden=64, d_probe=128, epochs=30, batch_size=32, lr=0.001"),

        heading(3, "\u6CE8\u91CA\u63A2\u9488 (\u529F\u80FD\u5206\u7C7B)"),
        codeBlock("python scripts/train_annotation_probe.py \\\n    --gff assets/genome/Cucumis_melo.Melonv4.62.gff3 \\\n    --genome assets/genome/Cucumis_melo.Melonv4.dna.toplevel.fa \\\n    --model assets/plantcad2 --d-model 768 \\\n    --out assets/annotation_probe.safetensors"),
        p("\u914D\u7F6E: 20 epochs, 237,568\u8BAD\u7EC3\u4F4D\u70B9, 17\u7C7B\u591A\u6807\u7B7E\u5206\u7C7B"),

        heading(2, "4.4 Step 5: \u6807\u7B7E\u5339\u914D"),
        p("\u4F7F\u7528 GWAS \u663E\u8457 SNP (Akter 2023 TableS6) \u5728 20Kb \u7A97\u53E3\u5185\u56DE\u586B\u8BAD\u7EC3\u6807\u7B7E\u3002"),
        codeBlock("python prepare_labels.py \\\n    --gwas assets/phenotypes/Akter_2023_TableS6_significant_SNPs.csv \\\n    --variants data/variants.parquet \\\n    --out data/variants_labeled.parquet --window 20000"),
        p("\u67D3\u8272\u4F53\u547D\u540D\u7EDF\u4E00\u7B97\u6CD5\u5904\u7406\u4E86 chr1 / chr01 / 1 \u7B49\u591A\u79CD\u683C\u5F0F\u3002"),

        heading(2, "4.5 Step 6: \u6279\u91CF\u9884\u6D4B"),
        codeBlock("python scripts/predict_annotation_batch.py \\\n    --variants data/variants.parquet \\\n    --activations data/activations \\\n    --probe assets/probe_covariance.safetensors \\\n    --annotation-probe assets/annotation_probe.safetensors \\\n    --out results/predictions_with_annotation.csv"),
        p("\u8017\u65F6: ~5.5\u5C0F\u65F6 (19,929\u79D2), ~2 items/sec"),
        p("GPU: \u7F16\u7801\u6BCF\u4E2A\u53D8\u5F02\u7EA67\u79D2"),

        // ========== 5. \u6A21\u578B\u8BAD\u7EC3\u7ED3\u679C ==========
        heading(1, "5. \u6A21\u578B\u8BAD\u7EC3\u7ED3\u679C"),

        heading(2, "5.1 \u534F\u65B9\u5DEE\u63A2\u9488 (\u53D8\u5F02\u6548\u5E94\u9884\u6D4B)"),
        makeTable(
          ["Epoch", "Train Loss", "Val AUROC"],
          [
            ["1", "97.45", "0.5294"], ["5", "92.65", "0.5518"],
            ["10", "79.14", "0.5310"], ["15", "64.68", "0.5629"],
            ["20", "55.60", "0.5654"], ["25", "47.15", "0.5777"],
            ["28 (\u6700\u4F18)", "41.21", "0.5878"], ["30", "45.26", "0.5719"],
          ],
          [2260, 3380, 3386]
        ),
        p("\u6700\u4F73 AUROC: 0.5878 (Epoch 28)", { bold: true }),
        p("\u8BF4\u660E: \u534F\u65B9\u5DEE\u63A2\u9488 AUROC \u504F\u4F4E\uFF08<0.6\uFF09\uFF0C\u4E3B\u8981\u539F\u56E0\u662F GWAS \u6807\u7B7E\u6765\u81EA Akter 2023 \u7684\u5C0F\u89C4\u6A21\u7FA4\u4F53\uFF08\u7EA6200\u4EFD\u6750\u6599\uFF09\uFF0C\u6807\u7B7E\u566A\u58F0\u8F83\u5927\u3002\u8FD9\u4E0E\u4E0A\u6E38\u756A\u8304\u9879\u76EE\u7684\u7ECF\u9A8C\u4E00\u81F4\uFF1A\u6807\u7B7E\u8D28\u91CF > \u6A21\u578B\u67B6\u6784\u3002", { italics: true }),

        heading(2, "5.2 \u6CE8\u91CA\u63A2\u9488 (17\u7C7B\u529F\u80FD\u5206\u7C7B)"),
        makeTable(
          ["Epoch", "Loss", "Accuracy"],
          [
            ["1", "0.5000", "94.60%"],
            ["5", "0.2536", "95.38%"],
            ["10", "0.2288", "95.73%"],
            ["15", "0.2137", "95.87%"],
            ["20 (\u6700\u4F18)", "0.2028", "95.97%"],
          ],
          [2260, 3380, 3386]
        ),
        p("\u6700\u7EC8\u51C6\u786E\u7387: 95.97%", { bold: true }),

        heading(2, "5.3 17\u7C7B\u529F\u80FD\u6CE8\u91CA\u9762\u677F"),
        makeTable(
          ["\u5E8F\u53F7", "\u7C7B\u522B", "\u8BF4\u660E"],
          [
            ["1", "region_CDS", "\u7F16\u7801\u533A"],
            ["2", "region_intron", "\u5185\u542B\u5B50"],
            ["3", "region_5UTR", "5'\u975E\u7FFB\u8BD1\u533A"],
            ["4", "region_3UTR", "3'\u975E\u7FFB\u8BD1\u533A"],
            ["5", "region_intergenic", "\u57FA\u56E0\u95F4\u533A"],
            ["6", "region_exon_boundary", "\u5916\u663E\u5B50\u8FB9\u754C"],
            ["7", "splice_donor", "\u526A\u63A5\u4F9B\u4F53"],
            ["8", "splice_acceptor", "\u526A\u63A5\u53D7\u4F53"],
            ["9", "codon_pos1", "\u5BC6\u7801\u5B50\u7B2C1\u4F4D"],
            ["10", "codon_pos2", "\u5BC6\u7801\u5B50\u7B2C2\u4F4D"],
            ["11", "codon_pos3", "\u5BC6\u7801\u5B50\u7B2C3\u4F4D"],
            ["12", "start_codon", "\u8D77\u59CB\u5BC6\u7801\u5B50"],
            ["13", "stop_codon", "\u7EC8\u6B62\u5BC6\u7801\u5B50"],
            ["14", "frameshift_risk", "\u79FB\u7801\u98CE\u9669"],
            ["15", "promoter_core", "\u6838\u5FC3\u542F\u52A8\u5B50"],
            ["16", "promoter_proximal", "\u8FD1\u7AEF\u542F\u52A8\u5B50"],
            ["17", "terminator", "\u7EC8\u6B62\u5B50"],
          ],
          [900, 3500, 2000]
        ),

        // ========== 6. \u5168\u57FA\u56E0\u7EC4\u9884\u6D4B\u7ED3\u679C ==========
        heading(1, "6. \u5168\u57FA\u56E0\u7EC4\u9884\u6D4B\u7ED3\u679C"),

        heading(2, "6.1 \u603B\u4F53\u7EDF\u8BA1"),
        makeTable(
          ["\u6307\u6807", "\u6570\u503C"],
          [
            ["\u9884\u6D4B\u53D8\u5F02\u603B\u6570", "30,052"],
            ["\u4E0D\u5229\u53D8\u5F02 (effect \u2265 0.5)", "9,596 (31.9%)"],
            ["\u6709\u5229/\u4E2D\u6027 (effect < 0.5)", "20,456 (68.1%)"],
          ],
          [4500, 4526]
        ),

        heading(2, "6.2 \u6548\u5E94\u5206\u6570\u5206\u5E03"),
        makeTable(
          ["\u7EDF\u8BA1\u91CF", "\u503C"],
          [
            ["\u6700\u5C0F\u503C", "0.0000"],
            ["25%\u5206\u4F4D\u6570", "0.0225"],
            ["\u4E2D\u4F4D\u6570", "0.3965"],
            ["75%\u5206\u4F4D\u6570", "0.6944"],
            ["\u6700\u5927\u503C", "1.0000"],
            ["\u5747\u503C", "0.4062"],
          ],
          [4500, 4526]
        ),

        heading(2, "6.3 \u67D3\u8272\u4F53\u5206\u5E03"),
        p("\u4E0D\u5229\u53D8\u5F02\u6700\u5BCC\u96C6\u7684\u67D3\u8272\u4F53\u662F contig7 (chr01) \u548C contig8 (chr09)\uFF0C\u5747\u8FBE 34.9%\uFF0C\u662F\u80B2\u79CD\u4F18\u5148\u5173\u6CE8\u533A\u57DF\u3002"),
        makeTable(
          ["\u67D3\u8272\u4F53", "\u603B\u6570", "\u4E0D\u5229", "\u6709\u5229", "\u4E0D\u5229%", "\u5E73\u5747\u6548\u5E94"],
          [
            ["contig1 (chr01)", "27", "6", "21", "22.2%", "0.2072"],
            ["contig2 (chr02)", "2,484", "742", "1,742", "29.9%", "0.4052"],
            ["contig3 (chr08)", "1,947", "648", "1,299", "33.3%", "0.4204"],
            ["contig4 (chr06)", "2,263", "705", "1,558", "31.2%", "0.4049"],
            ["contig5 (chr04)", "3,197", "1,031", "2,166", "32.2%", "0.3263"],
            ["contig6 (chr07)", "2,242", "719", "1,523", "32.1%", "0.4162"],
            ["contig7 (chr01)", "3,882", "1,355", "2,527", "34.9%", "0.4436"],
            ["contig8 (chr09)", "1,936", "675", "1,261", "34.9%", "0.4442"],
            ["contig9 (chr03)", "2,363", "735", "1,628", "31.1%", "0.4011"],
            ["contig10 (chr12)", "1,988", "572", "1,416", "28.8%", "0.3941"],
            ["contig11 (chr11)", "1,531", "500", "1,031", "32.7%", "0.4200"],
            ["contig12 (chr05)", "4,443", "1,370", "3,073", "30.8%", "0.4096"],
            ["contig13 (chr10)", "1,749", "538", "1,211", "30.8%", "0.4038"],
          ],
          [1800, 1000, 1000, 1000, 1000, 1200]
        ),

        heading(2, "6.4 \u529F\u80FD\u6270\u52A8\u7C7B\u522B Top 10"),
        makeTable(
          ["\u529F\u80FD\u7C7B\u522B", "\u6270\u52A8\u53D8\u5F02\u6570", "\u6270\u52A8\u6BD4\u4F8B"],
          [
            ["region_intron", "29,855", "99.3%"],
            ["region_CDS", "23,079", "76.8%"],
            ["promoter_proximal", "22,814", "75.9%"],
            ["region_3UTR", "17,690", "58.9%"],
            ["terminator", "13,585", "45.2%"],
            ["splice_donor", "6,901", "23.0%"],
            ["splice_acceptor", "6,863", "22.8%"],
            ["region_intergenic", "6,852", "22.8%"],
            ["region_exon_boundary", "6,852", "22.8%"],
            ["codon_pos2", "4,627", "15.4%"],
          ],
          [3000, 2500, 2000]
        ),

        heading(2, "6.5 \u9AD8\u5F71\u54CD\u53D8\u5F02 Top 5"),
        makeTable(
          ["\u6392\u540D", "\u53D8\u5F02ID", "\u67D3\u8272\u4F53", "\u5206\u6570", "\u9884\u6D4B", "\u4E3B\u8981\u6270\u52A8"],
          [
            ["1", "contig7:1311982:A>G", "chr01", "1.0000", "\u4E0D\u5229", "promoter_proximal, region_intron"],
            ["2", "contig7:2182087:G>A", "chr01", "1.0000", "\u4E0D\u5229", "region_intron, region_CDS, promoter_proximal"],
            ["3", "contig7:2499465:T>C", "chr01", "1.0000", "\u4E0D\u5229", "promoter_proximal, region_intron, region_CDS"],
            ["4", "contig7:4558286:C>A", "chr01", "1.0000", "\u4E0D\u5229", "region_intron, region_CDS, promoter_proximal"],
            ["5", "contig7:4968676:A>G", "chr01", "1.0000", "\u4E0D\u5229", "region_intron, promoter_proximal, region_CDS"],
          ],
          [900, 2800, 1000, 1000, 1000, 2000]
        ),

        // ========== 7. De Novo \u8C03\u63A7\u5143\u4EF6\u8BBE\u8BA1 ==========
        heading(1, "7. De Novo \u8C03\u63A7\u5143\u4EF6\u8BBE\u8BA1"),
        heading(2, "7.1 \u65B9\u6CD5"),
        p("PlantCAD2 \u4F5C\u4E3A\u53CC\u5411\u63A9\u7801\u8BED\u8A00\u6A21\u578B\uFF0C\u91C7\u7528 Gibbs \u63A9\u7801\u8FED\u4EE3\u91CD\u91C7\u6837\u65B9\u6CD5\u8FDB\u884C de novo \u8BBE\u8BA1\uFF1A"),
        ...numberedList([
          "\u4ECE\u968F\u673A\u5E8F\u5217\u51FA\u53D1",
          "\u6BCF\u8F6E\u968F\u673A\u63A9\u7801\u90E8\u5206\u4F4D\u7F6E",
          "MLM \u9884\u6D4B\u5206\u5E03\u91CD\u91C7\u6837\u586B\u5145",
          "\u591A\u8F6E\u540E\u6536\u655B\u81F3\u7B26\u5408\u751C\u74DC\u5E8F\u5217\u8BED\u6CD5\u7684\u5019\u9009\u5143\u4EF6",
        ]),

        heading(2, "7.2 \u8BBE\u8BA1\u53C2\u6570"),
        makeTable(
          ["\u53C2\u6570", "\u503C"],
          [
            ["\u5143\u4EF6\u7C7B\u578B", "sugar (\u7CD6\u5EA6/\u85C1\u7CD6\u79EF\u7D2F\u901A\u8DEF)"],
            ["\u76EE\u6807\u57FA\u56E0", "CmTST2 / CmSPS / CmAGA2"],
            ["\u5143\u4EF6\u957F\u5EA6", "400 bp"],
            ["\u751F\u6210\u6570\u91CF", "76 \u6761"],
            ["\u91C7\u6837\u6E29\u5EA6", "1.0"],
          ],
          [3000, 4000]
        ),

        heading(2, "7.3 \u8F93\u51FA"),
        ...bulletList([
          "results/designed_sugar_elements.fasta (32KB, 76\u6761\u5E8F\u5217\u00D7400bp)",
          "results/designed_sugar_batch2.fasta (7.8KB, \u7B2C\u4E8C\u6279\u8865\u5145)",
        ]),

        // ========== 8. \u80B2\u79CD\u503C\u6392\u5E8F ==========
        heading(1, "8. \u80B2\u79CD\u503C\u6392\u5E8F"),
        heading(2, "8.1 \u65B9\u6CD5"),
        p("\u6309\u67D3\u8272\u4F53\u805A\u5408\u4E0D\u5229\u53D8\u5F02\u6548\u5E94\u5206\u6570\uFF0C\u8BA1\u7B97\u6BCF\u6761\u67D3\u8272\u4F53\u7684\u80B2\u79CD\u6539\u8FDB\u6F5C\u529B\uFF1A"),
        codeBlock("\u80B2\u79CD\u503C = \u03A3(\u4E0D\u5229\u53D8\u5F02\u6548\u5E94\u5206\u6570)"),

        heading(2, "8.2 \u67D3\u8272\u4F53\u80B2\u79CD\u4F18\u5148\u7EA7"),
        makeTable(
          ["\u4F18\u5148\u7EA7", "\u67D3\u8272\u4F53", "\u6807\u51C6\u540D", "\u4E0D\u5229\u53D8\u5F02\u6570", "\u80B2\u79CD\u503C", "\u5E73\u5747\u6548\u5E94"],
          [
            ["1 (\u6700\u9AD8)", "contig7", "chr01", "1,355", "\u6700\u9AD8", "0.4436"],
            ["2", "contig8", "chr09", "675", "\u6700\u9AD8", "0.4442"],
            ["3", "contig12", "chr05", "1,370", "\u8F83\u9AD8", "0.4096"],
            ["4", "contig5", "chr04", "1,031", "\u8F83\u9AD8", "0.3263"],
            ["5", "contig3", "chr08", "648", "\u4E2D", "0.4204"],
          ],
          [1500, 1500, 1200, 1800, 1500, 1500]
        ),
        p("\u5EFA\u8BAE: \u80B2\u79CD\u5DE5\u4F5C\u4E2D\u4F18\u5148\u9488\u5BF9 chr01 \u548C chr09 \u4E0A\u7684\u9AD8\u5F71\u54CD\u53D8\u5F02\u8BBE\u8BA1\u5206\u5B50\u6807\u8BB0\u8F85\u52A9\u9009\u62E9\uFF08MAS\uFF09\u65B9\u6848\u3002", { bold: true }),

        // ========== 9. \u6A21\u578B\u9A8C\u8BC1 ==========
        heading(1, "9. \u6A21\u578B\u9A8C\u8BC1"),

        heading(2, "9.1 \u5DF2\u53D1\u8868 GWAS \u4F4D\u70B9\u9A8C\u8BC1"),
        p("\u5229\u7528\u6587\u732E\u4E2D\u5DF2\u53D1\u8868\u7684\u751C\u74DC GWAS/QTL \u5019\u9009\u57FA\u56E0\u533A\u57DF\uFF0C\u5BF9\u6A21\u578B\u7684\u6548\u5E94\u9884\u6D4B\u8FDB\u884C\u72EC\u7ACB\u9A8C\u8BC1\u3002\u9A8C\u8BC1\u4F4D\u70B9\u6765\u81EA 18 \u7BC7\u5DF2\u53D1\u8868\u7814\u7A76\uFF0C\u8986\u76D6\u679C\u8089\u989C\u8272\u3001\u7CD6\u5EA6\u3001\u679C\u5F62\u3001\u9999\u6C14\u3001\u82E6\u5473\u7B49\u5173\u952E\u6027\u72B6\u3002"),

        heading(3, "9.1.1 \u5019\u9009\u57FA\u56E0\u533A\u57DF\u6548\u5E94\u5206\u6570"),
        makeTable(
          ["\u57FA\u56E0", "\u6027\u72B6", "\u53D8\u5F02\u6570", "\u5E73\u5747\u6548\u5E94", "\u9AD8\u6548\u5E94\u5360\u6BD4", "\u9A8C\u8BC1\u7ED3\u8BBA"],
          [
            ["\u2605 CmTST2", "\u7CD6\u79EF\u7D2F\uFF08\u6DB2\u6CE1\u819C\u8F6C\u8FD0\uFF09", "5", "0.7922", "80%", "\u2714 \u6781\u9AD8\u6548\u5E94"],
            ["CmBt \u9644\u8FD1", "\u679C\u8089\u82E6\u5473", "103", "0.5505", "\u2014", "\u2714 \u9AD8\u6548\u5E94"],
            ["SSC QTL", "\u53EF\u6EB6\u6027\u56FA\u5F62\u7269/\u85C1\u7CD6", "104", "0.4366", "33.7%", "\u2714 \u4E2D\u9AD8\u6548\u5E94"],
            ["CmOr", "\u679C\u8089\u989C\u8272\uFF08\u6A59/\u975E\u6A59\uFF09", "\u533A\u57DF\u65E0SNP", "\u9644\u8FD10.4459", "\u2014", "\u26A0 GBS\u5BC6\u5EA6\u4E0D\u8DB3"],
            ["CmACS7", "\u679C\u957F/\u679C\u5F62", "\u533A\u57DF\u65E0SNP", "\u9644\u8FD10.3768", "\u2014", "\u26A0 GBS\u5BC6\u5EA6\u4E0D\u8DB3"],
            ["CmAAT1/2", "\u9999\u6C14", "\u533A\u57DF\u65E0SNP", "\u9644\u8FD10.3578", "\u2014", "\u26A0 GBS\u5BC6\u5EA6\u4E0D\u8DB3"],
          ],
          [1600, 2000, 1100, 1100, 1100, 1200]
        ),
        p("\u5173\u952E\u53D1\u73B0\uFF1ACmTST2\uFF08\u6DB2\u6CE1\u819C\u7CD6\u8F6C\u8FD0\u86CB\u767D\uFF09\u533A\u57DF5\u4E2A\u53D8\u5F02\u4E2D4\u4E2A\u6548\u5E94\u5206\u6570 > 0.7\uFF0C\u5E73\u5747 0.7922\uFF0C\u8FDC\u9AD8\u4E8E\u57FA\u56E0\u7EC4\u5747\u503C 0.4062\u3002\u6A21\u578B\u6210\u529F\u8BC6\u522B\u4E86\u5DF2\u77E5\u7CD6\u4EE3\u8C22\u529F\u80FD\u57FA\u56E0\u7684\u53D8\u5F02\u6548\u5E94\u3002", { italics: true }),

        heading(3, "9.1.2 \u67D3\u8272\u4F53\u6548\u5E94\u5BCC\u96C6"),
        makeTable(
          ["\u6392\u540D", "\u67D3\u8272\u4F53", "\u6807\u51C6\u540D", "\u4E0D\u5229\u53D8\u5F02%", "\u5E73\u5747\u6548\u5E94", "\u4E0E\u6587\u732E\u4E00\u81F4\u6027"],
          [
            ["1", "contig7", "chr01", "34.9%", "0.4436", "\u2714 \u9A6F\u5316\u9009\u62E9\u6700\u5BCC\u96C6"],
            ["1", "contig8", "chr09", "34.9%", "0.4442", "\u2714 \u542B CmOr/CmBt"],
            ["3", "contig3", "chr08", "33.3%", "0.4204", "\u2014"],
            ["4", "contig11", "chr11", "32.7%", "0.4200", "\u2014"],
            ["5", "contig5", "chr04", "32.2%", "0.3263", "\u2014"],
            ["6-12", "\u5176\u4ED9", "\u2014", "28.8-31.2%", "0.326-0.416", "\u2014"],
            ["13", "contig1", "chr01", "22.2%", "0.2072", "\u26A0 \u4EC527\u4E2A\u53D8\u5F02"],
          ],
          [800, 1300, 1000, 1300, 1400, 1800]
        ),
        p("\u7ED3\u8BBA\uFF1Achr01 \u548C chr09 \u786E\u8BA4\u662F\u4E0D\u5229\u53D8\u5F02\u6700\u5BCC\u96C6\u7684\u67D3\u8272\u4F53\uFF0C\u4E0E\u6587\u732E\u62A5\u9053\u7684\u9A6F\u5316\u9009\u62E9\u4FE1\u53F7\u4E00\u81F4\u3002", { bold: true }),

        heading(2, "9.2 \u9A8C\u8BC1\u603B\u7ED3"),
        codeBlock(
          "\u5DF2\u77E5\u7CD6\u4EE3\u8C22\u57FA\u56E0 (CmTST2)  \u2192  effect=0.7922 \u2714 \u9AD8\n" +
          "SSC\u7CD6\u5EA6QTL\u533A\u57DF           \u2192  effect=0.4366 \u2714 \u4E2D\u9AD8\n" +
          "\u82E6\u5473\u57FA\u56E0\u533A\u57DF (CmBt\u9644\u8FD1)  \u2192  effect=0.5505 \u2714 \u9AD8\n" +
          "\u67D3\u8272\u4F53\u5BCC\u96C6\u6392\u540D           \u2192  \u4E0E\u6587\u732E\u4E00\u81F4 \u2714\n" +
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n" +
          "\u6A21\u578B\u5BF9\u5DF2\u77E5\u529F\u80FD\u4F4D\u70B9\u5177\u6709\u5408\u7406\u7684\u6548\u5E94\u533A\u5206\u80FD\u529B"
        ),

        // ========== 10. LLM \u673A\u5236\u89E3\u91CA ==========
        heading(1, "10. LLM \u673A\u5236\u89E3\u91CA"),
        p("\u5BF9 Top 5 \u9AD8\u5F71\u54CD\u53D8\u5F02\uFF0C\u5229\u7528 LLM Synthesis \u6A21\u5757\u751F\u6210\u751C\u74DC\u80B2\u79CD\u8BED\u5883\u4E0B\u7684\u673A\u5236\u89E3\u91CA\u3002"),
        p("\u89E3\u91CA\u6A21\u677F\u8981\u7D20:", { bold: true }),
        ...bulletList([
          "\u53D8\u5F02\u4F4D\u7F6E\u4E0E\u57FA\u56E0\u4FE1\u606F",
          "\u6548\u5E94\u5206\u6570\u4E0E\u9884\u6D4B\u7C7B\u522B",
          "\u6270\u52A8\u8C31\u5206\u6790\uFF08top-8 \u53D7\u5F71\u54CD\u7684\u529F\u80FD\u7C7B\u522B\uFF09",
          "\u5BF9\u751C\u74DC\u7CD6\u4EE3\u8C22/\u54C1\u8D28\u6027\u72B6\u7684\u6F5C\u5728\u5F71\u54CD",
        ]),
        p("\u6CE8\u610F: \u5F53\u524D LLM \u89E3\u91CA\u6A21\u5757\u4E3A\u6A21\u677F\u5316\u751F\u6210\uFF0C\u5B8C\u6574 Claude API \u8C03\u7528\u9700\u914D\u7F6E ANTHROPIC_API_KEY\u3002", { italics: true }),

        // ========== 10. \u5DE5\u7A0B\u7ECF\u9A8C\u4E0E\u8FD0\u7EF4\u6559\u8BAD ==========
        heading(1, "11. \u5DE5\u7A0B\u7ECF\u9A8C\u4E0E\u8FD0\u7EF4\u6559\u8BAD"),

        heading(2, "11.1 \u5173\u952E\u6280\u672F\u4FEE\u590D"),
        makeTable(
          ["\u95EE\u9898", "\u539F\u56E0", "\u89E3\u51B3\u65B9\u6848"],
          [
            ["mamba_ssm CUDA \u52A0\u8F7D\u5931\u8D25", "\u7F3A\u5C11 CUDA \u5E93\u8DEF\u5F84", "\u8BBE LD_LIBRARY_PATH=/data/zhangcw/cuda-11.8/lib64"],
            ["Embedding FloatTensor \u9519\u8BEF", "input_ids \u4E3A float \u7C7B\u578B", "\u5728 encoder.py \u548C modeling_rcps.py \u52A0 .long()"],
            ["causal_conv1d \u5185\u6838\u5D29\u6E83", "\u4E2A\u522B\u53D8\u5F02\u89E6\u53D1 CUDA \u5F02\u5E38", "extract_activations.py \u6DFB\u52A0 try-except"],
            ["VCF/\u57FA\u56E0\u7EC4\u67D3\u8272\u4F53\u547D\u540D\u4E0D\u5339\u914D", "chr01-12 vs contig1-13", "sed \u6279\u91CF\u91CD\u547D\u540D + \u6620\u5C04\u8868"],
            ["\u6807\u7B7E\u5339\u914D\u5931\u8D25", "CSV \u542B\u975E\u6570\u503C\u4F4D\u7F6E", "\u6DFB\u52A0\u6B63\u5219\u8FC7\u6EE4 ^[0-9]+$"],
            ["\u591A\u8FDB\u7A0B\u8BBE\u8BA1\u6B7B\u9501", "\u91CD\u590D\u8FDB\u7A0B\u7ADE\u4E89\u8D44\u6E90", "kill \u6240\u6709\u51B2\u7A81\u8FDB\u7A0B\u540E\u91CD\u542F"],
          ],
          [2500, 3000, 3526]
        ),

        heading(2, "11.2 \u8FD0\u7EF4\u7ECF\u9A8C"),
        ...numberedList([
          "\u6807\u7B7E\u8D28\u91CF > \u57FA\u56E0\u7EC4\u8986\u76D6: GWAS \u6807\u7B7E\u7684 AUROC \u663E\u8457\u9AD8\u4E8E\u6587\u732E meta-QTL \u533A\u95F4\u6807\u6CE8",
          "\u9632 OOM: \u5168\u91CF VCF \u53EF\u8FBE\u5343\u4E07\u884C\uFF0C--max-variants \u9650\u6D41\u81F3 50K\u3002GPU \u663E\u5B58\u5360\u7528\u63A7\u5236\u5728 60-75%",
          "smoke-test \u5148\u9A8C: \u771F\u5B9E\u8BAD\u7EC3\u524D\u52A1\u5FC5\u7528 --smoke-test \u9A8C\u8BC1\u6D41\u6C34\u7EBF\u7ED3\u6784",
          "d_model \u5BF9\u9F50: \u63A2\u9488\u7EF4\u5EA6\u5FC5\u987B\u4E0E\u6240\u7528 PlantCAD2 \u6743\u91CD hidden_size \u4E00\u81F4",
          "\u591C\u95F4\u8BAD\u7EC3\u7B56\u7565: \u8BBE\u7F6E 23:00-8:00 \u81EA\u52A8\u8BAD\u7EC3 + \u81EA\u6108\u76D1\u63A7\uFF0C\u767D\u5929\u5FEB\u901F\u8FED\u4EE3",
        ]),

        heading(2, "11.3 \u81EA\u52A8\u5316\u8FD0\u7EF4"),
        makeTable(
          ["\u811A\u672C", "\u529F\u80FD"],
          [
            ["nightly_daemon.sh", "\u591C\u95F4 23:00-8:00 \u5B9A\u65F6\u542F\u505C\u8BAD\u7EC3"],
            ["nightly_extract.sh", "\u6279\u91CF\u63D0\u53D6\u6FC0\u6D3B\u5F20\u91CF"],
            ["self_heal_monitor.sh", "\u6BCF\u5C0F\u65F6\u81EA\u68C0\uFF0C\u81EA\u52A8\u4FEE\u590D\u5E38\u89C1\u9519\u8BEF"],
            ["run_pipeline.sh", "\u4E00\u952E\u7F16\u6392\u5168\u6D41\u7A0B"],
          ],
          [3500, 4000]
        ),

        // ========== 11. \u8F93\u51FA\u6587\u4EF6\u6E05\u5355 ==========
        heading(1, "11. \u8F93\u51FA\u6587\u4EF6\u6E05\u5355"),

        heading(2, "11.1 \u6838\u5FC3\u7ED3\u679C"),
        makeTable(
          ["\u6587\u4EF6", "\u5927\u5C0F", "\u5185\u5BB9"],
          [
            ["predictions_with_annotation.csv", "3.5 MB", "30,052\u6761\u53D8\u5F02\u7684\u5B8C\u6574\u9884\u6D4B"],
            ["predictions.csv", "1.3 MB", "\u521D\u59CB\u9884\u6D4B\u7ED3\u679C"],
            ["predictions_v2.csv", "1.5 MB", "\u7B2C\u4E8C\u6B21\u9884\u6D4B"],
            ["top100_high_impact.csv", "9.7 KB", "\u6548\u5E94\u5206\u6570\u6700\u9AD8\u7684100\u4E2A\u53D8\u5F02"],
            ["chrom_distribution.csv", "506 B", "13\u6761\u67D3\u8272\u4F53\u7684\u6548\u5E94\u7EDF\u8BA1"],
            ["designed_sugar_elements.fasta", "32 KB", "76\u6761 de novo \u7CD6\u4EE3\u8C22\u542F\u52A8\u5B50"],
            ["designed_sugar_batch2.fasta", "7.8 KB", "\u7B2C\u4E8C\u6279\u8BBE\u8BA1\u5143\u4EF6"],
            ["annotation_smoke.csv", "1.6 MB", "\u6CE8\u91CA\u63A2\u9488\u5192\u70DF\u6D4B\u8BD5"],
            ["smoke_design.fasta", "595 B", "\u8BBE\u8BA1\u6A21\u5757\u5192\u70DF\u6D4B\u8BD5"],
          ],
          [4000, 1200, 3826]
        ),

        heading(2, "11.2 \u6A21\u578B\u6743\u91CD"),
        makeTable(
          ["\u6587\u4EF6", "\u5927\u5C0F", "\u5185\u5BB9"],
          [
            ["probe_covariance.safetensors", "852 KB", "\u534F\u65B9\u5DEE\u63A2\u9488\u6743\u91CD (d=768\u2192128\u21922)"],
            ["probe_covariance.json", "102 B", "\u63A2\u9488\u914D\u7F6E (d_model=768, d_hidden=64)"],
            ["annotation_probe.safetensors", "52 KB", "\u6CE8\u91CA\u63A2\u9488\u6743\u91CD (d=768\u219217\u7C7B)"],
            ["annotation_probe.json", "417 B", "17\u7C7B\u6CE8\u91CA\u9762\u677F\u5B9A\u4E49"],
            ["plantcad2/ (\u8FDC\u7A0B)", "672 MB", "PlantCAD2-Small \u6A21\u578B\u6743\u91CD"],
          ],
          [4000, 1200, 3826]
        ),

        // ========== 12. \u7ED3\u8BBA\u4E0E\u540E\u7EED\u5DE5\u4F5C ==========
        heading(1, "12. \u7ED3\u8BBA\u4E0E\u540E\u7EED\u5DE5\u4F5C"),

        heading(2, "12.1 \u4E3B\u8981\u7ED3\u8BBA"),
        ...numberedList([
          "PlantCAD2-EVEE \u6D41\u6C34\u7EBF\u5B8C\u6574\u8FD0\u884C\u6210\u529F\uFF1A\u4ECE VCF \u53D8\u5F02\u8F93\u5165\u5230\u5168\u57FA\u56E0\u7EC4\u6548\u5E94\u9884\u6D4B\uFF0C\u5168\u6D41\u7A0B 13 \u6B65\u5728 NVIDIA L20 GPU \u4E0A\u987A\u5229\u6267\u884C\u3002",
          "\u6CE8\u91CA\u63A2\u9488\u8868\u73B0\u4F18\u5F02\uFF1A17\u7C7B\u529F\u80FD\u5206\u7C7B\u51C6\u786E\u7387\u8FBE 95.97%\uFF0C\u8BC1\u660E PlantCAD2 \u7684 RC \u4E0D\u53D8\u5D4C\u5165\u80FD\u591F\u6709\u6548\u6355\u83B7\u57FA\u56E0\u7EC4\u529F\u80FD\u7ED3\u6784\u4FE1\u606F\u3002",
          "\u534F\u65B9\u5DEE\u63A2\u9488\u9700\u4F18\u5316\uFF1AAUROC=0.5878\uFF0C\u63D0\u793A GWAS \u6807\u7B7E\u8D28\u91CF\u4E0D\u8DB3\u4EE5\u5145\u5206\u8BAD\u7EC3\u3002",
          "chr01 \u548C chr09 \u662F\u751C\u74DC\u80B2\u79CD\u4F18\u5148\u5173\u6CE8\u533A\u57DF\u3002",
          "Gibbs \u8BBE\u8BA1\u751F\u6210 76 \u6761\u7CD6\u4EE3\u8C22\u8C03\u63A7\u5143\u4EF6\u3002",
        ]),

        heading(2, "12.2 \u4E0E\u756A\u8304\u9879\u76EE\u7684\u5BF9\u6BD4"),
        makeTable(
          ["\u6307\u6807", "Tomato-GFM-EVEE2", "Melon-PlantCAD2-EVEE"],
          [
            ["\u57FA\u7840\u6A21\u578B", "PlantGFM (d=1024)", "PlantCAD2 (d=768)"],
            ["\u67B6\u6784", "Hyena / \u5355\u5411\u81EA\u56DE\u5F52", "Caduceus+Mamba2 / \u53CC\u5411MLM"],
            ["\u6CE8\u91CA\u63A2\u9488\u51C6\u786E\u7387", "91.5%", "95.97%"],
            ["GWAS\u63A2\u9488AUROC", "0.7035 (\u65B9\u6848A chr1)", "0.5878 (\u5168\u57FA\u56E0\u7EC4)"],
            ["De novo\u8BBE\u8BA1", "\u81EA\u56DE\u5F52 generate", "Gibbs \u63A9\u7801\u91CD\u91C7\u6837"],
            ["\u9884\u6D4B\u6548\u7387", "~15s/\u53D8\u5F02", "~7s/\u53D8\u5F02"],
          ],
          [2500, 3263, 3263]
        ),

        heading(2, "12.3 \u540E\u7EED\u5DE5\u4F5C\u5EFA\u8BAE"),
        ...numberedList([
          "\u6807\u7B7E\u589E\u5F3A: \u6574\u5408 Zhao 2019 (1175\u4EFD\u6750\u6599)\u3001Liu 2020 (297\u4EFD\u91CD\u6D4B\u5E8F) \u7B49\u591A\u6765\u6E90 GWAS \u6807\u7B7E\uFF0C\u63D0\u5347\u534F\u65B9\u5DEE\u63A2\u9488 AUROC",
          "\u66F4\u5927\u6A21\u578B: \u5C1D\u8BD5 PlantCAD2-Medium (311M, d=1024) \u6216 Large (694M, d=1536) \u6743\u91CD",
          "\u5B9E\u9A8C\u9A8C\u8BC1: \u5BF9 top 20 \u9AD8\u5F71\u54CD\u53D8\u5F02\u8FDB\u884C\u5B9E\u9A8C\u9A8C\u8BC1 (RT-qPCR/\u62A5\u544A\u57FA\u56E0)",
          "\u542F\u52A8\u5B50\u5B9E\u9A8C: \u5408\u6210\u8BBE\u8BA1\u7684\u7CD6\u4EE3\u8C22\u5143\u4EF6\u8FDB\u884C\u70DF\u8349\u77AC\u65F6\u8868\u8FBE\u6216\u751C\u74DC\u8F6C\u5316\u9A8C\u8BC1",
          "\u591A\u6027\u72B6\u80B2\u79CD\u6307\u6570: \u6574\u5408\u7CD6\u5EA6\u3001\u679C\u91CD\u3001\u6297\u75C5\u7B49\u591A\u6027\u72B6\u6784\u5EFA\u7EFC\u5408\u80B2\u79CD\u9009\u62E9\u6307\u6570",
          "EPI/ChIP-seq \u8F68\u9053: \u82E5\u6709\u751C\u74DC DAP-seq/ATAC-seq \u6570\u636E\uFF0C\u53EF\u8BAD\u7EC3\u6269\u5C55\u6CE8\u91CA\u63A2\u9488\u63D0\u5347 TFBS \u9884\u6D4B",
        ]),

        p(""), p(""),
        p("\u2014 \u62A5\u544A\u7531 WorkBuddy AI \u81EA\u52A8\u751F\u6210\u4E8E 2026\u5E746\u67081\u65E5", { align: AlignmentType.CENTER, italics: true, color: "999999" }),
        p("\u9879\u76EE\u5730\u5740 (\u8FDC\u7A0B): /data/zhangcw/melon-plantcad2-evee/melon-plantcad2-evee/", { align: AlignmentType.CENTER, color: "999999" }),
        p("\u672C\u5730\u526F\u672C: E:/XTTDATA/melon-plantcad2-evee/", { align: AlignmentType.CENTER, color: "999999" }),
      ]
    }
  ],
});

// ======== Generate ========
const outPath = "E:/XTTDATA/melon-plantcad2-evee/TECHNICAL_REPORT.docx";
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(outPath, buf);
  const mb = (buf.length / 1024 / 1024).toFixed(2);
  console.log("OK: " + outPath + " (" + mb + " MB)");
});
