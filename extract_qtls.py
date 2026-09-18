import fitz, re

doc = fitz.open('C:/Users/Administrator/Desktop/41588_2019_522_MOESM1_ESM.pdf')

# ====== Supplementary 10: QTLs overlapping with domestication sweeps ======
txt10 = doc[17].get_text()
lines = txt10.split('\n')
qtl10 = []
in_table = False
for line in lines:
    line = line.strip()
    if line.startswith('Chr'):
        in_table = True
    if in_table and line.startswith('*'):
        in_table = False
    if in_table and line.startswith('Chr'):
        parts = re.split(r'\s{2,}', line)
        if len(parts) >= 6:
            chrom = parts[0].strip()
            qtl_name = parts[1].strip().replace('*', '')
            pos_parts = [p for p in parts if p.strip().replace(',','').replace('.','').isdigit() and len(p) > 5]
            if len(pos_parts) >= 2:
                start = int(pos_parts[0].replace(',',''))
                end = int(pos_parts[1].replace(',',''))
                pop = ''
                for p in parts:
                    if p.strip() in ['melo', 'agrestis']:
                        pop = p.strip()
                qtls10.append(f'{chrom}\t{start}\t{end}\t{qtl_name}\t{pop}')

print(f"=== Supplementary 10: {len(qtls10)} QTLs ===")
for q in qtls10:
    print(q)

# ====== Supplementary 14: QTLs overlapping with divergent genomic regions ======
txt14 = doc[20].get_text()
lines = txt14.split('\n')
qtl14 = []
for line in lines:
    line = line.strip()
    if line.startswith('FCONV') or line.startswith('FFP') or line.startswith('SSC') or line.startswith('TSUG') or line.startswith('Eay'):
        parts = re.split(r'\s{2,}', line)
        if len(parts) >= 6:
            chrom = parts[1].strip() if len(parts) > 1 else ''
            if not chrom.startswith('chr'):
                chrom = 'chr' + chrom
            start = int(parts[2].replace(',',''))
            end = int(parts[3].replace(',',''))
            div = parts[4].strip() if len(parts) > 4 else ''
            name = parts[0].strip()
            qtls14.append(f'{chrom}\t{start}\t{end}\t{name}\t{div}')

print(f"\n=== Supplementary 14: {len(qtls14)} QTLs ===")
for q in qtls14:
    print(q)
doc.close()
