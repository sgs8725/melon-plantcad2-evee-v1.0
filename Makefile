# Melon-PlantCAD2-EVEE 便捷命令
.PHONY: install download smoke dataset activations train annot annot-tracks tracks-bed predict design labels evaluate pipeline smoke-all

install:
	pip install -r requirements.txt

download:
	bash scripts/download_all.sh

smoke:
	python scripts/smoke_test.py

dataset:
	python scripts/build_variant_dataset.py \
	  --vcf assets/variants/melon_variants.vcf.gz \
	  --genome assets/genome/melon_DHL92.fa \
	  --gff assets/genome/melon_DHL92.gff3 \
	  --out data/variants.parquet --window 8192

activations:
	python scripts/extract_activations.py \
	  --variants data/variants.parquet --model assets/plantcad2 \
	  --out data/activations --topk 256

train:
	python scripts/train_probe.py \
	  --activations data/activations --labels data/variants.parquet \
	  --out assets/probe_covariance.safetensors --task classification

predict:
	python scripts/predict.py \
	  --variants data/variants.parquet --activations data/activations \
	  --probe assets/probe_covariance.safetensors --out results/predictions.csv

design:
	python scripts/design_elements.py --model assets/plantcad2 \
	  --element sugar --n 1000 --length 400 \
	  --out results/designed_sugar_elements.fasta

labels:
	python scripts/build_phenotype_labels.py --mode gwas \
	  --gwas data/phenotype/gwas_hits.csv \
	  --variants data/variants.parquet \
	  --out data/variants_labeled.parquet

annot:
	python scripts/train_annotation_probe.py \
	  --gff assets/genome/melon_DHL92.gff3 \
	  --genome assets/genome/melon_DHL92.fa \
	  --model assets/plantcad2 --chrom 1 --d-model 768 \
	  --out assets/annotation_probe.safetensors

tracks-bed:
	python scripts/tracks_to_bed.py --input-dir raw_tracks/ --out-dir assets/tracks/ \
	  --strip-chr --chrom 1

annot-tracks:
	python scripts/train_annotation_tracks.py \
	  --tracks data/tracks.json --genome assets/genome/melon_DHL92.fa \
	  --model assets/plantcad2 --d-model 768 \
	  --out assets/annotation_probe_tracks.safetensors

evaluate:
	python scripts/evaluate.py \
	  --pred results/predictions.csv \
	  --labels data/variants_labeled.parquet --task classification

pipeline:
	bash runners/run_full_pipeline.sh

smoke-all:
	bash runners/run_full_pipeline.sh smoke
