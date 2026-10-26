#!/usr/bin/env bash
# Full pipeline, in order. Each step can also be run on its own.
set -e

# 1. Real data (Places365 subset + your photos in data/own/<class>/)
python scripts/prepare_data.py

# 2. Baseline ("A1" model): ResNet-18, real images only
python train.py --arch resnet18 --run baseline_resnet18

# 3. SOTA data step: TA-TiTok + MaskGen, more images for the classes the baseline is worst at
[ -d external/1d-tokenizer ] || git clone https://github.com/bytedance/1d-tokenizer external/1d-tokenizer
python scripts/generate_maskgen.py --weights-from runs/baseline_resnet18/metrics.json --total 600

# 4. SOTA classifier + ablations
python train.py --arch vit_small --run vit_real                                   # ViT, real only
python train.py --arch vit_small --run vit_real_syn   --synthetic-frac 1.0        # ViT + all synthetic (final model)
python train.py --arch vit_small --run vit_real_syn50 --synthetic-frac 0.5        # ablation: half the synthetic set
python train.py --arch vit_small --run vit_real_syn_noaug --synthetic-frac 1.0 --no-aug   # ablation: no augmentation

# 5. Extra seeds for mean ± std
for seed in 1 2; do
  python train.py --arch vit_small --run vit_real_s$seed --seed $seed
  python train.py --arch vit_small --run vit_real_syn_s$seed --synthetic-frac 1.0 --seed $seed
done

# 6. Report tables and figures -> figures/
python make_figures.py

# 7. Demo app
python app.py
