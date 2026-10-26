@echo off
REM Full pipeline for Windows. Each line can also be run on its own.

python scripts\prepare_data.py || exit /b 1
python train.py --arch resnet18 --run baseline_resnet18 || exit /b 1

if not exist external\1d-tokenizer git clone https://github.com/bytedance/1d-tokenizer external\1d-tokenizer
python scripts\generate_maskgen.py --weights-from runs\baseline_resnet18\metrics.json --total 600 || exit /b 1

python train.py --arch vit_small --run vit_real || exit /b 1
python train.py --arch vit_small --run vit_real_syn --synthetic-frac 1.0 || exit /b 1
python train.py --arch vit_small --run vit_real_syn50 --synthetic-frac 0.5 || exit /b 1
python train.py --arch vit_small --run vit_real_syn_noaug --synthetic-frac 1.0 --no-aug || exit /b 1

python train.py --arch vit_small --run vit_real_s1 --seed 1 || exit /b 1
python train.py --arch vit_small --run vit_real_s2 --seed 2 || exit /b 1
python train.py --arch vit_small --run vit_real_syn_s1 --synthetic-frac 1.0 --seed 1 || exit /b 1
python train.py --arch vit_small --run vit_real_syn_s2 --synthetic-frac 1.0 --seed 2 || exit /b 1

python scripts\filter_synthetic.py || exit /b 1
python train.py --arch vit_small --run vit_real_synf --synthetic-frac 1.0 --filter-synthetic || exit /b 1
python train.py --arch vit_small --run vit_real_synf_s1 --synthetic-frac 1.0 --filter-synthetic --seed 1 || exit /b 1
python train.py --arch vit_small --run vit_real_synf_s2 --synthetic-frac 1.0 --filter-synthetic --seed 2 || exit /b 1

python make_figures.py || exit /b 1
python app.py
