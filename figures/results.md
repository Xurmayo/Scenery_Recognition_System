| run | arch | synthetic | aug | train imgs (real+syn) | test acc | test macro-F1 | indoor/outdoor acc | own-photo acc | own-photo F1 | latency ms | params M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_resnet18 | resnet18 | 0% | yes | 360+0 | 0.813 | 0.811 | 0.973 | 0.797 | 0.792 | 4.08 | 11.18 |
| vit_real | vit_small | 0% | yes | 360+0 | 0.913 | 0.913 | 0.993 | 0.924 | 0.905 | 3.85 | 21.67 |
| vit_real_s1 | vit_small | 0% | yes | 360+0 | 0.920 | 0.919 | 0.993 | 0.924 | 0.916 | 3.76 | 21.67 |
| vit_real_s2 | vit_small | 0% | yes | 360+0 | 0.887 | 0.885 | 0.993 | 0.962 | 0.953 | 6.99 | 21.67 |
| vit_real_syn | vit_small | 100% | yes | 360+601 | 0.893 | 0.893 | 0.987 | 0.975 | 0.969 | 4.01 | 21.67 |
| vit_real_syn50 | vit_small | 50% | yes | 360+300 | 0.887 | 0.887 | 0.987 | 0.949 | 0.936 | 4.1 | 21.67 |
| vit_real_syn_noaug | vit_small | 100% | no | 360+601 | 0.867 | 0.865 | 0.973 | 0.949 | 0.936 | 4.1 | 21.67 |
| vit_real_syn_s1 | vit_small | 100% | yes | 360+601 | 0.920 | 0.920 | 0.987 | 0.949 | 0.944 | 5.6 | 21.67 |
| vit_real_syn_s2 | vit_small | 100% | yes | 360+601 | 0.893 | 0.892 | 0.993 | 0.975 | 0.969 | 4.09 | 21.67 |
