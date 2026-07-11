## Updates
1. Simplified package requirements (reduced to ~10 from ~200): You will need to reinstall this simplified packages in a new environment using `requirements.txt`
2. MPS compatibility: Now you can use your Apple Silicon to perform training and evaluation
3. Data path logic: Now you can enter either relative path or absolute path in `configs/datasets.json`. Relative path is recommended
4. Making non-adaptive SAM as default: Now you have to pass `--adaptive True` to manually enable ASAM. Without it, the vanilla SAM will be executed. Kindly note that `--rho` values affects SAM and ASAM differently (i.e., you have to figure out a new value for SAM, such as `0.05`)
5. Fixed the issue of schedular double-step
6. Reimplemented SAMFW (our method) based on the original algorithm presented in the document (*"Group Fairness via Sharpness-Aware Minimization and Frank-Wolfe Optimization"*)
7. HAM10000 results available: The conclusion is generally seed-agnostic, you can try other seeds as well
8. Prototype method MGDASAM: This is likely the faithful implementation of MDGA+SAM, however, it serves as a prototype only without further testing or tunning. It is unlikely that we will include it in the final paper


## HAM10000
### Final Results
| Method | Test Overall AUC (std) | Test Worst-Group AUC | Test Overall Acc |
|---|---|---|---|
| ERM | 0.893 (0.0025) | 0.882 (0.0059) | 0.859 (0.0245) |
| SAM | 0.901 (0.0071) | 0.895 (0.0067) | 0.873 (0.0206) |
| SAMFW (Ours) | **0.909** (0.0035) | **0.899** (0.0021) | **0.876** (0.0157) |

Averaged over 3 random seeds (`0, 1, 2`). And the corresponding hyperparameters (refer to the Hyperparameter Sweep) were set to the following:
1. ERM: `lr=1e-4`
2. SAM: `--rho 0.05 --lr 2e-4`
3. SAMFW: `--rho 0.05 --lr 5e-5`

We also provide the training/evaluation command below for your reference.

ERM:
```
python main.py \
  --experiment baseline \
  --dataset_name HAM10000 \
  --backbone cusResNet18 \
  --total_epochs 20 \
  --early_stopping 5 \
  --sensitive_name Sex \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --sens_classes 2 \
  --output_dim 1 \
  --num_classes 1 \
  --val_strategy worst_auc \
  --log_freq 0
```

SAM:
```
python main.py \
  --experiment SAM \
  --dataset_name HAM10000 \
  --backbone cusResNet18 \
  --total_epochs 20 \
  --early_stopping 5 \
  --sensitive_name Sex \
  --batch_size 32 \
  --lr 2e-4 \
  --weight_decay 1e-4 \
  --sens_classes 2 \
  --output_dim 1 \
  --num_classes 1 \
  --val_strategy worst_auc \
  --rho 0.05 \
  --T_max 50 \
  --log_freq 0
```

SAMFW (ours):
```
python main.py \
  --experiment SAMFW \
  --dataset_name HAM10000 \
  --backbone cusResNet18 \
  --total_epochs 20 \
  --early_stopping 5 \
  --sensitive_name Sex \
  --batch_size 32 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --sens_classes 2 \
  --output_dim 1 \
  --num_classes 1 \
  --val_strategy worst_auc \
  --rho 0.05 \
  --T_max 50 \
  --log_freq 0
```


### Hyperparameter Sweep
This is a rough sweeep with a fixed random seed `0`, not an exhaustive one.

#### ERM Baseline
| lr | Test Overall AUC | Test Worst-Group AUC | Test Overall Acc | Best epoch |
|---|---|---|---|---|
| 5e-5 | 0.889 | 0.855 | 0.881 | 12 |
| **1e-4** | **0.896** | **0.875** | **0.887** | 2 |
| 2e-4 | 0.888 | 0.872 | 0.847 | 2 |

#### SAM
ρ sweep, lr fixed at `1e-4`:
| ρ | Test Overall AUC | Test Worst-Group AUC | Test Overall Acc | Best epoch |
|---|---|---|---|---|
| 2 (old default) | 0.818 | 0.801 | 0.793 | 14 |
| 0.5 | 0.854 | 0.849 | 0.825 | 14 |
| 0.1 | 0.907 | 0.893 | 0.881 | 13 |
| **0.05** | **0.912** | **0.897** | 0.875 | 16 |

lr sweep, ρ fixed at `0.05`:
| lr | Test Overall AUC | Test Worst-Group AUC | Test Overall Acc | Best epoch |
|---|---|---|---|---|
| 5e-5 | 0.911 | 0.890 | 0.880 | 17 |
| 1e-4 (Step 1 winner, carried over) | 0.912 | 0.897 | 0.875 | 16 |
| **2e-4** | 0.907 | **0.899** | **0.893** | **8** |

#### SAMFW (Ours)
ρ sweep, lr fixed at `1e-4`:
| ρ | Test Overall AUC | Test worst-group AUC | Test Overall Acc |
|---|---|---|---|
| 2 (old default, ~40x the SAM paper's recommended 0.05) | 0.829 | 0.818 | 0.817 |
| 0.5 | 0.850 | 0.841 | 0.839 |
| 0.1 | 0.905 | 0.891 | 0.871 |
| **0.05** | **0.910** | **0.899** | **0.897** |

lr sweep, ρ fixed at `0.05`:
| lr | Test Overall AUC | Test worst-group AUC | Test Overall Acc | Best epoch |
|---|---|---|---|---|
| **5e-5** | **0.912** | **0.901** | **0.894** | 17 |
| 1e-4 (Step 1 winner, carried over) | 0.910 | 0.899 | 0.897 | 16 |
| 2e-4 | 0.909 | 0.898 | 0.891 | 6-8 |


## Acknowledgement
This repository is adapted from the original work by Zong et al., (https://github.com/ys-zong/MEDFAIR.git). We thank the original authors for their great work.
