
# Experimental Section of CausalRivers

This repository contains resources and documentation for the experiments conducted as part of the CausalRivers project.

## Repository Structure

- **grid_export1/** and **grid_export2/**: Contain the exported raw values from our experiments.
- **exp1.ipynb - exp3.ipynb/**: Document the extraction process for the final tables used in the paper.
- **extract_grid_info.py**: Script for generating grid exports from raw experimental data (Not included but feel free to request).
- **Causal Discovery Zoo/**: Includes benchmarking scripts, all evaluated methods, and the finetuning script for CP (Causal Pretraining).







## Installation
To reproduce results or verify the experimental standards described in the paper, clone and install the [main repository](https://github.com/CausalRivers/causalrivers/) first and simple clone this repo inside

- **CP Weights**: To use the raw or finetuned CP weights, download them from the [release page](https://github.com/CausalRivers/experiments/releases) by running:

```bash
wget https://github.com/CausalRivers/experiments/releases/download/weights/cp_models.zip
unzip cp_models.zip
mv finetuned_weights causal_discovery_zoo/methods/cp_models
mv pretrained_weights causal_discovery_zoo/methods/cp_models
rm cp_models.zip
```
We also upload the raw exprimental results [here](https://zenodo.org/records/16797284?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjhiMDc4YTEyLWY0MTAtNDk4MS05NmU5LTkwYTNhYWI3NWRhYiIsImRhdGEiOnt9LCJyYW5kb20iOiI2MDVlMjliNzAxNjI5ZTM3ZjYxMWE5Y2M3NTQyNGM4ZSJ9.pH4yqfkiT5ZTe3IcwsQC3CVQc27DjZCKJA16kqz14YgNY__9fVy_76SYEpnwBRsQbaraQB-ffm6hEQ6Putaaag) (not needed for reproductions): 



For running PCMCI and Varlingam you can simply install the following in the base environment: 
```bash
pip install lingam tigramite
```


Unfortunately for Dynotears, CP, and CDMI, a new environment is required.

For Dynotears: 
```bash

conda create -n dyno
conda activate dyno 
conda install python==3.9.0
pip install causalnex
pip install hydra-core
pip install hydra-submitit-launcher --upgrade
```







## Running our benchmarking script

If you want to conduct you own grid search on a specific graph set you can simply run benchmark.py with your custom configuration.

E.g., this script would reproduce the scoring of VAR on the flood set (The parameters can be checked in exp2.ipynb):

```bash
cd causal_discovery_zoo
python benchmark.py label_path=../../datasets/random_5/flood.p data_path=../../product/rivers_ts_flood.csv method=var data_preprocess.normalize=False data_preprocess.resolution=15min method.max_lag=5 method.var_absolute_values=False
```


This would reproduce the results for the close_5 dataset with a finetuned CP-architecture (Exp.3 )
```bash

python benchmark.py method=cp method.use_river_finetune=True label_path="../../datasets/close_5/east.p" data_preprocess.resolution="12H" data_preprocess.normalize=False
```



Note that this script is configured with hydra so you can leverage multirun as well as Slurm configurations.
Further, you might need to install one of the specific environments (included in causal_discovery_zoo/envs) to run your specified method.
