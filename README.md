
# Experimental Section of CausalRivers

This repository contains resources and documentation for the experiments conducted as part of the CausalRivers project.

## Repository Structure

- **grid_export1/** and **grid_export2/**: Contain the exported raw values from our experiments.
- **exp1.ipynb - exp3.ipynb/**: Document the extraction process for the final tables used in the paper.
- **extract_grid_info.py**: Script for generating grid exports from raw experimental data (Not included but feel free to request).
- **Causal Discovery Zoo/**: Includes benchmarking scripts, all evaluated methods, and the finetuning script for CP (Causal Process).
- **CP Weights**: To use the raw or finetuned CP weights, download them from the [release page](https://github.com/CausalRivers/experiments/releases).

## Installation
To reproduce results or verify the experimental standards described in the paper, clone and install the [main repository](https://github.com/CausalRivers/causalrivers/) first and simple clone this repo inside




## Running our benchmarkin script

If you want to conduct you own grid search on a specific graph set you can simply run benchmark.py with your custom configuration.

E.g., this script would reproduce the scoring of VAR on the flood set (The parameters can be checked in exp2.ipynb):

```bash
cd causal_discovery_zoo
python benchmark.py label_path=../../datasets/random_5/flood.p data_path=../../product/rivers_ts_flood.csv method=var data_preprocess.normalize=False data_preprocess.resolution=15min method.max_lag=5 method.var_absolute_values=False
```


Note that this script is configured with hydra so you can leverage multirun as well as Slurm configurations.
Further, you might need to install one of the specific environments (included in causal_discovery_zoo/envs) to run your specified method.
