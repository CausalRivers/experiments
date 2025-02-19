


# Causal Discovery zoo (Unifying Causal Discovery algorithms)


The following content is included in this repository:

1. Causal Discovery general Benchmarking sleeve
2. Various Causal Discovery methods with a unified in/out
3. Hydra configs for overview
4. Install env via env.yml 


Importantly, the sleeve supports two input types. 
1. 2 files per sample. One that holds the time series and one that holds the labels. In "linear samples" you can find an example for this.
2. 2 files in total. One .csv file that holds the time series of all samples and a pickle file that holds a list of networkx objects that specify the ground truth and which time series are present in a sample. All timeseries specified in a netowrkx object are assumed to be in the same time series. We further provide functionality to parse the graph into matrix format for scoring.




### Missing: 
- Parallel script (I guess not needed due to slurm)
- Envs should be properly sparsed out.
- Properly reference all method origins.


### Methods implemented:

Importantly, we often simply wrap the official implementations which can be often installed via pip:


- VAR
- VARLINGAM
- Naive strategies
- PCMCI
- Dynotears
- CDMI
- Causal Pretraining
- STIC
- TCDF 





