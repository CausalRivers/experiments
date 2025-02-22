

# Experimental section of CausalRivers


Install the [experiments repository](https://github.com/CausalRivers/experiments/) here if you are interested in reproducing results or confirming the experimental standards of the paper.

We document all three experiments that are presented in the paper.



To reproduce e.g. the best scoring of experiment 2 run:

```
python benchmark.py -m  label_path=../../datasets/random_5/flood.p data_path=../../product/rivers_ts_flood.csv method=dynotears  data_preprocess.normalize=False  data_preprocess.resolution=12H method.max_lag=3
```


