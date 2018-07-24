# Neural Variational Set Expansion (NVSE) tensorflow implementation

This is the tensorflow implementation of NVSE for the paper: 
“Neural Variational Entity Set Expansion for Automatically Populated Knowledge Graphs”, Pushpendre Rastogi, Adam Poliak, Vince Lyzinski and Benjamin Van Durme. 2018

## Commands 

```
make train 
make serve
make query
```

## Data. 

Releasing the whole dataset for the experiments will not be possible because the TinkerBell KB was based on LDC2017E25. Please email us or raise an issue here for getting the dataset.

## Baseline software.

- word2vecf: https://github.com/se4u/word2vecf.git
- SetExpan: https://github.com/se4u/SetExpan.git
- BM25: bm25.py requiring gensim.
- Bayesian Sets: Our java code in src/main/java/edu/jhu/hlt/cadet/search/BinaryBayesianSets.java
