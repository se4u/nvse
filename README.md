# Neural Variational Set Expansion (NVSE) tensorflow implementation

This is the tensorflow implementation of NVSE for the paper: 
“Neural Variational Entity Set Expansion for Automatically Populated Knowledge Graphs”, Pushpendre Rastogi, Adam Poliak, Vince Lyzinski and Benjamin Van Durme. 2018

```
@Article{Rastogi2018,
  author="Rastogi, Pushpendre and Poliak, Adam and Lyzinski, Vince and Van Durme, Benjamin",
  title="Neural variational entity set expansion for automatically populated knowledge graphs",
  journal="Information Retrieval Journal",
  year="2018",
  month="Oct",
  day="25",
  issn="1573-7659",
  doi="10.1007/s10791-018-9342-1",
  url="https://doi.org/10.1007/s10791-018-9342-1"
}
```




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
