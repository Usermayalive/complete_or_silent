"""G2+: graph neural network scoring (spec §2.3, §3). Needs torch and torch_geometric.

    features.py   coarse graph tensors and per-question node features (D4)
    model.py      CoarseGNN (typed GraphSAGE)
    train.py      supervised pre-training on Silver Paths, then PPO (D5)
    evaluate.py   paragraph-level ranking readings vs baselines
    scorer.py     GNNScorer behind the Scorer interface
"""

                                                              