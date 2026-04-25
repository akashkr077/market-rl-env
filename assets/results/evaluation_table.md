# Stage 1 Evaluation Results

| Policy | Avg P&L | 95% CI Low | 95% CI High | Avg Reward | Win Rate | Parse Fail | Participation |
|--------|--------:|-----------:|------------:|-----------:|---------:|-----------:|--------------:|
| InformedBot | 277.251 | 197.528 | 363.132 | 0.251 | 94% | 0.0% | 39% |
| MarketMakerBot | 44.002 | -29.303 | 115.322 | 0.068 | 76% | 0.0% | 100% |
| HoldBaseline | 0.000 | 0.000 | 0.000 | -0.020 | 0% | 0.0% | 0% |
| RandomBot | -121.324 | -203.333 | -47.564 | -0.107 | 22% | 0.0% | 59% |

*50 eval episodes per policy, medium difficulty, eval bot composition (incl. InformedBot)*