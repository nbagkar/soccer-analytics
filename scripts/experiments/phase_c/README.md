# Phase C experiments: rest days and gradient boosting (rejected)

Run 2026-11-17. Neither candidate improved the live model, so neither shipped.

```bash
pip install scikit-learn          # only these scripts need it; not a project dependency
cd scripts/experiments/phase_c
SOCCER_ENV_FILE=../../../.env python build_preds.py E0 SP1 D1 I1 F1 E1 N1 P1 I2 SP2 D2 F2 B1 T1
SOCCER_ENV_FILE=../../../.env python analyze.py
SOCCER_ENV_FILE=../../../.env python experiment.py 300     # shrinkage; also tried 50, 1000
```

`build_preds.py` writes walk-forward predictions of the live model (`_forecast_fit`, fit per
match date on the prior ~3 seasons) from 2016/17 on, with each side's rest days (days since its
previous **league** match; cup and European games aren't in the files). It only reads the store.
`experiment.py` tunes on E0/SP1/D1/I1/F1 and confirms on the other nine using parameters fit
only on the tuning leagues' earlier seasons. The score is the gap to the closing line
(model log loss minus closing-line log loss; lower is better).

| | Tuning (5 leagues, 14,718 matches) | Confirm (9 leagues, 27,066) |
|---|---|---|
| Live model | +0.0119 | +0.0125 |
| A: rest-regime goal factors | +0.0118 (−0.00009 ± 0.00022; 3/5 leagues better) | +0.0125 (+0.00003 ± 0.00016; 4/9) |
| B: gradient boosting + rest | +0.0172 | +0.0171 |
| B0: gradient boosting, no rest | +0.0170 | +0.0167 |

- **Rest days carry no usable signal.** No fatigue penalty shows on ≤3 days' rest. After long
  breaks (10+ days, i.e. international breaks and season starts) home sides score ~4% below
  expectation and away sides ~3.5% above. But a factor for it doesn't move the gap at any
  shrinkage (50, 300, 1000): the closing line already prices it.
- **Gradient boosting stacked on the model is worse in all 14 leagues** (~+0.005), with or
  without rest. A tree classifier on expected goals loses the score grid's structure. Per the
  plan's rule, LightGBM wasn't tried as well: a second booster doesn't address why the first
  one failed.
