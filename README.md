# Multivariate Cloud Workload Analysis

[![CI](https://github.com/JavadDogani/Multivariate-Cloud-workload-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/JavadDogani/Multivariate-Cloud-workload-analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

Is a machine's CPU load worth using to forecast its memory load, and the other way round? This
repository answers that for five heavily loaded machines of the
[Google cluster trace](https://github.com/google/cluster-data). It runs two standard checks:

1. **Stationarity** (ADF and KPSS tests). Most forecasting models assume stationary input, so this
   tells you whether the series, or their first differences, can be modelled directly.
2. **Granger causality.** Does the past of one resource improve the prediction of another? If it
   does, a multivariate forecaster (e.g. CPU + memory together) should beat univariate ones.

These checks motivate multivariate workload prediction for cloud autoscaling.

## Data

The Google cluster-usage trace (2011) records resource usage of about 12,500 machines over 29 days
in 5-minute measurement windows. The five machines that ran the most tasks were extracted into
`Workload of Machines with High Utilization/`, one CSV per machine (file name = machine id), with
8,351 windows each:

| Column | Meaning |
|---|---|
| `start-timestamp`, `finish-timestamp` | Measurement window |
| `CPU_usage` | Mean CPU usage (normalized, as in the trace) |
| `Canonical_memory_usage` | Canonical memory usage |
| `ass_mem` | Assigned memory |
| `Maximum_memory_usage` | Maximum memory usage in the window |

## Quickstart

```bash
pip install -r requirements.txt

python workload_analysis.py stationarity            # ADF + KPSS for every machine and variable
python workload_analysis.py granger                 # Granger p-value matrix per machine
python workload_analysis.py granger --variables CPU_usage Canonical_memory_usage Maximum_memory_usage
python workload_analysis.py stationarity --out stationarity.csv   # also save the table
```

Both analyses take a few seconds. Run `python workload_analysis.py -h` for all options
(`--data-dir`, `--max-lag`, `--alpha`, `--out`). The functions (`load_machines`,
`stationarity_table`, `granger_matrix`, ...) can also be imported and used on your own traces.

## Method

All tests run on the **first difference** of each series.

- **ADF** (Augmented Dickey–Fuller): null hypothesis = unit root (non-stationary).
- **KPSS**: null hypothesis = stationary. statsmodels reports KPSS p-values only within
  [0.01, 0.1], so 0.1 means "at least 0.1".
- A series is reported as stationary when ADF rejects (p < 0.05) **and** KPSS does not reject
  (p ≥ 0.05). Using both tests guards against the low power of either one alone.
- **Granger causality**: the χ² (SSR) test for lags 1–15. Each entry is the smallest p-value over
  those lags for "the column variable does not Granger-cause the row variable".

## Results

Output of the two commands above on the bundled data.

**Stationarity.** After first differencing, all 15 series (5 machines × CPU, canonical memory,
maximum memory) are stationary: ADF p < 10⁻²⁹ and KPSS p ≥ 0.1 in every case.

**Granger causality** between CPU and canonical memory (smallest p-value over lags 1–15):

| Machine | CPU → memory | memory → CPU |
|---|---|---|
| 1093209 | 7.4 × 10⁻⁵ | 9.2 × 10⁻⁵ |
| 1093855 | 3.4 × 10⁻⁴ | 2.9 × 10⁻⁴ |
| 1094095 | 7.7 × 10⁻¹¹ | 3.8 × 10⁻⁶ |
| 1094470 | 8.4 × 10⁻⁷ | 2.0 × 10⁻¹⁸ |
| 1119618555 | 9.9 × 10⁻⁶ | 0.38 |

Past CPU usage helps predict memory usage on all five machines, and past memory usage helps
predict CPU on four of them. This supports forecasting the resources jointly rather than one at a
time.

Taking the smallest p-value over 15 lags is not corrected for multiple testing, so treat the
values as indicative. Even a Bonferroni factor of 15 leaves every result above significant except
memory → CPU on machine 1119618555.

## Testing

```bash
pip install pytest
pytest -q
```

The tests use synthetic series with known properties: white noise vs. random walk for ADF/KPSS, and
a lagged dependency for Granger. They also include a smoke test on the bundled data.

## Related work

A related publication by the author on multivariate workload prediction:

> Javad Dogani, Farshad Khunjush, Mohammad Reza Mahmoudi, Mehdi Seydali. "Multivariate Workload and
> Resource Prediction in Cloud Computing Using CNN and GRU with an Attention Mechanism."
> *The Journal of Supercomputing*, 79, 3437–3470, 2023.

## License

Code: MIT. See [LICENSE](LICENSE). The data is derived from the Google cluster-usage trace; see
[google/cluster-data](https://github.com/google/cluster-data) for its license (CC-BY) and attribution terms.
