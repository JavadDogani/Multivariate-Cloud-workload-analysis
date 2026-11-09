"""Stationarity and Granger-causality analysis of multivariate host-load traces.

Each CSV in the data directory holds the resource-usage time series of one machine from the
Google cluster trace (one row per 5-minute measurement window). The analysis works on first
differences of each series, as is standard before ADF/KPSS and Granger tests.

Usage:
    python workload_analysis.py stationarity            # ADF + KPSS per machine and variable
    python workload_analysis.py granger                 # pairwise Granger p-value matrix per machine
    python workload_analysis.py granger --variables CPU_usage Canonical_memory_usage Maximum_memory_usage
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "Workload of Machines with High Utilization"
DEFAULT_VARIABLES = ["CPU_usage", "Canonical_memory_usage", "Maximum_memory_usage"]


def load_machines(data_dir: Path | str = DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Return {machine_id: dataframe} for every CSV in ``data_dir``, sorted by machine id."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files found in {data_dir}")
    return {f.stem: pd.read_csv(f) for f in files}


def adf_test(series: pd.Series) -> dict[str, float]:
    """Augmented Dickey-Fuller test. Null hypothesis: the series has a unit root (non-stationary)."""
    stat, p_value, n_lags, _, crit, _ = adfuller(series.to_numpy())
    return {"adf_stat": stat, "adf_p": p_value, "adf_lags": n_lags, "adf_crit_5%": crit["5%"]}


def kpss_test(series: pd.Series) -> dict[str, float]:
    """KPSS test. Null hypothesis: the series is (level-)stationary.

    statsmodels reports p-values only within [0.01, 0.1]; values outside are clipped to the bounds.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        stat, p_value, n_lags, crit = kpss(series.to_numpy(), regression="c", nlags="auto")
    return {"kpss_stat": stat, "kpss_p": p_value, "kpss_lags": n_lags, "kpss_crit_5%": crit["5%"]}


def stationarity_table(
    machines: dict[str, pd.DataFrame],
    variables: list[str] = DEFAULT_VARIABLES,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """ADF and KPSS results on the first difference of each variable of each machine.

    A series is reported as stationary when ADF rejects a unit root (p < alpha) and KPSS does not
    reject stationarity (p >= alpha).
    """
    rows = []
    for machine_id, df in machines.items():
        for var in variables:
            diffed = df[var].diff().dropna()
            row = {"machine": machine_id, "variable": var, **adf_test(diffed), **kpss_test(diffed)}
            row["stationary"] = bool(row["adf_p"] < alpha and row["kpss_p"] >= alpha)
            rows.append(row)
    return pd.DataFrame(rows)


def granger_matrix(
    data: pd.DataFrame,
    variables: list[str],
    test: str = "ssr_chi2test",
    max_lag: int = 15,
) -> pd.DataFrame:
    """Pairwise Granger-causality p-values.

    Entry (row ``Y_y``, column ``X_x``) is the smallest p-value over lags 1..max_lag of the test
    "X does not Granger-cause Y". Small values mean X helps predict Y. The diagonal is not
    meaningful and is set to NaN.
    """
    out = pd.DataFrame(np.nan, index=variables, columns=variables)
    for y in variables:
        for x in variables:
            if x == y:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                res = grangercausalitytests(data[[y, x]], maxlag=max_lag, verbose=False)
            out.loc[y, x] = min(res[lag][0][test][1] for lag in range(1, max_lag + 1))
    out.index = [f"{v}_y" for v in variables]
    out.columns = [f"{v}_x" for v in variables]
    return out


def granger_tables(
    machines: dict[str, pd.DataFrame],
    variables: list[str] = DEFAULT_VARIABLES,
    max_lag: int = 15,
) -> dict[str, pd.DataFrame]:
    """Granger p-value matrix on first differences, per machine."""
    return {
        machine_id: granger_matrix(df[variables].diff().dropna(), variables, max_lag=max_lag)
        for machine_id, df in machines.items()
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis", choices=["stationarity", "granger"])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--variables", nargs="+", default=None, help="columns to analyse")
    ap.add_argument("--max-lag", type=int, default=15, help="Granger: largest lag tested")
    ap.add_argument("--alpha", type=float, default=0.05, help="stationarity: significance level")
    ap.add_argument("--out", type=Path, default=None, help="optional CSV file to write the results to")
    a = ap.parse_args(argv)

    machines = load_machines(a.data_dir)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")

    if a.analysis == "stationarity":
        table = stationarity_table(machines, a.variables or DEFAULT_VARIABLES, alpha=a.alpha)
        print(table.to_string(index=False))
        if a.out:
            table.to_csv(a.out, index=False)
    else:
        variables = a.variables or ["CPU_usage", "Canonical_memory_usage"]
        tables = granger_tables(machines, variables, max_lag=a.max_lag)
        for machine_id, table in tables.items():
            print(f"\nMachine {machine_id}  (min p-value over lags 1..{a.max_lag}; row = effect, column = cause)")
            print(table.to_string())
        if a.out:
            pd.concat(tables, names=["machine", "effect"]).to_csv(a.out)


if __name__ == "__main__":
    main()
