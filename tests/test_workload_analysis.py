import numpy as np
import pandas as pd
import pytest

import workload_analysis as wa

N = 600


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def write_machine(path, df):
    df.to_csv(path, index=False)


def synthetic_machine(rng, n=N):
    """CPU follows a random walk; memory follows CPU with a one-step lag."""
    cpu = np.cumsum(rng.normal(size=n))
    mem = np.empty(n)
    mem[0] = 0.0
    mem[1:] = 0.8 * cpu[:-1] + rng.normal(scale=0.3, size=n - 1)
    peak = mem + np.abs(rng.normal(scale=0.1, size=n))
    return pd.DataFrame({"CPU_usage": cpu, "Canonical_memory_usage": mem, "Maximum_memory_usage": peak})


def test_load_machines_sorted(tmp_path, rng):
    for name in ["b", "a"]:
        write_machine(tmp_path / f"{name}.csv", synthetic_machine(rng, 50))
    machines = wa.load_machines(tmp_path)
    assert list(machines) == ["a", "b"]
    assert machines["a"].shape == (50, 3)


def test_load_machines_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        wa.load_machines(tmp_path)


def test_adf_separates_noise_from_random_walk(rng):
    noise = pd.Series(rng.normal(size=N))
    walk = pd.Series(np.cumsum(rng.normal(size=N)))
    assert wa.adf_test(noise)["adf_p"] < 0.01
    assert wa.adf_test(walk)["adf_p"] > 0.05


def test_kpss_separates_noise_from_random_walk(rng):
    noise = pd.Series(rng.normal(size=N))
    walk = pd.Series(np.cumsum(rng.normal(size=N)))
    assert wa.kpss_test(noise)["kpss_p"] >= 0.05
    assert wa.kpss_test(walk)["kpss_p"] <= 0.05


def test_stationarity_table_on_differences(rng):
    machines = {"m1": synthetic_machine(rng), "m2": synthetic_machine(rng)}
    table = wa.stationarity_table(machines)
    assert len(table) == 2 * len(wa.DEFAULT_VARIABLES)
    assert set(table["machine"]) == {"m1", "m2"}
    # first differences of a random walk are white noise, hence stationary
    assert table.loc[table["variable"] == "CPU_usage", "stationary"].all()


def test_granger_detects_known_direction(rng):
    df = synthetic_machine(rng)
    variables = ["CPU_usage", "Canonical_memory_usage"]
    m = wa.granger_matrix(df[variables].diff().dropna(), variables, max_lag=3)
    assert m.shape == (2, 2)
    assert np.isnan(m.loc["CPU_usage_y", "CPU_usage_x"])
    # CPU drives memory with a one-step lag ...
    assert m.loc["Canonical_memory_usage_y", "CPU_usage_x"] < 1e-6
    # ... and is not driven by it.
    assert m.loc["CPU_usage_y", "Canonical_memory_usage_x"] > 0.01


def test_cli_writes_csv(tmp_path, rng):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_machine(data_dir / "123.csv", synthetic_machine(rng, 200))
    out = tmp_path / "out.csv"
    wa.main(["granger", "--data-dir", str(data_dir), "--max-lag", "2", "--out", str(out)])
    written = pd.read_csv(out)
    assert {"machine", "effect"} <= set(written.columns)
    wa.main(["stationarity", "--data-dir", str(data_dir), "--out", str(out)])
    assert len(pd.read_csv(out)) == 3


def test_bundled_dataset_smoke():
    machines = wa.load_machines()
    assert len(machines) == 5
    for df in machines.values():
        assert set(wa.DEFAULT_VARIABLES) <= set(df.columns)
