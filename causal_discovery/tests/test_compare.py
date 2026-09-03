from pathlib import Path
import pandas as pd
import pytest
from causal_discovery.compare import run_comparison


def test_run_comparison_returns_dataframe(tmp_path):
    df = run_comparison(
        preset_name="small_business",
        algorithms=["granger", "pcmci_cmiknn"],  # subset for speed
        output_dir=tmp_path,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "precision" in df.columns
    assert "recall" in df.columns
    assert "f1" in df.columns
    assert "fdr" in df.columns
    assert "shd" in df.columns


def test_run_comparison_writes_files(tmp_path):
    run_comparison(
        preset_name="small_business",
        algorithms=["granger"],
        output_dir=tmp_path,
    )
    out = tmp_path / "small_business"
    assert (out / "metrics.csv").exists()
    assert (out / "adjacency_comparison.png").exists()
    assert (out / "metrics_bar.png").exists()
    assert (out / "runtime_vs_f1.png").exists()
