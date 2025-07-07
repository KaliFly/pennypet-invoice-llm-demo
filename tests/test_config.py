import pytest
from pandas import DataFrame

def test_actes_loaded(config):
    assert isinstance(config.actes_df, DataFrame)
    assert "pattern" in config.actes_df.columns

def test_medicaments_loaded(config):
    assert hasattr(config, "medicaments_df")
    assert not config.medicaments_df.empty

def test_mapping_et_formules(config):
    assert isinstance(config.mapping_amv, dict)
    assert "INTEGRAL" in config.formules
