import pytest

from rscene.config import DEFAULT_CONFIG, merged_config


def test_spec_tolerances_have_their_specified_values():
    assert DEFAULT_CONFIG["tau_fit_m"] == 0.003
    assert DEFAULT_CONFIG["tau_feature_m"] == 0.008


def test_merged_config_overrides_without_mutating_the_default():
    cfg = merged_config({"tau_fit_m": 0.005})
    assert cfg["tau_fit_m"] == 0.005
    assert cfg["min_patch_points"] == DEFAULT_CONFIG["min_patch_points"]
    assert DEFAULT_CONFIG["tau_fit_m"] == 0.003


def test_unknown_config_keys_are_rejected():
    with pytest.raises(KeyError, match="unknown config key"):
        merged_config({"tau_fitt_m": 0.005})


def test_merged_config_of_none_equals_the_default():
    assert merged_config(None) == DEFAULT_CONFIG
