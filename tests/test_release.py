from pathlib import Path

import numpy as np
import torch
import yaml

import predict
from infer import resolve_config_path
from weaver.data_preprocessing.variables import ALL_CHANNELS, SPLIT_YEARS
from weaver.models import Weaver
from weaver.models import backbone


ROOT = Path(__file__).resolve().parents[1]


def variables():
    with (ROOT / "configs/weaver.yaml").open() as handle:
        return yaml.safe_load(handle)["data"]["variables"]


def test_release_assets_match_configuration():
    names = variables()
    assert len(names) == 71
    assert tuple(names) == ALL_CHANNELS
    assert SPLIT_YEARS == {
        "train": tuple(range(2008, 2017)),
        "val": (2017,),
        "test": (2018,),
    }
    stats = ROOT / "assets/statistics"
    for filename in (
        "normalize_mean.npz", "normalize_std.npz",
        "normalize_diff_std_6.npz", "normalize_diff_std_12.npz",
        "normalize_diff_std_24.npz", "clim.npz",
    ):
        with np.load(stats / filename) as archive:
            assert set(names) == set(archive.files)
            if "std" in filename:
                ordered = np.concatenate([
                    np.asarray(archive[name]).reshape(-1) for name in names
                ])
                assert np.isfinite(ordered).all()
                assert (ordered > 0).all()
    assert np.load(stats / "lat.npy").shape == (128,)
    assert np.load(stats / "lon.npy").shape == (256,)
    labels = np.load(ROOT / "assets/region_maps/region_labels_era5_agglomerative_patch2.npy")
    bias = np.load(ROOT / "assets/region_maps/region_to_expert_prior_init_era5_agglomerative_patch2_climv2.npy")
    assert labels.shape == (64 * 128,)
    assert set(np.unique(labels)) == set(range(9))
    assert bias.shape == (9, 9)
    np.testing.assert_allclose(np.diag(bias), 1.0, atol=1e-6)


def test_configuration_matches_paper_architecture():
    with (ROOT / "configs/weaver.yaml").open() as handle:
        config = yaml.safe_load(handle)
    net = config["model"]["net"]["init_args"]
    assert config["model"]["net"]["class_path"] == "weaver.models.weaver.Weaver"
    assert net["in_img_size"] == [128, 256]
    assert net["patch_size"] == 2
    assert net["hidden_size"] == 1024
    assert net["depth"] == 24
    assert net["num_heads"] == 16
    assert net["d_ch"] == 128
    assert net["channel_num_heads"] == 4
    assert net["pool_k"] == 4
    assert net["routed_num_experts"] == 9
    assert net["selected_experts"] == 3
    assert net["num_regions"] == 9


def test_bundled_config_resolves_assets_from_project_root():
    config_path = ROOT / "configs/weaver.yaml"
    assert resolve_config_path("assets/statistics", config_path) == ROOT / "assets/statistics"
    assert resolve_config_path(
        "assets/region_maps/region_labels_era5_agglomerative_patch2.npy",
        config_path,
    ) == ROOT / "assets/region_maps/region_labels_era5_agglomerative_patch2.npy"


def test_sdpa_attention_fallback_shape():
    original = backbone._xformers_attention
    failed = backbone._xformers_operator_unsupported
    try:
        backbone._xformers_attention = None
        query = torch.randn(2, 5, 2, 4)
        output = backbone.memory_efficient_attention(query, query, query)
        assert output.shape == query.shape
        assert torch.isfinite(output).all()
    finally:
        backbone._xformers_attention = original
        backbone._xformers_operator_unsupported = failed


def test_physical_rollout_and_static_field():
    class UnitIncrement(torch.nn.Module):
        def forward(self, state, names, interval):
            return torch.ones_like(state)

    initial = torch.zeros(1, 2, 2, 2)
    mean = torch.zeros(1, 2, 1, 1)
    std = torch.ones(1, 2, 1, 1)
    diff_std = {6: torch.tensor([2.0, 1.0]).view(1, 2, 1, 1)}
    result = predict.rollout(
        UnitIncrement(), initial,
        ["2m_temperature", "land_sea_mask"],
        6, 3, mean, std, diff_std,
    )
    torch.testing.assert_close(result[:, 0], torch.full((1, 2, 2), 6.0))
    torch.testing.assert_close(result[:, 1], torch.zeros(1, 2, 2))


def test_paper_facing_model_has_canonical_name():
    assert Weaver.__name__ == "Weaver"
    assert Weaver.__module__ == "weaver.models.weaver"
