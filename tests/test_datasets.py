"""Dataset licensing tests.

These guard a legal constraint, not a behavioural one. StatsBomb's public data is
governed by a proprietary EULA that bars both redistribution and commercial
exploitation -- including of derived analysis. If a future change flips those flags
or adds an export path that ignores them, these fail loudly.
"""

from __future__ import annotations

import pytest

from soccer.sources.datasets import (
    DATASETS,
    DataKind,
    Dataset,
    DatasetId,
    commercially_usable,
    redistributable,
    total_size_gb,
)


class TestStatsBombIsNotOpen:
    """The plan described StatsBomb as 'attribution required', implying CC BY. It is not."""

    def test_redistribution_prohibited(self) -> None:
        assert not DATASETS[DatasetId.STATSBOMB].may_redistribute

    def test_commercial_use_prohibited(self) -> None:
        # EULA s1.2.2 bars exploiting the data OR any analysis derived from it.
        assert not DATASETS[DatasetId.STATSBOMB].commercial_use

    def test_logo_attribution_required(self) -> None:
        # s1.4 requires the brand logo, not a text credit.
        assert DATASETS[DatasetId.STATSBOMB].logo_attribution_required

    def test_licence_source_is_not_the_readme(self) -> None:
        # The README's softer wording is not the binding document.
        assert "LICENSE.pdf" in DATASETS[DatasetId.STATSBOMB].licence_source

    def test_excluded_from_redistributable_set(self) -> None:
        assert DatasetId.STATSBOMB not in {d.id for d in redistributable()}

    def test_excluded_from_commercial_set(self) -> None:
        assert DatasetId.STATSBOMB not in {d.id for d in commercially_usable()}


class TestPermissiveDatasets:
    def test_wyscout_is_fully_permissive(self) -> None:
        wyscout = DATASETS[DatasetId.WYSCOUT]
        assert wyscout.may_redistribute
        assert wyscout.commercial_use
        assert wyscout.licence == "CC BY 4.0"

    def test_wyscout_attributes_to_figshare_not_the_repo(self) -> None:
        # The GitHub repo has no LICENSE file; figshare is the source of record.
        assert "figshare" in DATASETS[DatasetId.WYSCOUT].licence_source
        assert "Pappalardo" in DATASETS[DatasetId.WYSCOUT].attribution

    def test_skillcorner_is_mit(self) -> None:
        skillcorner = DATASETS[DatasetId.SKILLCORNER]
        assert skillcorner.licence == "MIT"
        assert skillcorner.may_redistribute
        assert skillcorner.commercial_use

    def test_exactly_two_datasets_are_safe_to_redistribute(self) -> None:
        assert {d.id for d in redistributable()} == {
            DatasetId.WYSCOUT,
            DatasetId.SKILLCORNER,
        }


class TestKnownTraps:
    def test_skillcorner_documents_the_extrapolation_bias(self) -> None:
        # Broadcast tracking fills off-camera players by extrapolation, so frames
        # look complete when they are not. This must stay documented on the dataset.
        caveats = " ".join(DATASETS[DatasetId.SKILLCORNER].caveats).lower()
        assert "is_detected" in caveats
        assert "extrapolated" in caveats

    def test_statsbomb_documents_the_360_field_trap(self) -> None:
        caveats = " ".join(DATASETS[DatasetId.STATSBOMB].caveats)
        assert "match_available_360" in caveats

    def test_only_skillcorner_provides_tracking(self) -> None:
        # socceraction is event-only; tracking needs custom code. Keep this explicit
        # so nobody assumes SPADL/VAEP can consume it.
        tracking = {d.id for d in DATASETS.values() if DataKind.TRACKING in d.kinds}
        assert tracking == {DatasetId.SKILLCORNER}


class TestIntegrity:
    @pytest.mark.parametrize("dataset", DATASETS.values(), ids=lambda d: d.id.value)
    def test_dataset_is_self_consistent(self, dataset: Dataset) -> None:
        assert dataset.kinds
        assert dataset.licence
        assert dataset.licence_source
        assert dataset.attribution
        assert dataset.matches > 0
        assert dataset.size_gb > 0

    @pytest.mark.parametrize("dataset", DATASETS.values(), ids=lambda d: d.id.value)
    def test_non_commercial_implies_non_redistributable(self, dataset: Dataset) -> None:
        # A licence permitting redistribution but barring commercial use would be
        # unusual; if one appears, it needs deliberate handling rather than a default.
        if not dataset.commercial_use:
            assert not dataset.may_redistribute

    def test_storage_estimate_is_realistic(self) -> None:
        # Guards against a plan that budgets for "a few GB". StatsBomb alone is 16 GB.
        assert total_size_gb() > 16
        assert total_size_gb(include_360=False) < total_size_gb()
