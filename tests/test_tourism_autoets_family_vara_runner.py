import pytest

from scripts.run_tourism_autoets_family_vara import _candidate_ids


def test_candidate_ids_are_nonempty_and_unique() -> None:
    assert _candidate_ids("aaa_damped, zzz_undamped") == [
        "aaa_damped",
        "zzz_undamped",
    ]
    with pytest.raises(ValueError, match="unique"):
        _candidate_ids("aaa_damped,aaa_damped")
