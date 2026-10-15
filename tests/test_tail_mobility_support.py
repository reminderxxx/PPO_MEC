"""External-geometry support selection never relaxes its eligibility for a score."""
from copy import deepcopy
import pytest
from scripts.joint_tail_mobility_support import select_cohort
from tests.test_parameterized_control_learning import four_site_env


def test_support_keeps_real_tracks_and_does_not_mutate_source():
    e=four_site_env();before=deepcopy(e.frames)
    frames,ids=select_cohort(e.frames,e.rsus,count=2)
    assert len(ids)==2 and frames==before and e.frames==before


def test_support_fails_when_only_one_site_is_available():
    e=four_site_env();sites=deepcopy(e.rsus)
    for row in sites[1:]:row.position_x=1e9
    with pytest.raises(ValueError,match='not enough'):
        select_cohort(e.frames,sites,count=2)
