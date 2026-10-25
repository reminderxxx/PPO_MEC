import pytest
from scripts.prepare_event_joint_inputs import assert_disjoint


def unit(index,start,end):return dict(index=index,split='dev',offset=index*1000,frames=[dict(time_index=start),dict(time_index=end)])


def test_raw_time_overlap_is_rejected_even_when_rank_and_id_differ():
    with pytest.raises(ValueError,match='Raw time'):assert_disjoint([unit(99,50,150)],[unit(0,0,100)])
    with pytest.raises(ValueError,match='Raw time'):assert_disjoint([unit(1,100,200)],[unit(0,0,100)])
    assert_disjoint([unit(99,101,200)],[unit(0,0,100)])
