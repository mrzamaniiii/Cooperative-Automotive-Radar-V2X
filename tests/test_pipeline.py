import numpy as np
from src.geometry import is_occluded
from src.scenario import generate
from src.tracking import tracker_at
from src.evaluation import evaluate_multi_tracker

def test_occlusion():
    assert is_occluded(np.array([0.,0.]),np.array([20.,0.]),np.array([10.,0.]),2.)
    assert not is_occluded(np.array([0.,0.]),np.array([20.,10.]),np.array([10.,0.]),2.)

def test_stream_deterministic():
    t,a,e=generate(); t2,b,f=generate()
    assert np.array_equal(a,b)
    assert len(e)==len(f)
    assert all(np.array_equal(x.z,y.z) for x,y in zip(e,f))
    assert all(x.arrival_time >= x.measurement_time for x in e)

def test_tracker_smoke():
    ts,truth,events=generate()
    out=[tracker_at(float(t),events,True) for t in ts]
    m=evaluate_multi_tracker(out,truth,ts)
    assert np.isfinite(m['rmse'])
    assert m['missed_target_frames'] >= 0
