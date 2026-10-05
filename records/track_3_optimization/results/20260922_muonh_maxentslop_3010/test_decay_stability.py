import numpy as np
import pytest
from decay_stability import effective_kernel,matching_boundary,matching_error,curvature_from_anchor,critical_lr


def test_decay_mass_is_retained_and_input_unchanged():
    c=np.array([.5,.5]);modified=effective_kernel(c,2.,.4)
    np.testing.assert_array_equal(c,[.5,.5])
    np.testing.assert_allclose(modified,[.7,.5])
    assert modified.sum()==pytest.approx(1.2)


def test_fixed_decay_memoryless_exact_boundary():
    eta,_=critical_lr([1.],2.,weight_decay=.3)
    assert eta==pytest.approx(2/2.3)
    assert matching_error(eta,[1.],2.,.3)==pytest.approx(0,abs=1e-12)


def test_lr_following_decay_memoryless_exact_boundary():
    eta,_=critical_lr([1.],2.,decay_per_lr=4.)
    exact=(-2+np.sqrt(4+32))/8
    assert eta==pytest.approx(exact,rel=1e-12)
    assert matching_error(eta,[1.],2.,4*eta)==pytest.approx(0,abs=1e-12)


def test_delayed_kernel_including_complex_crossing():
    eta,theta=critical_lr([0.,1.],2.,decay_per_lr=.25)
    assert eta==pytest.approx(.5,rel=1e-10)
    assert 0<theta<np.pi
    assert matching_error(eta,[0.,1.],2.,.25*eta)==pytest.approx(0,abs=1e-10)
    def roots(rate):return np.roots([1.,-(1-.25*rate**2),rate*2])
    assert max(abs(roots(eta*.999)))<1
    assert max(abs(roots(eta*1.001)))>1


def test_anchor_recovered_with_fixed_and_rate_following_decay():
    for gamma in [0.,4.]:
        anchor=.2;decay=.1+gamma*anchor
        mu=curvature_from_anchor([1.],anchor,decay)
        eta,_=critical_lr([1.],mu,weight_decay=.1,decay_per_lr=gamma)
        assert eta==pytest.approx(anchor,rel=1e-12)


def test_no_decay_reduces_to_old_boundary():
    eta,_=critical_lr([.5,.5],3.)
    assert eta==pytest.approx(matching_boundary([.5,.5],3.)[0]/3.)


def test_negative_decay_rejected():
    with pytest.raises(ValueError):effective_kernel([1.],2.,-.1)


def test_constrained_entropy_matches_decay_adjusted_kernel():
    from constrained_maxentslop import Shape,solve_stability_matched
    from kernel_boundary import stability_boundary
    shape=Shape(.07235,.02235,90.,3.7153849427)
    offset=.002
    result=solve_stability_matched(shape,.02,1000.,offset*1000.,stability_boundary,phase_points=64)
    c=result['weights'];shifted=c.copy();shifted[0]+=offset
    assert c.sum()==pytest.approx(1,abs=1e-10)
    assert c[0]==shape.c0 and c[1]==shape.c1
    assert stability_boundary(shifted)[0]==pytest.approx(20.,rel=1e-7)
    assert abs(stability_boundary(c)[0]/20-1)>1e-3
