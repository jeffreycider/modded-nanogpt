"""Decay-sensitive frozen positive-mode matching and maximum LR.

Physical recurrence: x_next=(1-eta*lambda(eta))*x
                          -eta*mu_sum*sum c[k]*x_past[k].
mu_sum includes the training-token normalization; decay does not.
No normalization of c+(lambda/mu_sum)*delta0 is permitted.
This handles Muon decay; it is NOT a MuonH hyperball stability model.
"""
import math
import numpy as np
from kernel_boundary import stability_boundary


def effective_kernel(weights,mu_sum,weight_decay):
    if not np.isfinite(mu_sum) or mu_sum<=0 or not np.isfinite(weight_decay) or weight_decay<0:
        raise ValueError('expected positive curvature and nonnegative decay')
    c=np.array(weights,dtype=np.float64,copy=True)
    if c.ndim!=1 or not len(c):raise ValueError('expected a nonempty kernel')
    c[0]+=weight_decay/mu_sum
    return c


def matching_boundary(weights,mu_sum,weight_decay=0.0):
    """Match eta*mu_sum against this κ*, retaining the extra decay mass."""
    return stability_boundary(effective_kernel(weights,mu_sum,weight_decay))


def matching_error(eta,weights,mu_sum,weight_decay=0.0):
    return eta*mu_sum/matching_boundary(weights,mu_sum,weight_decay)[0]-1


def curvature_from_anchor(weights,eta,weight_decay=0.0):
    """Infer fixed spatial curvature from an existing marginal LR+decay anchor.

    At that anchor retention=1-eta*decay is fixed, and eta*mu_sum is the
    first positive spatial gain. Recheck critical_lr along the intended LR ray.
    """
    if not np.isfinite(eta) or eta<=0:raise ValueError('expected positive anchor LR')
    if not np.isfinite(weight_decay) or weight_decay<0:raise ValueError('invalid decay')
    return stability_boundary(weights,retention=1-eta*weight_decay)[0]/eta


def critical_lr(weights,mu_sum,weight_decay=0.0,decay_per_lr=0.0,grid_size=32768):
    """First unit-circle boundary with lambda(eta)=weight_decay+decay_per_lr*eta.

    Original Muon policy: lambda=.1*(eta/.025)=4*eta, so decay_per_lr=4.
    This follows that dependency during the LR scan instead of freezing decay.
    At z=exp(i theta), eta=-sin(theta)/(mu_sum*Im C).
    Eliminating eta gives gamma*sin² - sin*a*b - (1-cos)*b² = 0,
    a=mu_sum*Re C+weight_decay, b=mu_sum*Im C, gamma=decay_per_lr.
    """
    if not np.isfinite(decay_per_lr) or decay_per_lr<0:raise ValueError('invalid decay/LR ratio')
    c=effective_kernel(weights,mu_sum,0.0)
    if not np.isfinite(weight_decay) or weight_decay<0:raise ValueError('invalid fixed decay')
    if decay_per_lr==0:
        kappa,theta=matching_boundary(c,mu_sum,weight_decay)
        return kappa/mu_sum,theta
    if grid_size<8*len(c):raise ValueError('phase grid too small')
    theta=np.linspace(0,np.pi,grid_size+1)
    h=np.fft.rfft(c,n=2*grid_size)
    ages=np.arange(len(c))
    def phase(t,response):
        sine=np.sin(t);a=mu_sum*response.real+weight_decay;b=mu_sum*response.imag
        return decay_per_lr*sine*sine-sine*a*b-(1-np.cos(t))*b*b
    values=phase(theta,h)
    crossings=np.flatnonzero(values[1:-1]*values[2:]<=0)+1
    candidates=[]
    for index in crossings:
        lo,hi=theta[index],theta[index+1];flo=values[index]
        for _ in range(45):
            mid=(lo+hi)/2;response=np.dot(c,np.exp(-1j*ages*mid));fm=phase(mid,response)
            if flo*fm<=0:hi=mid
            else:lo,flo=mid,fm
        t=(lo+hi)/2;response=np.dot(c,np.exp(-1j*ages*t))
        if abs(response.imag)<1e-14:continue
        eta=-math.sin(t)/(mu_sum*response.imag)
        residual=np.exp(1j*t)-1+eta*(weight_decay+decay_per_lr*eta)+eta*mu_sum*response
        if eta>0 and abs(residual)<1e-7:
            candidates.append((float(eta),float(t)))
    # The real root z=-1 obeys gamma*eta²+(mu_sum*C(pi)+lambda0)*eta=2.
    linear=mu_sum*np.dot(c,(-1.)**ages)+weight_decay
    root=math.sqrt(linear*linear+8*decay_per_lr)
    eta_pi=4/(linear+root) if linear>=0 else (-linear+root)/(2*decay_per_lr)
    candidates.append((float(eta_pi),math.pi))
    return min(candidates)
