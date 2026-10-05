"""Three-waypoint desiderata and a prescribed, independently anchored κ*.

At a phase θ, the marginal-mode equation is linear in the kernel:
H(θ) = (1-exp(iθ))/κ*. We maximize entropy subject to this equation and
the original six constraints, then certify that this is the FIRST boundary.
The phase is derived by entropy maximization, never a sweep hyperparameter.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import linprog, minimize_scalar
from scipy.special import logsumexp


@dataclass(frozen=True)
class Shape:
    c0: float
    c1: float
    mean_lag: float
    log_moment: float


@dataclass(frozen=True)
class Waypoint:
    iteration: int
    shape: Shape


class InfeasibleKernel(ValueError):
    pass


def interpolate_shape(waypoints, iteration):
    if len(waypoints) != 3 or not all(a.iteration < b.iteration for a,b in zip(waypoints,waypoints[1:])):
        raise ValueError('expected three strictly ordered waypoints')
    if iteration <= waypoints[0].iteration:
        return waypoints[0].shape
    if iteration >= waypoints[-1].iteration:
        return waypoints[-1].shape
    a,b = next((a,b) for a,b in zip(waypoints,waypoints[1:]) if a.iteration <= iteration <= b.iteration)
    fraction = (iteration-a.iteration)/(b.iteration-a.iteration)
    return Shape(**{key:value+(getattr(b.shape,key)-value)*fraction for key,value in asdict(a.shape).items()})


def _fixed_phase(shape, kappa, theta, length=256, initial=None):
    """Strict concave entropy solve, with exact moment/phase constraints."""
    ages = np.arange(2,length,dtype=float)
    mass = 1-shape.c0-shape.c1
    if mass <= 0 or min(shape.c0,shape.c1) <= 0 or kappa <= 0:
        raise InfeasibleKernel('nonpositive mass, tap or κ*')
    feats=np.vstack([(-1.)**ages, ages, np.log1p(ages), np.cos(ages*theta), -np.sin(ages*theta)])
    response=(1-np.exp(1j*theta))/kappa-shape.c0-shape.c1*np.exp(-1j*theta)
    target=np.array([shape.c1-shape.c0, shape.mean_lag-shape.c1,
                     shape.log_moment-shape.c1*np.log(2),response.real,response.imag])/mass
    scales=np.array([2.,float(length),np.log(length),2.,2.])
    phi=feats/scales[:,None]; desired=target/scales
    lam=np.zeros(5) if initial is None else np.array(initial,copy=True)
    def evaluate(lam):
        z=phi.T@lam; p=np.exp(z-logsumexp(z))
        grad=phi@p-desired
        hess=(phi*p)@phi.T-np.outer(phi@p,phi@p)
        return logsumexp(z)-lam@desired,p,grad,hess
    for _ in range(100):
        objective,p,grad,hess=evaluate(lam)
        if np.max(np.abs(grad)) < 2e-11:
            weights=np.r_[shape.c0,shape.c1,mass*p]
            entropy=-np.dot(weights,np.log(np.maximum(weights,1e-300)))
            return weights,float(entropy),lam
        try:
            direction=-np.linalg.solve(hess,grad)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(direction).all() or np.linalg.norm(direction)>1e10:
            break
        rate=1.
        while rate>1e-10:
            trial=evaluate(lam+rate*direction)[0]
            if trial <= objective+1e-4*rate*np.dot(grad,direction):
                break
            rate*=.5
        if rate<=1e-10:
            break
        lam+=rate*direction
    raise InfeasibleKernel('entropy dual did not converge')


def phase_feasible(shape,kappa,theta,length=256):
    ages=np.arange(2,length,dtype=float)
    response=(1-np.exp(1j*theta))/kappa-shape.c0-shape.c1*np.exp(-1j*theta)
    a=np.vstack([np.ones(len(ages)),(-1.)**ages,ages/length,np.log1p(ages),
                 np.cos(ages*theta),-np.sin(ages*theta)])
    b=np.array([1-shape.c0-shape.c1,shape.c1-shape.c0,(shape.mean_lag-shape.c1)/length,
                shape.log_moment-shape.c1*np.log(2),response.real,response.imag])
    return linprog(np.zeros(len(ages)),A_eq=a,b_eq=b,bounds=(0,None),method='highs').success


def unconstrained(shape,length=256):
    ages=np.arange(2,length,dtype=float);mass=1-shape.c0-shape.c1
    scales=np.array([2.,float(length),np.log(length)])
    phi=np.vstack([(-1.)**ages,ages,np.log1p(ages)])/scales[:,None]
    target=np.array([shape.c1-shape.c0,shape.mean_lag-shape.c1,
                     shape.log_moment-shape.c1*np.log(2)])/mass/scales
    lam=np.zeros(3)
    for _ in range(100):
        z=phi.T@lam;p=np.exp(z-logsumexp(z));gradient=phi@p-target
        if max(abs(gradient))<2e-11:
            weights=np.r_[shape.c0,shape.c1,mass*p]
            return weights,np.r_[lam,0.,0.]
        hess=(phi*p)@phi.T-np.outer(phi@p,phi@p)
        try:direction=-np.linalg.solve(hess,gradient)
        except np.linalg.LinAlgError:break
        objective=logsumexp(z)-lam@target;rate=1.
        while rate>1e-10:
            trial=lam+rate*direction
            if logsumexp(phi.T@trial)-trial@target<=objective+1e-4*rate*np.dot(gradient,direction):break
            rate*=.5
        lam+=rate*direction
    raise InfeasibleKernel('original shape constraints infeasible')


def solve(shape,kappa,boundary,previous=None,phase_points=128,length=256):
    """Search phase branches; accept only independently certified first crossings.

    The returned solution is the highest entropy certified candidate found by
    this numerical search. No claim of a global nonconvex optimum is made.
    """
    if previous is None:
        try:
            weights,lam=unconstrained(shape,length)
            actual,theta=boundary(weights)
            if abs(actual/kappa-1)<2e-6:
                return {'weights':weights,'entropy':float(-weights@np.log(weights)),
                        'theta':theta,'dual':lam,'kappa':kappa}
        except InfeasibleKernel:
            pass
    # Follow a previously certified entropy maximum while the inputs vary
    # smoothly. Only the completed local candidate needs a spectral audit.
    if previous is not None and 'dual' in previous:
        local=[]
        def local_objective(theta):
            try:
                weights,entropy,lam=_fixed_phase(shape,kappa,theta,length,previous['dual'])
                local.append((entropy,theta,weights,lam))
                return -entropy
            except InfeasibleKernel:
                return 1e3
        theta=previous['theta']
        width=max(.002,theta*.015)
        lo=max(1e-7,theta-width);hi=min(np.pi-1e-7,theta+width)
        local_objective(theta)
        minimize_scalar(local_objective,bounds=(lo,hi),method='bounded',options={'xatol':1e-8,'maxiter':35})
        if local:
            entropy,theta,weights,lam=max(local,key=lambda x:x[0])
            actual,_=boundary(weights)
            if abs(actual/kappa-1)<2e-6:
                return {'weights':weights,'entropy':entropy,'theta':theta,'dual':lam,'kappa':kappa}
    grid=np.unique(np.r_[np.geomspace(1e-5,.12,phase_points//2),
                         np.linspace(.12,np.pi-1e-5,phase_points)])
    if previous is not None:
        grid=np.unique(np.r_[grid,previous['theta'],previous['theta']+np.linspace(-.03,.03,9)])
    candidates=[]
    for theta in grid:
        if not 0<theta<np.pi or abs(1-np.exp(1j*theta))>kappa:
            continue
        try:
            weights,entropy,lam=_fixed_phase(shape,kappa,theta,length)
        except InfeasibleKernel:
            continue
        actual,first_phase=boundary(weights)
        if abs(actual/kappa-1)<2e-6:
            candidates.append((entropy,theta,weights,lam))
    if not candidates:
        raise InfeasibleKernel(f'no certified first-boundary solution for {shape}, κ*={kappa}')
    best=max(candidates,key=lambda x:x[0])
    center=best[1];index=np.searchsorted(grid,center)
    lo=grid[max(0,index-1)];hi=grid[min(len(grid)-1,index+1)]
    def objective(theta):
        try:
            weights,entropy,lam=_fixed_phase(shape,kappa,theta,length)
            actual,_=boundary(weights)
            if abs(actual/kappa-1)<2e-6:
                candidates.append((entropy,theta,weights,lam))
                return -entropy
        except InfeasibleKernel:
            pass
        return 1e3
    minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'xatol':1e-8,'maxiter':50})
    entropy,theta,weights,lam=max(candidates,key=lambda x:x[0])
    return {'weights':weights,'entropy':entropy,'theta':theta,'dual':lam,'kappa':kappa}
