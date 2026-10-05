"""Generate independent 14-coordinate local proposals around a measured center.

CPU screen only: accepted proposals still require prepare_waypoint_schedule.py
at every update before GPU training. Neither curvature nor LR is a search input.
Run with OPENBLAS_NUM_THREADS=1 for small entropy dual solves.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from constrained_maxentslop import Shape,Waypoint,interpolate_shape,solve_stability_matched,InfeasibleKernel,unconstrained
from kernel_boundary import stability_boundary
from decay_stability import curvature_from_anchor
from prepare_waypoint_schedule import TRAIN_STEPS,REFERENCE_LR,original_lr_function

SCALES=[dict(c0=.004,c1=.002,mean_lag=4.,log_moment=.025),
        dict(c0=.006,c1=.004,mean_lag=4.,log_moment=.05),
        dict(c0=.008,c1=.008,mean_lag=.7,log_moment=.006)]

def propose(center,rng,scale=1.):
    config=json.loads(json.dumps(center));wps=config['waypoints']
    wps[0]['iteration']=int(np.clip(wps[0]['iteration']+rng.normal(0,110*scale),256,TRAIN_STEPS-3))
    wps[1]['iteration']=int(np.clip(wps[1]['iteration']+rng.normal(0,4*scale),wps[0]['iteration']+1,TRAIN_STEPS-2))
    wps[2]['iteration']=TRAIN_STEPS-1
    for w,stds in zip(wps,SCALES):
        for hp,std in stds.items():w['shape'][hp]+=float(rng.normal(0,std*scale))
    return config

def screen(config):
    reference,_=unconstrained(Shape(.07235,.02235,90.,3.7153849427))
    mu=curvature_from_anchor(reference.astype(np.float32),REFERENCE_LR,.1)
    lr=original_lr_function(None)
    wps=[Waypoint(w['iteration'],Shape(**w['shape'])) for w in config['waypoints']]
    points=np.unique(np.r_[np.linspace(wps[0].iteration,wps[1].iteration,16).astype(int),
                           np.arange(wps[1].iteration,TRAIN_STEPS)])
    previous=None;passed=[]
    for t in points:
        try:
            previous=solve_stability_matched(interpolate_shape(wps,int(t)),lr(t),mu,4*lr(t),
                                             stability_boundary,previous=previous,phase_points=128)
            passed.append(int(t))
        except (InfeasibleKernel,ValueError) as e:
            return dict(sampled_feasible=False,passed=passed,rejection=str(e))
    return dict(sampled_feasible=True,passed=passed)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('center');p.add_argument('output_dir')
    p.add_argument('--seed',type=int,default=4873);p.add_argument('--attempts',type=int,default=40)
    p.add_argument('--accept',type=int,default=4);p.add_argument('--scale',type=float,default=1.)
    args=p.parse_args();center=json.loads(Path(args.center).read_text());rng=np.random.default_rng(args.seed)
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True);accepted=[]
    with (root/'screen.jsonl').open('w') as log:
        for trial in range(args.attempts):
            config=propose(center,rng,args.scale);result=screen(config);name=f'proposal_s{args.seed}_{trial:03d}'
            row=dict(name=name,config=config,**result);log.write(json.dumps(row)+'\n');log.flush()
            print(json.dumps(row),flush=True)
            if result['sampled_feasible']:
                (root/(name+'.json')).write_text(json.dumps(config));accepted.append(name)
                if len(accepted)>=args.accept:break
    (root/'accepted.json').write_text(json.dumps(accepted)+'\n')
