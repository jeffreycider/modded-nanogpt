"""Solve and audit every FP32 training kernel before reserving training GPUs."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from constrained_maxentslop import Shape,Waypoint,interpolate_shape,solve,InfeasibleKernel
from kernel_boundary import stability_boundary

# Immutable original-record anchors; these are not candidate hyperparameters.
REFERENCE_KAPPA=31.62336703513191
REFERENCE_START=750
TRAIN_STEPS=3010
REFERENCE_LR=0.01711068502877683
ORIGINAL_SHA256='4a8b026ea4c97aa7f56055ec6b2dcce3761b783f92bd159ea7eda4a4a8015030'

def original_lr_function(source):
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fast_slow_decay_lr')
    ns={'math':math}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'original_record','exec'),ns)
    return lambda t:ns['fast_slow_decay_lr'](t,TRAIN_STEPS,warmup_end=100,
        fast_decay_end=1750,peak_lr=.03,floor_lr=.006,min_lr=.0005018182,
        slow_decay_schedule='linear',fast_decay_exponent=.6,plateau_end=200)

def prepare(config,original_source,output):
    assert hashlib.sha256(original_source.encode()).hexdigest()==ORIGINAL_SHA256
    waypoints=[Waypoint(w['iteration'],Shape(**w['shape'])) for w in config['waypoints']]
    assert len(waypoints)==3 and 256<=waypoints[0].iteration<waypoints[1].iteration<waypoints[2].iteration==TRAIN_STEPS-1
    lr=original_lr_function(original_source)
    assert lr(REFERENCE_START)==REFERENCE_LR
    start=waypoints[0].iteration
    kernels=[];diagnostics=[];previous=None;began=time.time()
    ages=np.arange(256)
    for t in range(start,TRAIN_STEPS):
        shape=interpolate_shape(waypoints,t)
        target=REFERENCE_KAPPA*lr(t)/REFERENCE_LR
        try:previous=solve(shape,target,stability_boundary,previous=previous,phase_points=128)
        except InfeasibleKernel as e:raise InfeasibleKernel(f'iteration {t}: {e}') from e
        kernel=previous['weights'].astype(np.float32)
        actual,theta=stability_boundary(kernel)
        if abs(actual/target-1)>5e-5:
            raise InfeasibleKernel(f'FP32 boundary drift at {t}: {actual} vs {target}')
        moments=np.array([kernel.sum(dtype=np.float64),((-1.)**ages)@kernel,ages@kernel,np.log1p(ages)@kernel])
        if not np.allclose(moments,[1,0,shape.mean_lag,shape.log_moment],rtol=0,atol=1e-5):
            raise InfeasibleKernel(f'FP32 moments drift at {t}: {moments}')
        if not np.isfinite(kernel).all() or kernel.min()<0:
            raise InfeasibleKernel(f'Invalid FP32 weights at {t}')
        kernels.append(kernel)
        diagnostics.append({'iteration':t,'lr':lr(t),'kappa_target':target,'kappa_actual':actual,
                            'theta':theta,'entropy':previous['entropy'],'shape':shape.__dict__})
        if (t-start)%100==0:print(json.dumps({'iteration':t,'elapsed':time.time()-began,'target':target,'actual':actual}),flush=True)
    table=np.stack(kernels)
    metadata={'config':config,'start':start,'train_steps':TRAIN_STEPS,'original_sha256':ORIGINAL_SHA256,
        'reference_start':REFERENCE_START,'reference_lr':REFERENCE_LR,'reference_kappa':REFERENCE_KAPPA,
        'mu_sum_fixed':REFERENCE_KAPPA/REFERENCE_LR,'mu_token_mean_fixed':REFERENCE_KAPPA/REFERENCE_LR/524288,
        'max_relative_kappa_error':max(abs(d['kappa_actual']/d['kappa_target']-1) for d in diagnostics),
        'max_kernel_step_l1':float(np.abs(np.diff(table.astype(float),axis=0)).sum(axis=1).max()),
        'kernel_sha256':hashlib.sha256(table.tobytes()).hexdigest(),'elapsed_seconds':time.time()-began}
    np.savez_compressed(output,kernels=table,metadata=np.array(json.dumps(metadata)))
    Path(str(output)+'.json').write_text(json.dumps({'metadata':metadata,'diagnostics':diagnostics})+'\n')
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('original_source');p.add_argument('output');args=p.parse_args()
    prepare(json.loads(Path(args.config).read_text()),Path(args.original_source).read_text(),args.output)
