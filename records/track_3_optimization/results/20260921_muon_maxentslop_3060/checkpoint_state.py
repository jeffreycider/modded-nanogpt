"""Lossless, rank-local training state for frozen-checkpoint segment tuning."""
import json
import os
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
import torch.distributed as dist

FORMAT=1
_cached_payloads={}

def clone_tree(value):
    if isinstance(value,torch.Tensor):return value.clone()
    if isinstance(value,dict):return {k:clone_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [clone_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(clone_tree(v) for v in value)
    return value

def pack_optimizer(optimizer):
    state=optimizer.state_dict()
    packed={**state,'state':{}}
    for pid,values in state['state'].items():
        packed['state'][pid]={}
        for name,value in values.items():
            if hasattr(value,'buffer') and hasattr(value,'newest_row') and hasattr(value,'length'):
                value={'_raw_history':True,'buffer':value.buffer,'newest_row':value.newest_row,'length':value.length}
            packed['state'][pid][name]=value
    return packed

def unpack_optimizer(optimizer,state,history_class):
    for values in state['state'].values():
        for key,value in list(values.items()):
            if isinstance(value,dict) and value.get('_raw_history'):
                history=history_class.__new__(history_class)
                history.buffer=value['buffer'];history.newest_row=value['newest_row'];history.length=value['length']
                values[key]=history
    optimizer.load_state_dict(state)

def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':(n[0],n[1].tolist(),n[2],n[3],n[4]),
            'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state()}

def restore_rng(state):
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch_cpu'].cpu());torch.cuda.set_rng_state(state['torch_cuda'].cpu())

def save(root,step,model,optimizers,seed,train_steps):
    rank=dist.get_rank();world=dist.get_world_size();root=Path(root)/f'step_{step:05d}'
    root.mkdir(parents=True,exist_ok=True);dist.barrier();torch.cuda.synchronize()
    payload={'format':FORMAT,'step':step,'world_size':world,'rank':rank,'seed':seed,'train_steps':train_steps,
             'optimizers':[pack_optimizer(o) for o in optimizers],
             'optimizer_steps':[getattr(o,'_step',None) for o in optimizers],'rng':rng_state()}
    if rank==0:payload['model']=model.state_dict()
    path=root/f'rank_{rank:02d}.pt';temporary=root/f'rank_{rank:02d}.tmp'
    torch.save(payload,temporary);os.replace(temporary,path);dist.barrier()
    if rank==0:
        manifest={k:payload[k] for k in ['format','step','world_size','seed','train_steps']}
        manifest['files']={f'rank_{r:02d}.pt':(root/f'rank_{r:02d}.pt').stat().st_size for r in range(world)}
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        subprocess.run(['sync',str(root)],check=True)
    dist.barrier()

def load(root,model,optimizers,history_class,seed,train_steps,device):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());rank=dist.get_rank()
    assert m['format']==FORMAT and m['world_size']==dist.get_world_size()
    assert (m['seed'],m['train_steps'])==(seed,train_steps)
    path=root/f'rank_{rank:02d}.pt';assert path.stat().st_size==m['files'][path.name]
    key=(str(path),str(device))
    cache=os.environ.get('CACHE_CHECKPOINT_GPU')=='1'
    if cache and key in _cached_payloads:
        state=_cached_payloads[key]
    else:
        if cache:_cached_payloads.clear()
        state=torch.load(path,map_location=device,weights_only=True)
        if cache:_cached_payloads[key]=state
    assert (state['rank'],state['step'])==(rank,m['step'])
    if rank==0:model.load_state_dict(state['model'])
    for value in model.state_dict().values():dist.broadcast(value,0)
    for opt,packed,count in zip(optimizers,state['optimizers'],state['optimizer_steps']):
        unpack_optimizer(opt,clone_tree(packed) if cache else packed,history_class)
        if count is not None:opt._step=count
    restore_rng(state['rng']);dist.barrier()
    return m['step']

def data_position(files,batches,batch_size):
    """Match the original generator's strict shard rollover condition."""
    if batches<0:raise ValueError('negative batch position')
    for index,path in enumerate(files):
        header=np.fromfile(path,dtype=np.int32,count=256)
        assert header[0]==20240520 and header[1]==1
        capacity=(int(header[2])-2)//batch_size
        if batches<capacity:return index,batches*batch_size
        batches-=capacity
    raise ValueError('resume data exhausted')

def compare(root,model,optimizers,device):
    """Compare resumed state with a frozen uninterrupted checkpoint."""
    rank=dist.get_rank()
    reference=torch.load(Path(root)/f'rank_{rank:02d}.pt',map_location=device,weights_only=True)
    differences=[]
    def walk(a,b,path):
        if isinstance(a,torch.Tensor):
            if not torch.equal(a,b):
                differences.append({'path':path,'max_abs':float((a.float()-b.float()).abs().max()),'relative_l2':float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-30))})
        elif isinstance(a,dict):
            assert set(a)==set(b),(path,set(a),set(b))
            for key in a:walk(a[key],b[key],path+'/'+str(key))
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
        else:
            if a!=b:differences.append({'path':path,'a':a,'b':b})
    walk(reference['optimizers'],[pack_optimizer(o) for o in optimizers],'optimizers')
    if rank==0:walk(reference['model'],model.state_dict(),'model')
    report={'rank':rank,'exact':not differences,'differences':differences}
    reports=[None for _ in range(dist.get_world_size())];dist.all_gather_object(reports,report)
    return reports
