"""Backward ordinary-MaxEntSlop fitting; every candidate scores the fixed suffix."""
import argparse,ast,json,os,re,shutil,subprocess,time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from waypoint_schedule import load_schedule

def original_solver(source):
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and
           n.name in {'MomentumDesiderata','solve_for_momentum_kernel_given_desiderata'}]
    ns={'dataclass':dataclass,'np':np,'torch':torch,'MAXENTSLOP_HISTORY_LENGTH':256,'MomentumKernel':torch.Tensor}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'record_entropy_solver','exec'),ns)
    return ns['solve_for_momentum_kernel_given_desiderata'],ns['MomentumDesiderata']

def reference_shape(t):
    a=dict(c0=.07235,c1=.02235,mean_lag=90.,log_moment=3.7153849427)
    b=dict(c0=.137,c1=.087,mean_lag=24.05,log_moment=2.3786333904)
    f=min(1.,max(0.,(t-780)/2340))
    return {k:a[k]+f*(b[k]-a[k]) for k in a}

def population(center,free,rng,count,scale,solver,desiderata,out):
    result=[];attempts=0
    while len(result)<count and attempts<count*100:
        attempts+=1;c=json.loads(json.dumps(center))
        valid=True
        if result:
            for i in free:
                s=c['waypoints'][i]['shape']
                widths={'c0':.012,'c1':.008,'mean_lag':max(2.,s['mean_lag']*.12),'log_moment':.08}
                for hp,width in widths.items():s[hp]+=float(rng.normal(0,width*scale))
                if min(s['c0'],s['c1'])<=0 or s['c0']+s['c1']>=.8:
                    valid=False;break
        if not valid:continue
        name=f'trial_{len(result):03d}';path=out/(name+'.json');path.write_text(json.dumps(c))
        try:load_schedule(path,solver,desiderata)
        except (AssertionError,ValueError,np.linalg.LinAlgError):continue
        result.append((name,path,c))
    if len(result)<count:raise RuntimeError('Too few feasible ordinary entropy kernels')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint_root');p.add_argument('output_dir')
    p.add_argument('--rounds',type=int,default=2);p.add_argument('--population',type=int,default=10)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--points',default='780,1250,1750,2250,2500,2750,2900,3000')
    args=p.parse_args();root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('train_gpt_muon_maxentslop.py');solver,d=original_solver(source.read_text())
    rng=np.random.default_rng(51005);suffix=None;stage_results=[]
    for start in sorted(map(int,args.points.split(',')),reverse=True):
        checkpoint=Path(args.checkpoint_root)/f'step_{start:05d}'
        assert (checkpoint/'manifest.json').exists(),checkpoint
        left={'iteration':start,'shape':reference_shape(start)}
        center={'waypoints':[left,{'iteration':3060,'shape':reference_shape(3060)}]} if suffix is None else {'waypoints':[left]+suffix['waypoints']}
        free=[0,1] if suffix is None else [0];best_loss=float('inf')
        for round_id in range(args.rounds):
            work=root/f'stage_{start:05d}'/f'round_{round_id}';work.mkdir(parents=True,exist_ok=True)
            for file in ['checkpoint_state.py','waypoint_schedule.py']:
                shutil.copy2(source.with_name(file),work/file)
            shutil.copy2(source,work/'train.py');(work/'data').mkdir(exist_ok=True)
            (work/'data/fineweb10B').symlink_to('/tmp/kappa_lr_data',target_is_directory=True)
            candidates=population(center,free,rng,args.population,.6**round_id,solver,d,work)
            specs=[{'name':name,'schedule_path':str(path),'resume_checkpoint':str(checkpoint)} for name,path,_ in candidates]
            (work/'batch.json').write_text(json.dumps(specs))
            env=dict(os.environ,TRIAL_BATCH_PATH=str(work/'batch.json'),SEGMENT_FINAL_ONLY='1',PYTHONUNBUFFERED='1')
            for key in ['CHECKPOINT_ROOT','CHECKPOINT_STEPS','RESUME_CHECKPOINT','STOP_STEP','WAYPOINT_SCHEDULE_PATH','CONSTRAINED_SCHEDULE_PATH','COMPARE_CHECKPOINT']:env.pop(key,None)
            with (work/'console.log').open('w') as log:
                proc=subprocess.Popen(['/opt/venv211/bin/torchrun','--standalone','--nproc_per_node=8','train.py','--seed',str(args.seed),'--train_steps','3060'],cwd=work,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                losses={}
                for line in proc.stdout:
                    log.write(line);log.flush()
                    if line.startswith('TRIAL_RESULT '):
                        record=json.loads(line[len('TRIAL_RESULT '):]);losses[record['name']]=record['final_loss'];print(json.dumps({'stage':start,'round':round_id,**record}),flush=True)
                rc=proc.wait()
            if rc or len(losses)!=len(candidates):raise RuntimeError(f'Stage failed: {work}, exit={rc}')
            winner=min(candidates,key=lambda c:losses[c[0]])
            if losses[winner[0]]<best_loss:best_loss=losses[winner[0]];center=winner[2]
            (work/'results.json').write_text(json.dumps({'losses':losses,'best_loss':best_loss,'best_schedule':center},indent=2))
        suffix=center;stage_results.append({'start':start,'loss':best_loss,'schedule':suffix})
        (root/'best_schedule.json').write_text(json.dumps(suffix,indent=2));(root/'stages.json').write_text(json.dumps(stage_results,indent=2))
        subprocess.run(['sync',str(root)],check=True)
    work=root/'forward_validation';work.mkdir(exist_ok=True)
    for file in ['checkpoint_state.py','waypoint_schedule.py']:
        shutil.copy2(source.with_name(file),work/file)
    shutil.copy2(source,work/'train.py');(work/'data').mkdir(exist_ok=True)
    (work/'data/fineweb10B').symlink_to('/tmp/kappa_lr_data',target_is_directory=True)
    env=dict(os.environ,WAYPOINT_SCHEDULE_PATH=str(root/'best_schedule.json'),PYTHONUNBUFFERED='1')
    for key in ['CHECKPOINT_ROOT','CHECKPOINT_STEPS','RESUME_CHECKPOINT','STOP_STEP','TRIAL_BATCH_PATH','SEGMENT_FINAL_ONLY','CONSTRAINED_SCHEDULE_PATH','COMPARE_CHECKPOINT']:env.pop(key,None)
    with (work/'console.log').open('w') as log:
        subprocess.run(['/opt/venv211/bin/torchrun','--standalone','--nproc_per_node=8','train.py','--seed',str(args.seed),'--train_steps','3060'],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    curve=[(int(t),float(loss)) for t,loss in re.findall(r'step:(\d+)/\d+ val_loss:([\d.]+)',(work/'console.log').read_text())]
    (work/'results.json').write_text(json.dumps({'seed':args.seed,'validation':curve,'final_loss':curve[-1][1]},indent=2))
    subprocess.run(['sync',str(root)],check=True)
    print(json.dumps({'state':'done','schedule':suffix,'stages':stage_results}),flush=True)

if __name__=='__main__':main()
