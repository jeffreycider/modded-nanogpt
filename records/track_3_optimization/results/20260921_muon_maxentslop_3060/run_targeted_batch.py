from pathlib import Path
import os,json,subprocess,shutil,sys,time
import wandb
# Explicit experiment design; this worker chooses no new experiments.
design=json.loads(Path(sys.argv[1]).read_text());seed=int(sys.argv[2]);name=sys.argv[3]
root=Path('/scratch/artifacts/critroll/backward_maxentslop_20261005/targeted_c0')/name;root.mkdir(parents=True,exist_ok=True)
source=Path('/tmp/backward_tuner_source')
for src,dst in [('train_gpt_muon_maxentslop.py','train.py'),('checkpoint_state.py','checkpoint_state.py'),('waypoint_schedule.py','waypoint_schedule.py')]:shutil.copy2(source/src,root/dst)
(root/'data').mkdir(exist_ok=True);(root/'data/fineweb10B').symlink_to('/tmp/kappa_lr_data',target_is_directory=True)
checkpoint_root='/scratch/checkpoints/critroll/backward_maxentslop_defazio_'+('s1_' if seed==1 else '')+'20261005'
checkpoint=checkpoint_root+f'/step_{design["stage"]:05d}'
assert Path(checkpoint,'manifest.json').exists(),checkpoint
specs=[]
for t in design['trials']:
 path=root/(t['name']+'.json');path.write_text(json.dumps(t['schedule'],indent=2));specs.append({'name':t['name'],'schedule_path':str(path),'resume_checkpoint':checkpoint})
(root/'batch.json').write_text(json.dumps(specs));(root/'design.json').write_text(json.dumps(design,indent=2))
env=dict(os.environ,TRIAL_BATCH_PATH=str(root/'batch.json'),SEGMENT_FINAL_ONLY='1',PYTHONUNBUFFERED='1')
for key in ['RESUME_CHECKPOINT','STOP_STEP','CHECKPOINT_ROOT','CHECKPOINT_STEPS','WAYPOINT_SCHEDULE_PATH','COMPARE_CHECKPOINT','COMPARE_RESTORE_CHECKPOINT']:env.pop(key,None)
run=wandb.init(mode='offline',project='nanogpt-backward-waypoints',name=name,dir=str(root),config={'seed':seed,'stage':design['stage'],'purpose':design['purpose']})
results=[];began=time.time()
with (root/'console.log').open('w') as log:
 p=subprocess.Popen(['/opt/venv211/bin/torchrun','--standalone','--nproc_per_node=8','train.py','--seed',str(seed),'--train_steps','3060'],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 for line in p.stdout:
  log.write(line);log.flush()
  if line.startswith('TRIAL_RESULT '):
   r=json.loads(line[len('TRIAL_RESULT '):]);r.update(seed=seed,elapsed_seconds=time.time()-began);results.append(r);run.log({'final_loss':r['final_loss'],'elapsed_seconds':r['elapsed_seconds']});(root/'results.json').write_text(json.dumps(results,indent=2));subprocess.run(['sync',str(root)],check=True);print(json.dumps(r),flush=True)
 rc=p.wait()
run.finish();(root/'status.json').write_text(json.dumps({'state':'done' if rc==0 else 'failed','returncode':rc,'seed':seed,'results':results},indent=2));subprocess.run(['sync',str(root)],check=True)
if rc:sys.exit(rc)
