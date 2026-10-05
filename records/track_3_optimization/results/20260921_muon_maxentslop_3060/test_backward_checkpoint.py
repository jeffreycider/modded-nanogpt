import ast
from pathlib import Path
import numpy as np
import torch
from checkpoint_state import pack_optimizer,unpack_optimizer,data_position
from waypoint_schedule import interpolate_waypoint_kernels

class History:
    def __init__(self):
        self.buffer=torch.arange(24,dtype=torch.float32).reshape(4,6)
        self.newest_row=2;self.length=4

def test_optimizer_history_roundtrip_preserves_next_update(tmp_path):
    p=torch.nn.Parameter(torch.arange(6,dtype=torch.float32))
    opt=torch.optim.AdamW([p],lr=.01)
    p.square().sum().backward();opt.step();opt.zero_grad()
    opt.state[p]['history']=History()
    torch.save(pack_optimizer(opt),tmp_path/'state.pt')
    q=torch.nn.Parameter(p.detach().clone());other=torch.optim.AdamW([q],lr=.01)
    unpack_optimizer(other,torch.load(tmp_path/'state.pt',weights_only=True),History)
    h=other.state[q]['history'];assert h.newest_row==2 and h.length==4
    torch.testing.assert_close(h.buffer,opt.state[p]['history'].buffer,rtol=0,atol=0)
    p.square().sum().backward();q.square().sum().backward();opt.step();other.step()
    torch.testing.assert_close(p,q,rtol=0,atol=0)

def test_data_resume_matches_strict_reference_rollover(tmp_path):
    files=[]
    for i,n in enumerate([17,19,14]):
        h=np.zeros(256,dtype=np.int32);h[:3]=[20240520,1,n]
        f=tmp_path/str(i);h.tofile(f);files.append(f)
    reference=[]
    for i,n in enumerate([17,19,14]):
        pos=0
        while pos+4+1<n:
            reference.append((i,pos));pos+=4
    for batch,position in enumerate(reference):assert data_position(files,batch,4)==position

def test_linear_kernel_anneal_matches_reference_at_every_update():
    kernels=torch.rand(3,256);wps=[{'iteration':780},{'iteration':1750},{'iteration':3060}]
    for t in range(780,3060):
        i=0 if t<1750 else 1;a=wps[i]['iteration'];b=wps[i+1]['iteration']
        torch.testing.assert_close(interpolate_waypoint_kernels(kernels,wps,t),
                                   kernels[i].lerp(kernels[i+1],(t-a)/(b-a)),rtol=0,atol=0)

def test_original_lr_and_model_functions_are_unchanged():
    import subprocess
    p=Path(__file__).with_name('train_gpt_muon_maxentslop.py')
    original=subprocess.check_output(['git','show','36bff7fe0c4427e5801a78e91bd4fbcc7b992e3c:'+str(p.relative_to(Path(__file__).parents[4]))],text=True,cwd=p.parent)
    a=ast.parse(original);b=ast.parse(p.read_text())
    def nodes(t,kind):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,kind)}
    fa=nodes(a,ast.FunctionDef);fb=nodes(b,ast.FunctionDef)
    for name in ['set_hparams','zeropower_via_newtonschulz5','muon_update','muon_update_maxentslop','_weighted_sum_over_history','solve_for_momentum_kernel_given_desiderata']:
        assert fa[name]==fb[name]
    for name in ['GPT','Block','Linear','RMSNorm','RawGradientHistory']:
        ca=nodes(a,ast.ClassDef);cb=nodes(b,ast.ClassDef)
        if name in ca:assert ca[name]==cb[name]


def test_cached_optimizer_template_stays_immutable():
    from checkpoint_state import clone_tree
    p=torch.nn.Parameter(torch.arange(6,dtype=torch.float32))
    opt=torch.optim.AdamW([p],lr=.01)
    p.square().sum().backward();opt.step();opt.zero_grad()
    opt.state[p]['history']=History()
    template=pack_optimizer(opt)
    frozen=clone_tree(template)
    q=torch.nn.Parameter(p.detach().clone());other=torch.optim.AdamW([q],lr=.01)
    unpack_optimizer(other,clone_tree(template),History)
    other.state[q]['history'].buffer.zero_()
    q.square().sum().backward();other.step()
    for pid,values in frozen['state'].items():
        for key,value in values.items():
            actual=template['state'][pid][key]
            if isinstance(value,torch.Tensor):torch.testing.assert_close(actual,value,rtol=0,atol=0)
            elif isinstance(value,dict):torch.testing.assert_close(actual['buffer'],value['buffer'],rtol=0,atol=0)
