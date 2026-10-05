import ast
import os
from pathlib import Path
import random
import types
import numpy as np
import pytest
import torch
from constrained_maxentslop import Shape,Waypoint,interpolate_shape,solve,unconstrained,InfeasibleKernel
from kernel_boundary import stability_boundary

EARLY=Shape(.07235,.02235,90.,3.7153849427)

def test_three_waypoint_interpolation():
    middle=Shape(.1,.05,60.,3.2);end=Shape(.137,.087,24.05,2.3786333904)
    w=[Waypoint(600,EARLY),Waypoint(1500,middle),Waypoint(3009,end)]
    assert interpolate_shape(w,600)==EARLY
    assert interpolate_shape(w,1500)==middle
    assert interpolate_shape(w,3009)==end
    assert interpolate_shape(w,1050).mean_lag==75.
    assert interpolate_shape(w,1050).c0==pytest.approx((EARLY.c0+middle.c0)/2)
    assert interpolate_shape(w,1050).c1==pytest.approx((EARLY.c1+middle.c1)/2)
    assert interpolate_shape(w,1050).log_moment==pytest.approx((EARLY.log_moment+middle.log_moment)/2)

def test_invalid_waypoint_order():
    with pytest.raises(ValueError):interpolate_shape([Waypoint(600,EARLY)]*3,700)

def test_extra_constraint_preserves_all_original_moments():
    result=solve(EARLY,20.,stability_boundary,phase_points=64)
    c=result['weights'];ages=np.arange(256)
    np.testing.assert_allclose([c[0],c[1],c.sum(),((-1.)**ages)@c,ages@c,np.log1p(ages)@c],
        [EARLY.c0,EARLY.c1,1,0,EARLY.mean_lag,EARLY.log_moment],rtol=0,atol=1e-8)
    assert stability_boundary(c)[0]==pytest.approx(20.,rel=1e-7)

def test_original_entropy_solution_preserved_when_constraint_already_met():
    c,_=unconstrained(EARLY)
    result=solve(EARLY,stability_boundary(c)[0],stability_boundary)
    np.testing.assert_array_equal(result['weights'],c)

def test_infeasible_shape_rejected():
    with pytest.raises(InfeasibleKernel):solve(Shape(.4,.4,200.,1.),1.,stability_boundary,phase_points=16)

def test_training_source_and_rng_invariance(tmp_path):
    original=ast.parse(Path(os.environ['ORIGINAL_RECORD_SOURCE']).read_text())
    modified=ast.parse(Path(__file__).with_name('train_gpt_muon_maxentslop.py').read_text())
    def functions(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    # Includes the EXACT original Muon and every Adam LR assignment, as well
    # as the model, data, compiled updates, and validation functions.
    assert functions(original)==functions(modified)
    class Strip(ast.NodeTransformer):
        def visit_Assign(self,node):
            targets=[t.id for t in node.targets if isinstance(t,ast.Name)]
            if any(t in {'CONSTRAINED_SCHEDULE_PATH','constrained_kernel_table','constrained_schedule_metadata'} for t in targets):return None
            if any(isinstance(t,ast.Attribute) and t.attr=='constrained_schedule_gpu' for t in node.targets):return None
            return node
        def visit_If(self,node):
            test=ast.unparse(node.test)
            if test=='CONSTRAINED_SCHEDULE_PATH' or test=='constrained_schedule_metadata is not None':return None
            if test=='constrained_kernel_table is not None':return node.orelse
            return self.generic_visit(node)
    assert ast.dump(original,include_attributes=False)==ast.dump(Strip().visit(modified),include_attributes=False)
    # Execute only the new CPU table-load block and prove it consumes no RNG.
    modified=ast.parse(Path(__file__).with_name('train_gpt_muon_maxentslop.py').read_text())
    names={'CONSTRAINED_SCHEDULE_PATH','constrained_kernel_table','constrained_schedule_metadata'}
    block=[n for n in modified.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)) or (isinstance(n,ast.If) and ast.unparse(n.test)=='CONSTRAINED_SCHEDULE_PATH')]
    import hashlib,json
    c,_=unconstrained(EARLY)
    table=np.tile(c.astype(np.float32),(3060-780,1))
    metadata={'start':780,'kernel_sha256':hashlib.sha256(table.tobytes()).hexdigest(),
              'reference_kappa':31.62336703513191,'reference_lr':.025}
    schedule=tmp_path/'loader_fixture.npz'
    np.savez(schedule,kernels=table,metadata=np.array(json.dumps(metadata)))
    ns={'os':types.SimpleNamespace(environ={'CONSTRAINED_SCHEDULE_PATH':str(schedule)}),
        'np':np,'torch':torch,'TRAIN_STEPS':3060,'MAXENTSLOP_HISTORY_LENGTH':256}
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    exec(compile(ast.Module(body=block,type_ignores=[]),'table_loader','exec'),ns)
    assert random.getstate()==before[0]
    assert np.array_equal(np.random.get_state()[1],before[1][1])
    assert torch.equal(torch.get_rng_state(),before[2])
    assert ns['MAXENTSLOP_START']==780


def test_final_low_lr_boundary_is_found_without_phase_hint():
    from constrained_maxentslop import solve_stability_matched
    from decay_stability import matching_boundary
    shape=Shape(.01,.01,122.34179281473588,4.6855738760604755)
    eta=.025*(1-3059/3060)/.7;mu=1263.4044831967965
    result=solve_stability_matched(shape,eta,mu,4*eta,stability_boundary,phase_points=128)
    assert matching_boundary(result['weights'],mu,4*eta)[0]==pytest.approx(eta*mu,rel=2e-6)
