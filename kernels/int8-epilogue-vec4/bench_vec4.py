from __future__ import annotations
import gc, importlib.util, json, statistics
from pathlib import Path
import torch

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

root=Path(__file__).resolve().parent
ref=load("h3_hip_epilogue","kernels/int8-epilogue/build/h3_hip_epilogue.so")
new=load("h3_exact_epilogue_vec4",root/"build/h3_exact_epilogue_vec4.so")
M=11819
shapes=[("qkv",21504),("out",5376),("fc1",28672),("fc2",5376)]

def timed(fn,reps=10):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
    for _ in range(reps):fn()
    b.record();b.synchronize();return float(a.elapsed_time(b))/reps

results=[]
for idx,(name,n) in enumerate(shapes):
    g=torch.Generator(device="cuda").manual_seed(20260812+idx)
    acc=torch.randint(-2000000,2000001,(M,n),device="cuda",dtype=torch.int32,generator=g)
    row=torch.rand((M,1),device="cuda",dtype=torch.float32,generator=g)*.02+1e-5
    col=torch.rand((n,),device="cuda",dtype=torch.float32,generator=g)*.02+1e-5
    empty=torch.empty(0,device="cuda",dtype=torch.bfloat16)
    r=ref.epilogue(acc,row,col,empty);v=new.epilogue(acc,row,col);torch.cuda.synchronize()
    mismatch=int(torch.count_nonzero(r!=v).item());max_abs=float((r.float()-v.float()).abs().max().item())
    del r,v
    def refcall():return ref.epilogue(acc,row,col,empty)
    def newcall():return new.epilogue(acc,row,col)
    for _ in range(12):refcall();newcall()
    torch.cuda.synchronize();rs=[];vs=[]
    for j in range(11):
        if j%2==0:rs.append(timed(refcall));vs.append(timed(newcall))
        else:vs.append(timed(newcall));rs.append(timed(refcall))
    rm,vm=statistics.median(rs),statistics.median(vs)
    results.append({"name":name,"shape":[M,n],"mismatch":mismatch,"max_abs":max_abs,"reference_samples_ms":rs,"candidate_samples_ms":vs,"reference_median_ms":rm,"candidate_median_ms":vm,"saved_ms":rm-vm,"speedup":rm/vm})
    del acc,row,col,empty;gc.collect();torch.cuda.empty_cache()

saved=sum(x["saved_ms"] for x in results)
print(json.dumps({"results":results,"bitwise_pass":all(x["mismatch"]==0 for x in results),"saved_ms_per_block":saved,"predicted_full20_saved_seconds":saved*50*20/1000},indent=2,sort_keys=True))
