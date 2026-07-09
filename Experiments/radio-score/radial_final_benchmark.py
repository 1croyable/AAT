from __future__ import annotations
import argparse, csv, math, random, time
from pathlib import Path
from typing import List, Dict
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

KAPPA=6.0; DROP=0.15; GATE_INIT=1.0

def seed_all(s:int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); torch.backends.cudnn.benchmark=True

def params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
def split_csv(s): return [x.strip() for x in s.split(',') if x.strip()]

def f1_macro(pred,y,C):
    p=pred.cpu().numpy(); t=y.cpu().numpy(); out=[]
    for c in range(C):
        tp=((p==c)&(t==c)).sum(); fp=((p==c)&(t!=c)).sum(); fn=((p!=c)&(t==c)).sum(); d=2*tp+fp+fn
        if d: out.append(2*tp/d)
    return float(np.mean(out)) if out else 0.0

def polar_stats(x):
    c=x.float().mean(0,keepdim=True); r=(x-c).norm(dim=1); r0=torch.quantile(r,.01); r1=torch.quantile(r,.99)
    if bool((r1<=r0+1e-6).item()): r0=r.min(); r1=r.max()+1e-6
    return c.detach(),r0.detach(),r1.detach()

def make_checker3d(n, grid=4, seed=0):
    g=torch.Generator().manual_seed(seed); x=torch.rand(n,3,generator=g); cell=torch.floor(x*grid).long().clamp(max=grid-1)
    y=((cell[:,0]+cell[:,1]+cell[:,2])%2).long(); return x,y

def make_bubbles3d(n, seed=0, noise=.025):
    g=torch.Generator().manual_seed(seed); z=torch.randn(n,3,generator=g); u=F.normalize(z,dim=1); r=torch.rand(n,1,generator=g).pow(1/3)
    x=(.5+.48*r*u+noise*torch.randn(n,3,generator=g)).clamp(0,1); xc=x-.5; rr=xc.norm(dim=1)
    wob=.045*torch.sin(9*torch.atan2(xc[:,1],xc[:,0])); y=((rr+wob)>.30).long(); return x.float(),y

class AATLayer(nn.Module):
    def __init__(self,D,R,variant):
        super().__init__(); self.D=D; self.R=R; self.v=variant
        self.base=nn.Parameter(torch.randn(R,D)/math.sqrt(D)); self.bias=nn.Parameter(torch.zeros(R))
        self.rho1=nn.Parameter(torch.zeros(R)) if variant in {'rho1','rho12','exp_rho1','exp_rho12'} else None
        self.rho2=nn.Parameter(torch.zeros(R)) if variant in {'rho12','exp_rho12'} else None
        self.gamma=nn.Parameter(torch.zeros(R)) if variant in {'conc_exp','conc_poly2','exp_rho1','exp_rho12'} else None
        self.slope=nn.Parameter(torch.randn(R,D)*.02/math.sqrt(D)) if variant=='curved_s' else None
        self.dr=nn.Parameter(torch.randn(R)*.02); self.du=nn.Parameter(torch.randn(R,D)*.02/math.sqrt(D)); self.gate=nn.Parameter(torch.tensor(GATE_INIT))
    def drop(self,a):
        if (not self.training) or DROP<=0: return a
        m=torch.rand_like(a)>=DROP; empty=m.sum(1,keepdim=True)==0
        if bool(empty.any().item()):
            m=m.clone(); rows=empty.squeeze(1).nonzero().squeeze(1); cols=a.index_select(0,rows).argmax(1); m[rows,cols]=True
        a=a*m.to(a.dtype); return a/a.sum(1,keepdim=True).clamp_min(1e-8)
    def curved_cos(self,rho,u):
        b=self.base; s=self.slope
        num=F.linear(u,b)+rho*F.linear(u,s)
        den=((b*b).sum(1).view(1,-1)+2*rho*(b*s).sum(1).view(1,-1)+rho.square()*(s*s).sum(1).view(1,-1)).clamp_min(1e-8).sqrt()
        return num/den
    def score(self,rho,u):
        if self.v=='curved_s': return KAPPA*self.curved_cos(rho,u)+self.bias.view(1,-1)
        cos=F.linear(u,F.normalize(self.base,dim=1,eps=1e-8)); b=self.bias.view(1,-1)
        if self.v=='rho1': return KAPPA*cos+rho*self.rho1.view(1,-1)+b
        if self.v=='rho12': return KAPPA*cos+rho*self.rho1.view(1,-1)+rho.square()*self.rho2.view(1,-1)+b
        if self.v=='conc_exp': return KAPPA*torch.exp((rho*self.gamma.view(1,-1)).clamp(-2,2))*cos+b
        if self.v=='conc_poly2':
            z=rho*self.gamma.view(1,-1); sc=(1+z+.5*z.square()).clamp_min(.05).clamp_max(8); return KAPPA*sc*cos+b
        if self.v=='exp_rho1': return KAPPA*torch.exp((rho*self.gamma.view(1,-1)).clamp(-2,2))*cos+rho*self.rho1.view(1,-1)+b
        if self.v=='exp_rho12': return KAPPA*torch.exp((rho*self.gamma.view(1,-1)).clamp(-2,2))*cos+rho*self.rho1.view(1,-1)+rho.square()*self.rho2.view(1,-1)+b
        raise ValueError(self.v)
    def forward(self,rho,u):
        a=self.drop(F.softmax(self.score(rho,u),dim=1)); rho=rho+(a@self.dr[:,None])*self.gate; u=F.normalize(u+(a@self.du)*self.gate,dim=1,eps=1e-8); return rho,u

class AAT(nn.Module):
    def __init__(self,D,C,L,R,variant,center,r0,r1):
        super().__init__(); self.register_buffer('center',center.float().view(1,-1).clone()); self.register_buffer('r0',torch.as_tensor(float(r0))); self.register_buffer('r1',torch.as_tensor(float(r1)))
        self.layers=nn.ModuleList([AATLayer(D,R,variant) for _ in range(L)]); self.head=nn.Linear(D+1,C)
    def polar(self,x):
        z=x.float(); c=self.center.to(z); r0=self.r0.to(z); r1=self.r1.to(z); xc=z-c; r=xc.norm(dim=1,keepdim=True).clamp_min(1e-8); u=xc/r; rho=2*(r-r0)/(r1-r0).clamp_min(1e-8)-1; return rho,u
    def forward(self,x):
        rho,u=self.polar(x)
        for l in self.layers: rho,u=l(rho,u)
        return self.head(torch.cat([rho,u],1))

class MLP(nn.Module):
    def __init__(self,D,C,H,layers=2):
        super().__init__(); a=[]; d=D
        for _ in range(layers): a += [nn.Linear(d,H),nn.ReLU()]; d=H
        a.append(nn.Linear(d,C)); self.net=nn.Sequential(*a)
    def forward(self,x): return self.net(x.float())

def mlp_hidden(D,C,target,layers=2):
    best=(1,10**18)
    for h in range(1,4096):
        p=params(MLP(D,C,h,layers)); gap=abs(p-target)
        if gap<best[1]: best=(h,gap)
        if p>target and h>4: break
    return best[0]

def acc_f1(model,x,y,C):
    model.eval();
    with torch.no_grad(): pred=model(x).argmax(1)
    return float((pred==y).float().mean().item()), f1_macro(pred,y,C)

def train_tensor(task,model_name,variant,seed,device,xtr,ytr,xva,yva,D,C,args,out):
    c,r0,r1=polar_stats(xtr); hidden=''
    if model_name=='aat': model=AAT(D,C,args.layers,args.rays,variant,c.to(device),r0,r1).to(device)
    else:
        target=args.mlp_target_params
        if target<=0: target=params(AAT(D,C,args.layers,args.rays,args.mlp_match_variant,c.to(device),r0,r1))
        hidden=mlp_hidden(D,C,target,args.mlp_layers); model=MLP(D,C,hidden,args.mlp_layers).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); best={'epoch':0,'train':0,'val':-1,'f1':0}; st=None; t=time.time(); p=params(model)
    print(f"\n[{task}] {model_name} {variant} seed={seed} params={p} hidden={hidden}",flush=True)
    for e in range(1,args.toy_epochs+1):
        model.train(); perm=torch.randperm(len(xtr),device=device); loss_sum=0; n=0
        for s in range(0,len(xtr),args.batch_size):
            idx=perm[s:s+args.batch_size]; xb=xtr[idx]; yb=ytr[idx]; opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(xb),yb); loss.backward(); opt.step(); loss_sum+=loss.item()*len(xb); n+=len(xb)
        va,f1=acc_f1(model,xva,yva,C)
        if va>best['val']:
            tr,_=acc_f1(model,xtr,ytr,C); best={'epoch':e,'train':tr,'val':va,'f1':f1}; st={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if e==1 or e%args.toy_eval_every==0 or e==args.toy_epochs: print(f"epoch={e:04d} loss={loss_sum/max(n,1):.4f} val={va:.4f} best={best['val']:.4f}@{best['epoch']}",flush=True)
    row={'task':task,'model':model_name,'variant':variant,'seed':seed,'layers':args.layers if model_name=='aat' else args.mlp_layers,'rays':args.rays if model_name=='aat' else '', 'params':p,'hidden':hidden,'best_epoch':best['epoch'],'train_acc':best['train'],'val_acc':best['val'],'test_acc':'','test_f1':best['f1'],'seconds':time.time()-t}; write_row(out,row); return row

def read_airline(root,seed,val_frac=.15):
    tr=pd.read_csv(root/'train.csv').dropna(subset=['satisfaction']).reset_index(drop=True); te=pd.read_csv(root/'test.csv').dropna(subset=['satisfaction']).reset_index(drop=True)
    labs=sorted(tr['satisfaction'].astype(str).unique()); mp={v:i for i,v in enumerate(labs)}; y=np.array([mp[v] for v in tr['satisfaction'].astype(str)])
    rng=np.random.default_rng(seed); tri=[]; vai=[]
    for c in np.unique(y):
        idx=np.where(y==c)[0]; rng.shuffle(idx); nv=max(1,int(round(len(idx)*val_frac))); vai+=idx[:nv].tolist(); tri+=idx[nv:].tolist()
    rng.shuffle(tri); rng.shuffle(vai); trn=tr.iloc[tri].reset_index(drop=True); val=tr.iloc[vai].reset_index(drop=True)
    return prep_tab(trn,val,te)

def prep_tab(tr,val,te):
    for df in (tr,val,te):
        for c in ['Unnamed: 0','id']:
            if c in df.columns: df.drop(columns=[c],inplace=True)
    labs=sorted(tr['satisfaction'].astype(str).unique()); mp={v:i for i,v in enumerate(labs)}
    ys=[df['satisfaction'].astype(str).map(mp).astype(np.int64).to_numpy() for df in (tr,val,te)]
    cols=[c for c in tr.columns if c!='satisfaction']; num=[c for c in cols if pd.api.types.is_numeric_dtype(tr[c])]; cat=[c for c in cols if c not in num]
    xs=[]
    if num:
        mean=tr[num].apply(pd.to_numeric,errors='coerce').mean(0)
        nums=[df[num].apply(pd.to_numeric,errors='coerce').fillna(mean).to_numpy(np.float32) for df in (tr,val,te)]
    else: nums=[np.zeros((len(df),0),np.float32) for df in (tr,val,te)]
    cats=[[] for _ in range(3)]
    for c in cat:
        base=tr[c].astype(str).fillna('__NA__'); vals=sorted(base.unique()); mp2={v:i for i,v in enumerate(vals)}
        for k,df in enumerate((tr,val,te)):
            s=df[c].astype(str).fillna('__NA__'); a=np.zeros((len(s),len(vals)),np.float32)
            for i,v in enumerate(s.tolist()):
                j=mp2.get(v)
                if j is not None: a[i,j]=1
            cats[k].append(a)
    for k in range(3): xs.append(np.concatenate([nums[k],np.concatenate(cats[k],1) if cats[k] else np.zeros((len((tr,val,te)[k]),0),np.float32)],1).astype(np.float32))
    mu=xs[0].mean(0,keepdims=True); sd=xs[0].std(0,keepdims=True); sd=np.where(sd<1e-6,1,sd); xs=[((x-mu)/sd).astype(np.float32) for x in xs]
    return xs[0],ys[0],xs[1],ys[1],xs[2],ys[2],xs[0].shape[1],len(labs)

def loaders(data,device,batch,eval_batch):
    pin=device.type=='cuda'
    def ds(x,y): return TensorDataset(torch.from_numpy(x.copy()).float(),torch.from_numpy(y.copy()).long())
    xtr,ytr,xva,yva,xte,yte,D,C=data
    return {'train':DataLoader(ds(xtr,ytr),batch_size=batch,shuffle=True,pin_memory=pin),'train_eval':DataLoader(ds(xtr,ytr),batch_size=eval_batch,pin_memory=pin),'val':DataLoader(ds(xva,yva),batch_size=eval_batch,pin_memory=pin),'test':DataLoader(ds(xte,yte),batch_size=eval_batch,pin_memory=pin)}

def eval_loader(model,loader,device,C):
    model.eval(); ps=[]; ys=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True); ps.append(model(xb).argmax(1).cpu()); ys.append(yb.cpu())
    p=torch.cat(ps); y=torch.cat(ys); return float((p==y).float().mean().item()),f1_macro(p,y,C)

def train_air(task,model_name,variant,seed,device,data,lds,args,out):
    xtr,ytr,xva,yva,xte,yte,D,C=data; center=xtr.mean(0).astype(np.float32); r=np.linalg.norm(xtr-center[None,:],axis=1); r0=float(np.quantile(r,.01)); r1=float(np.quantile(r,.99))
    c=torch.tensor(center,device=device); hidden=''
    if model_name=='aat': model=AAT(D,C,args.layers,args.rays,variant,c,r0,r1).to(device)
    else:
        target=args.mlp_target_params
        if target<=0: target=params(AAT(D,C,args.layers,args.rays,args.mlp_match_variant,c,r0,r1))
        hidden=mlp_hidden(D,C,target,args.mlp_layers); model=MLP(D,C,hidden,args.mlp_layers).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); scaler=torch.amp.GradScaler('cuda',enabled=args.amp and device.type=='cuda'); best={'epoch':0,'train':0,'val':-1,'test':0,'f1':0}; t=time.time(); p=params(model)
    print(f"\n[{task}] {model_name} {variant} seed={seed} params={p} hidden={hidden}",flush=True)
    for e in range(1,args.airline_epochs+1):
        model.train(); loss_sum=0; n=0
        for xb,yb in lds['train']:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=args.amp and device.type=='cuda'): loss=F.cross_entropy(model(xb),yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); loss_sum+=loss.item()*len(xb); n+=len(xb)
        if e==1 or e%args.airline_eval_every==0 or e==args.airline_epochs:
            tr,_=eval_loader(model,lds['train_eval'],device,C); va,_=eval_loader(model,lds['val'],device,C); te,f1=eval_loader(model,lds['test'],device,C)
            if va>best['val']: best={'epoch':e,'train':tr,'val':va,'test':te,'f1':f1}
            print(f"epoch={e:04d} loss={loss_sum/max(n,1):.4f} val={va:.4f} test={te:.4f} best={best['val']:.4f}@{best['epoch']}",flush=True)
            if e-best['epoch']>=args.airline_patience: break
    row={'task':task,'model':model_name,'variant':variant,'seed':seed,'layers':args.layers if model_name=='aat' else args.mlp_layers,'rays':args.rays if model_name=='aat' else '', 'params':p,'hidden':hidden,'best_epoch':best['epoch'],'train_acc':best['train'],'val_acc':best['val'],'test_acc':best['test'],'test_f1':best['f1'],'seconds':time.time()-t}; write_row(out,row); return row

def write_row(path,row):
    path.parent.mkdir(exist_ok=True,parents=True); fields=['task','model','variant','seed','layers','rays','params','hidden','best_epoch','train_acc','val_acc','test_acc','test_f1','seconds']; ex=path.exists()
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); 
        if not ex: w.writeheader()
        w.writerow({k:row.get(k,'') for k in fields})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tasks',default='checker3d,airline'); ap.add_argument('--variants',default='rho1,rho12,conc_exp,conc_poly2,exp_rho1,exp_rho12,curved_s'); ap.add_argument('--seeds',default='0,1'); ap.add_argument('--include-mlp',action='store_true'); ap.add_argument('--device',default='cuda'); ap.add_argument('--amp',action='store_true')
    ap.add_argument('--layers',type=int,default=4); ap.add_argument('--rays',type=int,default=32); ap.add_argument('--lr',type=float,default=3e-3); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--batch-size',type=int,default=512); ap.add_argument('--eval-batch-size',type=int,default=8192)
    ap.add_argument('--toy-epochs',type=int,default=1500); ap.add_argument('--toy-eval-every',type=int,default=25); ap.add_argument('--toy-train',type=int,default=4096); ap.add_argument('--toy-val',type=int,default=2048); ap.add_argument('--checker-grid',type=int,default=4)
    ap.add_argument('--airline-root',default=''); ap.add_argument('--airline-epochs',type=int,default=60); ap.add_argument('--airline-patience',type=int,default=15); ap.add_argument('--airline-eval-every',type=int,default=5)
    ap.add_argument('--mlp-layers',type=int,default=2); ap.add_argument('--mlp-target-params',type=int,default=0); ap.add_argument('--mlp-match-variant',default='exp_rho12')
    args=ap.parse_args(); tasks=split_csv(args.tasks); vars=split_csv(args.variants); seeds=[int(x) for x in split_csv(args.seeds)]; dev=torch.device(args.device if args.device=='cuda' and torch.cuda.is_available() else 'cpu')
    out=Path(__file__).resolve().parent/'outputs'/'radial_final_benchmark.csv'; rows=[]
    print('='*110+'\nFinal radial benchmark\n'+'='*110); print(f'device={dev} tasks={tasks} variants={vars} seeds={seeds} include_mlp={args.include_mlp}')
    for task in tasks:
        for sd in seeds:
            seed_all(sd)
            if task in {'checker3d','bubbles3d'}:
                maker=make_checker3d if task=='checker3d' else make_bubbles3d
                xtr,ytr=maker(args.toy_train, args.checker_grid, sd) if task=='checker3d' else maker(args.toy_train, sd)
                xva,yva=maker(args.toy_val, args.checker_grid, sd+1000) if task=='checker3d' else maker(args.toy_val, sd+1000)
                xtr,ytr=xtr.to(dev),ytr.to(dev); xva,yva=xva.to(dev),yva.to(dev)
                for v in vars: rows.append(train_tensor(task,'aat',v,sd,dev,xtr,ytr,xva,yva,3,2,args,out))
                if args.include_mlp: rows.append(train_tensor(task,'mlp','matched',sd,dev,xtr,ytr,xva,yva,3,2,args,out))
            elif task=='airline':
                root=Path(args.airline_root) if args.airline_root else Path(__file__).resolve().parent.parent/'data'/'AirlineSatisfaction'; data=read_airline(root,sd); lds=loaders(data,dev,args.batch_size,args.eval_batch_size)
                for v in vars: rows.append(train_air(task,'aat',v,sd,dev,data,lds,args,out))
                if args.include_mlp: rows.append(train_air(task,'mlp','matched',sd,dev,data,lds,args,out))
            else: raise ValueError(task)
    print('\nSUMMARY')
    groups={}
    for r in rows: groups.setdefault((r['task'],r['model'],r['variant']),[]).append(r)
    for k,vs in sorted(groups.items()):
        vals=np.array([float(v['val_acc']) for v in vs]); msg=f'{k[0]:<10} {k[1]:<4} {k[2]:<10} val={vals.mean():.4f}±{vals.std():.4f} params={vs[0]["params"]}'
        tests=[float(v['test_acc']) for v in vs if v['test_acc']!='']; f1s=[float(v['test_f1']) for v in vs if v['test_f1']!='']
        if tests: msg+=f' test={np.mean(tests):.4f} f1={np.mean(f1s):.4f}'
        if vs[0].get('hidden','')!='': msg+=f' hidden={vs[0]["hidden"]}'
        print(msg)
    print(f'\nsaved: {out}')
if __name__=='__main__': main()
