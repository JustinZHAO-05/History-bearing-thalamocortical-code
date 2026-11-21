"""Round-3 evaluator, inherited from the independently validated Round-2 evaluator.

Published model: Hosford et al. (2026), DOI 10.1016/j.neuron.2026.04.045.
No weight training or modification. Each step is 5 ms. See source attribution
and the restored Round-2 validation records. Pickles are read only from the
explicitly downloaded official model database archive.
"""
from pathlib import Path
import os, sqlite3, pickle, hashlib
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DATA=ROOT.parent/'round2_consciousness/external/hosford_history_rnn'
DATA=Path(os.environ.get('R3_HOSFORD_DATA',str(DEFAULT_DATA)))

class Model:
    def __init__(self,path):
        self.path=Path(path)
        self.name=self.path.parent.name.replace('TBPTT_bio_','').replace('_valid','')+'_'+self.path.stem
        db=sqlite3.connect('file:'+str(self.path)+'?mode=ro',uri=True)
        db.row_factory=sqlite3.Row
        row=dict(db.execute('select * from currentRNNParameters order by id').fetchall()[-1]);db.close()
        self.meta={k:v for k,v in row.items() if not isinstance(v,bytes)}
        self.arrays={k:pickle.loads(v) for k,v in row.items() if isinstance(v,bytes)}
        self.N=row['N'];self.C=row['PPC_N'];self.P=self.C+row['PULexh_N']
        assert (self.N,self.C,self.P)==(500,400,470)
        assert (row['ppc_rate'],row['pul_rate'],row['trn_rate'])==('relu','relu','alpha_relu')
        self.W=np.asarray(np.maximum(self.arrays['raw_Wr'],0)*self.arrays['SynapseMask'],dtype=np.float64)
        self.Win=np.asarray(self.arrays['Win'],dtype=np.float64).reshape(self.N)
        self.Wout=np.asarray(self.arrays['Wout'],dtype=np.float64).reshape(self.N)
        self.bias=float(self.arrays['b_out'])
        lo,hi=self.arrays['tausRange']
        self.tau=lo+(hi-lo)*expit(np.asarray(self.arrays['raw_tauS'],dtype=np.float64).reshape(self.N))
        self.alpha=1/self.tau
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
    def rate(self,x):
        r=np.clip(x,0,20);r[...,470:]=np.clip(10*(x[...,470:]-9),0,20);return r
    def step(self,x,u=0):
        return (1-self.alpha)*x+self.alpha*(self.rate(x)@self.W.T+np.asarray(u)[...,None]*self.Win)
    def output(self,x):
        return self.rate(x)@self.Wout+self.bias

def models():return sorted(DATA.glob('*/*.db'))

def paired_histories(n,seed,task='history',probe_counts=(10,)):
    """Lawful mirrored sensory histories, then identical future pip trains.

    History assay: five nonambiguous trials, then one ambiguous trial. Current
    assay: four nonambiguous trials, then a discrimination trial. Histories are
    sampled uniformly over the author's 20 nonambiguous mixture levels and
    oriented to have positive mean on member 0. Exact-zero histories are
    resampled, before any network is evaluated. Pair members have equal x0.
    """
    rng=np.random.default_rng(seed);old=5 if task=='history' else 4
    start=(old-1)*400+200;end=old*400+200
    u=np.zeros((n,2,end));counts=np.empty((n,2,old+1),int)
    allowed=np.r_[np.arange(10),np.arange(11,21)]
    for k in range(n):
        while True:
            c=rng.choice(allowed,old)
            if np.sum(c-10)!=0:break
        if np.sum(c-10)<0:c=20-c
        for j,q in enumerate(c):
            p=np.r_[np.ones(q),-np.ones(20-q)];rng.shuffle(p)
            a=j*400+50;u[k,0,a:a+60]=np.repeat(p,3);u[k,1,a:a+60]=np.repeat(-p,3)
            counts[k,:,j]=[q,20-q]
        q=int(probe_counts[k%len(probe_counts)])
        p=np.r_[np.ones(q),-np.ones(20-q)];rng.shuffle(p)
        a=old*400+50;u[k,:,a:a+60]=np.repeat(p,3)
        counts[k,:,-1]=q
    x0=rng.normal(0,1/500,(n,500))
    return dict(u=u,x0=x0,counts=counts,start=np.array(start),end=np.array(end),seed=np.array(seed))

def initialize(model,context):
    u=context['u'];n=len(u);start=int(context['start'])
    x=np.repeat(context['x0'][:,None,:],2,axis=1).reshape(n*2,500).copy()
    for t in range(start):x=model.step(x,u[:,:,t].reshape(-1))
    return x.reshape(n,2,500)
