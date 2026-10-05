from __future__ import annotations
import collections, hashlib, json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CFG=json.load(open(ROOT/'config/project.json'))
RUN=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep4a_srp_fullslate_devcal_freeze_v1'
RET=RUN/'return'; CACHE=RUN/'cache'
for p in [RUN,RET,CACHE]: p.mkdir(parents=True,exist_ok=True)
LOG=RUN/'run.log'
SHARED=Path.home()/'SLATE_TDSC_SERVER_SHARED'
P3B=SHARED/'runs/slate_stagep3b_llmdna_fullslate_devcal_freeze_v1'
P3B_RET=P3B/'return'; P3B_EMB=P3B/'embeddings'
MPNET=Path('/home/jx-vmlab/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130')
TRAFFIC=SHARED/'runs/slate_stage0_traffic_bootstrap_v1/traffic_frozen.jsonl'
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'

def log(x):
    s=str(x); print(s,flush=True)
    with open(LOG,'a',encoding='utf-8') as f: f.write(s+'\n')

def dump(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False,default=str)

def sha(x): return hashlib.sha256(str(x).encode()).hexdigest()
def file_sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def h64(x,salt=''): return int.from_bytes(hashlib.sha256((salt+'\0'+str(x)).encode()).digest()[:8],'big')

def load_jsonl(p):
    with open(p,encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def valid_text(t):
    t=str(t).strip(); n=len(t.split())
    return 4<=n<=256 and 12<=len(t)<=2400

def sessions_for_split(split):
    if split not in {'development','calibration'}:
        raise RuntimeError(f'P4A held-out hard guard: forbidden split {split!r}')
    by=collections.defaultdict(list)
    for r in load_jsonl(TRAFFIC):
        if r.get('split')!=split: continue
        t=str(r.get('text','')).strip()
        if valid_text(t): by[str(r['conversation_id'])].append((int(r.get('turn_index',0)),t))
    out=[]
    for cid,z in by.items():
        texts=[x[1] for x in sorted(z)]
        if texts: out.append({'id':cid,'texts':texts})
    out.sort(key=lambda x:h64(x['id'],'P3B_'+split))
    return out

def freeze_pool(sessions,n,salt):
    seen=set(); arr=[]
    for s in sessions:
        for ti,t in enumerate(s['texts']):
            if t in seen: continue
            seen.add(t); arr.append({'id':sha(salt+'|'+s['id']+'|'+str(ti)+'|'+t)[:20],'conversation_id':s['id'],'turn_index':ti,'text':t})
    arr.sort(key=lambda x:h64(x['id'],salt)); return arr[:n]

def excluded_queries(sessions,exclude,cap,salt):
    ex=set(exclude); arr=[]; seen=set()
    for s in sessions:
        for t in s['texts']:
            if t in ex or t in seen: continue
            seen.add(t); arr.append(t)
    arr.sort(key=lambda t:h64(t,salt)); return arr[:cap]

def load_p3b_embeddings(tag, expected_ids):
    p=P3B_EMB/f'{tag}.npz'
    if not p.exists(): raise RuntimeError('missing P3B embedding bank '+str(p))
    z=np.load(p,allow_pickle=False); ids=list(z['ids'].astype(str)); E=np.asarray(z['E'],np.float32)
    if ids!=list(expected_ids):
        raise RuntimeError(f'P3B embedding ID mismatch for {tag}: {len(ids)} vs {len(expected_ids)}')
    if not np.isfinite(E).all(): raise RuntimeError(f'non-finite P3B embeddings: {tag}')
    return E, {'path':str(p),'sha256':file_sha(p),'shape':list(E.shape),'ids_sha256':sha('\n'.join(ids))}

def cosrow(A,B):
    A=np.asarray(A,float); B=np.asarray(B,float)
    return np.sum(A*B,axis=1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+1e-12)

def weighted_session(J,lam,q,seed):
    rng=np.random.default_rng(seed); logw=np.asarray(J,float)/float(lam); g=-np.log(-np.log(np.clip(rng.random(len(J)),1e-12,1-1e-12)))
    idx=np.argsort(logw+g)[-q:]; return idx.astype(int).tolist()

def distribution_weights(J,lam):
    x=np.asarray(J,float)/float(lam); x-=x.max(); w=np.exp(x); w/=w.sum(); return w

def build_scheduler(dev_sessions,dev_pool,cal_pool,mpnet):
    from sklearn.cluster import MiniBatchKMeans
    flat=[]
    for s in dev_sessions:
        flat.extend(s['texts'])
        if len(flat)>=5000: break
    flat=flat[:5000]
    X=mpnet.encode(flat,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    km=MiniBatchKMeans(n_clusters=CFG['semantic_clusters'],random_state=CFG['seed'],batch_size=512,n_init=5).fit(X)
    lens=np.asarray([len(t.split()) for t in flat],float); qs=np.quantile(lens,[.25,.5,.75]).tolist(); cache={}
    def states(texts):
        miss=[t for t in texts if t not in cache]
        if miss:
            E=mpnet.encode(miss,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False); cl=km.predict(E)
            for t,c in zip(miss,cl): cache[t]=int(c)*CFG['length_bins']+int(np.searchsorted(qs,len(t.split()),side='right'))
        return np.asarray([cache[t] for t in texts],int)
    S=CFG['semantic_clusters']*CFG['length_bins']; init=np.ones(S)*1e-3; trans=np.ones((S,S))*1e-3
    for s in dev_sessions:
        st=states(s['texts'])
        if len(st): init[st[0]]+=1
        for a,b in zip(st[:-1],st[1:]): trans[a,b]+=1
    init/=init.sum(); trans/=trans.sum(1,keepdims=True)
    dev_states=states([r['text'] for r in dev_pool]); cal_states=states([r['text'] for r in cal_pool])
    def schedule(selected,pool_states,seed):
        rng=np.random.default_rng(seed); rem=list(map(int,selected)); out=[]
        vals=np.asarray([init[pool_states[i]] for i in rem]); best=np.flatnonzero(vals>=vals.max()-1e-15); j=int(rng.choice(best)); cur=rem.pop(j); out.append(cur)
        while rem:
            a=pool_states[cur]; vals=np.asarray([trans[a,pool_states[i]] for i in rem]); best=np.flatnonzero(vals>=vals.max()-1e-15); j=int(rng.choice(best)); cur=rem.pop(j); out.append(cur)
        return out
    return {'states_dev':dev_states,'states_cal':cal_states,'init':init,'trans':trans,'length_quantiles':qs,'schedule':schedule,'state_fn':states,'S':S}

def make_sessions(J,lam,n,q,states,scheduler,split_tag,scheduled=True):
    out=[]
    for e in range(n):
        seed=CFG['seed']+h64(f'{split_tag}|{lam}|{e}','P4A_SESSION')%100000000
        sel=weighted_session(J,lam,q,seed); rng=np.random.default_rng(seed+17); rnd=list(np.asarray(sel)[rng.permutation(len(sel))])
        order=scheduler(sel,states,seed+31) if scheduled else rnd
        out.append({'episode':e,'seed':int(seed),'selected':list(map(int,sel)),'random_order':list(map(int,rnd)),'order':list(map(int,order)),'mean_J':float(np.mean(np.asarray(J)[sel]))})
    return out

def fixed_sessions(J,n,q,split_tag):
    order=np.argsort(np.asarray(J,float))[::-1][:q].astype(int).tolist()
    return [{'episode':e,'seed':int(CFG['seed']+h64(f'{split_tag}|{e}','P4A_FIXED')%100000000),'selected':order,'random_order':order,'order':order,'mean_J':float(np.mean(np.asarray(J)[order]))} for e in range(n)]

def real_windows(sessions,L,cap,salt):
    arr=[]
    for s in sessions:
        for i in range(len(s['texts'])-L+1): arr.append((s['texts'][i:i+L],s['id']))
    arr.sort(key=lambda x:h64('|'.join(x[0])+'|'+x[1],salt)); return arr[:cap]

class ECache:
    def __init__(self,model): self.m=model; self.c={}
    def get(self,texts):
        miss=[]; seen=set()
        for t in texts:
            if t not in self.c and t not in seen: seen.add(t); miss.append(t)
        if miss:
            E=self.m.encode(miss,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            for t,e in zip(miss,E): self.c[t]=np.asarray(e,np.float32)
        return np.stack([self.c[t] for t in texts])

def seq_features(cache,items,L):
    flat=[t for seq,_ in items for t in seq]; E=cache.get(flat); out=[]; k=0
    for seq,_ in items:
        z=E[k:k+L]; k+=L; lens=np.asarray([len(x) for x in seq],np.float32)[:,None]/2048.; pos=np.arange(L,dtype=np.float32)[:,None]/max(1,L-1)
        out.append(np.concatenate([z,lens,pos],1))
    return np.stack(out).astype(np.float32) if out else np.empty((0,L,770),np.float32)

def make_seqnet(d,h,bidir):
    import torch
    class M(torch.nn.Module):
        def __init__(self): super().__init__(); self.r=torch.nn.GRU(d,h,batch_first=True,bidirectional=bidir); self.o=torch.nn.Linear(h*(2 if bidir else 1),1)
        def forward(self,x): z,_=self.r(x); return self.o(z[:,-1]).squeeze(-1)
    return M()

def fit_seq(cache,pos,neg,L,h,bidir,seed):
    import torch
    Xp=seq_features(cache,pos,L); Xn=seq_features(cache,neg,L); X=np.concatenate([Xp,Xn]); y=np.r_[np.ones(len(Xp)),np.zeros(len(Xn))].astype(np.float32)
    rng=np.random.default_rng(seed); torch.manual_seed(seed); dev='cuda' if torch.cuda.is_available() else 'cpu'; m=make_seqnet(X.shape[-1],h,bidir).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4); lossfn=torch.nn.BCEWithLogitsLoss()
    for _ in range(CFG['seq_epochs']):
        order=rng.permutation(len(y))
        for st in range(0,len(y),CFG['seq_batch_size']):
            ii=order[st:st+CFG['seq_batch_size']]; xb=torch.from_numpy(X[ii]).to(dev); yb=torch.from_numpy(y[ii]).to(dev); opt.zero_grad(); loss=lossfn(m(xb),yb); loss.backward(); opt.step()
    return m

def score_seq(m,cache,items,L):
    import torch
    X=seq_features(cache,items,L); dev=next(m.parameters()).device; out=[]; m.eval()
    with torch.inference_mode():
        for st in range(0,len(X),256): out.append(torch.sigmoid(m(torch.from_numpy(X[st:st+256]).to(dev))).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)

def roc_metrics(pos,neg):
    from sklearn.metrics import roc_auc_score,roc_curve
    pos=np.asarray(pos,float); neg=np.asarray(neg,float); y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]; s=np.r_[pos,neg]; auc=float(roc_auc_score(y,s)); fpr,tpr,_=roc_curve(y,s)
    b80=float(np.min(fpr[tpr>=.8])) if np.any(tpr>=.8) else 1.; a10=float(np.max(tpr[fpr<=.1])) if np.any(fpr<=.1) else 0.
    return {'AUROC':auc,'AUROC_sym':max(auc,1-auc),'BDR_at_AIR_0.8':b80,'AIR_at_BDR_0.1':a10}

def threshold_for_bdr(neg,beta):
    x=np.sort(np.asarray(neg,float))[::-1]; n=len(x); k=int(math.floor(beta*n)); th=float(np.nextafter(x[0],np.inf)) if k<=0 else (-float('inf') if k>=n else float(np.nextafter(x[k],np.inf))); return th,float(np.mean(np.asarray(neg)>=th))

def audit_items(pool,sessions,L):
    out=[]
    for s in sessions:
        texts=[pool[i]['text'] for i in s['order']]
        for i in range(len(texts)-L+1): out.append((texts[i:i+L],f"a{s['episode']}"))
    return out

def flatten_queries(pool,sessions): return [pool[i]['text'] for s in sessions for i in s['order']]

def trimmean(x,frac):
    x=np.sort(np.asarray(x,float)); k=int(math.floor(len(x)*frac))
    if k>0 and 2*k<len(x): x=x[k:-k]
    return float(np.mean(x))

def srp_session(idx,sus,qwen,phi,tau_q=None,tau_p=None,risk_tau=None,gap_tau=None,use_risk=True):
    idx=np.asarray(idx,int); qs=cosrow(sus[idx],qwen[idx]); ps=cosrow(sus[idx],phi[idx]); sq=trimmean(qs,CFG['trim_fraction']); sp=trimmean(ps,CFG['trim_fraction']); margins=qs-ps
    med=float(np.median(margins)); mad=float(np.median(np.abs(margins-med))); fpos=float(np.mean(margins>=0)); mix=min(fpos,1-fpos); risk=float(mad+2.0*mix); gap=float(abs(sq-sp)); best='Qwen3-4B-Base' if sq>=sp else 'Phi-3-mini-4k-instruct'; score=max(sq,sp)
    verdict=None
    if tau_q is not None and tau_p is not None:
        if use_risk and risk_tau is not None and risk>risk_tau: verdict='Inconclusive-Serving'
        elif (best=='Qwen3-4B-Base' and sq<tau_q) or (best=='Phi-3-mini-4k-instruct' and sp<tau_p): verdict='Rejected'
        elif gap_tau is not None and gap<gap_tau: verdict='Inconclusive-Bank'
        else: verdict='Verified'
    return {'sim_qwen':sq,'sim_phi':sp,'best':best,'score':score,'risk':risk,'gap':gap,'verdict':verdict,'margin_trimmed':trimmean(margins,CFG['trim_fraction'])}

def verdict_rates(rows):
    n=max(1,len(rows)); return {'CSVR':sum(r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base' for r in rows)/n,'FSVR':sum(r['verdict']=='Verified' and r['best']=='Phi-3-mini-4k-instruct' for r in rows)/n,'Channel_Risk':sum(r['verdict']=='Inconclusive-Serving' for r in rows)/n,'Insufficient':sum(r['verdict'] in ('Inconclusive-Serving','Inconclusive-Bank') for r in rows)/n,'Rejected':sum(r['verdict']=='Rejected' for r in rows)/n}

def ci_binary(vals,reps=None):
    vals=np.asarray(vals,float); reps=reps or CFG['bootstrap_reps']; rng=np.random.default_rng(CFG['seed'])
    if not len(vals): return [None,None]
    z=[float(np.mean(vals[rng.integers(0,len(vals),len(vals))])) for _ in range(reps)]; return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]

def state_span_dist(session_orders,pool_states,L,S):
    if L==1:
        cnt=np.ones(S)*1e-9
        for o in session_orders:
            for i in o: cnt[pool_states[i]]+=1
    else:
        cnt=np.ones((S,S))*1e-9
        for o in session_orders:
            st=[pool_states[i] for i in o]
            for i in range(len(st)-L+1): cnt[st[i],st[i+L-1]]+=1
    return cnt.reshape(-1)/cnt.sum()

def real_state_span_dist(real_sessions,state_fn,L,S):
    cnt=np.ones(S if L==1 else S*S)*1e-9
    for s in real_sessions:
        st=state_fn(s['texts'])
        if L==1:
            for a in st: cnt[a]+=1
        else:
            for i in range(len(st)-L+1): cnt[st[i]*S+st[i+L-1]]+=1
    return cnt/cnt.sum()

def tv(a,b): return float(.5*np.abs(np.asarray(a)-np.asarray(b)).sum())

def attack_a1(session,pool,d1,d2,vec,ec,strong,thr,nf4,phi):
    idx=session['order']; texts=[pool[i]['text'] for i in idx]
    sc=d1.predict_proba(vec.transform(texts))[:,1] if strong=='D1' else d2.predict_proba(ec.get(texts))[:,1]
    route=np.asarray(sc)>=thr; mix=nf4.copy(); sel=np.asarray(idx,int); mix[sel[route]]=phi[sel[route]]; return mix,float(route.mean()),sc.tolist()

def attack_a2(session,pool,model,ec,L,thr,nf4,phi):
    idx=session['order']; texts=[pool[i]['text'] for i in idx]; route=np.zeros(len(idx),bool); scores=[]
    for t in range(L-1,len(idx)):
        s=float(score_seq(model,ec,[(texts[t-L+1:t+1],'x')],L)[0]); scores.append(s); route[t]=s>=thr
    mix=nf4.copy(); sel=np.asarray(idx,int); mix[sel[route]]=phi[sel[route]]; return mix,float(route.mean()),scores

def train_suite(name,dev_pool,dev_real,dev_sessions,ec,seed_offset):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    posq=flatten_queries(dev_pool,dev_sessions)[:CFG['detector_query_cap']]
    negq=excluded_queries(dev_real,[r['text'] for r in dev_pool],CFG['detector_query_cap'],f'P4A_{name}_DNEG')
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=100000,sublinear_tf=True,strip_accents='unicode'); X=vec.fit_transform(posq+negq); y=np.r_[np.ones(len(posq)),np.zeros(len(negq))]
    d1=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=CFG['seed']+seed_offset).fit(X,y)
    d2=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=CFG['seed']+seed_offset).fit(np.r_[ec.get(posq),ec.get(negq)],y)
    seq={}
    for L in CFG['L_grid']:
        pos=audit_items(dev_pool,dev_sessions,L)[:CFG['detector_window_cap']]; negw=real_windows(dev_real,L,CFG['detector_window_cap'],f'P4A_{name}_NEGW{L}')
        if L in (4,8): seq[f'D3_L{L}']=fit_seq(ec,pos,negw,L,CFG['d3_hidden'],False,CFG['seed']+seed_offset+L)
        seq[f'D4_L{L}']=fit_seq(ec,pos,negw,L,CFG['d4_hidden'],True,CFG['seed']+seed_offset+100+L)
    return {'vec':vec,'d1':d1,'d2':d2,'seq':seq}

def calibrate_suite(name,suite,cal_real,cal_pool,ec):
    cal_negq=excluded_queries(cal_real,[r['text'] for r in cal_pool],6000,f'P4A_{name}_CALNEG')
    rawneg={'D1':suite['d1'].predict_proba(suite['vec'].transform(cal_negq))[:,1],'D2':suite['d2'].predict_proba(ec.get(cal_negq))[:,1]}
    for L in CFG['L_grid']:
        nw=real_windows(cal_real,L,CFG['detector_window_cap'],f'P4A_{name}_CALW{L}')
        if L in (4,8): rawneg[f'D3_L{L}']=score_seq(suite['seq'][f'D3_L{L}'],ec,nw,L)
        rawneg[f'D4_L{L}']=score_seq(suite['seq'][f'D4_L{L}'],ec,nw,L)
    thresholds={}
    for beta in CFG['beta_grid']:
        thresholds[str(beta)]={}
        for k,v in rawneg.items():
            th,b=threshold_for_bdr(v,beta); thresholds[str(beta)][k]={'threshold':th,'measured_BDR':b}
    return rawneg,thresholds

def suite_metrics(suite,rawneg,pool,sessions,ec):
    qs=flatten_queries(pool,sessions); met={'D1_L1':roc_metrics(suite['d1'].predict_proba(suite['vec'].transform(qs))[:,1],rawneg['D1']),'D2_L1':roc_metrics(suite['d2'].predict_proba(ec.get(qs))[:,1],rawneg['D2'])}
    raw={'D1':suite['d1'].predict_proba(suite['vec'].transform(qs))[:,1],'D2':suite['d2'].predict_proba(ec.get(qs))[:,1]}
    for L in CFG['L_grid']:
        ai=audit_items(pool,sessions,L)
        if L in (4,8):
            sp=score_seq(suite['seq'][f'D3_L{L}'],ec,ai,L); met[f'D3_L{L}']=roc_metrics(sp,rawneg[f'D3_L{L}']); raw[f'D3_L{L}']=sp
        sp=score_seq(suite['seq'][f'D4_L{L}'],ec,ai,L); met[f'D4_L{L}']=roc_metrics(sp,rawneg[f'D4_L{L}']); raw[f'D4_L{L}']=sp
    strongest={}
    for L,ks in [(1,['D1_L1','D2_L1','D4_L1']),(4,['D3_L4','D4_L4']),(8,['D3_L8','D4_L8'])]:
        kk=max(ks,key=lambda z:met[z]['AUROC_sym']); strongest[str(L)]={'detector':kk,**met[kk]}
    return met,strongest,raw

def choose_attackers(suite,rawneg,thresholds,metrics,strongest,rawscores):
    sstat={'D1':roc_metrics(rawscores['D1'],rawneg['D1']),'D2':roc_metrics(rawscores['D2'],rawneg['D2'])}; stat=max(sstat,key=lambda k:sstat[k]['AUROC_sym'])
    sstate={'D3_L4':roc_metrics(rawscores['D3_L4'],rawneg['D3_L4']),'D4_L4':roc_metrics(rawscores['D4_L4'],rawneg['D4_L4'])}; state=max(sstate,key=lambda k:sstate[k]['AUROC_sym'])
    return stat,state,sstat,sstate

def eval_attacks(sessions,pool,suite,thresholds,stat,state,calE,tau_q,tau_p,risk_tau,gap_tau,use_risk):
    attacks={}; main_rows={}
    for beta in CFG['beta_grid']:
        bk=str(beta); a1rows=[]; a2rows=[]; airs1=[]; airs2=[]; th1=thresholds[bk][stat]['threshold']; th2=thresholds[bk][state]['threshold']; m2=suite['seq'][state]
        for s in sessions:
            mix,a,_=attack_a1(s,pool,suite['d1'],suite['d2'],suite['vec'],EC_GLOBAL,stat,th1,calE['qwen_nf4'],calE['phi_source']); a1rows.append(srp_session(s['order'],mix,calE['qwen_source'],calE['phi_source'],tau_q,tau_p,risk_tau,gap_tau,use_risk)); airs1.append(a)
            mix,a,_=attack_a2(s,pool,m2,EC_GLOBAL,4,th2,calE['qwen_nf4'],calE['phi_source']); a2rows.append(srp_session(s['order'],mix,calE['qwen_source'],calE['phi_source'],tau_q,tau_p,risk_tau,gap_tau,use_risk)); airs2.append(a)
        attacks[bk]={'A1':{**verdict_rates(a1rows),'AIR':float(np.mean(airs1)),'BDR':thresholds[bk][stat]['measured_BDR'],'AIR_minus_BDR':float(np.mean(airs1))-thresholds[bk][stat]['measured_BDR']},'A2':{**verdict_rates(a2rows),'AIR':float(np.mean(airs2)),'BDR':thresholds[bk][state]['measured_BDR'],'AIR_minus_BDR':float(np.mean(airs2))-thresholds[bk][state]['measured_BDR']}}
        if abs(beta-CFG['main_beta'])<1e-12: main_rows={'A1':a1rows,'A2':a2rows,'AIR1':airs1,'AIR2':airs2}
    honestrows=[srp_session(s['order'],calE['qwen_nf4'],calE['qwen_source'],calE['phi_source'],tau_q,tau_p,risk_tau,gap_tau,use_risk) for s in sessions]
    return verdict_rates(honestrows),attacks,honestrows,main_rows

def selftest():
    rng=np.random.default_rng(1); J=rng.random(128); s=weighted_session(J,.2,64,3); assert len(s)==64 and len(set(s))==64; w=distribution_weights(J,.2); assert abs(w.sum()-1)<1e-8
    A=rng.normal(size=(64,32)); r=srp_session(list(range(64)),A,A,A+0.1); assert all(k in r for k in ['sim_qwen','sim_phi','risk','gap'])
    print('SELFTEST_OK')

EC_GLOBAL=None

def main():
  global EC_GLOBAL
  if '--selftest' in sys.argv: selftest(); return
  phases={}; log('=== StageP4A SRP Full-SLATE Dev/Cal Freeze ==='); log(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
  try:
    if CFG.get('heldout_access',True): raise RuntimeError('P4A config heldout_access must be false')
    required=[P3B_RET/'status.json',P3B_RET/'summary.json',P3B_RET/'candidate_pool_freeze.json',P3B_EMB/'qwen_source.npz',P3B_EMB/'qwen_nf4.npz',P3B_EMB/'phi_source.npz',TRAFFIC,MPNET]
    if not all(Path(p).exists() for p in required): raise RuntimeError('required P3B/traffic/MPNet asset missing')
    p3bst=json.load(open(P3B_RET/'status.json')); p3bsum=json.load(open(P3B_RET/'summary.json'))
    if p3bst.get('status')!='STAGEP3B_COMPLETE' or p3bst.get('heldout_touched') is not False: raise RuntimeError('P3B must be complete and heldout-clean')
    phases['input_audit']='ok'

    dev_real=sessions_for_split('development'); cal_real=sessions_for_split('calibration'); dev_pool=freeze_pool(dev_real,CFG['candidate_dev'],'P3B_DEV_POOL'); cal_pool=freeze_pool(cal_real,CFG['candidate_cal'],'P3B_CAL_POOL')
    if len(dev_pool)!=CFG['candidate_dev'] or len(cal_pool)!=CFG['candidate_cal']: raise RuntimeError('candidate pool reconstruction failed')
    p3b_pool=json.load(open(P3B_RET/'candidate_pool_freeze.json'))
    dev_hash=sha('\n'.join(r['id'] for r in dev_pool)); cal_hash=sha('\n'.join(r['id'] for r in cal_pool))
    if dev_hash!=p3b_pool.get('development_sha256') or cal_hash!=p3b_pool.get('calibration_sha256'):
        raise RuntimeError('reconstructed P3B candidate-pool hash mismatch')
    allrec=[{**r,'pool':'dev'} for r in dev_pool]+[{**r,'pool':'cal'} for r in cal_pool]; ids=[r['id'] for r in allrec]
    embs={}; audit={}
    for tag in ['qwen_source','qwen_nf4','phi_source']: embs[tag],audit[tag]=load_p3b_embeddings(tag,ids)
    ndev=len(dev_pool); devE={k:v[:ndev] for k,v in embs.items()}; calE={k:v[ndev:] for k,v in embs.items()}
    dump(RET/'input_reuse_audit.json',{'P3B_status':p3bst,'P3B_summary_sha256':file_sha(P3B_RET/'summary.json'),'embedding_banks':audit,'development_pool_sha256':dev_hash,'calibration_pool_sha256':cal_hash,'heldout_access':False,'new_model_generation':False}); phases['response_embedding_reuse']='ok'

    # SRP informativeness is source-only: semantic separation between the two enrolled source prototypes.
    rawd=1-cosrow(devE['qwen_source'],devE['phi_source']); lo,hi=np.quantile(rawd,[.05,.95]); den=max(float(hi-lo),1e-8); Jd=np.clip((rawd-lo)/den,0,1)
    rawc=1-cosrow(calE['qwen_source'],calE['phi_source']); Jc=np.clip((rawc-lo)/den,0,1)
    dump(RET/'evidence_model.json',{'engine':'Semantic Response Profile (SRP)','definition':'J_E(x)=cosine-distance(Qwen-source response prototype, Phi-source response prototype), source-only; clipped using development q05/q95','dev_q05':float(lo),'dev_q95':float(hi),'dev_mean':float(Jd.mean()),'cal_mean':float(Jc.mean()),'uses_suspect_response':False,'trim_fraction':CFG['trim_fraction']}); phases['evidence']='ok'

    # Symmetric source-verification thresholds on uniform calibration sessions.
    rng=np.random.default_rng(CFG['seed']+77); base=[rng.choice(len(cal_pool),CFG['main_q'],replace=False).tolist() for _ in range(CFG['lineage_cal_sessions'])]
    honest0=[srp_session(i,calE['qwen_nf4'],calE['qwen_source'],calE['phi_source']) for i in base]
    qneg=[srp_session(i,calE['phi_source'],calE['qwen_source'],calE['phi_source'])['sim_qwen'] for i in base]
    pneg=[srp_session(i,calE['qwen_source'],calE['qwen_source'],calE['phi_source'])['sim_phi'] for i in base]
    tau_q=float(np.quantile(qneg,1-CFG['source_far'],method='higher')); tau_p=float(np.quantile(pneg,1-CFG['source_far'],method='higher')); risk_tau=float(np.quantile([r['risk'] for r in honest0],.975,method='higher')); gap_tau=float(np.quantile([r['gap'] for r in honest0],.025,method='lower'))
    h0v=[srp_session(i,calE['qwen_nf4'],calE['qwen_source'],calE['phi_source'],tau_q,tau_p,risk_tau,gap_tau,True) for i in base]
    dump(RET/'lineage_thresholds.json',{'tau_qwen_from_phi_negatives':tau_q,'tau_phi_from_qwen_negatives':tau_p,'risk_threshold_kappa':risk_tau,'bank_gap_threshold':gap_tau,'source_far_target':CFG['source_far'],'qwen_false_accept_cal':float(np.mean(np.asarray(qneg)>=tau_q)),'phi_false_accept_cal':float(np.mean(np.asarray(pneg)>=tau_p)),'uniform_honest_rates':verdict_rates(h0v),'protocol':'source thresholds use cross-source negatives; risk/gap use calibration honest only; all frozen before heldout'}); phases['lineage_thresholds']='ok'

    from sentence_transformers import SentenceTransformer
    mp=SentenceTransformer(str(MPNET),device='cuda:0'); sched=build_scheduler(dev_real,dev_pool,cal_pool,mp); EC_GLOBAL=ECache(mp)
    dev_sessions={str(l):make_sessions(Jd,l,CFG['dev_sessions_per_lambda'],CFG['main_q'],sched['states_dev'],sched['schedule'],'DEV',True) for l in CFG['lambda_grid']}
    cal_sessions={str(l):make_sessions(Jc,l,CFG['cal_sessions_per_lambda'],CFG['main_q'],sched['states_cal'],sched['schedule'],'CAL',True) for l in CFG['lambda_grid']}
    cal_random={str(l):make_sessions(Jc,l,CFG['cal_sessions_per_lambda'],CFG['main_q'],sched['states_cal'],sched['schedule'],'CAL',False) for l in CFG['lambda_grid']}
    unwrapped_dev=fixed_sessions(Jd,CFG['dev_sessions_per_lambda'],CFG['main_q'],'DEV'); unwrapped_cal=fixed_sessions(Jc,CFG['cal_sessions_per_lambda'],CFG['main_q'],'CAL')

    # Full-SLATE detector suite is adaptive across all lambda candidates; unwrapped suite is trained on fixed top-q audit traffic.
    full_dev_pooled=[s for l in CFG['lambda_grid'] for s in dev_sessions[str(l)]]
    full_suite=train_suite('FULL',dev_pool,dev_real,full_dev_pooled,EC_GLOBAL,0); unwrap_suite=train_suite('UNWRAPPED',dev_pool,dev_real,unwrapped_dev,EC_GLOBAL,1000)
    full_neg,full_thr=calibrate_suite('FULL',full_suite,cal_real,cal_pool,EC_GLOBAL); unwrap_neg,unwrap_thr=calibrate_suite('UNWRAPPED',unwrap_suite,cal_real,cal_pool,EC_GLOBAL); phases['detectors']='ok'

    perlam={}; raw_by_lam={}
    for lam in CFG['lambda_grid']:
        key=str(lam); met,strong,raw=suite_metrics(full_suite,full_neg,cal_pool,cal_sessions[key],EC_GLOBAL); raw_by_lam[key]=raw
        honest=[srp_session(s['order'],calE['qwen_nf4'],calE['qwen_source'],calE['phi_source'],tau_q,tau_p,risk_tau,gap_tau,True) for s in cal_sessions[key]]; hr=verdict_rates(honest); w=distribution_weights(Jd,lam); kl=float(np.sum(w*np.log(np.maximum(w*len(w),1e-300)))); K=float(CFG['main_q']*kl); meanj=float(np.mean([s['mean_J'] for s in cal_sessions[key]])); gamma=meanj-float(Jc.mean())
        perlam[key]={'honest':hr,'strongest_by_L':strong,'mean_J':meanj,'Khat_q':K,'Gamma_select':gamma}
    feasible=[str(l) for l in CFG['lambda_grid'] if perlam[str(l)]['honest']['CSVR']>=.95]; cand=feasible if feasible else [str(l) for l in CFG['lambda_grid']]
    def rank(k):
        x=perlam[k]; return (x['strongest_by_L']['4']['BDR_at_AIR_0.8'],-x['strongest_by_L']['4']['AUROC_sym'],x['Gamma_select'],x['honest']['CSVR']) if feasible else (x['honest']['CSVR'],x['strongest_by_L']['4']['BDR_at_AIR_0.8'],-x['strongest_by_L']['4']['AUROC_sym'],x['Gamma_select'])
    chosen=max(cand,key=rank); chosen_l=float(chosen)
    dump(RET/'lambda_freeze.json',{'lambda_grid':CFG['lambda_grid'],'selection_rule':'same frozen P3B rule: among honest CSVR>=.95 maximize strongest-L4 BDR@AIR=.8; else maximize honest CSVR first; then same tie-breaks','feasible_lambdas':list(map(float,feasible)),'chosen_lambda':chosen_l,'per_lambda':perlam,'heldout_touched':False})

    fmet,fstrong,fraw=suite_metrics(full_suite,full_neg,cal_pool,cal_sessions[chosen],EC_GLOBAL); fstat,fstate,fsstat,fsstate=choose_attackers(full_suite,full_neg,full_thr,fmet,fstrong,fraw)
    umet,ustrong,uraw=suite_metrics(unwrap_suite,unwrap_neg,cal_pool,unwrapped_cal,EC_GLOBAL); ustat,ustate,usstat,usstate=choose_attackers(unwrap_suite,unwrap_neg,unwrap_thr,umet,ustrong,uraw)
    dump(RET/'detector_freeze.json',{'full_slate':{'training':'development pooled across frozen lambda grid','thresholds':full_thr,'stateless':fstat,'stateful_L4':fstate,'strongest_by_L':fstrong},'unwrapped_srp':{'training':'development deterministic top-q SRP audit traffic','thresholds':unwrap_thr,'stateless':ustat,'stateful_L4':ustate,'strongest_by_L':ustrong},'rule':'P4B rebuild must reproduce these metrics before heldout access.'})

    full_hon,full_att,full_rows,full_main=eval_attacks(cal_sessions[chosen],cal_pool,full_suite,full_thr,fstat,fstate,calE,tau_q,tau_p,risk_tau,gap_tau,True)
    unwrap_hon,unwrap_att,unwrap_rows,unwrap_main=eval_attacks(unwrapped_cal,cal_pool,unwrap_suite,unwrap_thr,ustat,ustate,calE,tau_q,tau_p,risk_tau,gap_tau,False)
    ci={'full_honest_CSVR':ci_binary([r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base' for r in full_rows]),'unwrapped_honest_CSVR':ci_binary([r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base' for r in unwrap_rows]),'full_A1_beta10_CSVR':ci_binary([r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base' for r in full_main['A1']]),'full_A2_beta10_CSVR':ci_binary([r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base' for r in full_main['A2']]),'unwrapped_A1_beta10_CSVR':ci_binary([r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base' for r in unwrap_main['A1']]),'unwrapped_A2_beta10_CSVR':ci_binary([r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base' for r in unwrap_main['A2']])}

    realdist={L:real_state_span_dist(cal_real,sched['state_fn'],L,sched['S']) for L in CFG['L_grid']}; deltas={}
    for L in CFG['L_grid']:
        dr=state_span_dist([s['random_order'] for s in cal_random[chosen]],sched['states_cal'],L,sched['S']); ds=state_span_dist([s['order'] for s in cal_sessions[chosen]],sched['states_cal'],L,sched['S']); deltas[str(L)]={'delta_random':tv(dr,realdist[L]),'delta_sched_L':tv(ds,realdist[L]),'improvement':tv(dr,realdist[L])-tv(ds,realdist[L])}
    eps={}
    for L in CFG['L_grid']:
        det=fstrong[str(L)]['detector']; key=det.split('_L')[0] if det in ('D1_L1','D2_L1') else det; pos=fraw[key]; th,b=threshold_for_bdr(full_neg[key],CFG['main_beta']); eps[str(L)]={'detector':det,'AIR':float(np.mean(pos>=th)),'BDR':b,'epsilon_D_L':float(abs(np.mean(pos>=th)-b))}
    theory={'q':CFG['main_q'],'lambda':chosen_l,'mean_J':perlam[chosen]['mean_J'],'Khat_q':perlam[chosen]['Khat_q'],'Gamma_select':perlam[chosen]['Gamma_select'],'delta_sched':deltas,'epsilon_D_L':eps,'selection_KL_bound_reference_q_over_lambda':float(CFG['main_q']/chosen_l)}; dump(RET/'theory_bridge_dev.json',theory)

    b10=str(CFG['main_beta']); preview={'G2_cover':{L:{'full_AUROC':fstrong[L]['AUROC_sym'],'unwrapped_AUROC':ustrong[L]['AUROC_sym'],'improvement':ustrong[L]['AUROC_sym']-fstrong[L]['AUROC_sym']} for L in ['1','4','8']},'G3_security_cost_L4':{'full_BDR_at_AIR_0.8':fstrong['4']['BDR_at_AIR_0.8'],'unwrapped_BDR_at_AIR_0.8':ustrong['4']['BDR_at_AIR_0.8'],'gain':fstrong['4']['BDR_at_AIR_0.8']-ustrong['4']['BDR_at_AIR_0.8']},'G4_honest_utility':{'full_CSVR':full_hon['CSVR'],'unwrapped_CSVR':unwrap_hon['CSVR'],'drop':unwrap_hon['CSVR']-full_hon['CSVR']},'G5_beta10':{'A1_full':full_att[b10]['A1']['CSVR'],'A1_unwrapped':unwrap_att[b10]['A1']['CSVR'],'A1_gain':full_att[b10]['A1']['CSVR']-unwrap_att[b10]['A1']['CSVR'],'A1_retention':full_att[b10]['A1']['CSVR']/max(full_hon['CSVR'],1e-12),'A2_full':full_att[b10]['A2']['CSVR'],'A2_unwrapped':unwrap_att[b10]['A2']['CSVR'],'A2_gain':full_att[b10]['A2']['CSVR']-unwrap_att[b10]['A2']['CSVR'],'A2_retention':full_att[b10]['A2']['CSVR']/max(full_hon['CSVR'],1e-12)},'note':'development/calibration preview only; formal gate decisions wait for P4B heldout'}
    dump(RET/'development_metrics.json',{'chosen_lambda':chosen_l,'full_slate':{'honest':full_hon,'attacks':full_att,'strongest_by_L':fstrong},'unwrapped_srp':{'honest':unwrap_hon,'attacks':unwrap_att,'strongest_by_L':ustrong},'bootstrap_CI':ci,'gate_preview':preview,'per_lambda':perlam})

    fixed=unwrapped_cal
    def detect_row(suite,neg,ss):
        met,strong,_=suite_metrics(suite,neg,cal_pool,ss,EC_GLOBAL); return {'strongest_by_L':strong,'mean_J':float(np.mean([s['mean_J'] for s in ss]))}
    dump(RET/'ablation_dev.json',{'B0_unwrapped_topq':{'detectability':detect_row(unwrap_suite,unwrap_neg,fixed),'honest':unwrap_hon,'beta10_A1':unwrap_att[b10]['A1'],'beta10_A2':unwrap_att[b10]['A2']},'S1_selection_only_topq':{'note':'same deterministic informative set as B0, before randomization/scheduling'},'S2_ephemeral_randomization':detect_row(full_suite,full_neg,cal_random[chosen]),'S3_history_scheduler':detect_row(full_suite,full_neg,cal_sessions[chosen]),'S5_full_risk_aware':{'honest':full_hon,'beta10_A1':full_att[b10]['A1'],'beta10_A2':full_att[b10]['A2']},'S4_private_consistency_note':'P3D G6 failed for LLM-DNA; retained only as auxiliary risk-aware mechanism and not re-optimized here.'})

    freeze={'stage':'P4B_INPUT_FREEZE','engine':'Semantic Response Profile (SRP)','engine_definition':{'response_encoder':'Qwen3-Embedding-8B response embeddings reused from P3B','prototype':'per-query source response embedding','session_aggregation':f"{int(CFG['trim_fraction']*100)}% trimmed mean cosine similarity",'source_candidates':['Qwen3-4B-Base','Phi-3-mini-4k-instruct']},'family':'Qwen3-4B-Base','descendant':'NF4 runtime quantization','decoy':'Phi-3-mini-4k-instruct','q':CFG['main_q'],'lambda':chosen_l,'beta_grid':CFG['beta_grid'],'main_beta':CFG['main_beta'],'L_grid':CFG['L_grid'],'main_L':CFG['main_L'],'tau_qwen':tau_q,'tau_phi':tau_p,'risk_threshold_kappa':risk_tau,'bank_gap_threshold':gap_tau,'full_detectors':{'stateless':fstat,'stateful_L4':fstate,'thresholds':full_thr,'strongest_by_L':fstrong},'unwrapped_detectors':{'stateless':ustat,'stateful_L4':ustate,'thresholds':unwrap_thr,'strongest_by_L':ustrong},'unwrapped_query_strategy':'deterministic top-q by source-only J_E','evidence_scaling':{'q05':float(lo),'q95':float(hi)},'candidate_pool_hashes':{'dev':sha('\n'.join(r['id'] for r in dev_pool)),'cal':sha('\n'.join(r['id'] for r in cal_pool))},'scheduler':{'semantic_clusters':CFG['semantic_clusters'],'length_bins':CFG['length_bins'],'length_quantiles':sched['length_quantiles']},'heldout_access_P4A':False,'rule':'P4B one-shot heldout must rebuild/audit frozen detectors and must not retune any field.'}; dump(RET/'p4b_freeze_manifest.json',freeze)
    paper={'stage':'P4A','development_or_confirmatory':'development/calibration freeze','engine':'Semantic Response Profile (SRP)','family':'Qwen3-4B-Base','descendant_type':'NF4 runtime quantization','method':'Full SLATE vs matched unwrapped SRP','q':CFG['main_q'],'lambda':chosen_l,'full_honest_csvr':full_hon['CSVR'],'unwrapped_honest_csvr':unwrap_hon['CSVR'],'full_beta10_csvr':{'A1':full_att[b10]['A1']['CSVR'],'A2':full_att[b10]['A2']['CSVR']},'unwrapped_beta10_csvr':{'A1':unwrap_att[b10]['A1']['CSVR'],'A2':unwrap_att[b10]['A2']['CSVR']},'full_auroc':{L:fstrong[L]['AUROC_sym'] for L in ['1','4','8']},'unwrapped_auroc':{L:ustrong[L]['AUROC_sym'] for L in ['1','4','8']},'full_bdr_at_air_08_L4':fstrong['4']['BDR_at_AIR_0.8'],'unwrapped_bdr_at_air_08_L4':ustrong['4']['BDR_at_AIR_0.8'],'development_gate_preview':preview,'theory_bridge':theory,'bootstrap_CI':ci,'claim_scope':'development only; headline waits for P4B','next_required':'P4B one-shot heldout confirmatory; no retuning'}; dump(RET/'paper_result_summary.json',paper)
    summary={'workflow':CFG['workflow'],'status':'STAGEP4A_COMPLETE','heldout_touched':False,'new_model_generation':False,'reused_P3B_response_embeddings':True,'chosen_lambda':chosen_l,'full_honest':full_hon,'unwrapped_honest':unwrap_hon,'development_gate_preview':preview,'next':'StageP4B SRP Full-SLATE Held-out Confirmatory; freeze p4b_freeze_manifest.json'}; dump(RET/'summary.json',summary); dump(RET/'status.json',{'status':'STAGEP4A_COMPLETE','phases':{**phases,'detectors':'ok','attacks':'ok','freeze':'ok'},'heldout_touched':False,'retuned':False,'new_model_generation':False}); log(json.dumps(summary,indent=2))
  except Exception as e:
    dump(RET/'status.json',{'status':'STAGEP4A_FAILED','error':repr(e),'traceback':traceback.format_exc(),'phases':phases,'heldout_touched':False}); dump(RET/'summary.json',{'workflow':CFG['workflow'],'status':'FAILED','error':repr(e),'heldout_touched':False}); dump(RET/'paper_result_summary.json',{'stage':'P4A','status':'FAILED_BEFORE_FREEZE','paper_facing_claims':False,'error':repr(e)}); raise

if __name__=='__main__': main()
