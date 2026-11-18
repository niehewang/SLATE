from __future__ import annotations
import argparse, collections, dataclasses, gc, hashlib, inspect, json, math, os, pathlib, random, shutil, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CFG=json.load(open(ROOT/'config/project.json'))
RUN=Path.home()/'slate_shared/runs/slate_stagep3b_llmdna_fullslate_devcal_freeze_v1'
RET=RUN/'return'; BANK=RUN/'banks'; EMB=RUN/'embeddings'; CACHE=RUN/'cache'
for p in [RUN,RET,BANK,EMB,CACHE]: p.mkdir(parents=True,exist_ok=True)
LOG=RUN/'run.log'
PYTHON='/path/to/python'
SHARED=Path.home()/'slate_shared'
P3A=SHARED/'runs/slate_stagep3a_llmdna_official_freeze_v1_2'
LLMDNA_SRC=P3A/'frozen_release/LLM-DNA-v1.0.1/src'
P1_VENDOR=SHARED/'runs/slate_stagep1_published_baselines_honest_native_reproduction_v1_1/vendor'
QWEN=Path('/path/to/Qwen3-4B-Base')
PHI_CONFIGURED=SHARED/'public_models/Phi-3-mini-4k-instruct'
PHI_P2B_VERIFIED=Path('/path/to/slate_shared/public_models/Phi-3-mini-4k-instruct')

def model_complete_path(p):
    p=Path(p)
    return (p/'config.json').exists() and bool(list(p.glob('*.safetensors'))+list(p.glob('*.bin')))

def resolve_phi():
    # v1.2.1 asset-path resume: the successful P2B return proves this checkpoint
    # lives under /data/user after the storage migration. No online download.
    candidates=[PHI_P2B_VERIFIED, PHI_CONFIGURED]
    cache_roots=[
        Path.home()/'.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots',
        Path('/path/to/Phi-3-mini-4k-instruct'),
    ]
    for cache in cache_roots:
        if cache.exists():
            candidates += sorted([x for x in cache.iterdir() if x.is_dir()], key=lambda x:x.stat().st_mtime, reverse=True)
    for cand in candidates:
        if model_complete_path(cand): return cand
    return PHI_P2B_VERIFIED

PHI=resolve_phi()
ENC=SHARED/'public_models/Qwen3-Embedding-8B'
MPNET=Path('/path/to/all-mpnet-base-v2')
TRAFFIC=SHARED/'runs/slate_stage0_traffic_bootstrap_v1/traffic_frozen.jsonl'

os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['PYTHONPATH']=os.pathsep.join([str(LLMDNA_SRC),str(P1_VENDOR),os.environ.get('PYTHONPATH','')])
if str(LLMDNA_SRC) not in sys.path: sys.path.insert(0,str(LLMDNA_SRC))
if str(P1_VENDOR) not in sys.path: sys.path.insert(0,str(P1_VENDOR))

def log(x):
    s=str(x); print(s,flush=True)
    with open(LOG,'a',encoding='utf-8') as f: f.write(s+'\n')

def dump(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False,default=str)

def sha(x): return hashlib.sha256(str(x).encode()).hexdigest()
def h64(x,salt=''): return int.from_bytes(hashlib.sha256((salt+'\0'+str(x)).encode()).digest()[:8],'big')

def model_complete(p): return model_complete_path(p)

def load_jsonl(p):
    with open(p,encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def valid_text(t):
    t=str(t).strip(); n=len(t.split())
    return 4<=n<=256 and 12<=len(t)<=2400

def sessions_for_split(split):
    if split not in {'development','calibration'}:
        raise RuntimeError(f'P3B held-out hard guard: forbidden split {split!r}')
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
    arr.sort(key=lambda x:h64(x['id'],salt))
    return arr[:n]

def excluded_queries(sessions,exclude,cap,salt):
    ex=set(exclude); arr=[]; seen=set()
    for s in sessions:
        for t in s['texts']:
            if t in ex or t in seen: continue
            seen.add(t); arr.append(t)
    arr.sort(key=lambda t:h64(t,salt)); return arr[:cap]

def native_p3a_summary():
    need=['official_release_freeze.json','source_q64.json','nf4_q64.json','official_distance_q64.json','embedding_asset.json']
    out={}
    for fn in need:
        p=P3A/fn
        if not p.exists(): raise RuntimeError('missing P3A artifact '+str(p))
        out[fn]=json.load(open(p))
    return out

def signature_vector(obj, expected=128):
    audit={'type':type(obj).__name__,'module':type(obj).__module__,'selected':None,'candidates':[]}
    def tryv(name,v):
        try:
            a=np.asarray(v,dtype=np.float64).reshape(-1)
            audit['candidates'].append({'name':name,'size':int(a.size),'finite':bool(np.isfinite(a).all())})
            if a.size==expected and np.isfinite(a).all(): return a
        except Exception as e:
            audit['candidates'].append({'name':name,'error':repr(e)})
        return None
    for name in ['vector','dna_vector','dna','signature','values','embedding','features','data']:
        if hasattr(obj,name):
            a=tryv(name,getattr(obj,name))
            if a is not None: audit['selected']=name; return a,audit
    try:
        d=dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else None
        if d:
            for k,v in d.items():
                a=tryv('dataclass.'+k,v)
                if a is not None: audit['selected']='dataclass.'+k; return a,audit
    except Exception: pass
    if hasattr(obj,'to_dict'):
        try:
            d=obj.to_dict()
            if isinstance(d,dict):
                for k,v in d.items():
                    a=tryv('to_dict.'+str(k),v)
                    if a is not None: audit['selected']='to_dict.'+str(k); return a,audit
        except Exception: pass
    if hasattr(obj,'__dict__'):
        for k,v in vars(obj).items():
            a=tryv('__dict__.'+str(k),v)
            if a is not None: audit['selected']='__dict__.'+str(k); return a,audit
    raise TypeError('Cannot locate 128-D numeric vector in DNASignature; audit='+json.dumps(audit))

class DNAAdapter:
    def __init__(self):
        from llm_dna import TextDNAExtractor
        self.obj=TextDNAExtractor(dna_dim=128,pre_agg_embed_dim=64,normalize_embeddings=False,random_seed=42,
                                  aggregation_method='concat',reduction_method='random_projection',encoder_name=str(ENC),device='cuda:0')
        self.audit=None
    def vec(self,E,name='model'):
        sig=self.obj.extract_dna_from_embeddings(np.asarray(E,np.float32),model_name=name)
        v,a=signature_vector(sig,128)
        if self.audit is None: self.audit=a
        return v

def adapter_preflight():
    ad=DNAAdapter(); rng=np.random.default_rng(CFG['seed']); E=rng.normal(size=(CFG['main_q'],1024)).astype(np.float32)
    a=ad.vec(E,'preflightA'); b=ad.vec(E,'preflightB')
    out={'return_object_audit':ad.audit,'shape':list(a.shape),'finite':bool(np.isfinite(a).all()),'deterministic_max_abs_diff':float(np.max(np.abs(a-b))),'norm':float(np.linalg.norm(a))}
    if out['shape']!=[128] or not out['finite'] or out['deterministic_max_abs_diff']>1e-7: raise RuntimeError('LLM-DNA adapter preflight failed: '+json.dumps(out))
    return ad,out

def generate_bank(tag,path,kind,records):
    """Generate a resumable response bank with stable batch identities.

    A resumed run iterates the original full-record batch boundaries and replays a
    partially completed batch with the same batch seed. This avoids the v1/v1.1
    behavior where removing already-complete rows shifted later batch seeds.
    """
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    out=BANK/f'{tag}.json'; state={'tag':tag,'kind':kind,'rows':[]}
    if out.exists():
        try: state=json.load(open(out))
        except Exception: pass
    done={r['id']:r for r in state.get('rows',[]) if r.get('response') is not None}
    if all(r['id'] in done for r in records): return [done[r['id']] for r in records]
    tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=(kind=='phi'))
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side='left'
    kw={'local_files_only':True,'low_cpu_mem_usage':True,'trust_remote_code':(kind=='phi')}
    if kind=='nf4':
        kw.update(device_map='auto',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16))
    else: kw.update(torch_dtype=torch.float16)
    if kind=='phi': kw['attn_implementation']='eager'
    m=AutoModelForCausalLM.from_pretrained(str(path),**kw)
    if kind!='nf4': m.to('cuda:0')
    m.eval()
    if kind=='phi': m.config.use_cache=False
    bs=int(CFG['generation_batch_size'])
    for st in range(0,len(records),bs):
        batch=records[st:st+bs]
        missing_ids={r['id'] for r in batch if r['id'] not in done}
        if not missing_ids: continue
        texts=[]
        for r in batch:
            t=r['text']
            if kind=='phi' and hasattr(tok,'apply_chat_template'):
                try: t=tok.apply_chat_template([{'role':'user','content':t}],tokenize=False,add_generation_prompt=True)
                except Exception: pass
            texts.append(t)
        enc=tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=768)
        dev=m.device if hasattr(m,'device') else next(m.parameters()).device
        enc={k:v.to(dev) for k,v in enc.items()}
        seed=CFG['seed']+h64(tag+'|'+str(st),'GEN')%1000000; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        gkw=dict(**enc,max_new_tokens=CFG['response_max_new_tokens'],do_sample=True,temperature=CFG['generation_temperature'],top_p=CFG['generation_top_p'],top_k=0,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        if kind=='phi': gkw['use_cache']=False
        with torch.inference_mode(): y=m.generate(**gkw)
        inlen=enc['input_ids'].shape[1]
        for r,yy in zip(batch,y):
            if r['id'] not in missing_ids: continue
            resp=tok.decode(yy[inlen:],skip_special_tokens=True).strip()
            rec={**r,'response':resp,'response_chars':len(resp),'generation_batch_start':int(st),'generation_seed':int(seed)}; done[r['id']]=rec
        state['rows']=[done[k] for k in sorted(done)]; dump(out,state); log(f'{tag}: {len(done)}/{len(records)}')
    del m,tok; gc.collect(); torch.cuda.empty_cache()
    return [done[r['id']] for r in records]

def embed_bank(tag,rows,model):
    p=EMB/f'{tag}.npz'
    ids=[r['id'] for r in rows]
    if p.exists():
        z=np.load(p,allow_pickle=False)
        if list(z['ids'].astype(str))==ids: return np.asarray(z['E'],np.float32)
    texts=[r['response'] if r['response'].strip() else ' ' for r in rows]
    E=model.encode(texts,batch_size=16,normalize_embeddings=False,convert_to_numpy=True,show_progress_bar=True)
    E=np.asarray(E,np.float32); np.savez_compressed(p,ids=np.asarray(ids),E=E)
    return E

def cosrow(A,B):
    A=np.asarray(A,float);B=np.asarray(B,float)
    return np.sum(A*B,axis=1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+1e-12)

def cosine(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float); return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def weighted_session(J,lam,q,seed):
    rng=np.random.default_rng(seed); logw=np.asarray(J,float)/float(lam); g=-np.log(-np.log(np.clip(rng.random(len(J)),1e-12,1-1e-12)))
    idx=np.argsort(logw+g)[-q:]; return idx.astype(int).tolist()

def distribution_weights(J,lam):
    x=np.asarray(J,float)/float(lam); x-=x.max();w=np.exp(x);w/=w.sum();return w

def build_scheduler(dev_sessions,dev_pool,cal_pool,mpnet):
    from sklearn.cluster import MiniBatchKMeans
    flat=[]
    for s in dev_sessions:
        flat.extend(s['texts'])
        if len(flat)>=5000: break
    flat=flat[:5000]
    X=mpnet.encode(flat,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    km=MiniBatchKMeans(n_clusters=CFG['semantic_clusters'],random_state=CFG['seed'],batch_size=512,n_init=5).fit(X)
    lens=np.asarray([len(t.split()) for t in flat],float); qs=np.quantile(lens,[.25,.5,.75]).tolist()
    cache={}
    def states(texts):
        miss=[t for t in texts if t not in cache]
        if miss:
            E=mpnet.encode(miss,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False); cl=km.predict(E)
            for t,c in zip(miss,cl):
                lb=int(np.searchsorted(qs,len(t.split()),side='right')); cache[t]=int(c)*CFG['length_bins']+lb
        return np.asarray([cache[t] for t in texts],int)
    S=CFG['semantic_clusters']*CFG['length_bins']; init=np.ones(S)*1e-3; trans=np.ones((S,S))*1e-3
    for s in dev_sessions:
        st=states(s['texts']);
        if len(st): init[st[0]]+=1
        for a,b in zip(st[:-1],st[1:]): trans[a,b]+=1
    init/=init.sum(); trans/=trans.sum(1,keepdims=True)
    dev_states=states([r['text'] for r in dev_pool]); cal_states=states([r['text'] for r in cal_pool])
    def schedule(selected,pool_states,seed):
        rng=np.random.default_rng(seed); rem=list(map(int,selected)); out=[]
        # choose best initial probability, random tie
        vals=np.asarray([init[pool_states[i]] for i in rem]); best=np.flatnonzero(vals>=vals.max()-1e-15); j=int(rng.choice(best)); cur=rem.pop(j); out.append(cur)
        while rem:
            a=pool_states[cur]; vals=np.asarray([trans[a,pool_states[i]] for i in rem]); best=np.flatnonzero(vals>=vals.max()-1e-15); j=int(rng.choice(best)); cur=rem.pop(j); out.append(cur)
        return out
    return {'states_dev':dev_states,'states_cal':cal_states,'init':init,'trans':trans,'length_quantiles':qs,'schedule':schedule,'state_fn':states,'S':S}

def make_sessions(J,lam,n,q,states,scheduler,split_tag,scheduled=True):
    out=[]
    for e in range(n):
        seed=CFG['seed']+h64(f'{split_tag}|{lam}|{e}','SESSION')%100000000
        sel=weighted_session(J,lam,q,seed); rng=np.random.default_rng(seed+17); rnd=list(np.asarray(sel)[rng.permutation(len(sel))])
        order=scheduler(sel,states,seed+31) if scheduled else rnd
        out.append({'episode':e,'seed':int(seed),'selected':list(map(int,sel)),'random_order':list(map(int,rnd)),'order':list(map(int,order)),'mean_J':float(np.mean(np.asarray(J)[sel]))})
    return out

def real_windows(sessions,L,cap,salt):
    arr=[]
    for s in sessions:
        for i in range(len(s['texts'])-L+1): arr.append((s['texts'][i:i+L],s['id']))
    arr.sort(key=lambda x:h64('|'.join(x[0])+'|'+x[1],salt)); return arr[:cap]

class ECache:
    def __init__(self,model): self.m=model; self.c={}
    def get(self,texts):
        miss=[];seen=set()
        for t in texts:
            if t not in self.c and t not in seen:seen.add(t);miss.append(t)
        if miss:
            E=self.m.encode(miss,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            for t,e in zip(miss,E): self.c[t]=np.asarray(e,np.float32)
        return np.stack([self.c[t] for t in texts])

def seq_features(cache,items,L):
    flat=[t for seq,_ in items for t in seq]; E=cache.get(flat);out=[];k=0
    for seq,_ in items:
        z=E[k:k+L];k+=L; lens=np.asarray([len(x) for x in seq],np.float32)[:,None]/2048.; pos=np.arange(L,dtype=np.float32)[:,None]/max(1,L-1)
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
    Xp=seq_features(cache,pos,L);Xn=seq_features(cache,neg,L);X=np.concatenate([Xp,Xn]);y=np.r_[np.ones(len(Xp)),np.zeros(len(Xn))].astype(np.float32)
    rng=np.random.default_rng(seed);torch.manual_seed(seed);dev='cuda' if torch.cuda.is_available() else 'cpu';m=make_seqnet(X.shape[-1],h,bidir).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4);lossfn=torch.nn.BCEWithLogitsLoss()
    for _ in range(CFG['seq_epochs']):
        order=rng.permutation(len(y))
        for st in range(0,len(y),CFG['seq_batch_size']):
            ii=order[st:st+CFG['seq_batch_size']]; xb=torch.from_numpy(X[ii]).to(dev);yb=torch.from_numpy(y[ii]).to(dev);opt.zero_grad();loss=lossfn(m(xb),yb);loss.backward();opt.step()
    return m

def score_seq(m,cache,items,L):
    import torch
    X=seq_features(cache,items,L);dev=next(m.parameters()).device;out=[];m.eval()
    with torch.inference_mode():
        for st in range(0,len(X),256): out.append(torch.sigmoid(m(torch.from_numpy(X[st:st+256]).to(dev))).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)

def roc_metrics(pos,neg):
    from sklearn.metrics import roc_auc_score,roc_curve
    pos=np.asarray(pos,float);neg=np.asarray(neg,float);y=np.r_[np.ones(len(pos)),np.zeros(len(neg))];s=np.r_[pos,neg];auc=float(roc_auc_score(y,s));fpr,tpr,thr=roc_curve(y,s)
    b80=float(np.min(fpr[tpr>=.8])) if np.any(tpr>=.8) else 1.;a10=float(np.max(tpr[fpr<=.1])) if np.any(fpr<=.1) else 0.
    return {'AUROC':auc,'AUROC_sym':max(auc,1-auc),'BDR_at_AIR_0.8':b80,'AIR_at_BDR_0.1':a10}

def threshold_for_bdr(neg,beta):
    x=np.sort(np.asarray(neg,float))[::-1];n=len(x);k=int(math.floor(beta*n));th=float(np.nextafter(x[0],np.inf)) if k<=0 else (-float('inf') if k>=n else float(np.nextafter(x[k],np.inf)));return th,float(np.mean(np.asarray(neg)>=th))

def audit_items(pool,sessions,L):
    out=[]
    for s in sessions:
        texts=[pool[i]['text'] for i in s['order']]
        for i in range(len(texts)-L+1): out.append((texts[i:i+L],f"a{s['episode']}"))
    return out

def flatten_queries(pool,sessions): return [pool[i]['text'] for s in sessions for i in s['order']]

def lineage_session(ad,idx,sus,src,phi,risk_tau=None,gap_tau=None,tau=None):
    idx=np.asarray(idx,int); ds=ad.vec(sus[idx], 'suspect'); dq=ad.vec(src[idx], 'qwen_source'); dp=ad.vec(phi[idx], 'phi_source'); ss=cosine(ds,dq);sp=cosine(ds,dp)
    A=sus[idx,:64];B=src[idx,:64];C=phi[idx,:64];ms=cosrow(A,B)-cosrow(A,C); med=float(np.median(ms));mad=float(np.median(np.abs(ms-med)));fpos=float(np.mean(ms>=0));mix=min(fpos,1-fpos);risk=float(mad+2.0*mix);gap=float(abs(ss-sp));best='Qwen3-4B-Base' if ss>=sp else 'Phi-3-mini-4k-instruct';score=max(ss,sp)
    verdict=None
    if tau is not None:
        if risk_tau is not None and risk>risk_tau: verdict='Inconclusive-Serving'
        elif score<tau: verdict='Rejected'
        elif gap_tau is not None and gap<gap_tau: verdict='Inconclusive-Bank'
        else: verdict='Verified'
    return {'sim_qwen':ss,'sim_phi':sp,'best':best,'score':score,'risk':risk,'gap':gap,'verdict':verdict}

def verdict_rates(rows):
    n=max(1,len(rows)); return {'CSVR':sum(r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base' for r in rows)/n,'FSVR':sum(r['verdict']=='Verified' and r['best']=='Phi-3-mini-4k-instruct' for r in rows)/n,'Channel_Risk':sum(r['verdict']=='Inconclusive-Serving' for r in rows)/n,'Insufficient':sum(r['verdict'] in ('Inconclusive-Serving','Inconclusive-Bank') for r in rows)/n,'Rejected':sum(r['verdict']=='Rejected' for r in rows)/n}

def ci_binary(vals,reps=None):
    vals=np.asarray(vals,float);reps=reps or CFG['bootstrap_reps'];rng=np.random.default_rng(CFG['seed']);
    if not len(vals):return [None,None]
    z=[float(np.mean(vals[rng.integers(0,len(vals),len(vals))])) for _ in range(reps)];return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]

def state_span_dist(session_orders,pool_states,L,S):
    if L==1:
        cnt=np.ones(S)*1e-9
        for o in session_orders:
            for i in o:cnt[pool_states[i]]+=1
    else:
        cnt=np.ones((S,S))*1e-9
        for o in session_orders:
            st=[pool_states[i] for i in o]
            for i in range(len(st)-L+1):cnt[st[i],st[i+L-1]]+=1
    return cnt.reshape(-1)/cnt.sum()

def real_state_span_dist(real_sessions,state_fn,L,S):
    cnt=np.ones(S if L==1 else S*S)*1e-9
    for s in real_sessions:
        st=state_fn(s['texts'])
        if L==1:
            for x in st:cnt[x]+=1
        else:
            mat=cnt.reshape(S,S)
            for i in range(len(st)-L+1):mat[st[i],st[i+L-1]]+=1
    return cnt/cnt.sum()

def tv(a,b): return float(.5*np.abs(np.asarray(a)-np.asarray(b)).sum())

def attack_a1(session,pool,d1,d2,vec,ec,strong,thr,nf4,phi):
    idx=session['order'];texts=[pool[i]['text'] for i in idx]
    sc=d1.predict_proba(vec.transform(texts))[:,1] if strong=='D1' else d2.predict_proba(ec.get(texts))[:,1]
    route=np.asarray(sc)>=thr;mix=nf4.copy(); sel=np.asarray(idx,int);mix[sel[route]]=phi[sel[route]];return mix,float(route.mean()),sc.tolist()

def attack_a2(session,pool,model,ec,L,thr,nf4,phi):
    idx=session['order'];texts=[pool[i]['text'] for i in idx];route=np.zeros(len(idx),bool);scores=[]
    for t in range(L-1,len(idx)):
        item=[(texts[t-L+1:t+1],'x')];s=float(score_seq(model,ec,item,L)[0]);scores.append(s);route[t]=s>=thr
    mix=nf4.copy();sel=np.asarray(idx,int);mix[sel[route]]=phi[sel[route]];return mix,float(route.mean()),scores

def selftest():
    rng=np.random.default_rng(1);J=rng.random(128);s=weighted_session(J,.2,64,3);assert len(s)==64 and len(set(s))==64
    w=distribution_weights(J,.2);assert abs(w.sum()-1)<1e-8
    print('SELFTEST_OK')

