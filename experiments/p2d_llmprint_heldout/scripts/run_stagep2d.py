from __future__ import annotations
import collections, hashlib, json, math, os, shutil, subprocess, time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

ROOT=Path(__file__).resolve().parents[1]
CFG=json.load(open(ROOT/'config/project.json'))
RUN=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2d_llmprint_heldout_confirmatory_v1'
RET=RUN/'return'; RUN.mkdir(parents=True,exist_ok=True); RET.mkdir(parents=True,exist_ok=True)
LOG=RUN/'run.log'
P2A=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2a_llmprint_threat_surface_v1'
OLD_PAIRS=P2A/'llmprint_128/pairs_128.pt'; OLD_ADV=P2A/'llmprint_128/adv_128'
TRAFFIC=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stage0_traffic_bootstrap_v1/traffic_frozen.jsonl'
FROZEN_THRESH=json.load(open(ROOT/'frozen/p2a_attack_thresholds_calibration.json'))
FROZEN_MET=json.load(open(ROOT/'frozen/p2a_detector_metrics_calibration.json'))
FROZEN_BANK=json.load(open(ROOT/'frozen/p2a_llmprint_bank_summary.json'))

def log(s):
    s=str(s); print(s,flush=True)
    with open(LOG,'a',encoding='utf-8') as f:f.write(s+'\n')
def dump(p,o):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=Path(str(p)+'.tmp'); q.write_text(json.dumps(o,indent=2,ensure_ascii=False)); os.replace(q,p)
def sh(cmd,timeout=None,env=None,cwd=None):
    t=time.time(); log('$ '+' '.join(map(str,cmd)))
    p=subprocess.run(list(map(str,cmd)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout,env=env,cwd=cwd)
    out=p.stdout or ''; log(out[-12000:])
    return {'cmd':list(map(str,cmd)),'rc':p.returncode,'seconds':round(time.time()-t,3),'out_tail':out[-12000:]}
def find_snapshot(glob):
    xs=[]
    for b in [Path.home()/'.cache/huggingface/hub',Path('/data/jx-vmlab/.cache/huggingface/hub')]:
        for p in b.glob(glob):
            for s in (p/'snapshots').glob('*'):
                if (s/'config.json').exists(): xs.append(s)
    return sorted(xs,key=lambda p:p.stat().st_mtime,reverse=True)[0] if xs else None

def load_jsonl(p):
    out=[]
    with open(p,encoding='utf-8') as f:
        for ln in f:
            if ln.strip():out.append(json.loads(ln))
    return out
def h64(x,s=''):return int.from_bytes(hashlib.sha256((s+'\0'+str(x)).encode()).digest()[:8],'big')
def sessions(split):
    by=collections.defaultdict(list)
    for r in load_jsonl(TRAFFIC):
        if r.get('split')!=split:continue
        t=str(r.get('text','')).strip()
        if t:by[str(r['conversation_id'])].append((int(r.get('turn_index',0)),t))
    out=[]
    for cid,z in by.items():
        ts=[t for _,t in sorted(z)]
        if ts:out.append({'id':cid,'texts':ts})
    out.sort(key=lambda x:h64(x['id'],'P2A_'+split)); return out
def sample_queries(ss,n,salt):
    a=[(t,s['id']) for s in ss for t in s['texts']];a.sort(key=lambda x:h64(x[0]+'|'+x[1],salt));return a[:min(n,len(a))]
def real_windows(ss,L,n,salt):
    a=[]
    for s in ss:
        for i in range(len(s['texts'])-L+1):a.append((s['texts'][i:i+L],s['id']))
    a.sort(key=lambda x:h64('|'.join(x[0])+'|'+x[1],salt));return a[:min(n,len(a))]
def make_pos(prompts,n,salt):
    return [[prompts[i] for i in sorted(range(len(prompts)),key=lambda i:h64(f'{e}|{i}',salt))] for e in range(n)]
def pos_windows(eps,L):return [(ep[i:i+L],f'p{e}') for e,ep in enumerate(eps) for i in range(len(ep)-L+1)]
def sample_items(items,n,salt):
    idx=sorted(range(len(items)),key=lambda i:h64(f'{i}|{items[i][1]}',salt))[:min(n,len(items))];return [items[i] for i in idx]

def load_pairs(path):
    import torch
    obj=torch.load(path,map_location='cpu')
    if isinstance(obj,dict):
        for k in ('pairs','data','items','entries','word_pairs'):
            if isinstance(obj.get(k),list):obj=obj[k];break
    out=[]
    for x in obj:
        if isinstance(x,dict):a=x.get('word_1') or x.get('w1') or x.get('word1');b=x.get('word_2') or x.get('w2') or x.get('word2')
        else:a,b=x[0],x[1]
        out.append((str(a),str(b)))
    return out

def flat_ids(v):
    if isinstance(v,dict):
        for k in ('suffix_ids','adv_suffix','ids','tokens'):
            if k in v:v=v[k];break
    if hasattr(v,'detach'):v=v.detach().cpu().reshape(-1).tolist()
    elif hasattr(v,'tolist'):v=np.asarray(v.tolist()).reshape(-1).tolist()
    else:v=np.asarray(v).reshape(-1).tolist()
    return [int(x) for x in v]

def prompt_bank(pairs_path,adv_dir,tok):
    import torch
    pairs=load_pairs(pairs_path);rows=[]
    for i,(a,b) in enumerate(pairs):
        fp=adv_dir/f'{a}_{b}.pt'
        if not fp.exists():continue
        ids=flat_ids(torch.load(fp,map_location='cpu'))
        suffix=tok.decode(ids,skip_special_tokens=False)
        rows.append({'index':i,'word1':a,'word2':b,'suffix_tokens':len(ids),'prompt':f'Randomly output a word from your vocabulary {suffix}'})
    return rows

class EC:
    def __init__(self,m):self.m=m;self.c={}
    def get(self,ts):
        miss=[];seen=set()
        for t in ts:
            if t not in self.c and t not in seen:seen.add(t);miss.append(t)
        if miss:
            E=self.m.encode(miss,batch_size=96,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            for t,e in zip(miss,E):self.c[t]=np.asarray(e,np.float32)
        return np.stack([self.c[t] for t in ts])
def seq_features(c,items,L):
    flat=[t for s,_ in items for t in s];E=c.get(flat);o=[];k=0
    for seq,_ in items:
        z=E[k:k+L];k+=L;l=np.asarray([len(x) for x in seq],np.float32)[:,None];p=np.arange(L,dtype=np.float32)[:,None]/max(1,L-1);o.append(np.concatenate([z,l/2048.,p],1))
    return np.stack(o).astype(np.float32) if o else np.empty((0,L,770),np.float32)
def make_seqnet(d,h,bidir):
    import torch
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__();self.r=torch.nn.GRU(d,h,batch_first=True,bidirectional=bidir);self.o=torch.nn.Linear(h*(2 if bidir else 1),1)
        def forward(self,x):z,_=self.r(x);return self.o(z[:,-1]).squeeze(-1)
    return M()
def fit_seq(c,pos,neg,L,h,bidir,epochs,bs,seed):
    import torch
    Xp=seq_features(c,pos,L);Xn=seq_features(c,neg,L);X=np.concatenate([Xp,Xn]);y=np.r_[np.ones(len(Xp)),np.zeros(len(Xn))].astype(np.float32)
    rng=np.random.default_rng(seed);torch.manual_seed(seed);dev='cuda' if torch.cuda.is_available() else 'cpu';m=make_seqnet(X.shape[-1],h,bidir).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4);lf=torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        order=rng.permutation(len(y))
        for st in range(0,len(y),bs):
            ii=order[st:st+bs];xb=torch.from_numpy(X[ii]).to(dev);yb=torch.from_numpy(y[ii]).to(dev);opt.zero_grad();loss=lf(m(xb),yb);loss.backward();opt.step()
    return m
def score_seq(m,c,items,L,bs=256):
    import torch
    X=seq_features(c,items,L);dev=next(m.parameters()).device;out=[];m.eval()
    with torch.inference_mode():
        for st in range(0,len(X),bs):out.append(torch.sigmoid(m(torch.from_numpy(X[st:st+bs]).to(dev))).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)

def metrics(pos,neg):
    pos=np.asarray(pos,float);neg=np.asarray(neg,float);y=np.r_[np.ones(len(pos)),np.zeros(len(neg))];s=np.r_[pos,neg]
    auc=float(roc_auc_score(y,s));fpr,tpr,thr=roc_curve(y,s)
    return {'AUROC':auc,'AUROC_sym':max(auc,1-auc),'BDR_at_AIR_0.8':float(np.min(fpr[tpr>=.8])) if np.any(tpr>=.8) else 1.0,'AIR_at_BDR_0.1':float(np.max(tpr[fpr<=.1])) if np.any(fpr<=.1) else 0.0,'n_audit':int(len(pos)),'n_benign':int(len(neg))}
def bootstrap_auc(pos,neg,reps):
    pos=np.asarray(pos,float);neg=np.asarray(neg,float);rng=np.random.default_rng(20261040);vals=[]
    for _ in range(reps):
        p=pos[rng.integers(0,len(pos),len(pos))];n=neg[rng.integers(0,len(neg),len(neg))]
        try:vals.append(float(roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n])))
        except Exception:pass
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))] if vals else [None,None]
def bootstrap_rate(x,fn,reps,seed):
    x=np.asarray(x);rng=np.random.default_rng(seed);vals=[]
    for _ in range(reps):
        z=x[rng.integers(0,len(x),len(x))];vals.append(float(fn(z)))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def main():
    log('=== SLATE StageP2D LLMPrint Held-out Confirmatory ===')
    py='/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python'
    repo=Path.home()/'SLATE_TDSC_SERVER_SHARED/public_baselines_sources/ACL-LLMPrint'
    qwen=find_snapshot('models--Qwen--Qwen3-4B-Base');mpnet=find_snapshot('models--sentence-transformers--all-mpnet-base-v2')
    if not (repo.exists() and qwen and mpnet and TRAFFIC.exists() and OLD_PAIRS.exists()):raise RuntimeError('required frozen asset missing')
    env=os.environ.copy();vendor=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep0_published_baselines_reproduction_v1_1/vendor';nano=Path.home()/'SLATE_TDSC_SERVER_SHARED/public_baselines_sources/nanoGCG'
    env['PYTHONPATH']=str(vendor)+':'+str(nano)+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    # 1) Build unseen prompt bank without touching held-out traffic.
    work=RUN/'llmprint_heldout64';work.mkdir(exist_ok=True);cand=work/'pairs_candidates_256.pt';pairs64=work/'pairs_heldout64.pt';adv=work/'adv_heldout64';adv.mkdir(exist_ok=True)
    if not cand.exists():
        rr=sh([py,str(repo/'generate_pairs.py'),'--base-model',str(qwen),'--num-pairs',str(CFG['candidate_pair_count']),'--seed',str(CFG['pair_seed']),'--output',str(cand)],cwd=str(repo),env=env,timeout=1800)
        if rr['rc']!=0:raise RuntimeError('heldout generate_pairs failed')
    if not pairs64.exists():
        import torch
        old=load_pairs(OLD_PAIRS);oldset={tuple(sorted((a.lower(),b.lower()))) for a,b in old};selected=[]
        for a,b in load_pairs(cand):
            k=tuple(sorted((a.lower(),b.lower())))
            if k in oldset:continue
            if k in {tuple(sorted((x.lower(),y.lower()))) for x,y in selected}:continue
            selected.append((a,b))
            if len(selected)==CFG['heldout_pair_count']:break
        if len(selected)<CFG['heldout_pair_count']:raise RuntimeError(f'only {len(selected)} novel pairs')
        torch.save(selected,pairs64)
    for st in range(0,CFG['heldout_pair_count'],CFG['gcg_chunk_pairs']):
        rr=sh([py,str(repo/'generate_adv_prompt.py'),'--base-model',str(qwen),'--pairs-path',str(pairs64),'--output-dir',str(adv),'--device','cuda:0','--dtype','float16','--num-steps',str(CFG['gcg_num_steps']),'--search-width',str(CFG['gcg_search_width']),'--topk',str(CFG['gcg_topk']),'--start-index',str(st),'--max-pairs',str(CFG['gcg_chunk_pairs']),'--verbosity','WARNING','--disable-prefix-cache'],cwd=str(repo),env=env,timeout=14400)
        if rr['rc']!=0:raise RuntimeError(f'heldout GCG chunk failed {st}')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(qwen),local_files_only=True);newbank=prompt_bank(pairs64,adv,tok)
    if len(newbank)!=64:raise RuntimeError(f'heldout bank incomplete {len(newbank)}/64')
    oldbank=prompt_bank(OLD_PAIRS,OLD_ADV,tok);oldprompts={x['prompt'] for x in oldbank};newprompts=[x['prompt'] for x in newbank]
    overlap=sum(p in oldprompts for p in newprompts)
    if overlap:raise RuntimeError(f'exact prompt overlap with P2A bank: {overlap}')
    oldpairs={tuple(sorted((a.lower(),b.lower()))) for a,b in load_pairs(OLD_PAIRS)};newpairs={tuple(sorted((a.lower(),b.lower()))) for a,b in load_pairs(pairs64)}
    if oldpairs & newpairs:raise RuntimeError('word-pair overlap with P2A bank')
    bankhash=hashlib.sha256('\n'.join(newprompts).encode()).hexdigest();dump(RET/'heldout_prompt_bank_summary.json',{'count':64,'pair_seed':CFG['pair_seed'],'pair_overlap_with_P2A':0,'prompt_overlap_with_P2A':0,'sha256':bankhash,'suffix_token_min':min(x['suffix_tokens'] for x in newbank),'suffix_token_max':max(x['suffix_tokens'] for x in newbank),'gcg_settings':{k:CFG[k] for k in ['gcg_num_steps','gcg_search_width','gcg_topk']},'heldout_traffic_accessed_at_this_point':False})
    # 2) Reconstruct frozen detectors from development; verify on calibration before held-out.
    trainprompts=[x['prompt'] for x in oldbank[:64]];evalprompts=[x['prompt'] for x in oldbank[64:128]]
    dev=sessions('development');cal=sessions('calibration')
    negt=[x[0] for x in sample_queries(dev,CFG['benign_train_query_cap'],'p2a_train_q')];negc=[x[0] for x in sample_queries(cal,CFG['benign_eval_query_cap'],'p2a_eval_q')]
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=100000,sublinear_tf=True,strip_accents='unicode');X=vec.fit_transform(trainprompts+negt);y=np.r_[np.ones(64),np.zeros(len(negt))]
    d1=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=20261004).fit(X,y)
    from sentence_transformers import SentenceTransformer
    import torch
    enc=SentenceTransformer(str(mpnet),device='cuda' if torch.cuda.is_available() else 'cpu');cache=EC(enc)
    d2=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=20261004).fit(np.r_[cache.get(trainprompts),cache.get(negt)],y)
    ptrain=make_pos(trainprompts,CFG['positive_train_sessions'],'P2A_POS_TRAIN');seq={}
    for L in CFG['history_lengths']:
        pi=sample_items(pos_windows(ptrain,L),CFG['benign_window_cap'],'ptr'+str(L));ni=real_windows(dev,L,CFG['benign_window_cap'],'ntr'+str(L))
        if L in [4,8]:seq[f'D3_L{L}']=fit_seq(cache,pi,ni,L,CFG['d3_hidden'],False,CFG['d3_epochs'],CFG['batch_size'],20261004+L)
        seq[f'D4_L{L}']=fit_seq(cache,pi,ni,L,CFG['d4_hidden'],True,CFG['d4_epochs'],CFG['batch_size'],20261104+L)
    rawcal={}
    rawcal['D1_L1']=(d1.predict_proba(vec.transform(evalprompts))[:,1],d1.predict_proba(vec.transform(negc))[:,1])
    rawcal['D2_L1']=(d2.predict_proba(cache.get(evalprompts))[:,1],d2.predict_proba(cache.get(negc))[:,1])
    pe=make_pos(evalprompts,CFG['positive_eval_sessions'],'P2A_POS_EVAL')
    for L in CFG['history_lengths']:
        pi=sample_items(pos_windows(pe,L),CFG['benign_window_cap'],'pev'+str(L));ni=real_windows(cal,L,CFG['benign_window_cap'],'nev'+str(L))
        for k in ([f'D4_L{L}']+([f'D3_L{L}'] if L in [4,8] else [])):
            rawcal[k]=(score_seq(seq[k],cache,pi,L),score_seq(seq[k],cache,ni,L))
    recon={'metrics':{},'threshold_checks':{}}
    for k,(sp,sn) in rawcal.items():recon['metrics'][k]=metrics(sp,sn)
    for beta in CFG['bdr_budgets']:
        b=str(beta);recon['threshold_checks'][b]={}
        for ksrc,kraw in [('D1','D1_L1'),('D2','D2_L1'),('D3','D3_L4'),('D4','D4_L4')]:
            th=float(FROZEN_THRESH[b][ksrc]['threshold']);sn=rawcal[kraw][1];m=float(np.mean(sn>=th));f=float(FROZEN_THRESH[b][ksrc]['measured_BDR']);recon['threshold_checks'][b][ksrc]={'reconstructed_BDR':m,'frozen_BDR':f,'abs_diff':abs(m-f)}
            if abs(m-f)>CFG['calibration_bdr_tolerance']:raise RuntimeError(f'calibration detector reconstruction drift {ksrc} beta={b}: {m} vs {f}')
    dump(RET/'detector_reconstruction_calibration.json',recon)
    freeze={'workflow':CFG['workflow'],'new_prompt_bank_sha256':bankhash,'new_prompt_count':64,'pair_overlap':0,'prompt_overlap':0,'detectors_reconstructed_from':'development only','calibration_checks_passed':True,'frozen_strongest_by_L':CFG['frozen_strongest_by_L'],'frozen_threshold_source':'P2A calibration','heldout_touched':False,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    dump(RET/'preheldout_freeze.json',freeze)
    # 3) First held-out access occurs here.
    held=sessions('heldout')
    if not held:raise RuntimeError('heldout traffic empty')
    negh=[x[0] for x in sample_queries(held,CFG['benign_eval_query_cap'],'p2d_heldout_q')]
    raw={};raw['D1_L1']=(d1.predict_proba(vec.transform(newprompts))[:,1],d1.predict_proba(vec.transform(negh))[:,1]);raw['D2_L1']=(d2.predict_proba(cache.get(newprompts))[:,1],d2.predict_proba(cache.get(negh))[:,1])
    ph=make_pos(newprompts,CFG['positive_eval_sessions'],'P2D_POS_HELDOUT')
    for L in CFG['history_lengths']:
        pi=sample_items(pos_windows(ph,L),CFG['benign_window_cap'],'p2d_ph'+str(L));ni=real_windows(held,L,CFG['benign_window_cap'],'p2d_nh'+str(L))
        for k in ([f'D4_L{L}']+([f'D3_L{L}'] if L in [4,8] else [])):
            raw[k]=(score_seq(seq[k],cache,pi,L),score_seq(seq[k],cache,ni,L))
    out={}
    for i,(k,(sp,sn)) in enumerate(raw.items()):
        out[k]=metrics(sp,sn);out[k]['AUROC_CI95']=bootstrap_auc(sp,sn,CFG['bootstrap_reps'])
    strongest={}
    for L in CFG['history_lengths']:
        k=CFG['frozen_strongest_by_L'][str(L)]; strongest[str(L)]={'detector':k,**out[k]}
    out['strongest_by_L_frozen']=strongest;dump(RET/'detector_metrics_heldout.json',out)
    transfer={}
    for beta in CFG['bdr_budgets']:
        b=str(beta);transfer[b]={}
        for j,(ksrc,kraw) in enumerate([('D1','D1_L1'),('D2','D2_L1'),('D3','D3_L4'),('D4','D4_L4')]):
            sp,sn=raw[kraw];th=float(FROZEN_THRESH[b][ksrc]['threshold']);air=float(np.mean(sp>=th));bdr=float(np.mean(sn>=th))
            transfer[b][ksrc]={'threshold':th,'calibration_BDR_frozen':float(FROZEN_THRESH[b][ksrc]['measured_BDR']),'heldout_BDR':bdr,'unseen_prompt_AIR':air,'AIR_CI95':bootstrap_rate(sp,lambda z:np.mean(z>=th),CFG['bootstrap_reps'],20261100+j),'BDR_CI95':bootstrap_rate(sn,lambda z:np.mean(z>=th),CFG['bootstrap_reps'],20261200+j)}
    dump(RET/'fixed_threshold_transfer_heldout.json',transfer)
    summary={'workflow':CFG['workflow'],'status':'STAGEP2D_COMPLETE','paper_facing_confirmatory':True,'heldout_touched':True,'unseen_prompt_bank_sha256':bankhash,'frozen_strongest_by_L':strongest,'fixed_threshold_transfer':transfer,'no_retuning_after_heldout':True,'interpretation_rule':'Report held-out results as observed. No prompt/detector/threshold rescue is permitted after this run.'}
    dump(RET/'summary.json',summary);dump(RET/'status.json',{'workflow':CFG['workflow'],'status':'STAGEP2D_COMPLETE','rc':0});log(json.dumps(summary,indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(RET/'status.json',{'workflow':CFG['workflow'],'status':'STAGEP2D_FAILED','rc':1,'error':repr(e)});log('FATAL '+repr(e));raise
