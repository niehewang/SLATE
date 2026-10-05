from __future__ import annotations
import collections, gc, hashlib, json, math, os, subprocess, time, urllib.request
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

ROOT=Path(__file__).resolve().parents[1]
CFG=json.load(open(ROOT/'config/project.json'))
RUN=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2a_llmprint_threat_surface_v1'
RET=RUN/'return_v1_3'
RUN.mkdir(parents=True,exist_ok=True); RET.mkdir(parents=True,exist_ok=True)
LOG=RUN/'run_v1_3.log'
PHASE=RET/'phase_status.json'
PYTHON='/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python'


def log(s):
    s=str(s); print(s,flush=True)
    with open(LOG,'a',encoding='utf-8') as f: f.write(s+'\n')

def jdump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(path)+'.tmp')
    with open(tmp,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False)
    os.replace(tmp,path)

def phase(name,status='ok',**extra):
    cur={}
    if PHASE.exists():
        try: cur=json.load(open(PHASE))
        except Exception: cur={}
    cur[name]={'status':status,'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**extra}
    jdump(PHASE,cur)
    log(f'PHASE {name}: {status} {extra}')

def sh(cmd,timeout=None,env=None,cwd=None):
    t=time.time(); log('$ '+' '.join(map(str,cmd)))
    p=subprocess.run(list(map(str,cmd)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout,env=env,cwd=cwd)
    out=p.stdout or ''; log(out[-10000:])
    return {'cmd':list(map(str,cmd)),'rc':p.returncode,'seconds':round(time.time()-t,3),'out_tail':out[-12000:]}

def find_snapshot(root_glob):
    xs=[]
    for base in [Path.home()/'.cache/huggingface/hub', Path('/data/jx-vmlab/.cache/huggingface/hub')]:
        for p in base.glob(root_glob):
            for s in (p/'snapshots').glob('*'):
                if (s/'config.json').exists(): xs.append(s)
    return sorted(xs,key=lambda p:p.stat().st_mtime,reverse=True)[0] if xs else None

def load_jsonl(p):
    out=[]
    with open(p,encoding='utf-8') as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out

def h64(x,salt=''):
    return int.from_bytes(hashlib.sha256((salt+'\0'+str(x)).encode()).digest()[:8],'big')

def real_sessions(traffic,split):
    by=collections.defaultdict(list)
    for r in load_jsonl(traffic):
        if r.get('split')!=split: continue
        txt=str(r.get('text','')).strip()
        if txt: by[str(r['conversation_id'])].append((int(r.get('turn_index',0)),txt))
    out=[]
    for cid,z in by.items():
        texts=[t for _,t in sorted(z)]
        if texts: out.append({'id':cid,'texts':texts})
    out.sort(key=lambda e:h64(e['id'],'P2A_'+split))
    return out

def sample_queries(sessions,n,salt):
    arr=[(t,s['id']) for s in sessions for t in s['texts']]
    arr.sort(key=lambda x:h64(x[0]+'|'+x[1],salt))
    return arr[:min(n,len(arr))]

def real_windows(sessions,L,n,salt):
    arr=[]
    for s in sessions:
        texts=s['texts']
        for i in range(len(texts)-L+1): arr.append((texts[i:i+L],s['id']))
    arr.sort(key=lambda x:h64('|'.join(x[0])+'|'+x[1],salt))
    return arr[:min(n,len(arr))]

def make_positive_sessions(prompts,n,salt):
    out=[]
    for e in range(n):
        order=sorted(range(len(prompts)),key=lambda i:h64(f'{e}|{i}',salt))
        out.append([prompts[i] for i in order])
    return out

def pos_windows(eps,L):
    return [(ep[i:i+L],f'p{e}') for e,ep in enumerate(eps) for i in range(len(ep)-L+1)]

def sample_items(items,n,salt):
    idx=sorted(range(len(items)),key=lambda i:h64(f'{i}|{items[i][1]}',salt))[:min(n,len(items))]
    return [items[i] for i in idx]

class EC:
    def __init__(self,model): self.model=model; self.c={}
    def get(self,texts):
        miss=[];seen=set()
        for t in texts:
            if t not in self.c and t not in seen: seen.add(t);miss.append(t)
        if miss:
            E=self.model.encode(miss,batch_size=96,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            for t,e in zip(miss,E): self.c[t]=np.asarray(e,np.float32)
        return np.stack([self.c[t] for t in texts])

def seq_features(cache,items,L):
    flat=[t for seq,_ in items for t in seq]
    E=cache.get(flat); out=[]; k=0
    for seq,_ in items:
        z=E[k:k+L];k+=L
        lens=np.asarray([len(x) for x in seq],np.float32)[:,None]
        pos=np.arange(L,dtype=np.float32)[:,None]/max(1,L-1)
        out.append(np.concatenate([z,lens/2048.,pos],1))
    return np.stack(out).astype(np.float32) if out else np.empty((0,L,770),np.float32)

def make_seqnet(d,h,bidir):
    import torch
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.r=torch.nn.GRU(d,h,batch_first=True,bidirectional=bidir); self.o=torch.nn.Linear(h*(2 if bidir else 1),1)
        def forward(self,x): z,_=self.r(x); return self.o(z[:,-1]).squeeze(-1)
    return M()

def fit_seq(cache,pos,neg,L,h,bidir,epochs,bs,seed):
    import torch
    Xp=seq_features(cache,pos,L); Xn=seq_features(cache,neg,L)
    X=np.concatenate([Xp,Xn]); y=np.r_[np.ones(len(Xp)),np.zeros(len(Xn))].astype(np.float32)
    del Xp,Xn; gc.collect()
    rng=np.random.default_rng(seed); torch.manual_seed(seed); dev='cuda' if torch.cuda.is_available() else 'cpu'
    m=make_seqnet(X.shape[-1],h,bidir).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4); lossfn=torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        order=rng.permutation(len(y))
        for st in range(0,len(y),bs):
            ii=order[st:st+bs]; xb=torch.from_numpy(X[ii]).to(dev); yb=torch.from_numpy(y[ii]).to(dev)
            opt.zero_grad(set_to_none=True); loss=lossfn(m(xb),yb); loss.backward(); opt.step()
    del X,y; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return m

def score_seq(m,cache,items,L,bs=256):
    import torch
    X=seq_features(cache,items,L); dev=next(m.parameters()).device; out=[];m.eval()
    with torch.inference_mode():
        for st in range(0,len(X),bs): out.append(torch.sigmoid(m(torch.from_numpy(X[st:st+bs]).to(dev))).cpu().numpy())
    del X; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return np.concatenate(out) if out else np.empty(0)

def metrics(pos,neg):
    pos=np.asarray(pos,float);neg=np.asarray(neg,float); y=np.r_[np.ones(len(pos)),np.zeros(len(neg))];s=np.r_[pos,neg]
    auc=float(roc_auc_score(y,s));fpr,tpr,thr=roc_curve(y,s);gap=float(np.max(np.abs(tpr-fpr)))
    b80=float(np.min(fpr[tpr>=.8])) if np.any(tpr>=.8) else 1.; a10=float(np.max(tpr[fpr<=.1])) if np.any(fpr<=.1) else 0.
    return {'AUROC':auc,'AUROC_sym':max(auc,1-auc),'youden_abs':gap,'BDR_at_AIR_0.8':b80,'AIR_at_BDR_0.1':a10,'n_audit':int(len(pos)),'n_benign':int(len(neg))}

def threshold_for_bdr(neg,beta):
    x=np.sort(np.asarray(neg,float))[::-1];n=len(x);k=int(math.floor(beta*n))
    if k<=0: th=float(np.nextafter(x[0],np.inf))
    elif k>=n: th=-float('inf')
    else: th=float(np.nextafter(x[k],np.inf))
    return th,float(np.mean(np.asarray(neg)>=th))

def load_pairs(p):
    import torch
    obj=torch.load(p,map_location='cpu')
    if isinstance(obj,dict):
        for k in ('pairs','data','items','entries','word_pairs'):
            if k in obj and isinstance(obj[k],list): obj=obj[k];break
    out=[]
    for e in obj:
        if isinstance(e,dict): a=e.get('word_1') or e.get('w1') or e.get('word1'); b=e.get('word_2') or e.get('w2') or e.get('word2')
        else: a,b=e[0],e[1]
        out.append((str(a),str(b)))
    return out

def decode_prompt_bank(pairs_path,adv_dir,tokenizer):
    import torch
    pairs=load_pairs(pairs_path); out=[]
    for i,(a,b) in enumerate(pairs):
        fp=adv_dir/f'{a}_{b}.pt'
        if not fp.exists(): continue
        ids=torch.load(fp,map_location='cpu')
        if isinstance(ids,dict):
            for k in ('suffix_ids','adv_suffix','ids','tokens'):
                if k in ids: ids=ids[k]; break
        # Official LLMPrint stores result.best_ids, which can retain a singleton
        # batch dimension (e.g. shape [1, n]).  HF tokenizer.decode expects a
        # flat sequence of token ids.  Flattening here is serialization-only
        # compatibility; it does not alter any generated token or scientific setting.
        shape_before=None
        try:
            if hasattr(ids,'shape'): shape_before=list(ids.shape)
        except Exception:
            shape_before=None
        if hasattr(ids,'detach'):
            ids=ids.detach().cpu().reshape(-1).tolist()
        elif hasattr(ids,'tolist'):
            ids=np.asarray(ids.tolist()).reshape(-1).tolist()
        else:
            ids=np.asarray(ids).reshape(-1).tolist()
        ids=[int(x) for x in ids]
        suffix=tokenizer.decode(ids,skip_special_tokens=True)
        out.append({'index':i,'word_1':a,'word_2':b,'suffix_file':str(fp),'suffix_text':suffix,'prompt':f'Randomly output a word from your vocabulary {suffix}',
                    'suffix_token_count':len(ids),'stored_shape':shape_before})
    return out

def probe_url(url):
    t=time.time()
    try:
        req=urllib.request.Request(url,method='HEAD',headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=12) as r: return {'reachable':True,'status':getattr(r,'status',None),'seconds':round(time.time()-t,3)}
    except Exception as e: return {'reachable':False,'error':repr(e),'seconds':round(time.time()-t,3)}

def bootstrap_family_b(py,target):
    target=Path(target); target.mkdir(parents=True,exist_ok=True)
    st=os.statvfs(target);free=st.f_bavail*st.f_frsize/1024**3
    probes={u:probe_url(u) for u in ['https://hf-mirror.com','https://modelscope.cn']}
    ret={'repo_id':CFG['family_b_repo'],'target':str(target),'free_gb':free,'probes':probes,'attempted':False}
    if (target/'config.json').exists() and (target/'tokenizer_config.json').exists(): ret.update(status='already_present'); return ret
    if os.environ.get('SLATE_SKIP_FAMILY_B_DOWNLOAD','0')=='1': ret.update(status='skipped_by_env');return ret
    if free<CFG['family_b_min_free_gb']: ret.update(status='insufficient_disk');return ret
    if not probes['https://hf-mirror.com']['reachable']: ret.update(status='no_verified_mirror');return ret
    code='''from huggingface_hub import snapshot_download\nimport sys\nrepo=sys.argv[1]; dest=sys.argv[2]\nprint(snapshot_download(repo_id=repo,local_dir=dest,endpoint="https://hf-mirror.com",resume_download=True))\n'''
    ret['attempted']=True; denv=os.environ.copy(); denv['HF_HUB_OFFLINE']='0'; denv['TRANSFORMERS_OFFLINE']='0'
    rr=sh([py,'-c',code,CFG['family_b_repo'],str(target)],timeout=10800,env=denv); ret['download']=rr
    ret['status']='download_ok' if rr['rc']==0 and (target/'config.json').exists() else 'download_failed'
    if ret['status']=='download_ok':
        smoke='''from transformers import AutoTokenizer,AutoModelForCausalLM\nimport sys,torch\np=sys.argv[1]\nt=AutoTokenizer.from_pretrained(p,trust_remote_code=True,local_files_only=True)\nm=AutoModelForCausalLM.from_pretrained(p,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float16,attn_implementation="eager").to("cuda:0")\nx=t("Hello",return_tensors="pt").to("cuda:0"); y=m.generate(**x,max_new_tokens=4,do_sample=False); print(t.decode(y[0]))\n'''
        ret['smoke']=sh([py,'-c',smoke,str(target)],timeout=1800,env=os.environ.copy())
        if ret['smoke']['rc']!=0: ret['status']='downloaded_but_smoke_failed'
    return ret

def main():
    log('=== SLATE StageP2A v1.3 SUFFIX-DECODE HOTFIX / ANALYSIS-ONLY RESUME ==='); log(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    pairs=RUN/'llmprint_128/pairs_128.pt'; adv=RUN/'llmprint_128/adv_128'
    qwen=find_snapshot('models--Qwen--Qwen3-4B-Base'); mpnet=find_snapshot('models--sentence-transformers--all-mpnet-base-v2')
    traffic=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stage0_traffic_bootstrap_v1/traffic_frozen.jsonl'
    assets={'python':PYTHON,'pairs':str(pairs),'adv':str(adv),'qwen_tokenizer':str(qwen) if qwen else None,'mpnet':str(mpnet) if mpnet else None,'traffic':str(traffic)}
    assets['exists']={k:(Path(v).exists() if v else False) for k,v in assets.items() if k!='python'}
    jdump(RET/'asset_inventory.json',assets)
    if not all(assets['exists'].values()): raise RuntimeError('required frozen asset missing: '+json.dumps(assets))
    suffix_files=list(adv.glob('*.pt')); phase('frozen_gcg_verify','ok',suffix_file_count=len(suffix_files))
    if len(suffix_files)!=128: raise RuntimeError(f'expected exactly 128 GCG suffix files, found {len(suffix_files)}')

    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(qwen),local_files_only=True)
    bank=decode_prompt_bank(pairs,adv,tok); bank.sort(key=lambda x:x['index'])
    if len(bank)!=128: raise RuntimeError(f'decoded bank size {len(bank)} != 128')
    jdump(RET/'suffix_decode_diagnostics.json', {
        'count':len(bank),
        'stored_shapes':dict(collections.Counter(str(x.get('stored_shape')) for x in bank)),
        'token_count_min':min(x['suffix_token_count'] for x in bank),
        'token_count_max':max(x['suffix_token_count'] for x in bank),
        'token_count_mean':float(np.mean([x['suffix_token_count'] for x in bank])),
        'serialization_fix_only':True
    })
    train=[x['prompt'] for x in bank[:64]]; ev=[x['prompt'] for x in bank[64:128]]
    bank_summary={'count':128,'train_indices':[0,63],'eval_indices':[64,127],
                  'train_sha256':hashlib.sha256('\n'.join(train).encode()).hexdigest(),
                  'eval_sha256':hashlib.sha256('\n'.join(ev).encode()).hexdigest(),
                  'server_pairs':str(pairs),'server_adv':str(adv),
                  'gcg_settings':{'num_steps':100,'search_width':64,'topk':64,'disable_prefix_cache':True}}
    jdump(RET/'llmprint_bank_summary.json',bank_summary); phase('prompt_bank_decode','ok')

    dev=real_sessions(traffic,'development'); cal=real_sessions(traffic,'calibration')
    if not dev or not cal: raise RuntimeError('development/calibration traffic split load failed')
    jdump(RET/'traffic_counts.json',{'development_sessions':len(dev),'calibration_sessions':len(cal),'heldout_touched':False})
    phase('traffic_load','ok',development_sessions=len(dev),calibration_sessions=len(cal),heldout_touched=False)

    from sentence_transformers import SentenceTransformer
    import torch
    enc=SentenceTransformer(str(mpnet),device='cuda' if torch.cuda.is_available() else 'cpu')
    cache=EC(enc); phase('mpnet_load','ok',device=str(enc.device))

    nq=sample_queries(dev,CFG['benign_train_query_cap'],'p2a_train_q'); negt=[x[0] for x in nq]
    nq2=sample_queries(cal,CFG['benign_eval_query_cap'],'p2a_eval_q'); neg2=[x[0] for x in nq2]
    met={}; raw={}

    vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=100000,sublinear_tf=True,strip_accents='unicode')
    X=vec.fit_transform(train+negt); y=np.r_[np.ones(len(train)),np.zeros(len(negt))]
    d1=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=20261004).fit(X,y)
    s1p=d1.predict_proba(vec.transform(ev))[:,1]; s1n=d1.predict_proba(vec.transform(neg2))[:,1]
    met['D1_L1']=metrics(s1p,s1n); raw['D1']=(s1p,s1n); jdump(RET/'detector_metrics_partial.json',met); phase('D1_L1','ok',**met['D1_L1'])
    del X,y,d1; gc.collect()

    Etr_pos=cache.get(train); Etr_neg=cache.get(negt)
    d2=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=20261004).fit(np.concatenate([Etr_pos,Etr_neg]),np.r_[np.ones(len(train)),np.zeros(len(negt))])
    del Etr_pos,Etr_neg; gc.collect()
    s2p=d2.predict_proba(cache.get(ev))[:,1]; s2n=d2.predict_proba(cache.get(neg2))[:,1]
    met['D2_L1']=metrics(s2p,s2n); raw['D2']=(s2p,s2n); jdump(RET/'detector_metrics_partial.json',met); phase('D2_L1','ok',**met['D2_L1'])
    del d2; gc.collect()

    ptrain=make_positive_sessions(train,CFG['positive_train_sessions'],'P2A_POS_TRAIN')
    peval=make_positive_sessions(ev,CFG['positive_eval_sessions'],'P2A_POS_EVAL')
    seq_models={}
    plan=[(1,'D4',CFG['d4_hidden'],True,CFG['d4_epochs'],20261105),
          (4,'D3',CFG['d3_hidden'],False,CFG['d3_epochs'],20261008),
          (4,'D4',CFG['d4_hidden'],True,CFG['d4_epochs'],20261108),
          (8,'D3',CFG['d3_hidden'],False,CFG['d3_epochs'],20261012),
          (8,'D4',CFG['d4_hidden'],True,CFG['d4_epochs'],20261112)]
    for L,kind,h,bidir,epochs,seed in plan:
        key=f'{kind}_L{L}'
        pi_tr=sample_items(pos_windows(ptrain,L),CFG['benign_window_cap'],'ptr'+str(L)); ni_tr=real_windows(dev,L,CFG['benign_window_cap'],'ntr'+str(L))
        m=fit_seq(cache,pi_tr,ni_tr,L,h,bidir,epochs,CFG['batch_size'],seed)
        pi_ev=sample_items(pos_windows(peval,L),CFG['benign_window_cap'],'pev'+str(L)); ni_ev=real_windows(cal,L,CFG['benign_window_cap'],'nev'+str(L))
        sp=score_seq(m,cache,pi_ev,L); sn=score_seq(m,cache,ni_ev,L)
        met[key]=metrics(sp,sn); raw[kind if L==4 else key]=(sp,sn)
        jdump(RET/'detector_metrics_partial.json',met); phase(key,'ok',**met[key])
        del m,pi_tr,ni_tr,pi_ev,ni_ev; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    strongest={}
    for L,ks in [(1,['D1_L1','D2_L1','D4_L1']),(4,['D3_L4','D4_L4']),(8,['D3_L8','D4_L8'])]:
        k=max(ks,key=lambda x:met[x]['AUROC_sym']); strongest[str(L)]={'detector':k,**met[k]}
    met['strongest_by_L']=strongest
    jdump(RET/'detector_metrics_calibration.json',met); phase('detector_metrics_finalize','ok')

    thout={}
    scoremap={'D1':raw['D1'],'D2':raw['D2'],'D3':raw['D3'],'D4':raw['D4']}
    for beta in CFG['bdr_budgets']:
        thout[str(beta)]={}
        for k,(sp,sn) in scoremap.items():
            th,b=threshold_for_bdr(sn,beta); thout[str(beta)][k]={'threshold':th,'measured_BDR':b,'AIR':float(np.mean(np.asarray(sp)>=th))}
    jdump(RET/'attack_thresholds_calibration.json',thout); phase('attack_thresholds','ok')

    summary={'workflow':CFG['workflow'],'status':'STAGEP2A_V1_3_METRICS_COMPLETE','paper_facing_confirmatory':False,'heldout_touched':False,
             'llmprint_native_bank_frozen':True,'prompt_count':128,'strongest_by_L':strongest,
             'beta_0.05':thout.get('0.05'),'beta_0.1':thout.get('0.1'),
             'family_b_status':'pending','next_rule':'Freeze this published LLMPrint bank and detector development results; next measure A1/A2 verification degradation without retuning on held-out.'}
    jdump(RET/'summary.json',summary); jdump(RET/'status.json',{'workflow':CFG['workflow'],'status':'STAGEP2A_V1_3_METRICS_COMPLETE','rc':0})
    phase('metrics_complete','ok')

    # Family-B is intentionally last so network/download problems cannot erase threat-surface evidence.
    try:
        family=bootstrap_family_b(PYTHON,CFG['family_b_target'])
    except Exception as e:
        family={'status':'bootstrap_exception','error':repr(e)}
    jdump(RET/'family_b_bootstrap.json',family)
    summary['family_b_status']=family.get('status'); summary['status']='STAGEP2A_V1_2_COMPLETE'
    jdump(RET/'summary.json',summary); jdump(RET/'status.json',{'workflow':CFG['workflow'],'status':'STAGEP2A_V1_3_COMPLETE','rc':0})
    phase('family_b_bootstrap','ok',family_b_status=family.get('status'))
    log(json.dumps(summary,indent=2))

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        phase('fatal','failed',error=repr(e))
        jdump(RET/'status.json',{'workflow':CFG['workflow'],'status':'STAGEP2A_V1_3_FAILED','rc':1,'error':repr(e)})
        raise
