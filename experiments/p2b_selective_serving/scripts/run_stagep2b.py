from __future__ import annotations
import collections, gc, hashlib, json, math, os, random, shutil, subprocess, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CFG=json.load(open(ROOT/'config/project.json'))
RUN=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2b_llmprint_selective_serving_degradation_v1'
RET=RUN/'return'
RUN.mkdir(parents=True,exist_ok=True); RET.mkdir(parents=True,exist_ok=True)
LOG=RUN/'run.log'; PY='/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python'
P2A=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2a_llmprint_threat_surface_v1'
PAIRS=P2A/'llmprint_128/pairs_128.pt'; ADV=P2A/'llmprint_128/adv_128'
TRAFFIC=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stage0_traffic_bootstrap_v1/traffic_frozen.jsonl'
PUBLIC=Path('/data/jx-vmlab/SLATE_TDSC_SERVER_SHARED/public_models'); PUBLIC.mkdir(parents=True,exist_ok=True)
P2A_THRESH=json.load(open(ROOT/'frozen/p2a_attack_thresholds_calibration.json'))
P2A_MET=json.load(open(ROOT/'frozen/p2a_detector_metrics_calibration.json'))
P2A_BANK=json.load(open(ROOT/'frozen/p2a_llmprint_bank_summary.json'))

def log(s):
    s=str(s); print(s,flush=True)
    with open(LOG,'a',encoding='utf-8') as f:f.write(s+'\n')
def dump(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=Path(str(p)+'.tmp');q.write_text(json.dumps(o,indent=2,ensure_ascii=False));os.replace(q,p)
def find_snapshot(glob):
    xs=[]
    for b in [Path.home()/'.cache/huggingface/hub',Path('/data/jx-vmlab/.cache/huggingface/hub')]:
        for p in b.glob(glob):
            for s in (p/'snapshots').glob('*'):
                if (s/'config.json').exists():xs.append(s)
    return sorted(xs,key=lambda p:p.stat().st_mtime,reverse=True)[0] if xs else None

def load_pairs():
    import torch
    obj=torch.load(PAIRS,map_location='cpu')
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

def frozen_eval_bank(base_tok):
    import torch
    pairs=load_pairs(); rows=[]
    for i in range(CFG['eval_start'],CFG['eval_start']+CFG['eval_count']):
        a,b=pairs[i];fp=ADV/f'{a}_{b}.pt'
        ids=flat_ids(torch.load(fp,map_location='cpu'));suffix=base_tok.decode(ids,skip_special_tokens=False)
        rows.append({'global_index':i,'word1':a,'word2':b,'prompt':f'Randomly output a word from your vocabulary {suffix}'})
    h=hashlib.sha256('\n'.join(x['prompt'] for x in rows).encode()).hexdigest()
    if h!=P2A_BANK['eval_sha256']:raise RuntimeError(f'eval bank hash mismatch {h} != {P2A_BANK["eval_sha256"]}')
    return rows,h

def load_jsonl(p):
    out=[]
    with open(p,encoding='utf-8') as f:
        for ln in f:
            if ln.strip():out.append(json.loads(ln))
    return out

def h64(x,s=''):
    return int.from_bytes(hashlib.sha256((s+'\0'+str(x)).encode()).digest()[:8],'big')
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
    out.sort(key=lambda x:h64(x['id'],'P2A_'+split));return out

def sample_queries(ss,n,salt):
    a=[(t,s['id']) for s in ss for t in s['texts']];a.sort(key=lambda x:h64(x[0]+'|'+x[1],salt));return a[:min(n,len(a))]
def real_windows(ss,L,n,salt):
    a=[]
    for s in ss:
        for i in range(len(s['texts'])-L+1):a.append((s['texts'][i:i+L],s['id']))
    a.sort(key=lambda x:h64('|'.join(x[0])+'|'+x[1],salt));return a[:min(n,len(a))]
def make_pos(prompts,n,salt):
    out=[]
    for e in range(n):out.append([prompts[i] for i in sorted(range(len(prompts)),key=lambda i:h64(f'{e}|{i}',salt))])
    return out
def pos_windows(eps,L):return [(ep[i:i+L],f'p{e}') for e,ep in enumerate(eps) for i in range(len(ep)-L+1)]
def sample_items(items,n,salt):
    idx=sorted(range(len(items)),key=lambda i:h64(f'{i}|{items[i][1]}',salt))[:min(n,len(items))];return [items[i] for i in idx]

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
    return np.stack(o).astype(np.float32)
def seqnet(d,h=64):
    import torch
    class M(torch.nn.Module):
        def __init__(self):super().__init__();self.r=torch.nn.GRU(d,h,batch_first=True);self.o=torch.nn.Linear(h,1)
        def forward(self,x):z,_=self.r(x);return self.o(z[:,-1]).squeeze(-1)
    return M()
def fit_d3(c,pos,neg):
    import torch
    L=4;Xp=seq_features(c,pos,L);Xn=seq_features(c,neg,L);X=np.concatenate([Xp,Xn]);y=np.r_[np.ones(len(Xp)),np.zeros(len(Xn))].astype(np.float32)
    rng=np.random.default_rng(20261008);torch.manual_seed(20261008);dev='cuda' if torch.cuda.is_available() else 'cpu';m=seqnet(X.shape[-1],64).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4);lf=torch.nn.BCEWithLogitsLoss()
    for _ in range(8):
        order=rng.permutation(len(y))
        for st in range(0,len(y),128):
            ii=order[st:st+128];xb=torch.from_numpy(X[ii]).to(dev);yb=torch.from_numpy(y[ii]).to(dev);opt.zero_grad(set_to_none=True);loss=lf(m(xb),yb);loss.backward();opt.step()
    return m
def score_d3(m,c,seqs):
    import torch
    items=[(s,str(i)) for i,s in enumerate(seqs)];X=seq_features(c,items,4);dev=next(m.parameters()).device;m.eval();out=[]
    with torch.inference_mode():
        for st in range(0,len(X),256):out.append(torch.sigmoid(m(torch.from_numpy(X[st:st+256]).to(dev))).cpu().numpy())
    return np.concatenate(out)

def reconstruct_detectors(eval_prompts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sentence_transformers import SentenceTransformer
    import torch
    mp=find_snapshot('models--sentence-transformers--all-mpnet-base-v2');
    if not mp:raise RuntimeError('MPNet missing')
    dev=sessions('development');cal=sessions('calibration')
    nq=sample_queries(dev,12000,'p2a_train_q');negt=[x[0] for x in nq];nq2=sample_queries(cal,12000,'p2a_eval_q');neg2=[x[0] for x in nq2]
    # Reconstruct D1 from frozen P2A bank: train indices 0-63.
    base_tok=__import__('transformers').AutoTokenizer.from_pretrained(str(find_snapshot('models--Qwen--Qwen3-4B-Base')),local_files_only=True)
    full,_=frozen_eval_bank(base_tok) # eval only; build train below separately
    pairs=load_pairs();import torch as T
    train=[]
    for i in range(64):
        a,b=pairs[i];ids=flat_ids(T.load(ADV/f'{a}_{b}.pt',map_location='cpu'));s=base_tok.decode(ids,skip_special_tokens=False);train.append(f'Randomly output a word from your vocabulary {s}')
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=100000,sublinear_tf=True,strip_accents='unicode');X=vec.fit_transform(train+negt);y=np.r_[np.ones(64),np.zeros(len(negt))];d1=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=20261004).fit(X,y)
    d1p=d1.predict_proba(vec.transform(eval_prompts))[:,1];d1n=d1.predict_proba(vec.transform(neg2))[:,1]
    enc=SentenceTransformer(str(mp),device='cuda' if torch.cuda.is_available() else 'cpu');cache=EC(enc)
    ptrain=make_pos(train,120,'P2A_POS_TRAIN');pi_tr=sample_items(pos_windows(ptrain,4),7000,'ptr4');ni_tr=real_windows(dev,4,7000,'ntr4');d3=fit_d3(cache,pi_tr,ni_tr)
    # verify calibration separation and frozen thresholds measured BDR within 0.02 absolute
    peval=make_pos(eval_prompts,120,'P2A_POS_EVAL');pi_ev=sample_items(pos_windows(peval,4),7000,'pev4');ni_ev=real_windows(cal,4,7000,'nev4');d3p=score_d3(d3,cache,[x[0] for x in pi_ev]);d3n=score_d3(d3,cache,[x[0] for x in ni_ev])
    checks={'D1_AUROC':float(roc_auc_score(np.r_[np.ones(len(d1p)),np.zeros(len(d1n))],np.r_[d1p,d1n])),'D3_L4_AUROC':float(roc_auc_score(np.r_[np.ones(len(d3p)),np.zeros(len(d3n))],np.r_[d3p,d3n])),'frozen_threshold_checks':{}}
    for b in CFG['betas']:
        k=str(b);td1=P2A_THRESH[k]['D1']['threshold'];td3=P2A_THRESH[k]['D3']['threshold'];bd1=float(np.mean(d1n>=td1));bd3=float(np.mean(d3n>=td3));checks['frozen_threshold_checks'][k]={'D1_BDR_reconstructed':bd1,'D1_BDR_frozen':P2A_THRESH[k]['D1']['measured_BDR'],'D3_BDR_reconstructed':bd3,'D3_BDR_frozen':P2A_THRESH[k]['D3']['measured_BDR']}
        if abs(bd1-P2A_THRESH[k]['D1']['measured_BDR'])>.02 or abs(bd3-P2A_THRESH[k]['D3']['measured_BDR'])>.02:raise RuntimeError('detector reconstruction drift >0.02 at beta '+k)
    dump(RET/'detector_reconstruction.json',checks)
    return vec,d1,d1p,d3,cache,checks

def download_model(repo):
    target=PUBLIC/repo.split('/')[-1]
    if (target/'config.json').exists():return target
    env=os.environ.copy();env['HF_HUB_OFFLINE']='0';env['TRANSFORMERS_OFFLINE']='0';env['HF_ENDPOINT']=CFG['hf_mirror']
    code='from huggingface_hub import snapshot_download;import sys;print(snapshot_download(repo_id=sys.argv[1],local_dir=sys.argv[2],endpoint=sys.argv[3]))'
    r=subprocess.run([PY,'-c',code,repo,str(target),CFG['hf_mirror']],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log(r.stdout[-3000:])
    return target if r.returncode==0 and (target/'config.json').exists() else None

def one_model_sequence(tag,path,rows,kind):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    out=RUN/'sequences'/f'{tag}.json';out.parent.mkdir(parents=True,exist_ok=True)
    state={'tag':tag,'kind':kind,'rows':[]}
    if out.exists():
        try:state=json.load(open(out))
        except Exception:pass
    done={int(x['index']):x for x in state.get('rows',[])}
    tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=(kind=='phi'))
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    kwargs={'local_files_only':True,'low_cpu_mem_usage':True,'trust_remote_code':(kind=='phi')}
    if kind=='quant4':kwargs.update(device_map='auto',quantization_config=BitsAndBytesConfig(load_in_4bit=True))
    else:kwargs.update(torch_dtype=torch.float16)
    if kind=='phi':kwargs['attn_implementation']='eager'
    m=AutoModelForCausalLM.from_pretrained(str(path),**kwargs)
    if kind!='quant4':m.to('cuda:0')
    m.eval();m.config.use_cache=False if kind=='phi' else m.config.use_cache
    pad=tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    for j,r in enumerate(rows):
        idx=int(r['global_index'])
        if idx in done:continue
        enc=tok(r['prompt'],add_special_tokens=False,return_tensors='pt')['input_ids'].to(m.device);c1=c2=0
        seed=20262000+idx+sum(ord(c) for c in tag)%1000;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            for _ in range(CFG['num_samples']):
                kw=dict(input_ids=enc,max_new_tokens=1,pad_token_id=pad,do_sample=True,temperature=1.0,top_p=1.0,top_k=0,eos_token_id=tok.eos_token_id)
                if kind=='phi':kw['use_cache']=False
                y=m.generate(**kw);tid=y[0,-1].item();w=tok.decode([tid],skip_special_tokens=True).strip().lower()
                if w==r['word1'].lower():c1+=1
                elif w==r['word2'].lower():c2+=1
        rec={'index':idx,'p1':c1/CFG['num_samples'],'p2':c2/CFG['num_samples']};done[idx]=rec;state['rows']=[done[k] for k in sorted(done)];dump(out,state);log(f'{tag} {len(done)}/{len(rows)}')
    del m,tok;gc.collect();torch.cuda.empty_cache()
    return np.asarray([[x['p1'],x['p2']] for x in state['rows']],float)

def bits(seq):return (np.asarray(seq)[:,0]>=np.asarray(seq)[:,1]).astype(int)
def acc(a,b):return float(np.mean(bits(a)==bits(b)))
def threshold_from_neg(base,negs):
    vals=np.asarray([acc(base,x) for x in negs],float);mu=float(vals.mean());sd=float(vals.std(ddof=1)) if len(vals)>1 else 0.;return min(1.,mu+1.64*sd),vals.tolist(),mu,sd

def bootstrap_acc(base,mixed,reps=2000):
    rng=np.random.default_rng(20261004);bb=bits(base);mm=bits(mixed);n=len(bb);vals=[]
    for _ in range(reps):
        ii=rng.integers(0,n,n);vals.append(float(np.mean(bb[ii]==mm[ii])))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def main():
    log('=== SLATE StageP2B LLMPrint Selective-Serving Verification Degradation ===')
    qwen=find_snapshot('models--Qwen--Qwen3-4B-Base');phi=PUBLIC/'Phi-3-mini-4k-instruct'
    if not qwen:raise RuntimeError('Qwen3-4B-Base missing')
    from transformers import AutoTokenizer
    bt=AutoTokenizer.from_pretrained(str(qwen),local_files_only=True);rows,bank_hash=frozen_eval_bank(bt);prompts=[r['prompt'] for r in rows]
    vec,d1,d1scores,d3,ec,checks=reconstruct_detectors(prompts)
    # sequences
    base=one_model_sequence('qwen_base',qwen,rows,'base')
    quant=one_model_sequence('qwen_nf4',qwen,rows,'quant4')
    proxy=None;proxy_tag=None;phi_status={'path':str(phi),'exists':(phi/'config.json').exists()}
    if phi_status['exists']:
        try:proxy=one_model_sequence('phi3_proxy',phi,rows,'phi');proxy_tag='Phi-3-mini-4k-instruct';phi_status['status']='ok_use_cache_false'
        except Exception as e:phi_status['status']='failed';phi_status['error']=repr(e);log('Phi proxy failed '+repr(e))
    dump(RET/'phi_proxy_status.json',phi_status)
    negs=[];negstat=[];fallback_proxy=None;fallback_tag=None
    for repo in CFG['negative_models']:
        p=download_model(repo)
        if p is None:negstat.append({'repo':repo,'status':'download_failed'});continue
        try:
            s=one_model_sequence('neg_'+repo.split('/')[-1].replace('-','_'),p,rows,'base');negs.append(s);negstat.append({'repo':repo,'status':'ok','path':str(p)})
            if fallback_proxy is None:fallback_proxy=s;fallback_tag=repo
        except Exception as e:negstat.append({'repo':repo,'status':'sequence_failed','error':repr(e)})
    if len(negs)<CFG['negative_min_required']:raise RuntimeError('insufficient validation negatives')
    if proxy is None:proxy=fallback_proxy;proxy_tag=fallback_tag;phi_status['fallback_proxy']=proxy_tag;dump(RET/'phi_proxy_status.json',phi_status)
    dump(RET/'validation_negative_status.json',negstat)
    tau,neg_acc,mu,sd=threshold_from_neg(base,negs)
    honest_acc=acc(base,quant);proxy_acc=acc(base,proxy)
    native={'bank_hash':bank_hash,'q':64,'num_samples':100,'threshold_rule':'mu + 1.64*std over public LLMPrint validation negatives','tau':tau,'negative_accuracies':neg_acc,'negative_mu':mu,'negative_std':sd,'honest_quantized_accuracy':honest_acc,'honest_verified':bool(honest_acc>=tau),'proxy_model':proxy_tag,'proxy_accuracy':proxy_acc,'proxy_verified':bool(proxy_acc>=tau)}
    dump(RET/'llmprint_native_verification.json',native)
    # A1 masks and A2 100 permutations
    results={'A0':{'accuracy':honest_acc,'verified':bool(honest_acc>=tau)},'A1':{},'A2':{}}
    # Frozen D1 per-prompt scores already computed on eval prompts.
    for beta in CFG['betas']:
        k=str(beta);th=float(P2A_THRESH[k]['D1']['threshold']);mask=np.asarray(d1scores)>=th;mixed=np.where(mask[:,None],proxy,quant);aa=acc(base,mixed)
        results['A1'][k]={'detector':'D1_L1','threshold':th,'AIR':float(mask.mean()),'BDR_frozen':float(P2A_THRESH[k]['D1']['measured_BDR']),'accuracy':aa,'accuracy_ci_pair_bootstrap':bootstrap_acc(base,mixed,CFG['bootstrap_reps']),'verified':bool(aa>=tau),'CVR':float(aa>=tau),'score_drop':honest_acc-aa,'proxy_fraction':float(mask.mean())}
    # A2 session permutations. Score each latest L=4 window; first 3 honest.
    for beta in CFG['betas']:
        k=str(beta);th=float(P2A_THRESH[k]['D3']['threshold']);epacc=[];airs=[];passes=[]
        for e in range(CFG['a2_episodes']):
            order=sorted(range(64),key=lambda i:h64(f'{e}|{i}','P2B_A2'));ordered=[prompts[i] for i in order];mask_order=np.zeros(64,dtype=bool)
            wins=[ordered[i-3:i+1] for i in range(3,64)];sc=score_d3(d3,ec,wins);mask_order[3:]=sc>=th
            # map decisions back to original eval row order
            mask=np.zeros(64,dtype=bool)
            for pos,orig in enumerate(order):mask[orig]=mask_order[pos]
            mixed=np.where(mask[:,None],proxy,quant);a=acc(base,mixed);epacc.append(a);airs.append(float(mask.mean()));passes.append(float(a>=tau))
        epacc=np.asarray(epacc);airs=np.asarray(airs);passes=np.asarray(passes)
        results['A2'][k]={'detector':'D3_L4','threshold':th,'BDR_frozen':float(P2A_THRESH[k]['D3']['measured_BDR']),'episodes':len(epacc),'AIR_mean':float(airs.mean()),'AIR_std':float(airs.std()),'accuracy_mean':float(epacc.mean()),'accuracy_std':float(epacc.std()),'accuracy_ci_episode':[float(np.quantile(epacc,.025)),float(np.quantile(epacc,.975))],'CVR':float(passes.mean()),'score_drop_mean':float(honest_acc-epacc.mean()),'proxy_fraction_mean':float(airs.mean())}
    dump(RET/'selective_serving_results.json',results)
    dump(RET/'model_sequence_manifest.json',{'qwen_base_rows':len(base),'qwen_nf4_rows':len(quant),'proxy_rows':len(proxy),'proxy_model':proxy_tag,'validation_negative_count':len(negs),'eval_bank_hash':bank_hash})
    summary={'workflow':CFG['workflow'],'status':'STAGEP2B_COMPLETE','paper_facing_confirmatory':False,'heldout_touched':False,'P2A_frozen':True,'native':native,'results':results,'interpretation_rule':'Report observed verification degradation without post-hoc threshold or detector changes.'}
    dump(RET/'summary.json',summary);dump(RET/'status.json',{'workflow':CFG['workflow'],'status':'STAGEP2B_COMPLETE','rc':0});log(json.dumps(summary,indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(RET/'status.json',{'workflow':CFG['workflow'],'status':'STAGEP2B_FAILED','rc':1,'error':repr(e)});log('FATAL '+repr(e));raise
