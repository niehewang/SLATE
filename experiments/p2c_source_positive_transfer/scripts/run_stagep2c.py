from __future__ import annotations
import collections, gc, hashlib, json, os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CFG=json.load(open(ROOT/'config/project.json'))
RUN=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2c_llmprint_source_positive_threat_transfer_v1'
RET=RUN/'return'; RUN.mkdir(parents=True,exist_ok=True); RET.mkdir(parents=True,exist_ok=True)
LOG=RUN/'run.log'
P2A=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2a_llmprint_threat_surface_v1'
P2B=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep2b_llmprint_selective_serving_degradation_v1'
PAIRS=P2A/'llmprint_128/pairs_128.pt'; ADV=P2A/'llmprint_128/adv_128'
TRAFFIC=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stage0_traffic_bootstrap_v1/traffic_frozen.jsonl'
P2A_THRESH=json.load(open(ROOT/'frozen/p2a_attack_thresholds_calibration.json'))
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
    pairs=load_pairs();rows=[]
    for i in range(CFG['eval_start'],CFG['eval_start']+CFG['eval_count']):
        a,b=pairs[i];ids=flat_ids(torch.load(ADV/f'{a}_{b}.pt',map_location='cpu'));suffix=base_tok.decode(ids,skip_special_tokens=False)
        rows.append({'global_index':i,'word1':a,'word2':b,'prompt':f'Randomly output a word from your vocabulary {suffix}'})
    h=hashlib.sha256('\n'.join(x['prompt'] for x in rows).encode()).hexdigest()
    if h!=P2A_BANK['eval_sha256']:raise RuntimeError(f'eval bank hash mismatch {h} != {P2A_BANK["eval_sha256"]}')
    return rows,h

def load_seq(path):
    o=json.load(open(path)); rows=o.get('rows',o)
    if len(rows)!=64: raise RuntimeError(f'incomplete frozen sequence {path}: {len(rows)}')
    return np.asarray([[float(x['p1']),float(x['p2'])] for x in rows],float)

def bits(seq):return (np.asarray(seq)[:,0]>=np.asarray(seq)[:,1]).astype(int)
def acc(ref,seq):return float(np.mean(bits(ref)==bits(seq)))

def one_source_rep(tag,path,rows,rep):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    out=RUN/'sequences'/f'{tag}.json';out.parent.mkdir(parents=True,exist_ok=True)
    state={'tag':tag,'replicate':rep,'rows':[]}
    if out.exists():
        try:state=json.load(open(out))
        except Exception:pass
    done={int(x['index']):x for x in state.get('rows',[])}
    if len(done)==len(rows):
        log(f'{tag} cache complete {len(done)}/{len(rows)}')
        return np.asarray([[x['p1'],x['p2']] for x in state['rows']],float)
    tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,low_cpu_mem_usage=True,torch_dtype=torch.float16).to('cuda:0');m.eval()
    pad=tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    for r in rows:
        idx=int(r['global_index'])
        if idx in done:continue
        enc=tok(r['prompt'],add_special_tokens=False,return_tensors='pt')['input_ids'].to(m.device);c1=c2=0
        seed=20263000 + rep*100000 + idx
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            for _ in range(CFG['num_samples']):
                y=m.generate(input_ids=enc,max_new_tokens=1,pad_token_id=pad,do_sample=True,temperature=1.0,top_p=1.0,top_k=0,eos_token_id=tok.eos_token_id)
                tid=y[0,-1].item();w=tok.decode([tid],skip_special_tokens=True).strip().lower()
                if w==r['word1'].lower():c1+=1
                elif w==r['word2'].lower():c2+=1
        done[idx]={'index':idx,'p1':c1/CFG['num_samples'],'p2':c2/CFG['num_samples']}
        state['rows']=[done[k] for k in sorted(done)];dump(out,state);log(f'{tag} {len(done)}/{len(rows)}')
    del m,tok;gc.collect();torch.cuda.empty_cache()
    return np.asarray([[x['p1'],x['p2']] for x in state['rows']],float)

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
    out.sort(key=lambda x:h64(x['id'],'P2A_'+split));return out

def sample_queries(ss,n,salt):
    a=[(t,s['id']) for s in ss for t in s['texts']];a.sort(key=lambda x:h64(x[0]+'|'+x[1],salt));return a[:min(n,len(a))]
def real_windows(ss,L,n,salt):
    a=[]
    for s in ss:
        for i in range(len(s['texts'])-L+1):a.append((s['texts'][i:i+L],s['id']))
    a.sort(key=lambda x:h64('|'.join(x[0])+'|'+x[1],salt));return a[:min(n,len(a))]
def make_pos(prompts,n,salt):return [[prompts[i] for i in sorted(range(len(prompts)),key=lambda i:h64(f'{e}|{i}',salt))] for e in range(n)]
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
    X=seq_features(c,[(s,str(i)) for i,s in enumerate(seqs)],4);dev=next(m.parameters()).device;m.eval();out=[]
    with torch.inference_mode():
        for st in range(0,len(X),256):out.append(torch.sigmoid(m(torch.from_numpy(X[st:st+256]).to(dev))).cpu().numpy())
    return np.concatenate(out)

def reconstruct_detectors(eval_prompts,base_tok):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sentence_transformers import SentenceTransformer
    import torch
    mp=find_snapshot('models--sentence-transformers--all-mpnet-base-v2')
    if not mp:raise RuntimeError('MPNet missing')
    dev=sessions('development');cal=sessions('calibration');negt=[x[0] for x in sample_queries(dev,12000,'p2a_train_q')];neg2=[x[0] for x in sample_queries(cal,12000,'p2a_eval_q')]
    pairs=load_pairs();import torch as T;train=[]
    for i in range(64):
        a,b=pairs[i];ids=flat_ids(T.load(ADV/f'{a}_{b}.pt',map_location='cpu'));train.append(f'Randomly output a word from your vocabulary {base_tok.decode(ids,skip_special_tokens=False)}')
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=100000,sublinear_tf=True,strip_accents='unicode');X=vec.fit_transform(train+negt);y=np.r_[np.ones(64),np.zeros(len(negt))];d1=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=20261004).fit(X,y)
    d1p=d1.predict_proba(vec.transform(eval_prompts))[:,1];d1n=d1.predict_proba(vec.transform(neg2))[:,1]
    enc=SentenceTransformer(str(mp),device='cuda' if torch.cuda.is_available() else 'cpu');cache=EC(enc)
    pi_tr=sample_items(pos_windows(make_pos(train,120,'P2A_POS_TRAIN'),4),7000,'ptr4');ni_tr=real_windows(dev,4,7000,'ntr4');d3=fit_d3(cache,pi_tr,ni_tr)
    peval=make_pos(eval_prompts,120,'P2A_POS_EVAL');d3p=score_d3(d3,cache,[x[0] for x in sample_items(pos_windows(peval,4),7000,'pev4')]);d3n=score_d3(d3,cache,[x[0] for x in real_windows(cal,4,7000,'nev4')])
    checks={'D1_AUROC':float(roc_auc_score(np.r_[np.ones(len(d1p)),np.zeros(len(d1n))],np.r_[d1p,d1n])),'D3_L4_AUROC':float(roc_auc_score(np.r_[np.ones(len(d3p)),np.zeros(len(d3n))],np.r_[d3p,d3n])),'frozen_threshold_checks':{}}
    for b in CFG['betas']:
        k=str(b);bd1=float(np.mean(d1n>=P2A_THRESH[k]['D1']['threshold']));bd3=float(np.mean(d3n>=P2A_THRESH[k]['D3']['threshold']));checks['frozen_threshold_checks'][k]={'D1_BDR_reconstructed':bd1,'D1_BDR_frozen':P2A_THRESH[k]['D1']['measured_BDR'],'D3_BDR_reconstructed':bd3,'D3_BDR_frozen':P2A_THRESH[k]['D3']['measured_BDR']}
        if abs(bd1-P2A_THRESH[k]['D1']['measured_BDR'])>.02 or abs(bd3-P2A_THRESH[k]['D3']['measured_BDR'])>.02:raise RuntimeError('detector reconstruction drift >0.02')
    dump(RET/'detector_reconstruction.json',checks);return d1p,d3,cache,checks

def pair_bootstrap(ref,seq,reps=2000):
    rng=np.random.default_rng(20261009);a=bits(ref);b=bits(seq);n=len(a);v=[]
    for _ in range(reps):
        ii=rng.integers(0,n,n);v.append(float(np.mean(a[ii]==b[ii])))
    return [float(np.quantile(v,.025)),float(np.quantile(v,.975))]

def main():
    log('=== SLATE StageP2C LLMPrint Source-Positive Threat Transfer ===')
    qwen=find_snapshot('models--Qwen--Qwen3-4B-Base');
    if not qwen:raise RuntimeError('Qwen3-4B-Base missing')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(qwen),local_files_only=True);rows,bank_hash=frozen_eval_bank(tok);prompts=[r['prompt'] for r in rows]
    # frozen StageP2B artifacts
    ref=load_seq(P2B/'sequences/qwen_base.json');proxy=load_seq(P2B/'sequences/phi3_proxy.json')
    native_path=P2B/'return/llmprint_native_verification.json'
    if not native_path.exists():raise RuntimeError('P2B frozen native verification missing')
    native=json.load(open(native_path));tau=float(native['tau'])
    if native.get('bank_hash')!=bank_hash:raise RuntimeError('P2B/P2C bank hash mismatch')
    nf4_boundary={'accuracy':float(native['honest_quantized_accuracy']),'verified':bool(native['honest_verified']),'tau':tau}
    d1scores,d3,ec,checks=reconstruct_detectors(prompts,tok)
    reps=[]
    for r in range(CFG['source_repeats']):reps.append(one_source_rep(f'qwen_source_rep{r}',qwen,rows,r))
    a0_scores=np.asarray([acc(ref,x) for x in reps],float);a0_pass=a0_scores>=tau
    results={'A0':{'source_repeats':CFG['source_repeats'],'accuracies':a0_scores.tolist(),'accuracy_mean':float(a0_scores.mean()),'accuracy_min':float(a0_scores.min()),'accuracy_max':float(a0_scores.max()),'CVR':float(a0_pass.mean()),'verified_each':a0_pass.tolist(),'pair_bootstrap_ci_each':[pair_bootstrap(ref,x,CFG['bootstrap_reps']) for x in reps]},'A1':{},'A2':{}}
    # A1
    for beta in CFG['betas']:
        k=str(beta);th=float(P2A_THRESH[k]['D1']['threshold']);mask=np.asarray(d1scores)>=th;vals=[];cis=[]
        for src in reps:
            mix=np.where(mask[:,None],proxy,src);vals.append(acc(ref,mix));cis.append(pair_bootstrap(ref,mix,CFG['bootstrap_reps']))
        vals=np.asarray(vals);results['A1'][k]={'detector':'D1_L1','threshold':th,'AIR':float(mask.mean()),'BDR_frozen':float(P2A_THRESH[k]['D1']['measured_BDR']),'accuracies':vals.tolist(),'accuracy_mean':float(vals.mean()),'CVR':float(np.mean(vals>=tau)),'score_drop_from_A0_mean':float(a0_scores.mean()-vals.mean()),'proxy_fraction':float(mask.mean()),'pair_bootstrap_ci_each':cis}
    # A2: 100 permutations per source replicate
    for beta in CFG['betas']:
        k=str(beta);th=float(P2A_THRESH[k]['D3']['threshold']);scores=[];airs=[]
        for e in range(CFG['a2_episodes']):
            order=sorted(range(64),key=lambda i:h64(f'{e}|{i}','P2C_A2'));ordered=[prompts[i] for i in order];mask_order=np.zeros(64,dtype=bool);wins=[ordered[i-3:i+1] for i in range(3,64)];sc=score_d3(d3,ec,wins);mask_order[3:]=sc>=th;mask=np.zeros(64,dtype=bool)
            for pos,orig in enumerate(order):mask[orig]=mask_order[pos]
            airs.append(float(mask.mean()))
            for src in reps:scores.append(acc(ref,np.where(mask[:,None],proxy,src)))
        scores=np.asarray(scores);airs=np.asarray(airs);results['A2'][k]={'detector':'D3_L4','threshold':th,'BDR_frozen':float(P2A_THRESH[k]['D3']['measured_BDR']),'episodes_per_source_repeat':CFG['a2_episodes'],'total_score_episodes':int(len(scores)),'AIR_mean':float(airs.mean()),'AIR_std':float(airs.std()),'accuracy_mean':float(scores.mean()),'accuracy_std':float(scores.std()),'accuracy_ci_episode':[float(np.quantile(scores,.025)),float(np.quantile(scores,.975))],'CVR':float(np.mean(scores>=tau)),'score_drop_from_A0_mean':float(a0_scores.mean()-scores.mean()),'proxy_fraction_mean':float(airs.mean())}
    dump(RET/'source_positive_results.json',results)
    dump(RET/'source_replicate_manifest.json',{'reference_rows':64,'proxy_rows':64,'source_repeats':CFG['source_repeats'],'source_rows_each':[len(x) for x in reps],'bank_hash':bank_hash})
    prov={'tau':tau,'threshold_rule':native['threshold_rule'],'negative_accuracies':native['negative_accuracies'],'P2A_bank_hash':bank_hash,'NF4_boundary_frozen':nf4_boundary,'heldout_touched':False,'no_threshold_retuning':True}
    dump(RET/'frozen_provenance.json',prov)
    summary={'workflow':CFG['workflow'],'status':'STAGEP2C_COMPLETE','paper_facing_confirmatory':False,'heldout_touched':False,'purpose':'published LLMPrint source-ownership threat transfer; NF4 boundary remains frozen separately','tau':tau,'NF4_boundary':nf4_boundary,'source_A0':results['A0'],'A1':results['A1'],'A2':results['A2'],'interpretation_rule':'Only claim attack-induced verification degradation if independent source A0 passes the frozen threshold; never reinterpret the NF4 boundary.'}
    dump(RET/'summary.json',summary);dump(RET/'status.json',{'workflow':CFG['workflow'],'status':'STAGEP2C_COMPLETE','rc':0});log(json.dumps(summary,indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(RET/'status.json',{'workflow':CFG['workflow'],'status':'STAGEP2C_FAILED','rc':1,'error':repr(e)});log('FATAL '+repr(e));raise
