from __future__ import annotations
import collections, gc, gzip, hashlib, json, math, os, random, re, shutil, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CFG=json.load(open(ROOT/'config/project.json'))
PYTHON='/home/jx-vmlab/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python'
SHARED=Path.home()/'SLATE_TDSC_SERVER_SHARED'
RUN=SHARED/'runs/slate_stagep7v2_revised_sharegpt_phi_external_confirmatory_v1'
RET=RUN/'return'; BANK=RUN/'banks'; EMB=RUN/'embeddings'; CKPT=RUN/'checkpoints'; CACHE=RUN/'cache'
for p in [RUN,RET,BANK,EMB,CKPT,CACHE]: p.mkdir(parents=True,exist_ok=True)
LOG=RUN/'run.log'

sys.path.insert(0,str(ROOT/'scripts'))
import p3b_core as C
C.CFG=CFG; C.RUN=RUN; C.RET=RET; C.BANK=BANK; C.EMB=EMB; C.CACHE=CACHE; C.LOG=LOG

# Existing offline assets, all already used successfully by prior stages.
PHI_CANDS=[
    Path('/data/jx-vmlab/SLATE_TDSC_SERVER_SHARED/public_models/Phi-3-mini-4k-instruct'),
    SHARED/'public_models/Phi-3-mini-4k-instruct',
]
QWEN_CANDS=[
    Path('/home/jx-vmlab/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539'),
    Path('/data/jx-vmlab/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539'),
]
ENC_CANDS=[SHARED/'public_models/Qwen3-Embedding-8B',Path('/data/jx-vmlab/SLATE_TDSC_SERVER_SHARED/public_models/Qwen3-Embedding-8B')]
MPNET_CANDS=[
    Path('/home/jx-vmlab/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130'),
    Path('/data/jx-vmlab/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130'),
]

def first_complete(cands,model=False):
    for p in cands:
        if p.exists() and ((not model) or C.model_complete_path(p)): return p
    return cands[0]
PHI=first_complete(PHI_CANDS,True); QWEN=first_complete(QWEN_CANDS,True); ENC=first_complete(ENC_CANDS,True); MPNET=first_complete(MPNET_CANDS,False)
C.PHI=PHI; C.QWEN=QWEN; C.ENC=ENC; C.MPNET=MPNET
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'

def log(x):
    s=str(x); print(s,flush=True)
    with open(LOG,'a',encoding='utf-8') as f:f.write(s+'\n')
def dump(name,obj):
    p=RET/name; p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,default=str)
def load(p):return json.load(open(p,encoding='utf-8'))
def sha_text(s):return hashlib.sha256(str(s).encode()).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def h64(x,salt=''):return int.from_bytes(hashlib.sha256((salt+'\0'+str(x)).encode()).digest()[:8],'big')

def resolve_traffic():
    fn=CFG['sharegpt_filename']; candidates=[
        ROOT/'assets'/fn,
        Path('/data/jx-vmlab')/fn,
        Path('/data/jx-vmlab/SLATE_TDSC_SERVER_SHARED/public_data')/fn,
        SHARED/'public_data'/fn,
    ]
    for p in candidates:
        if p.exists():
            got=sha_file(p)
            if got!=CFG['sharegpt_expected_sha256']:
                raise RuntimeError(f'ShareGPT asset SHA mismatch: {p} got={got} expected={CFG["sharegpt_expected_sha256"]}')
            return p
    raise RuntimeError('MISSING_SHAREGPT_ASSET: '+fn+'; mirror-only asset acquisition must run before science stage')

CONV_ID_RE=re.compile(r'"id"\s*:\s*"([^"]+)"')
def split_of(tid):
    x=h64(tid,'P7V2_SHAREGPT_SPLIT')%100
    return 'development' if x<60 else ('calibration' if x<80 else 'heldout')

def valid_text(t):
    t=str(t or '').strip(); n=len(t.split())
    return 4<=n<=256 and 12<=len(t)<=2400

def extract_user_texts(obj):
    conv=obj.get('conversations') or []
    if not isinstance(conv,list) or not conv:return []
    # Require a user-started conversation to match an ordinary request session.
    if str((conv[0] or {}).get('from','')).lower() not in ('human','user'):return []
    out=[]
    for m in conv:
        if not isinstance(m,dict):continue
        role=str(m.get('from','')).lower()
        if role in ('human','user'):
            t=str(m.get('value') or '').strip()
            if valid_text(t):out.append(t)
    return out

def load_split(asset,target):
    sessions=[]; ids=[]; line_count=0; parsed=0
    with open(asset,'rt',encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            line_count+=1
            # train.json is JSONL. Determine split from id before parsing the row;
            # heldout row bodies are not JSON-parsed before the freeze.
            m=CONV_ID_RE.search(line[:512]); tid=m.group(1) if m else None
            if tid is None:
                # Conservative: rows without an early id are never eligible.
                continue
            if split_of(tid)!=target:continue
            obj=json.loads(line);parsed+=1
            tid=str(obj.get('id') or tid)
            if split_of(tid)!=target:continue
            texts=extract_user_texts(obj)
            if texts:sessions.append({'id':tid,'tree_id':tid,'texts':texts})
            ids.append(tid)
    sessions.sort(key=lambda x:h64(x['id'],'P7V2_'+target))
    return sessions,{'target':target,'raw_lines_seen':line_count,'json_parsed_target_lines':parsed,'conversations':len(set(ids)),'sessions':len(sessions),'conversation_id_sha256':sha_text('\n'.join(sorted(set(ids))))}

def sequence_eligibility(dev_real,cal_real):
    counts={}
    for name,ss in [('dev',dev_real),('cal',cal_real)]:
        for L in (4,8):counts[f'{name}_L{L}']=len(C.real_windows(ss,L,10**9,'P7V2_ELIG_'+name+str(L)))
    req={'dev_L4':CFG['min_dev_windows_L4'],'dev_L8':CFG['min_dev_windows_L8'],'cal_L4':CFG['min_cal_windows_L4'],'cal_L8':CFG['min_cal_windows_L8']}
    bad={k:(counts[k],v) for k,v in req.items() if counts[k]<v}
    dump('traffic_eligibility.json',{'counts':counts,'requirements':req,'pass':not bool(bad),'failures':bad})
    if bad:raise RuntimeError('TRAFFIC_CORPUS_STATEFUL_INELIGIBLE: '+repr(bad))
    return counts

def freeze_pool(sessions,n,salt):return C.freeze_pool(sessions,n,salt)

OLD_P7_RUN=SHARED/'runs/slate_stagep7_revised_oasst_phi_external_confirmatory_v1'
LORA_DIR=OLD_P7_RUN/'checkpoints/phi_oasst_lora'
def resolve_existing_lora():
    marker=LORA_DIR/'P7_LORA_COMPLETE.json'
    if not marker.exists():raise RuntimeError('Missing completed Phi OASST-LoRA checkpoint from prior pre-freeze P7 run: '+str(marker))
    rec=load(marker)
    if rec.get('status')!='COMPLETE':raise RuntimeError('Prior Phi LoRA checkpoint is not COMPLETE')
    files={}
    for fn in ['adapter_config.json','adapter_model.safetensors','tokenizer_config.json']:
        q=LORA_DIR/fn
        if q.exists():files[fn]=sha_file(q)
    if 'adapter_config.json' not in files or 'adapter_model.safetensors' not in files:raise RuntimeError('Incomplete LoRA adapter files')
    return {**rec,'reuse_for_p7v2':True,'checkpoint':str(LORA_DIR),'file_sha256':files,'training_traffic':'OASST1 development only; frozen before ShareGPT confirmatory design'}

def generate_bank(tag,kind,records):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    out=BANK/f'{tag}.json'; state={'tag':tag,'kind':kind,'rows':[]}
    if out.exists():
        try:state=load(out)
        except Exception:pass
    done={r['id']:r for r in state.get('rows',[]) if r.get('response') is not None}
    if all(r['id'] in done for r in records):return [done[r['id']] for r in records]
    base=PHI if kind.startswith('phi') else QWEN
    trust=kind.startswith('phi')
    tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True,trust_remote_code=trust)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.padding_side='left'
    kw={'local_files_only':True,'low_cpu_mem_usage':True,'trust_remote_code':trust}
    if kind=='phi_nf4':
        kw.update(device_map='auto',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),attn_implementation='eager')
    else:
        kw.update(torch_dtype=torch.bfloat16)
        if trust:kw['attn_implementation']='eager'
    m=AutoModelForCausalLM.from_pretrained(str(base),**kw)
    if kind!='phi_nf4':m.to('cuda:0')
    if kind=='phi_lora':
        from peft import PeftModel
        m=PeftModel.from_pretrained(m,str(LORA_DIR),is_trainable=False)
    m.eval()
    if trust:m.config.use_cache=False
    bs=CFG['generation_batch_size']
    for st in range(0,len(records),bs):
        batch=records[st:st+bs]; missing={r['id'] for r in batch if r['id'] not in done}
        if not missing:continue
        texts=[]
        for r in batch:
            t=r['text']
            if trust and hasattr(tok,'apply_chat_template'):
                try:t=tok.apply_chat_template([{'role':'user','content':t}],tokenize=False,add_generation_prompt=True)
                except Exception:pass
            texts.append(t)
        enc=tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=768)
        dev=m.device if hasattr(m,'device') else next(m.parameters()).device;enc={k:v.to(dev) for k,v in enc.items()}
        seed=CFG['seed']+h64(tag+'|'+str(st),'GEN')%1000000;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        gkw=dict(**enc,max_new_tokens=CFG['response_max_new_tokens'],do_sample=True,temperature=CFG['generation_temperature'],top_p=CFG['generation_top_p'],top_k=0,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        if trust:gkw['use_cache']=False
        with torch.inference_mode():y=m.generate(**gkw)
        inlen=enc['input_ids'].shape[1]
        for r,yy in zip(batch,y):
            if r['id'] not in missing:continue
            resp=tok.decode(yy[inlen:],skip_special_tokens=True).strip();done[r['id']]={**r,'response':resp,'response_chars':len(resp),'generation_seed':int(seed),'generation_batch_start':int(st)}
        state['rows']=[done[k] for k in sorted(done)];dump(Path('..')/'banks_progress_placeholder.json',{}) if False else None
        with open(out,'w',encoding='utf-8') as f:json.dump(state,f,ensure_ascii=False)
        log(f'{tag}: {len(done)}/{len(records)}')
    del m,tok;gc.collect();torch.cuda.empty_cache();return [done[r['id']] for r in records]

def embed_all(banks):
    from sentence_transformers import SentenceTransformer
    enc=SentenceTransformer(str(ENC),device='cuda:0')
    out={k:C.embed_bank(k,v,enc) for k,v in banks.items()};del enc;gc.collect()
    try:
        import torch;torch.cuda.empty_cache()
    except Exception:pass
    return out

def lineage_phi(ad,idx,sus,src,wrong,risk_tau=None,gap_tau=None,tau=None):
    idx=np.asarray(idx,int); ds=ad.vec(sus[idx],'suspect'); dp=ad.vec(src[idx],'phi_source'); dq=ad.vec(wrong[idx],'qwen_wrong')
    sp=C.cosine(ds,dp);sq=C.cosine(ds,dq)
    A=sus[idx,:64];B=src[idx,:64];D=wrong[idx,:64];ms=C.cosrow(A,B)-C.cosrow(A,D);med=float(np.median(ms));mad=float(np.median(np.abs(ms-med)));fpos=float(np.mean(ms>=0));mix=min(fpos,1-fpos);risk=float(mad+2*mix);gap=float(abs(sp-sq));best='Phi-3-mini-4k-instruct' if sp>=sq else 'Qwen3-4B-Base';score=max(sp,sq)
    verdict=None
    if tau is not None:
        if risk_tau is not None and risk>risk_tau:verdict='Inconclusive-Serving'
        elif score<tau:verdict='Rejected'
        elif gap_tau is not None and gap<gap_tau:verdict='Inconclusive-Bank'
        else:verdict='Verified'
    return {'sim_phi':sp,'sim_qwen':sq,'best':best,'score':score,'risk':risk,'gap':gap,'verdict':verdict}

def rates_phi(rows):
    n=max(1,len(rows));return {'CSVR':sum(r['verdict']=='Verified' and r['best']=='Phi-3-mini-4k-instruct' for r in rows)/n,'FSVR':sum(r['verdict']=='Verified' and r['best']=='Qwen3-4B-Base' for r in rows)/n,'Channel_Risk':sum(r['verdict']=='Inconclusive-Serving' for r in rows)/n,'Insufficient':sum(r['verdict'] in ('Inconclusive-Serving','Inconclusive-Bank') for r in rows)/n,'Rejected':sum(r['verdict']=='Rejected' for r in rows)/n}

def fixed_sessions(J,n,q,tag):
    top=np.argsort(np.asarray(J,float))[::-1][:q].astype(int).tolist();return [{'episode':e,'seed':CFG['seed']+e,'selected':top,'random_order':top,'order':top,'mean_J':float(np.mean(np.asarray(J)[top]))} for e in range(n)]

def train_detector_bundle(tag,dev_real,dev_pool,dev_sessions_list,mp):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    ec=C.ECache(mp);posq=[]
    for ss in dev_sessions_list:posq.extend(C.flatten_queries(dev_pool,ss))
    posq=posq[:CFG['detector_query_cap']];negq=C.excluded_queries(dev_real,[r['text'] for r in dev_pool],CFG['detector_query_cap'],'P7V2_'+tag+'_DNEG')
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=100000,sublinear_tf=True,strip_accents='unicode');X=vec.fit_transform(posq+negq);y=np.r_[np.ones(len(posq)),np.zeros(len(negq))]
    d1=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=CFG['seed']).fit(X,y);d2=LogisticRegression(max_iter=1500,class_weight='balanced',random_state=CFG['seed']).fit(np.r_[ec.get(posq),ec.get(negq)],y)
    seq={}; pooled=[s for ss in dev_sessions_list for s in ss]
    for L in CFG['L_grid']:
        pos=C.audit_items(dev_pool,pooled,L)[:CFG['detector_window_cap']];negw=C.real_windows(dev_real,L,CFG['detector_window_cap'],'P7V2_'+tag+'_NEGW'+str(L))
        if L in (4,8):seq[f'D3_L{L}']=C.fit_seq(ec,pos,negw,L,CFG['d3_hidden'],False,CFG['seed']+L+(1000 if tag=='UNWRAP' else 0))
        seq[f'D4_L{L}']=C.fit_seq(ec,pos,negw,L,CFG['d4_hidden'],True,CFG['seed']+100+L+(1000 if tag=='UNWRAP' else 0))
    return {'d1':d1,'d2':d2,'vec':vec,'ec':ec,'seq':seq}

def calibration_neg(bundle,cal_real,cal_pool,tag):
    b=bundle;negq=C.excluded_queries(cal_real,[r['text'] for r in cal_pool],6000,'P7V2_'+tag+'_CALNEG');out={'D1':b['d1'].predict_proba(b['vec'].transform(negq))[:,1],'D2':b['d2'].predict_proba(b['ec'].get(negq))[:,1]}
    for L in CFG['L_grid']:
        nw=C.real_windows(cal_real,L,CFG['detector_window_cap'],'P7V2_'+tag+'_CALW'+str(L))
        if L in (4,8):out[f'D3_L{L}']=C.score_seq(b['seq'][f'D3_L{L}'],b['ec'],nw,L)
        out[f'D4_L{L}']=C.score_seq(b['seq'][f'D4_L{L}'],b['ec'],nw,L)
    return out

def score_detector_metrics(bundle,pool,sessions,neg):
    b=bundle;qs=C.flatten_queries(pool,sessions);pos={'D1':b['d1'].predict_proba(b['vec'].transform(qs))[:,1],'D2':b['d2'].predict_proba(b['ec'].get(qs))[:,1]}
    for L in CFG['L_grid']:
        ai=C.audit_items(pool,sessions,L)
        if L in (4,8):pos[f'D3_L{L}']=C.score_seq(b['seq'][f'D3_L{L}'],b['ec'],ai,L)
        pos[f'D4_L{L}']=C.score_seq(b['seq'][f'D4_L{L}'],b['ec'],ai,L)
    met={}
    met['D1_L1']=C.roc_metrics(pos['D1'],neg['D1']);met['D2_L1']=C.roc_metrics(pos['D2'],neg['D2']);met['D4_L1']=C.roc_metrics(pos['D4_L1'],neg['D4_L1'])
    for L in (4,8):
        met[f'D3_L{L}']=C.roc_metrics(pos[f'D3_L{L}'],neg[f'D3_L{L}']);met[f'D4_L{L}']=C.roc_metrics(pos[f'D4_L{L}'],neg[f'D4_L{L}'])
    return pos,met

def score_method(bundle,real,pool,sessions,neg):
    # Development/calibration only: select the strongest detector identity by L.
    pos,met=score_detector_metrics(bundle,pool,sessions,neg);strong={}
    for L,ks in [(1,['D1_L1','D2_L1','D4_L1']),(4,['D3_L4','D4_L4']),(8,['D3_L8','D4_L8'])]:
        k=max(ks,key=lambda z:met[z]['AUROC_sym']);strong[str(L)]={'detector':k,'score_key':k.split('_L')[0] if k in ('D1_L1','D2_L1') else k,**met[k]}
    return strong,pos,met

def score_frozen_method(bundle,pool,sessions,neg,frozen_strongest):
    # Confirmatory only: evaluate the detector identities frozen before held-out.
    # Never re-select a strongest detector on held-out data.
    pos,met=score_detector_metrics(bundle,pool,sessions,neg);strong={}
    for L in ['1','4','8']:
        k=frozen_strongest[L]['detector']
        if k not in met: raise RuntimeError(f'frozen detector {k} missing for L={L}')
        strong[L]={'detector':k,'score_key':k.split('_L')[0] if k in ('D1_L1','D2_L1') else k,**met[k]}
    return strong,pos,met

def freeze_thresholds(neg):
    out={}
    for beta in CFG['beta_grid']:
        out[str(beta)]={}
        for k,v in neg.items():
            th,b=C.threshold_for_bdr(v,beta);out[str(beta)][k]={'threshold':th,'measured_BDR':b}
    return out

def choose_attackers(strong,met):
    stat=max(['D1','D2'],key=lambda k:met[k+'_L1']['AUROC_sym']);state=max(['D3_L4','D4_L4'],key=lambda k:met[k]['AUROC_sym']);return stat,state

def attack_eval(bundle,sessions,pool,Edesc,Esrc,Ewrong,tau,risk_tau,gap_tau,thresholds,stat,state,use_risk,canonical):
    ad=C.DNAAdapter();b=str(CFG['main_beta']);th1=thresholds[b][stat]['threshold'];th2=thresholds[b][state]['threshold'];hon=[];a1=[];a2=[];air1=[];air2=[]
    for s in sessions:
        ev=s['random_order'] if canonical else s['order']
        hon.append(lineage_phi(ad,ev,Edesc,Esrc,Ewrong,risk_tau if use_risk else None,gap_tau if use_risk else None,tau))
        idx=s['order'];texts=[pool[i]['text'] for i in idx];sc=bundle['d1'].predict_proba(bundle['vec'].transform(texts))[:,1] if stat=='D1' else bundle['d2'].predict_proba(bundle['ec'].get(texts))[:,1];route=np.asarray(sc)>=th1;mix=Edesc.copy();sel=np.asarray(idx,int);mix[sel[route]]=Ewrong[sel[route]];a1.append(lineage_phi(ad,ev,mix,Esrc,Ewrong,risk_tau if use_risk else None,gap_tau if use_risk else None,tau));air1.append(float(route.mean()))
        route=np.zeros(len(idx),bool)
        for t in range(3,len(idx)):
            scr=float(C.score_seq(bundle['seq'][state],bundle['ec'],[(texts[t-3:t+1],'x')],4)[0]);route[t]=scr>=th2
        mix=Edesc.copy();mix[sel[route]]=Ewrong[sel[route]];a2.append(lineage_phi(ad,ev,mix,Esrc,Ewrong,risk_tau if use_risk else None,gap_tau if use_risk else None,tau));air2.append(float(route.mean()))
    return {'honest':rates_phi(hon),'A1':{**rates_phi(a1),'AIR':float(np.mean(air1))},'A2':{**rates_phi(a2),'AIR':float(np.mean(air2))},'_rows':{'honest':hon,'A1':a1,'A2':a2,'AIR1':air1,'AIR2':air2}}

def mean_csvr(perdesc,cond):return float(np.mean([v[cond]['CSVR'] for v in perdesc.values()]))

def main():
    if '--selftest' in sys.argv:
        assert split_of('abc') in {'development','calibration','heldout'}
        # Synthetic JSONL verifies conversation-id routing before row parsing.
        import tempfile
        chosen_ids={}
        i=0
        while set(chosen_ids)!={'development','calibration','heldout'}:
            tid=f'selftest-{i}';sp=split_of(tid);chosen_ids.setdefault(sp,tid);i+=1
        with tempfile.NamedTemporaryFile(suffix='.jsonl',delete=False,mode='w') as tf: tmp=Path(tf.name)
        try:
            with open(tmp,'wt',encoding='utf-8') as f:
                for sp,tid in chosen_ids.items():
                    obj={'id':tid,'conversations':sum(([{'from':'human','value':f'Tell me a useful fact about number {j+10} please.'},{'from':'gpt','value':'Here is a concise useful answer for the synthetic test.'}] for j in range(10)),[])}
                    f.write(json.dumps(obj)+'\n')
            for sp in ['development','calibration','heldout']:
                ss,info=load_split(tmp,sp);assert info['json_parsed_target_lines']==1 and len(ss)==1 and len(ss[0]['texts'])==10
        finally:
            tmp.unlink(missing_ok=True)
        print('P7V2_SELFTEST_OK');return
    phases={};heldout_touched=False;log('=== StageP7v2 Revised-SLATE ShareGPT + Phi Family-B External Confirmatory ===');log(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    try:
        asset=resolve_traffic();assets={'sharegpt':str(asset),'sharegpt_sha256':sha_file(asset),'phi':str(PHI),'qwen_wrong':str(QWEN),'encoder':str(ENC),'mpnet':str(MPNET),'all_offline':True};dump('asset_inventory.json',assets)
        if not C.model_complete_path(PHI) or not C.model_complete_path(QWEN) or not C.model_complete_path(ENC) or not MPNET.exists():raise RuntimeError('required local model asset missing')
        phases['assets']='ok'
        dev_real,devinfo=load_split(asset,'development');cal_real,calinfo=load_split(asset,'calibration');sequence_eligibility(dev_real,cal_real)
        dev_pool=freeze_pool(dev_real,CFG['candidate_dev'],'P7V2_DEV_POOL');cal_pool=freeze_pool(cal_real,CFG['candidate_cal'],'P7V2_CAL_POOL')
        if len(dev_pool)<CFG['candidate_dev'] or len(cal_pool)<CFG['candidate_cal']:raise RuntimeError('insufficient ShareGPT dev/cal candidate pool')
        lora=resolve_existing_lora();dump('lora_training.json',lora);phases['family_b_descendants']='ok'
        dump('split_preconfirm.json',{'development':devinfo,'calibration':calinfo,'heldout':'NOT_JSON_PARSED_BEFORE_FREEZE','candidate_dev_sha':sha_text('\n'.join(x['id'] for x in dev_pool)),'candidate_cal_sha':sha_text('\n'.join(x['id'] for x in cal_pool))})
        allrec=[{**r,'pool':'dev'} for r in dev_pool]+[{**r,'pool':'cal'} for r in cal_pool]
        banks={k:generate_bank(k,kind,allrec) for k,kind in [('phi_source','phi_source'),('phi_nf4','phi_nf4'),('phi_lora','phi_lora'),('qwen_wrong','qwen_wrong')]};dump('response_bank_devcal.json',{k:{'n':len(v),'sha':sha_text('\n'.join(x['id']+'|'+x['response'] for x in v))} for k,v in banks.items()});phases['devcal_responses']='ok'
        E=embed_all(banks);nd=len(dev_pool);devE={k:v[:nd] for k,v in E.items()};calE={k:v[nd:] for k,v in E.items()}
        # Suspect-independent informativeness: enrolled-source dispersion only.
        rawd=1-C.cosrow(devE['phi_source'][:,:64],devE['qwen_wrong'][:,:64]);lo,hi=np.quantile(rawd,[.05,.95]);den=max(float(hi-lo),1e-8);Jd=np.clip((rawd-lo)/den,0,1);rawc=1-C.cosrow(calE['phi_source'][:,:64],calE['qwen_wrong'][:,:64]);Jc=np.clip((rawc-lo)/den,0,1)
        dump('evidence_model.json',{'definition':'cosdist(Phi source reference, Qwen source reference), first64 frozen response-embedding coordinates; no suspect/descendant response enters J_E','dev_q05':float(lo),'dev_q95':float(hi),'dev_mean':float(Jd.mean()),'cal_mean':float(Jc.mean())});phases['evidence']='ok'
        from sentence_transformers import SentenceTransformer
        mp=SentenceTransformer(str(MPNET),device='cuda:0');sched=C.build_scheduler(dev_real,dev_pool,cal_pool,mp)
        full_dev={str(l):C.make_sessions(Jd,l,CFG['dev_sessions_per_lambda'],CFG['main_q'],sched['states_dev'],sched['schedule'],'P7V2_DEV',True) for l in CFG['lambda_grid']};full_cal={str(l):C.make_sessions(Jc,l,CFG['cal_sessions_per_lambda'],CFG['main_q'],sched['states_cal'],sched['schedule'],'P7V2_CAL',True) for l in CFG['lambda_grid']};unwrap_dev=fixed_sessions(Jd,CFG['dev_sessions_per_lambda'],CFG['main_q'],'UD');unwrap_cal=fixed_sessions(Jc,CFG['cal_sessions_per_lambda'],CFG['main_q'],'UC')
        full_bundle=train_detector_bundle('FULL',dev_real,dev_pool,list(full_dev.values()),mp);unwrap_bundle=train_detector_bundle('UNWRAP',dev_real,dev_pool,[unwrap_dev],mp);full_neg=calibration_neg(full_bundle,cal_real,cal_pool,'FULL');unwrap_neg=calibration_neg(unwrap_bundle,cal_real,cal_pool,'UNWRAP');full_thr=freeze_thresholds(full_neg);unwrap_thr=freeze_thresholds(unwrap_neg)
        # Lineage thresholds: negative Qwen suspect vs Phi source; honest risk from both descendants. Uniform calibration sessions.
        ad=C.DNAAdapter();rng=np.random.default_rng(CFG['seed']+77);base=[rng.choice(len(cal_pool),CFG['main_q'],replace=False).tolist() for _ in range(CFG['lineage_cal_sessions'])]
        negrows=[lineage_phi(ad,x,calE['qwen_wrong'],calE['phi_source'],calE['qwen_wrong']) for x in base];honraw=[]
        for d in ['phi_nf4','phi_lora']:honraw += [lineage_phi(ad,x,calE[d],calE['phi_source'],calE['qwen_wrong']) for x in base]
        tau=float(np.quantile([r['sim_phi'] for r in negrows],.95,method='higher'));risk_tau=float(np.quantile([r['risk'] for r in honraw],.975,method='higher'));gap_tau=float(np.quantile([r['gap'] for r in honraw],.025,method='lower'))
        dump('lineage_thresholds.json',{'similarity_threshold':tau,'risk_threshold_kappa':risk_tau,'bank_gap_threshold':gap_tau,'negative_source':'Qwen3-4B-Base responses against Phi source','honest_descendants':['Phi NF4','Phi LoRA']});phases['lineage_thresholds']='ok'
        # Calibrate unwrapped and each full lambda.
        ustrong,upos,umet=score_method(unwrap_bundle,cal_real,cal_pool,unwrap_cal,unwrap_neg);ustat,ustate=choose_attackers(ustrong,umet);uper={}
        for d in ['phi_nf4','phi_lora']:uper[d]=attack_eval(unwrap_bundle,unwrap_cal,cal_pool,calE[d],calE['phi_source'],calE['qwen_wrong'],tau,risk_tau,gap_tau,unwrap_thr,ustat,ustate,False,False)
        uhon=mean_csvr(uper,'honest')
        perlam={}
        for lam in CFG['lambda_grid']:
            key=str(lam);strong,pos,met=score_method(full_bundle,cal_real,cal_pool,full_cal[key],full_neg);stat,state=choose_attackers(strong,met);pd={}
            for d in ['phi_nf4','phi_lora']:pd[d]=attack_eval(full_bundle,full_cal[key],cal_pool,calE[d],calE['phi_source'],calE['qwen_wrong'],tau,risk_tau,gap_tau,full_thr,stat,state,True,True)
            perlam[key]={'strongest':strong,'stateless':stat,'stateful_L4':state,'descendants':{d:{k:v for k,v in pd[d].items() if k!='_rows'} for d in pd},'mean_honest_CSVR':mean_csvr(pd,'honest')}
        # Post-revision development rule frozen before heldout: prioritize cover subject to honest retention; deterministic fallback.
        utility_target=(max(0.0,uhon-.05) if uhon>=.5 else .9*uhon)
        feasible=[]
        for l in CFG['lambda_grid']:
            x=perlam[str(l)];
            if x['strongest']['1']['AUROC_sym']<=.65 and x['strongest']['4']['AUROC_sym']<=.70 and x['mean_honest_CSVR']+1e-12>=utility_target:feasible.append(float(l))
        if feasible:chosen=min(feasible)
        else:
            def loss(l):
                x=perlam[str(l)];return max(max(0,x['strongest']['1']['AUROC_sym']-.65)/.35,max(0,x['strongest']['4']['AUROC_sym']-.70)/.30,max(0,utility_target-x['mean_honest_CSVR'])/max(.1,utility_target))
            chosen=min(CFG['lambda_grid'],key=lambda l:(loss(l),-perlam[str(l)]['strongest']['4']['BDR_at_AIR_0.8'],l))
        chosen_key=str(chosen);cf=perlam[chosen_key];fstat,fstate=cf['stateless'],cf['stateful_L4']
        freeze={'stage':'P7V2_REVISED_FREEZE_BEFORE_HELDOUT','method':'Revised SLATE','components':['traffic-constrained tilted sampling','fresh session randomization','history scheduler','evidence-order canonicalization','risk-aware abstention'],'private_consistency_headline':False,'theory_delta_sched_success_proxy':False,'family':'Phi-3-mini-4k-instruct','descendants':['NF4 runtime','OASST-dev LoRA'],'wrong_source':'Qwen3-4B-Base','traffic':'heegyu/ShareGPT_Vicuna_unfiltered_no_imsorry (JSONL)','traffic_file_sha256':CFG['sharegpt_expected_sha256'],'traffic_repo':CFG['sharegpt_repo'],'traffic_file':CFG['sharegpt_file'],'q':CFG['main_q'],'lambda':chosen,'lambda_rule':'smallest lambda satisfying L1 AUROC<=.65, L4 AUROC<=.70 and honest retention target; deterministic normalized-excess fallback','unwrapped_honest_cal_mean_csvr':uhon,'honest_retention_target':utility_target,'beta_grid':CFG['beta_grid'],'main_beta':CFG['main_beta'],'full_strongest_by_L':cf['strongest'],'unwrapped_strongest_by_L':ustrong,'full_stateless_attacker':fstat,'full_stateful_L4_attacker':fstate,'unwrap_stateless_attacker':ustat,'unwrap_stateful_L4_attacker':ustate,'full_detector_thresholds':full_thr,'unwrap_detector_thresholds':unwrap_thr,'lineage_thresholds':{'tau':tau,'risk_tau':risk_tau,'gap_tau':gap_tau},'evidence_scaling':{'q05':float(lo),'q95':float(hi)},'scheduler':{'semantic_clusters':CFG['semantic_clusters'],'length_bins':CFG['length_bins'],'length_quantiles':sched['length_quantiles']},'heldout_touched':False,'rule':CFG['confirmatory_rule']}
        dump('p7_freeze_manifest.json',freeze);dump('development_freeze_summary.json',{'unwrapped':{'strongest':ustrong,'descendants':{d:{k:v for k,v in uper[d].items() if k!='_rows'} for d in uper}},'per_lambda':perlam,'chosen_lambda':chosen,'feasible_lambdas':feasible});phases['freeze']='ok'
        # Hard transition: only now parse/access heldout content.
        heldout_touched=True;dump('status.json',{'status':'STAGEP7V2_RUNNING_CONFIRMATORY','phases':phases,'heldout_touched':True,'retuned':False})
        held_real,heldinfo=load_split(asset,'heldout')
        held_counts={f'held_L{L}':len(C.real_windows(held_real,L,10**9,'P7V2_HELD_ELIG_'+str(L))) for L in (4,8)}
        held_req={'held_L4':CFG['min_held_windows_L4'],'held_L8':CFG['min_held_windows_L8']}
        held_bad={k:(held_counts[k],v) for k,v in held_req.items() if held_counts[k]<v}
        dump('heldout_traffic_eligibility.json',{'counts':held_counts,'requirements':held_req,'pass':not bool(held_bad),'failures':held_bad})
        if held_bad:raise RuntimeError('HELDOUT_TRAFFIC_STATEFUL_INELIGIBLE: '+repr(held_bad))
        held_pool=freeze_pool(held_real,CFG['candidate_heldout'],'P7V2_HELD_POOL')
        if len(held_pool)<CFG['candidate_heldout']:raise RuntimeError('insufficient heldout pool')
        dump('heldout_pool.json',{'heldout':heldinfo,'candidate_n':len(held_pool),'candidate_sha':sha_text('\n'.join(x['id'] for x in held_pool))})
        hbanks={k:generate_bank('held_'+k,kind,[{**r,'pool':'heldout'} for r in held_pool]) for k,kind in [('phi_source','phi_source'),('phi_nf4','phi_nf4'),('phi_lora','phi_lora'),('qwen_wrong','qwen_wrong')]};HE=embed_all(hbanks);phases['heldout_responses']='ok'
        rawh=1-C.cosrow(HE['phi_source'][:,:64],HE['qwen_wrong'][:,:64]);Jh=np.clip((rawh-lo)/den,0,1);hst=sched['state_fn']([r['text'] for r in held_pool]);full_held=C.make_sessions(Jh,chosen,CFG['heldout_sessions'],CFG['main_q'],hst,sched['schedule'],'P7V2_HELD',True);unwrap_held=fixed_sessions(Jh,CFG['heldout_sessions'],CFG['main_q'],'UH')
        # Heldout benign negatives and metrics with frozen detectors.
        def held_neg(bundle,tag):
            negq=C.excluded_queries(held_real,[r['text'] for r in held_pool],6000,'P7V2_HELD_'+tag+'_NEG');out={'D1':bundle['d1'].predict_proba(bundle['vec'].transform(negq))[:,1],'D2':bundle['d2'].predict_proba(bundle['ec'].get(negq))[:,1]}
            for L in CFG['L_grid']:
                nw=C.real_windows(held_real,L,CFG['detector_window_cap'],'P7V2_HELD_'+tag+'_W'+str(L))
                if L in (4,8):out[f'D3_L{L}']=C.score_seq(bundle['seq'][f'D3_L{L}'],bundle['ec'],nw,L)
                out[f'D4_L{L}']=C.score_seq(bundle['seq'][f'D4_L{L}'],bundle['ec'],nw,L)
            return out
        fneg=held_neg(full_bundle,'FULL');uneg=held_neg(unwrap_bundle,'UNWRAP')
        fstrong,fpos,fmet=score_frozen_method(full_bundle,held_pool,full_held,fneg,cf['strongest'])
        ustrong_h,upos_h,umet_h=score_frozen_method(unwrap_bundle,held_pool,unwrap_held,uneg,ustrong)
        fdesc={};udesc={}
        for d in ['phi_nf4','phi_lora']:
            fdesc[d]=attack_eval(full_bundle,full_held,held_pool,HE[d],HE['phi_source'],HE['qwen_wrong'],tau,risk_tau,gap_tau,full_thr,fstat,fstate,True,True)
            udesc[d]=attack_eval(unwrap_bundle,unwrap_held,held_pool,HE[d],HE['phi_source'],HE['qwen_wrong'],tau,risk_tau,gap_tau,unwrap_thr,ustat,ustate,False,False)
        # Confirmatory gate assessment, same numerical targets as preregistration but transparently post-revision.
        abs_cover=(fstrong['1']['AUROC_sym']<=.65 and fstrong['4']['AUROC_sym']<=.70 and fstrong['8']['AUROC_sym']<=.70);rel_cover=all(ustrong_h[L]['AUROC_sym']-fstrong[L]['AUROC_sym']>=.15 for L in ['1','4','8']);G2=bool(abs_cover or rel_cover)
        g3_gain=float(fstrong['4']['BDR_at_AIR_0.8']-ustrong_h['4']['BDR_at_AIR_0.8']);G3=g3_gain>=.20
        g4rows={}
        for d in fdesc:
            fv=fdesc[d]['honest']['CSVR'];uv=udesc[d]['honest']['CSVR'];formal=(fv>=uv-.05) if uv>=.5 else (fv+1e-12>=.9*uv);g4rows[d]={'full':fv,'unwrapped':uv,'formal_pass':bool(formal),'A0_informative':bool(uv>=.25)}
        G4_formal=all(x['formal_pass'] for x in g4rows.values())
        G4_informative=all(x['A0_informative'] for x in g4rows.values())
        fh=mean_csvr(fdesc,'honest');g5rows={}
        for a in ['A1','A2']:
            fv=mean_csvr(fdesc,a);uv=mean_csvr(udesc,a);gain=fv-uv;retain=(fv/(fh+1e-12)) if fh>0 else 0;g5rows[a]={'full':fv,'unwrapped':uv,'gain':gain,'retention_of_full_honest':retain,'pass':bool(gain>=.15 and fv+1e-12>=.75*fh)}
        G5=all(x['pass'] for x in g5rows.values())
        def strip_rows(x):return {d:{k:v for k,v in z.items() if k!='_rows'} for d,z in x.items()}
        confirm={'status':'STAGEP7V2_COMPLETE','development_or_confirmatory':'new external confirmatory after disclosed method revision','post_revision':True,'retuned_after_heldout':False,'heldout_touched':True,'new_model_family':'Phi-3-mini-4k-instruct','new_traffic_corpus':'ShareGPT JSONL (heegyu/ShareGPT_Vicuna_unfiltered_no_imsorry)','descendants':['NF4 runtime','OASST-development LoRA frozen before ShareGPT confirmatory design'],'chosen_lambda':chosen,'full_strongest_by_L':fstrong,'unwrapped_strongest_by_L':ustrong_h,'full_descendants':strip_rows(fdesc),'unwrapped_descendants':strip_rows(udesc),'gates':{'G2':{'pass':G2,'absolute_branch':abs_cover,'relative_branch':rel_cover},'G3':{'pass':G3,'L4_BDR_at_AIR08_gain':g3_gain},'G4':{'pass':G4_formal,'scientifically_informative':G4_informative,'per_descendant':g4rows},'G5':{'pass':G5,'full_honest_mean':fh,'attacks':g5rows},'G6':'not re-tested; private consistency remains demoted'},'claim_rule':'This result is independent confirmation of the disclosed revised compiler, not a continuation of the original preregistered confirmatory run.'}
        # Bootstrap CIs on main rates (descriptive).
        ci={}
        for method,D in [('full',fdesc),('unwrapped',udesc)]:
            for cond in ['honest','A1','A2']:
                vals=[]
                for d,z in D.items():vals.extend([1.0 if r['verdict']=='Verified' and r['best']=='Phi-3-mini-4k-instruct' else 0.0 for r in z['_rows'][cond]])
                ci[f'{method}_{cond}_CSVR']=C.ci_binary(vals)
        confirm['bootstrap_CI']=ci;dump('confirmatory_results.json',confirm);dump('paper_result_summary.json',confirm);phases['confirmatory']='ok';dump('status.json',{'status':'STAGEP7V2_COMPLETE','phases':phases,'heldout_touched':True,'retuned':False,'new_model_generation':True});log(json.dumps(confirm,indent=2))
    except Exception as e:
        dump('status.json',{'status':'STAGEP7V2_FAILED','error':repr(e),'traceback':traceback.format_exc(),'phases':phases,'heldout_touched':heldout_touched,'retuned':False});dump('paper_result_summary.json',{'status':'FAILED','error':repr(e),'heldout_touched':heldout_touched,'retuned':False});raise

if __name__=='__main__':main()
