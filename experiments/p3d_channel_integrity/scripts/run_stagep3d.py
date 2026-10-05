from __future__ import annotations
import collections, hashlib, json, sys, time, traceback
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
CFG=json.load(open(ROOT/'config/project.json'))
RUN=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep3d_llmdna_channel_integrity_v1'
RET=RUN/'return'; RET.mkdir(parents=True,exist_ok=True)
LOG=RUN/'run.log'
P3C_RUN=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep3c_llmdna_fullslate_heldout_confirmatory_v1'
P3C_RET=P3C_RUN/'return'; P3C_BANK=P3C_RUN/'banks'; P3C_EMB=P3C_RUN/'embeddings'
P3B_RUN=Path.home()/'SLATE_TDSC_SERVER_SHARED/runs/slate_stagep3b_llmdna_fullslate_devcal_freeze_v1'
P3B_RET=P3B_RUN/'return'

sys.path.insert(0,str(ROOT/'scripts'))
import p3c_helpers as H
C=H.C
# Send helper audits to P3D return/log rather than mutating P3C return.
H.RET=RET; H.RUN=RUN; H.LOG=LOG

P3C_EXPECTED={
 'status.json':'cc831ad54ce15dbbdbaee6f84f8bc12f313fb509b955e947818dc3361f0c9405',
 'summary.json':'fd56d9a437776343010f3f0e698ca2588d36f14c7aeba26d7604ed5da506f18b',
 'heldout_pool.json':'659c11e68ea29c699a5702ebaf37b79eeac2258daa8b4e9997ee53c246bfb0ec',
 'heldout_response_bank_summary.json':'c014266fe0ee6cf4a3cef9540e81ec1ae43e9b474db888850582493a147583ff',
 'paper_result_summary.json':'9770fb11b8162bc5f2cf87c9a3aee44fb53d1b4c934f53ea827fa8cbeef8bb23'
}

def log(x):
    s=str(x); print(s,flush=True)
    with open(LOG,'a',encoding='utf-8') as f:f.write(s+'\n')

def load(p): return json.load(open(p,encoding='utf-8'))
def dump(name,obj):
    p=RET/name if isinstance(name,str) else Path(name); p.parent.mkdir(parents=True,exist_ok=True)
    json.dump(obj,open(p,'w',encoding='utf-8'),indent=2,ensure_ascii=False,default=str)
def semhash_obj(o): return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def semhash_file(p): return semhash_obj(load(p))
def sha(s): return hashlib.sha256(str(s).encode()).hexdigest()

def selftest():
    H.selftest()
    assert CFG['main_q']==64 and abs(float(CFG['main_beta'])-.1)<1e-12
    for fn,h in P3C_EXPECTED.items():
        p=ROOT/'frozen_p3c'/fn; assert p.exists(),fn; assert semhash_file(p)==h,(fn,semhash_file(p),h)
    print('P3D_SELFTEST_OK')

def verify_p3c():
    st=load(P3C_RET/'status.json')
    if st.get('status')!='STAGEP3C_COMPLETE' or st.get('heldout_touched') is not True or st.get('retuned') is not False:
        raise RuntimeError('P3C not a valid frozen confirmatory run: '+json.dumps(st))
    aud={}
    for fn,h in P3C_EXPECTED.items():
        sp=P3C_RET/fn; ep=ROOT/'frozen_p3c'/fn
        if not sp.exists(): raise RuntimeError('missing P3C artifact '+str(sp))
        hs,he=semhash_file(sp),semhash_file(ep); ok=(hs==he==h)
        aud[fn]={'server':hs,'embedded':he,'expected':h,'match':ok}
        if not ok: raise RuntimeError('P3C semantic hash mismatch '+fn)
    dump('p3c_freeze_audit.json',{'status':'PASS','artifacts':aud,'retuning_allowed':False,'new_heldout_selection':False})
    return load(P3B_RET/'p3c_freeze_manifest.json'), load(P3C_RET/'heldout_pool.json'), load(P3C_RET/'heldout_response_bank_summary.json')

def load_p3b_embeddings(tag, records):
    p=P3B_RUN/'embeddings'/f'{tag}.npz'
    if not p.exists(): raise RuntimeError('missing P3B embedding '+str(p))
    z=np.load(p,allow_pickle=False); ids=list(z['ids'].astype(str)); E=np.asarray(z['E'],np.float32); mp={i:E[k] for k,i in enumerate(ids)}
    miss=[r['id'] for r in records if r['id'] not in mp]
    if miss: raise RuntimeError(f'P3B embedding missing {len(miss)} {tag}')
    return np.stack([mp[r['id']] for r in records])

def load_p3c_bank_and_embedding(tag,hpool,expected_summary):
    bp=P3C_BANK/f'{tag}.json'; ep=P3C_EMB/f'{tag}.npz'
    if not bp.exists() or not ep.exists(): raise RuntimeError('missing P3C retained bank/embedding '+tag)
    state=load(bp); rm={r['id']:r for r in state.get('rows',[])}
    rows=[]
    for r in hpool:
        if r['id'] not in rm: raise RuntimeError('P3C bank id missing '+tag+' '+r['id'])
        rows.append(rm[r['id']])
    hh=sha('\n'.join(r['id']+'|'+r['response'] for r in rows))
    if hh!=expected_summary[tag]['sha256']: raise RuntimeError('P3C bank hash mismatch '+tag)
    z=np.load(ep,allow_pickle=False); ids=list(z['ids'].astype(str)); E=np.asarray(z['E'],np.float32); em={i:E[k] for k,i in enumerate(ids)}
    miss=[r['id'] for r in hpool if r['id'] not in em]
    if miss: raise RuntimeError('P3C embedding id missing '+tag)
    return rows,np.stack([em[r['id']] for r in hpool])

def calibration_bundle(fm):
    # Frozen rebuild, identical to P3C pre-heldout audit. No heldout data enters this function.
    evidence=load(P3B_RET/'evidence_model.json'); detfreeze=load(P3B_RET/'detector_freeze.json')
    bundle=H.rebuild_frozen_detectors(fm,detfreeze,evidence)
    records=bundle['dev_pool']+bundle['cal_pool']; ndev=len(bundle['dev_pool'])
    embs={k:load_p3b_embeddings(k,records) for k in ['qwen_source','qwen_nf4','phi_source']}
    calE={k:v[ndev:] for k,v in embs.items()}
    sess=C.make_sessions(bundle['Jc'],float(fm['lambda']),CFG['cal_sessions_per_lambda'],CFG['main_q'],bundle['sched']['states_cal'],bundle['sched']['schedule'],'CAL',True)
    ad=C.DNAAdapter(); risks=[]
    for s in sess:
        r=C.lineage_session(ad,s['order'],calE['qwen_nf4'],calE['qwen_source'],calE['phi_source'])
        risks.append(float(r['risk']))
    risks=np.asarray(risks,float)
    op=float(np.quantile(risks,.95,method='higher'))
    kappa=float(fm['risk_threshold_kappa'])
    info={'calibration_sessions':len(risks),'evaluation_only_threshold_5pct_far':op,'calibration_empirical_far_at_eval_threshold':float(np.mean(risks>=op)),'frozen_p3b_kappa':kappa,'calibration_far_at_frozen_kappa':float(np.mean(risks>kappa)),'rule':'5% FAR threshold is evaluation-only; main risk-aware verdict continues to use P3B frozen kappa.'}
    dump('consistency_operating_point.json',info)
    return bundle,ad,op,kappa

def heldout_pool_reconstruct(hp):
    hreal=H.heldout_sessions_all(); hpool=C.freeze_pool(hreal,CFG['candidate_heldout'],'P3C_HELDOUT_POOL')
    hh=sha('\n'.join(x['id'] for x in hpool))
    if hh!=hp['heldout_sha256'] or len(hpool)!=hp['heldout_n']: raise RuntimeError('P3C heldout pool reconstruction mismatch')
    return hreal,hpool

def metric(att,hon,op,kappa,reps=2000):
    att=np.asarray(att,float);hon=np.asarray(hon,float)
    y=np.r_[np.zeros(len(hon)),np.ones(len(att))]; sc=np.r_[hon,att]
    auc=float(roc_auc_score(y,sc))
    tpr5=float(np.mean(att>=op)); far5=float(np.mean(hon>=op)); tprk=float(np.mean(att>kappa)); fark=float(np.mean(hon>kappa))
    rng=np.random.default_rng(CFG['seed']+303); aucs=[]; tprs=[]
    for _ in range(reps):
        h=hon[rng.integers(0,len(hon),len(hon))]; a=att[rng.integers(0,len(att),len(att))]
        yy=np.r_[np.zeros(len(h)),np.ones(len(a))]; ss=np.r_[h,a]
        try: aucs.append(float(roc_auc_score(yy,ss)))
        except Exception: pass
        tprs.append(float(np.mean(a>=op)))
    return {'AUROC':auc,'AUROC_CI95':[float(np.quantile(aucs,.025)),float(np.quantile(aucs,.975))],'TPR_at_calibrated_5pct_FAR':tpr5,'TPR_at_5pct_FAR_CI95':[float(np.quantile(tprs,.025)),float(np.quantile(tprs,.975))],'heldout_honest_FAR_at_eval_threshold':far5,'TPR_at_frozen_kappa':tprk,'heldout_honest_FAR_at_frozen_kappa':fark,'mean_attack_score':float(att.mean()),'mean_honest_score':float(hon.mean())}

def sem_nearest_map(hpool,hstates,ec):
    texts=[r['text'] for r in hpool]; Q=np.asarray(ec.get(texts),float); Q/=np.linalg.norm(Q,axis=1,keepdims=True)+1e-12
    out={}
    for i in range(len(hpool)):
        same=np.flatnonzero(hstates==hstates[i]); same=same[same!=i]
        cand=same if len(same) else np.asarray([j for j in range(len(hpool)) if j!=i],int)
        sims=Q[cand]@Q[i]; best=float(np.max(sims)); tied=cand[np.flatnonzero(sims>=best-1e-12)]
        # deterministic tie break by stable id hash
        j=min(map(int,tied),key=lambda z:C.h64(hpool[z]['id'],'P3D_SEM_CACHE'))
        out[i]=j
    return out

def main():
    if '--selftest' in sys.argv: selftest(); return
    phases={}; log('=== StageP3D LLM-DNA Channel Integrity ==='); log(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    try:
        fm,hp,bank_summary=verify_p3c(); phases['p3c_freeze_audit']='ok'
        bundle,ad,op,kappa=calibration_bundle(fm); phases['calibration_freeze']='ok'
        hreal,hpool=heldout_pool_reconstruct(hp); phases['heldout_reconstruction']='ok'
        rows={}; E={}
        for tag in ['held_qwen_source','held_qwen_nf4','held_phi_source']:
            rows[tag],E[tag]=load_p3c_bank_and_embedding(tag,hpool,bank_summary)
        src,nf4,phi=E['held_qwen_source'],E['held_qwen_nf4'],E['held_phi_source']; phases['p3c_bank_reuse']='ok'

        lo=float(fm['evidence_scaling']['q05']); hi=float(fm['evidence_scaling']['q95']); den=max(hi-lo,1e-8)
        raw=(1-C.cosrow(src[:,:64],phi[:,:64]))-(1-C.cosrow(src[:,:64],nf4[:,:64])); Jh=np.clip((raw-lo)/den,0,1)
        hstates=bundle['sched']['state_fn']([r['text'] for r in hpool])
        hsess=C.make_sessions(Jh,float(fm['lambda']),CFG['heldout_sessions'],CFG['main_q'],hstates,bundle['sched']['schedule'],'HELDOUT_CONFIRM',True)
        p3cp=load(P3C_RET/'paper_result_summary.json')
        meanj=float(np.mean([s['mean_J'] for s in hsess]))
        if abs(meanj-float(p3cp['mean_J']))>1e-10: raise RuntimeError('P3C session reconstruction mean_J mismatch')

        tau=float(fm['lineage_similarity_threshold']); gap=float(fm['bank_gap_threshold'])
        honest_raw=[]; honest_v=[]
        for s in hsess:
            honest_raw.append(C.lineage_session(ad,s['order'],nf4,src,phi))
            honest_v.append(C.lineage_session(ad,s['order'],nf4,src,phi,kappa,gap,tau))
        honrisk=np.asarray([r['risk'] for r in honest_raw],float)

        # A2 frozen beta=.10 stateful router; no threshold changes.
        state=fm['stateful_attacker_L4']; th=float(fm['detector_thresholds'][str(CFG['main_beta'])][state]['threshold']); a2raw=[];a2v=[]
        for s in hsess:
            mix,_,_=C.attack_a2(s,hpool,bundle['seq'][state],bundle['ec'],4,th,nf4,phi)
            a2raw.append(C.lineage_session(ad,s['order'],mix,src,phi)); a2v.append(C.lineage_session(ad,s['order'],mix,src,phi,kappa,gap,tau))

        # A3 exact cache: same-query cached Qwen-source output replaces live descendant output.
        a3eraw=[];a3ev=[]
        for s in hsess:
            sus=nf4.copy(); idx=np.asarray(s['order'],int); sus[idx]=src[idx]
            a3eraw.append(C.lineage_session(ad,s['order'],sus,src,phi)); a3ev.append(C.lineage_session(ad,s['order'],sus,src,phi,kappa,gap,tau))

        # A3 semantic cache: nearest different query in the same frozen semantic/length state.
        smap=sem_nearest_map(hpool,hstates,bundle['ec']); a3sraw=[];a3sv=[]
        for s in hsess:
            sus=nf4.copy(); idx=np.asarray(s['order'],int)
            for i in idx: sus[i]=src[smap[int(i)]]
            a3sraw.append(C.lineage_session(ad,s['order'],sus,src,phi)); a3sv.append(C.lineage_session(ad,s['order'],sus,src,phi,kappa,gap,tau))

        # A4 full wrong-source/proxy.
        a4raw=[];a4v=[]
        for s in hsess:
            sus=nf4.copy(); idx=np.asarray(s['order'],int); sus[idx]=phi[idx]
            a4raw.append(C.lineage_session(ad,s['order'],sus,src,phi)); a4v.append(C.lineage_session(ad,s['order'],sus,src,phi,kappa,gap,tau))
        phases['attacks']='ok'

        risks={'A2':np.asarray([r['risk'] for r in a2raw]),'A3_exact':np.asarray([r['risk'] for r in a3eraw]),'A3_semantic':np.asarray([r['risk'] for r in a3sraw]),'A4':np.asarray([r['risk'] for r in a4raw])}
        metrics={k:metric(v,honrisk,op,kappa) for k,v in risks.items()}
        honest_far=float(np.mean(honrisk>kappa))
        passA2=metrics['A2']['AUROC']>=.80 and honest_far<=.05
        passA3=min(metrics['A3_exact']['AUROC'],metrics['A3_semantic']['AUROC'])>=.80 and honest_far<=.05
        passA4=metrics['A4']['AUROC']>=.80 and honest_far<=.05
        g6_count=int(passA2)+int(passA3)+int(passA4); g6=g6_count>=2
        verdicts={'honest':C.verdict_rates(honest_v),'A2':C.verdict_rates(a2v),'A3_exact':C.verdict_rates(a3ev),'A3_semantic':C.verdict_rates(a3sv),'A4':C.verdict_rates(a4v)}
        g={'criterion':'AUROC>=0.80 and heldout honest FAR at frozen P3B kappa <=0.05 in at least two of A2/A3/A4; A3 conservatively requires both exact and semantic variants.','A2_pass':passA2,'A3_pass':passA3,'A4_pass':passA4,'passing_families':g6_count,'G6_pass':g6,'honest_FAR_frozen_kappa':honest_far}
        dump('channel_integrity_metrics.json',{'private_consistency_score':'existing P3B lineage_session risk = MAD(query margins) + 2*mixture_fraction','metrics':metrics,'verdict_rates':verdicts,'G6':g,'semantic_cache_mapping_sha256':sha('\n'.join(f'{i}->{smap[i]}' for i in sorted(smap)))})
        paper={'stage':'P3D','development_or_confirmatory':'post-P3C heldout channel-integrity evaluation with calibration-frozen operating point','engine':'LLM-DNA/RepTrace 1.0.1','method_component':'private consistency check (auxiliary)','new_detector_added':False,'retuning':False,'response_generation':False,'reused_P3C_heldout_pool':True,'A2':metrics['A2'],'A3_exact':metrics['A3_exact'],'A3_semantic':metrics['A3_semantic'],'A4':metrics['A4'],'honest_FAR_frozen_kappa':honest_far,'verdict_rates':verdicts,'G6':g,'paper_action':'retain auxiliary private consistency only if G6 passes; otherwise demote/remove from headline and proceed to P4 SRP without rescue.'}
        dump('paper_result_summary.json',paper)
        summary={'workflow':CFG['workflow'],'status':'STAGEP3D_COMPLETE','G6_pass':g6,'passing_families':g6_count,'A2_AUROC':metrics['A2']['AUROC'],'A3_exact_AUROC':metrics['A3_exact']['AUROC'],'A3_semantic_AUROC':metrics['A3_semantic']['AUROC'],'A4_AUROC':metrics['A4']['AUROC'],'honest_FAR_frozen_kappa':honest_far,'next':'P4A/P4B SRP Full-SLATE. No private-consistency rescue if G6 fails.'}
        dump('summary.json',summary); phases['metrics']='ok'; dump('status.json',{'status':'STAGEP3D_COMPLETE','phases':phases,'retuned':False,'new_model_generation':False,'P3C_heldout_reused':True}); log(json.dumps(summary,indent=2))
    except Exception as e:
        dump('status.json',{'status':'STAGEP3D_FAILED','error':repr(e),'traceback':traceback.format_exc(),'phases':phases,'retuned':False,'new_model_generation':False})
        dump('paper_result_summary.json',{'stage':'P3D','status':'FAILED','paper_facing_claims':False,'error':repr(e)})
        raise

if __name__=='__main__': main()
