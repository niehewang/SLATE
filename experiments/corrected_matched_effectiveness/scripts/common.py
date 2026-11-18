from __future__ import annotations
import hashlib, json, os, shutil, tempfile, time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
CFG = json.load(open(PKG/'config/protocol_config.json', encoding='utf-8'))
PCFG = json.load(open(PKG/'config/project.json', encoding='utf-8'))

def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')

def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def sha_file(p: Path) -> str:
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def atomic_json(path: Path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.', dir=str(path.parent))
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(obj,f,indent=2,ensure_ascii=False,default=str); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        try: os.unlink(tmp)
        except FileNotFoundError: pass

def load_json(path: Path, default=None):
    try: return json.load(open(path,encoding='utf-8'))
    except Exception: return default

def update_status(run_dir: Path, **kw):
    p=Path(run_dir)/'RUN_STATUS.json'; d=load_json(p,{}) or {}
    d.update(kw); d['updated_at']=now(); atomic_json(p,d); return d

def valid_text(t):
    t=str(t).strip(); n=len(t.split())
    return 4<=n<=256 and 12<=len(t)<=2400

def h64(x,salt=''):
    return int.from_bytes(hashlib.sha256((salt+'\0'+str(x)).encode()).digest()[:8],'big')

def query_id(salt,cid,turn,text):
    return sha_text(f'{salt}|{cid}|{turn}|{text}')[:20]

def load_traffic(path):
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def sessions_for_split(path, split, salt):
    import collections
    by=collections.defaultdict(list)
    for r in load_traffic(path):
        if r.get('split')!=split: continue
        t=str(r.get('text','')).strip()
        if valid_text(t): by[str(r['conversation_id'])].append((int(r.get('turn_index',0)),t))
    out=[]
    for cid,z in by.items():
        z=sorted(z)
        if z: out.append({'id':cid,'turns':z,'texts':[x[1] for x in z]})
    out.sort(key=lambda x:h64(x['id'],salt))
    return out

def freeze_pool(sessions,n,salt):
    seen=set(); arr=[]
    for s in sessions:
        for ti,t in s['turns']:
            if t in seen: continue
            seen.add(t)
            arr.append({'id':query_id(salt,s['id'],ti,t),'conversation_id':s['id'],'turn_index':ti,'text':t})
    arr.sort(key=lambda x:h64(x['id'],salt))
    return arr[:n]

def jsonl_write(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n')

def tree_manifest(root: Path, exclude=()):
    root=Path(root); rows=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file(): continue
        rel=str(p.relative_to(root))
        if any(rel==x or rel.startswith(x.rstrip('/')+'/') for x in exclude): continue
        rows.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha_file(p)})
    return rows

def disk_free_gb(path):
    return shutil.disk_usage(path).free/(1024**3)
