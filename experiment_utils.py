"""Data preparation, metadata encoding and common full-catalog evaluation."""
from pathlib import Path
import ast
import gzip
import hashlib
import json
import time
import urllib.request
import numpy as np
import pandas as pd
from scipy import sparse

SOURCE = 'https://amazon-reviews-2023.github.io/'
REVISION = '2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e'
BASE = 'https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/' + REVISION + '/raw/'
URLS = {
    'reviews': BASE+'review_categories/Video_Games.jsonl',
    'metadata': BASE+'meta_categories/meta_Video_Games.jsonl',
}

def download_data(root='data'):
    root = Path(root); root.mkdir(exist_ok=True)
    manifest = {}
    for name, url in URLS.items():
        path = root / url.rsplit('/', 1)[-1]
        if not path.exists():
            print('Download:', url, flush=True)
            temp = path.with_suffix('.part')
            for attempt in range(3):
                try:
                    last_report = [-1]
                    def progress(blocks, block_size, total):
                        mb = blocks * block_size // (64*1024*1024)
                        if mb > last_report[0]:
                            print(name, 'downloaded MB:', blocks*block_size//(1024*1024), flush=True)
                            last_report[0] = mb
                    urllib.request.urlretrieve(url, temp, reporthook=progress)
                    temp.replace(path)
                    break
                except Exception:
                    if attempt == 2: raise
                    time.sleep(2)
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''): digest.update(chunk)
        manifest[name] = {'url': url, 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
    return root, manifest

def records(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf8') as f:
        for line in f:
            try: yield json.loads(line)
            except json.JSONDecodeError: yield ast.literal_eval(line)

def load_reviews(root):
    def rows():
        for j, r in enumerate(records(Path(root)/'Video_Games.jsonl')):
            if j % 500_000 == 0: print('Parsed reviews:', j, flush=True)
            yield {'reviewerID':r.get('user_id'), 'asin':r.get('parent_asin'),
                   'overall':r.get('rating'),
                   'unixReviewTime':None if r.get('timestamp') is None else r['timestamp']//1000}
    return pd.DataFrame(rows())

def prepare(raw, max_items=6000, min_history=5):
    fields = ['reviewerID', 'asin', 'overall', 'unixReviewTime']
    clean = raw.dropna(subset=fields).copy()
    clean = clean[clean.overall.between(1, 5) & (clean.unixReviewTime > 0)]
    clean = clean.sort_values(['unixReviewTime', 'reviewerID', 'asin'], kind='stable')
    clean = clean.drop_duplicates(['reviewerID', 'asin'], keep='first')
    # Boundaries use only timestamps, never future ratings or user activity.
    t1, t2 = np.quantile(clean.unixReviewTime, [.8, .9]).astype(np.int64)
    prefix = clean[clean.unixReviewTime <= t1]
    counts = prefix[prefix.overall >= 4].groupby('asin').size().rename('count').reset_index()
    catalog = counts[counts['count'] >= 5].sort_values(['count','asin'], ascending=[False,True]).head(max_items).asin
    history = prefix[prefix.asin.isin(catalog)]
    users = history.groupby('reviewerID').size()
    active = set(users[users >= min_history].index) & set(history[history.overall >= 4].reviewerID)
    selected = clean[clean.asin.isin(catalog) & clean.reviewerID.isin(active)].copy()
    user_names = sorted(active)
    item_names = sorted(catalog)
    umap, imap = {u:j for j,u in enumerate(user_names)}, {i:j for j,i in enumerate(item_names)}
    selected['u'] = selected.reviewerID.map(umap).astype(int)
    selected['i'] = selected.asin.map(imap).astype(int)
    # Drop catalog items with no observations after user filtering using train alone.
    used = sorted(selected[selected.unixReviewTime <= t1].i.unique())
    selected = selected[selected.i.isin(used)].copy()
    item_names = [item_names[j] for j in used]
    imap = {i:j for j,i in enumerate(item_names)}
    selected['i'] = selected.asin.map(imap).astype(int)
    assert 100_000 <= len(selected) <= 2_000_000, f'Invalid size: {len(selected)}'
    assert len(user_names) >= 1000 and len(item_names) >= 1000
    train = selected[selected.unixReviewTime <= t1].copy()
    val = selected[(selected.unixReviewTime > t1) & (selected.unixReviewTime <= t2)].copy()
    test = selected[selected.unixReviewTime > t2].copy()
    assert train.unixReviewTime.max() < val.unixReviewTime.min()
    assert val.unixReviewTime.max() < test.unixReviewTime.min()
    audit = pd.DataFrame([
        ['source_raw', len(raw),raw.reviewerID.nunique(),raw.asin.nunique()],
        ['clean',len(clean),clean.reviewerID.nunique(),clean.asin.nunique()],
        ['selected',len(selected),len(user_names),len(item_names)],
        ['train',len(train),train.u.nunique(),train.i.nunique()],
        ['validation',len(val),val.u.nunique(),val.i.nunique()],
        ['test',len(test),test.u.nunique(),test.i.nunique()],
    ], columns=['stage','interactions','users','items'])
    split_audit = []
    for label, frame in [('validation',clean[(clean.unixReviewTime>t1)&(clean.unixReviewTime<=t2)]),('test',clean[clean.unixReviewTime>t2])]:
        split_audit.append({'split':label, 'source_events':len(frame),
                            'unknown_user_events':int((~frame.reviewerID.isin(active)).sum()),
                            'outside_catalog_events':int((~frame.asin.isin(item_names)).sum()),
                            'retained_events':len(val if label=='validation' else test)})
    return selected, train, val, test, user_names, item_names, (int(t1),int(t2)), audit, pd.DataFrame(split_audit)

def matrix(frame, shape):
    return sparse.csr_matrix((frame.overall.to_numpy(float),(frame.u,frame.i)),shape=shape)

def load_metadata(root, items):
    wanted = set(items); rows = {}
    for r in records(Path(root)/'meta_Video_Games.jsonl'):
        if r.get('parent_asin') in wanted and r['parent_asin'] not in rows:
            category = r.get('categories') or []
            if category and isinstance(category[0], list): category = category[0]
            rows[r['parent_asin']] = {'asin':r['parent_asin'], 'title':r.get('title') or '',
                'brand':str((r.get('details') or {}).get('Brand') or (r.get('details') or {}).get('Manufacturer') or r.get('store') or ''), 'categories':category}
    return pd.DataFrame([rows.get(i, {'asin':i,'title':'','brand':'','categories':[]}) for i in items])

def encode_metadata(metadata):
    # Each item: own ID, up to two category levels, brand; 0 is padding.
    n = len(metadata); vocabulary = {}; features = np.zeros((n,4),dtype=np.int64)
    features[:,0] = np.arange(1,n+1)
    for i,r in metadata.iterrows():
        cats = [str(x) for x in r.categories if str(x) != 'Video Games'][-2:]
        tokens = ['category:'+x for x in cats] + (['brand:'+r.brand] if r.brand else [])
        for j,token in enumerate(tokens[:3],1):
            if token not in vocabulary: vocabulary[token] = n+len(vocabulary)+1
            features[i,j] = vocabulary[token]
    return features,n+len(vocabulary)

def evaluation_users(heldout, seen, k=10):
    truth = heldout[heldout.overall>=4].groupby('u').i.apply(lambda x:np.array(sorted(set(x)), dtype=int)).to_dict()
    eligible = [u for u in sorted(truth) if seen.shape[1]-seen[u].nnz >= k]
    return np.array(eligible,dtype=int), truth

def recommend(model, users, seen, k=10, batch_size=128):
    result = []
    for start in range(0,len(users),batch_size):
        batch = users[start:start+batch_size]
        scores = np.asarray(model.score(batch), dtype=np.float64)
        assert scores.shape == (len(batch),seen.shape[1]) and np.isfinite(scores).all()
        for j,u in enumerate(batch):
            scores[j,seen[u].indices] = -np.inf
            # Stable sorting breaks ties by ascending catalog index / lexicographic ASIN.
            top = np.argsort(-scores[j],kind='stable')[:k]
            assert np.isfinite(scores[j,top]).all()
            result.append(top)
    return np.asarray(result,dtype=int)

def evaluate(model, users, truth, seen):
    start = time.perf_counter(); recs = recommend(model,users,seen)
    rows = []
    discounts = 1/np.log2(np.arange(2,12))
    for u,top in zip(users,recs):
        relevant = truth[u]
        assert not set(top) & set(seen[u].indices)
        hits = np.isin(top,relevant)
        rows.append((u,float((hits*discounts).sum()/discounts[:min(10,len(relevant))].sum()),float(hits.sum()/len(relevant))))
    per_user = pd.DataFrame(rows,columns=['user','NDCG@10','Recall@10'])
    return {'NDCG@10':per_user['NDCG@10'].mean(),'Recall@10':per_user['Recall@10'].mean(),
            'Coverage@10':len(np.unique(recs))/seen.shape[1], 'recommend_seconds':time.perf_counter()-start}, per_user, recs

def bootstrap_difference(first, second, seed=42, repeats=2000):
    a=first.set_index('user')['NDCG@10']; b=second.set_index('user')['NDCG@10']
    assert a.index.equals(b.index)
    diff=(a-b).to_numpy(); rng=np.random.default_rng(seed)
    samples=np.array([rng.choice(diff,len(diff),replace=True).mean() for _ in range(repeats)])
    return {'mean_difference':diff.mean(),'ci_low':np.quantile(samples,.025),'ci_high':np.quantile(samples,.975)}
