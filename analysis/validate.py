"""Independent checks of the new statistical code and persisted item-level results."""
from collections import defaultdict
from itertools import product
from pathlib import Path
import json
import math
import numpy as np
from scipy.stats import binomtest
import reproduce as pipeline

PAPER=Path(__file__).resolve().parent
checks=[]

def check(label,condition):
    checks.append({'check':label,'passed':bool(condition)})
    assert condition,label

def load_lines(p):
    with p.open(encoding='utf-8') as f:return [json.loads(l) for l in f if l.strip()]

# Exhaustive independent enumeration checks weighted sign-flip probabilities.
for d in [[0,0],[1,-1,0],[.5,.5,-1],[1,.5,-.5,0,1],[-1,-1,1,0],[-.5,.5,-.5]]:
    a=np.array(d);observed=abs(a.sum())
    exact=sum(abs(np.dot(a,s))>=observed-1e-12 for s in product([-1,1],repeat=len(d)))/(2**len(d))
    check('exact weighted sign flips '+str(d),abs(pipeline.exact_signflip(d)-exact)<1e-14)
check('Holm reference family',np.allclose(pipeline.holm([.01,.04,.03]),[.03,.06,.06]))

# Cases that caused the earlier stored-answer/strict-parser discrepancy.
for raw in ['{"answer":"A, C, G"}','{"answer":"A, G, H"}']:
    check('ambiguous option list rejected '+raw,pipeline.parse(raw,'mmlu_pro',[])=='')
check('unique explicit option',pipeline.parse('{"answer":"C"}','mmlu_pro',[])=='C')
check('option text matching',pipeline.parse('{"answer":"penetration pricing"}','mmlu_pro',['cost pricing','penetration pricing'])=='B')
check('GSM thousands formatting',pipeline.parse('{"answer":"240,000"}','gsm8k',[])=='240000')
check('GSM numeric grading',pipeline.hit('240000','240,000','gsm8k'))
check('empty prediction incorrect',not pipeline.hit('','0','gsm8k'))
check('no arbitrary MATH comma stripping',pipeline.normalized('(1,2)','math500')=='(1,2)')

R=json.loads((PAPER/'data/authoritative_v4.json').read_text(encoding='utf-8'))
items=load_lines(PAPER/'data/per_item_v4.jsonl')
base=load_lines(PAPER/'data/baseline_items_v4.jsonl')
check('raw record count equals manifest metadata',len(items)==sum(sum(m['records'].values()) for m in R['meta'].values()))
check('baseline record count equals result metadata',len(base)==sum(v['n'] for ds in R['baselines'].values() for v in ds.values()))
keys=[(r['dataset'],r['condition'],r['qid'],r['rep']) for r in items]
check('unique retained replicate keys',len(keys)==len(set(keys)))
check('repeated MMLU stem consistently excluded',not any(r['dataset']=='mmluhard' and r['qid']=='mmlu_pro:432' for r in items+base))
check('per-run accounting identity',all(abs(r['dA']-(r['regret_compression']-r['pool_loss']))<1e-12 for r in items))
check('union coverage bound',all(r['C_union']>=max(r['C_pre'],r['C_post']) for r in items))

# Independently aggregate correctness flags without pipeline aggregation helpers.
grouped=defaultdict(lambda:defaultdict(list))
for r in items:grouped[(r['dataset'],r['condition'])][r['qid']].append(r)
for (ds,c),qs in grouped.items():
    metric=R['accounting'][ds][c]
    for k in ['pre','post','C_pre','C_post','dA','pool_loss','readout_loss','regret_compression']:
        recomputed=sum(sum(row[k] for row in rows)/len(rows) for rows in qs.values())/len(qs)
        check(f'item-weighted mean {ds}/{c}/{k}',abs(recomputed-metric[k])<1e-12)

# Binary datasets give an independent exact binomial check of sign-flip p-values.
for ds in ['gsm8k','math500hard','mmluhard']:
    qs=grouped[(ds,'no_info')]
    differences=[rows[0]['guard']-rows[0]['post'] for rows in qs.values()]
    pos=sum(x>0 for x in differences);neg=sum(x<0 for x in differences)
    expected=binomtest(pos,pos+neg,p=.5).pvalue if pos+neg else 1.
    check('independent exact binomial guard p '+ds,abs(expected-R['guard'][ds]['test']['p'])<1e-14)
ps=[R['guard'][ds]['test']['p'] for ds in pipeline.DS]
order=np.argsort(ps)
expected=[0.]*len(ps)
for rank,i in enumerate(order):expected[i]=min(1.,max((len(ps)-j)*ps[k] for j,k in enumerate(order[:rank+1])))
check('primary five-test Holm family',np.allclose(expected,[R['guard'][ds]['holm_p'] for ds in pipeline.DS],atol=1e-15))

for ds in pipeline.DS:
    b=[r for r in base if r['dataset']==ds and r['model']=='qwen' and r['k']==9]
    ung=set(grouped[(ds,'no_info')]);covered={r['qid'] for r in b}
    check('full paired SC9 coverage '+ds,covered==ung)
    check('SC9 means '+ds,abs(sum(r['accuracy'] for r in b)/len(b)-R['baselines'][ds]['qwen_k9']['accuracy'])<1e-12)

manifest=json.loads((PAPER/'data/manifest_v4.json').read_text(encoding='utf-8'))
for group in ['input_hashes','implementation_hashes']:
    for name,digest in manifest[group].items():check('hash '+name,pipeline.sha(PAPER.parent/name)==digest)

reference=json.loads((PAPER.parent/'reference/authoritative_v4.json').read_text(encoding='utf-8'))
check('all numerical results equal the manuscript reference', R == reference)
for source in sorted((PAPER.parent/'reference/tables').glob('*.tex')):
    check('manuscript table '+source.name, source.read_text(encoding='utf-8') == (PAPER/'tables'/source.name).read_text(encoding='utf-8'))
check('all numerical prose macros match', (PAPER.parent/'reference/generated_numbers.tex').read_text(encoding='utf-8') == (PAPER/'data/generated_numbers.tex').read_text(encoding='utf-8'))
(PAPER/'audit').mkdir(exist_ok=True)
(PAPER/'audit/validation_v4.json').write_text(json.dumps({'passed':len(checks),'failed':0,'checks':checks},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'PASS {len(checks)} independent checks; FAIL 0')
