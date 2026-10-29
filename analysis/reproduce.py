"""Rebuild the v4 manuscript data from immutable raw outputs, without model calls.

python analysis/reproduce.py          # write results and generated tables
python analysis/reproduce.py --check  # independently recompute and compare all outputs
"""
from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_UP
import argparse
import hashlib
import json
import re
import sys
import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
RUNS = ROOT / 'data/raw_runs/runs'
CODE = ROOT / 'src'
sys.path.insert(0, str(CODE))
from lt1.data import normalize_answer, answers_equal
from lt1.debate import parse_response
from regrade_mmlu import extract_v3

VERSION = 'raw-v4-20261021'
B = 10000
SEED = 20261021
CONDS = ['no_info', 'expert_oracle_clean', 'expert_revealed_calib']
LABELS = ['Unguided', 'Ideal-info', 'Calibration']
DS = ['math500', 'mmlu_pro', 'gsm8k', 'math500hard', 'mmluhard']
NAMES = dict(zip(DS, ['MATH-500', 'MMLU-Pro', 'GSM8K', 'MATH-rep.', 'MMLU-rep.']))
FILES = dict(math500='v1/debates_math500.jsonl', mmlu_pro='v1/debates_mmlu_pro.jsonl',
             gsm8k='v4/debates_gsm8kdebate.jsonl', math500hard='v3/debates_math500ext.jsonl',
             mmluhard='v3/debates_mmluext.jsonl')
CAL = dict(math500='v1/calib_math500.json', mmlu_pro='v1/calib_mmlu_pro.json',
           gsm8k='v4/calib_gsm8kdebate.json', math500hard='v3/calib_math500ext.json',
           mmluhard='v3/calib_mmluext.json')
SPLITS = dict(math500='v1/split_math500.json', mmlu_pro='v1/split_mmlu_pro.json',
              gsm8k='v4/split_gsm8kdebate.json', math500hard='v3/split_math500ext.json',
              mmluhard='v3/split_mmluext.json')
INPUTS = set()

def load(p, lines=False):
    p=Path(p);INPUTS.add(p)
    with p.open(encoding='utf-8-sig') as f:
        return [json.loads(l) for l in f if l.strip()] if lines else json.load(f)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def prompt(r):
    p=r.get('prompt',r.get('question',''))
    if isinstance(p,list):p='\n'.join(x.get('value',x.get('content','')) for x in p)
    return ' '.join(str(p).split())

def normalized(a, ds):
    a=normalize_answer(a,'generic')
    if ds=='gsm8k' and re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?',a):
        a=a.replace(',','')
    return a

def parse(raw,ds,opts):
    if ds.startswith('mmlu'):
        # No first-character or last-letter fallback: ambiguous option lists abstain.
        return extract_v3(raw,opts)[0]
    return normalized(parse_response(raw)[0],ds)

def hit(a,g,ds):
    if ds.startswith('mmlu'):return bool(a and a==str(g).strip().upper())
    return answers_equal(a,normalized(g,ds),'generic')

def majority(v):
    c=Counter(a for a in v if a)
    return min(c,key=lambda a:(-c[a],a)) if c else ''

def guarded(entries,weights):
    """Each (phase, agent) contributes one vote; duplicate answers are NOT removed.

    Tie order: total support, post-phase support, largest member weight, lexical ASC.
    All weight comparisons use exact rational arithmetic.
    """
    sums=defaultdict(Fraction);post=defaultdict(Fraction);best=defaultdict(Fraction)
    for phase,agent,a in entries:
        if not a:continue
        w=weights[agent];sums[a]+=w;post[a]+=w*phase;best[a]=max(best[a],w)
    return min(sums,key=lambda a:(-sums[a],-post[a],-best[a],a)) if sums else ''

def exact_signflip(d):
    # All paired outcomes are means of one or two replicates, hence half-integers.
    z=np.asarray(d)*2
    assert np.allclose(z,np.rint(z)), 'Unexpected replicate denominator'
    v=np.rint(z).astype(int);weights=[abs(int(x)) for x in v if x]
    if not weights:return 1.
    counts={0:1}
    for w in weights:
        nxt=defaultdict(int)
        for s,n in counts.items():nxt[s]+=n;nxt[s+w]+=n
        counts=nxt
    total=sum(weights);observed=abs(int(v.sum()))
    return sum(n for s,n in counts.items() if abs(2*s-total)>=observed)/(2**len(weights))

def stats(d):
    d=np.asarray(d,float);assert len(d)>0
    rng=np.random.default_rng(SEED)
    ci=np.percentile(d[rng.integers(0,len(d),size=(B,len(d)))].mean(axis=1),[2.5,97.5])
    return {'n':len(d),'difference':float(d.mean()),'ci':[float(x) for x in ci],
            'p':float(exact_signflip(d))}

def holm(ps):
    order=sorted(range(len(ps)),key=lambda i:ps[i]);out=[0.]*len(ps);last=0.
    for j,i in enumerate(order):last=max(last,min(1.,ps[i]*(len(ps)-j)));out[i]=last
    return out

def mean_rows(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k],(float,int))}

def pooled_by_question(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['qid']].append(r)
    return {q:mean_rows(groups[q]) for q in sorted(groups)}

def compare(a,b,keya,keyb):
    qs=sorted(set(a)&set(b))
    s=stats([a[q][keya]-b[q][keyb] for q in qs])
    s.update(left=float(np.mean([a[q][keya] for q in qs])),
             right=float(np.mean([b[q][keyb] for q in qs])))
    return s

def compute():
    main_math=load(ROOT/'data/benchmarks/math500.json')
    ext_math=load(RUNS.parent/'_sources/math500_ext.json')
    main_mmlu=load(ROOT/'data/benchmarks/mmlu_pro.jsonl',True)
    ext_mmlu=load(RUNS.parent/'_sources/mmlu_pro_ext.jsonl',True)
    gsm_local=load(RUNS.parent/'_sources/gsm8k_debate.json')
    gsm_public=load(PAPER/'data/sources/gsm8k_test.jsonl',True)
    sources={'math500':{f'math500:{i:04d}':r for i,r in enumerate(main_math)},
             'math500hard':{f'math500:{i:04d}':r for i,r in enumerate(ext_math)},
             'mmlu_pro':{f"mmlu_pro:{r['question_id']}":r for r in main_mmlu},
             'mmluhard':{f"mmlu_pro:{r['question_id']}":r for r in ext_mmlu},
             'gsm8k':{f'math500:{i:04d}':r for i,r in enumerate(gsm_local)}}
    math_index={prompt(r):f'math500:{i:04d}' for i,r in enumerate(main_math)}
    gsm_index={prompt(r):f'gsm8k:{i:04d}' for i,r in enumerate(gsm_public)}
    mapping={ds:{q:(math_index[prompt(r)] if ds=='math500hard' else
                    gsm_index[prompt(r)] if ds=='gsm8k' else q) for q,r in src.items()}
             for ds,src in sources.items()}
    main_split_mmlu=load(RUNS/SPLITS['mmlu_pro'])
    main_stems={prompt(sources['mmlu_pro'][q]) for q in main_split_mmlu['calib_qids']+main_split_mmlu['eval_qids']}
    excluded_mmlu={q for q,r in sources['mmluhard'].items() if prompt(r) in main_stems}
    all_items=[];per={};summaries={};guard={};meta={};calib={}
    for ds in DS:
        recs=load(RUNS/FILES[ds],True);c=load(RUNS/CAL[ds]);split=load(RUNS/SPLITS[ds])
        calib[ds]=c
        weights={a:Fraction(round(v*c['n_calib']),c['n_calib']) for a,v in c['calib_acc'].items()}
        equal={a:Fraction(1) for a in weights}
        keys=[(r['condition'],r['qid'],r.get('rep',0)) for r in recs]
        assert len(keys)==len(set(keys)),f'Duplicate records: {ds}'
        assert not(set(split['calib_qids'])&set(split['eval_qids']))
        rows=defaultdict(list)
        for r in recs:
            if r['condition'] not in CONDS+['solo']:continue
            if ds=='mmluhard' and r['qid'] in excluded_mmlu:continue
            assert r['qid'] in split['eval_qids']
            source=sources[ds][r['qid']]
            source_gold=source.get('final_answer',source.get('answer'))
            assert hit(str(r['gold']).strip() if ds.startswith('mmlu') else normalized(r['gold'],ds),source_gold,ds),f'Gold/source mismatch {ds} {r["qid"]}'
            opts=source.get('options',[])
            last=max(t['round'] for t in r['turns'])
            pre={t['agent']:parse(t.get('raw',''),ds,opts) for t in r['turns'] if t['round']==0}
            post={t['agent']:parse(t.get('raw',''),ds,opts) for t in r['turns'] if t['round']==last}
            assert set(pre)==set(post)==set(weights),f'Incomplete turns: {ds}'
            entries=[(0,a,v) for a,v in pre.items()]+[(1,a,v) for a,v in post.items()]
            preds={'pre':majority(pre.values()),'post':majority(post.values()),
                   'union_equal':guarded(entries,equal),
                   'post_calibrated':guarded([e for e in entries if e[0]],weights),
                   'guard':guarded(entries,weights)}
            values={k:float(hit(v,r['gold'],ds)) for k,v in preds.items()}
            values.update(C_pre=float(any(hit(v,r['gold'],ds) for v in pre.values())),
                          C_post=float(any(hit(v,r['gold'],ds) for v in post.values())),
                          C_union=float(any(hit(e[2],r['gold'],ds) for e in entries)))
            values.update(pool_loss=values['C_pre']-values['C_post'],
                          readout_loss=values['C_post']-values['post'],
                          regret_pre=values['C_pre']-values['pre'])
            values['regret_compression']=values['regret_pre']-values['readout_loss']
            values['dA']=values['post']-values['pre']
            assert values['dA']==values['regret_compression']-values['pool_loss']
            assert values['C_union']>=max(values['C_pre'],values['C_post'])
            item={'dataset':ds,'condition':r['condition'],'qid':r['qid'],
                  'global_qid':mapping[ds][r['qid']],'rep':r.get('rep',0),
                  'gold':r['gold'],'pre_candidates':pre,'post_candidates':post,
                  'predictions':preds,**values}
            all_items.append(item);rows[r['condition']].append({'qid':r['qid'],**values})
        per[ds]={cond:pooled_by_question(rr) for cond,rr in rows.items()}
        summaries[ds]={}
        for cond,p in per[ds].items():
            mean=mean_rows(list(p.values()))
            if cond!='solo':
                mean['test']=stats([v['dA'] for v in p.values()])
                mean['pool_loss_test']=stats([v['pool_loss'] for v in p.values()])
                mean['compression_test']=stats([v['regret_compression'] for v in p.values()])
            mean.update(n=len(p),records=len(rows[cond]))
            summaries[ds][cond]=mean
        ung=per[ds]['no_info'];m=mean_rows(list(ung.values()))
        guard[ds]={k:m[k] for k in ['post','union_equal','post_calibrated','guard','C_pre','C_post','C_union']}
        guard[ds]['test']=compare(ung,ung,'guard','post')
        guard[ds]['union_gain']=m['union_equal']-m['post']
        guard[ds]['weight_gain']=m['guard']-m['union_equal']
        guard[ds]['union_share']=guard[ds]['union_gain']/guard[ds]['test']['difference']
        guard[ds]['union_ceiling_gain']=m['C_union']-m['C_post']
        meta[ds]={'calibration_n':len(split['calib_qids']),'evaluation_n':len(split['eval_qids']),
                  'excluded_repeated_stems':sorted(excluded_mmlu) if ds=='mmluhard' else [],
                  'actual_n':{cond:len(p) for cond,p in per[ds].items()},
                  'records':{cond:len(rr) for cond,rr in rows.items()},
                  'discussion_records':sum(len(rr) for cond,rr in rows.items() if cond!='solo'),
                  'solo_records':len(rows['solo'])}
    adjusted=holm([guard[ds]['test']['p'] for ds in DS])
    for ds,p in zip(DS,adjusted):guard[ds]['holm_p']=p
    increments={}
    for ds in DS:
        increments[ds]={}
        for cond in CONDS[1:]:
            increments[ds][cond]={'difference_in_changes':compare(per[ds][cond],per[ds]['no_info'],'dA','dA'),
                                 'post_accuracy_difference':compare(per[ds][cond],per[ds]['no_info'],'post','post')}
    # Reparse baseline samples under exactly the same answer rules.
    # Use archived calibration-selected Qwen; also report Llama/Mistral on main splits.
    baseline={ds:{} for ds in DS}
    baseline_items=[]
    for ds in DS:
        basekind='math500' if ds.startswith('math500') else 'mmlu_pro' if ds.startswith('mmlu') else 'gsm8k'
        candidates=[]
        if ds in ['math500','mmlu_pro']:
            for k in [3,9]:candidates.append((RUNS/f'v1/baseline_selfconsistency_{basekind}_k{k}.jsonl','qwen',k))
            for model in ['llama','mistral']:
                p=RUNS/f'v1/baseline_selfconsistency_{basekind}_k9_{model}.jsonl'
                if p.exists():candidates.append((p,model,9))
        else:
            main=RUNS/f'v1/baseline_selfconsistency_{basekind}_k9{ "_qwen" if ds=="gsm8k" else ""}.jsonl'
            candidates.append((main,'qwen',9))
            candidates += [(p,'qwen',9) for p in sorted((RUNS/'v2').glob(f'ext_{basekind}_*_k9_qwen.jsonl'))]
        reverse={mapping[ds][q]:q for q in per[ds]['no_info']}
        groups=defaultdict(dict)
        for path,model,k in candidates:
            for r in load(path,True):
                if r['qid'] not in reverse:continue
                local=reverse[r['qid']];src=sources[ds][local]
                assert hit(str(r['gold']).strip() if ds.startswith('mmlu') else normalized(r['gold'],ds),src.get('final_answer',src.get('answer')),ds)
                assert len(r['raw_samples'])==k
                aa=[parse(raw,ds,src.get('options',[])) for raw in r['raw_samples']]
                pred=majority(aa);v=float(hit(pred,r['gold'],ds));key=f'{model}_k{k}'
                assert local not in groups[key],f'Duplicate baseline {ds} {key} {local}'
                groups[key][local]={'accuracy':v}
                baseline_items.append({'dataset':ds,'model':model,'k':k,'qid':local,
                    'global_qid':r['qid'],'predicted':pred,'gold':r['gold'],'accuracy':v,
                    'answers':aa,'file':path.relative_to(ROOT).as_posix()})
        for key,p in groups.items():
            baseline[ds][key]={'n':len(p),'accuracy':float(np.mean([v['accuracy'] for v in p.values()])),
                'guard_comparison':compare(per[ds]['no_info'],p,'guard','accuracy'),
                'condition_comparisons':{c:compare(per[ds][c],p,'post','accuracy') for c in CONDS+['solo']}}
    # Secondary family: four MATH readouts x two independently generated SC budgets.
    family=[]
    for key in ['qwen_k3','qwen_k9']:
        for c in CONDS:
            family.append(baseline['math500'][key]['condition_comparisons'][c])
        family.append(baseline['math500'][key]['guard_comparison'])
    for s,p in zip(family,holm([s['p'] for s in family])):s['holm_p']=p
    for ds,hard in [('math500','math500hard'),('mmlu_pro','mmluhard')]:
        q1=set(per[ds]['no_info']);q2=set(per[hard]['no_info'])
        p1={prompt(sources[ds][q]) for q in q1};p2={prompt(sources[hard][q]) for q in q2}
        assert not p1&p2,'Main/replication question overlap'
    manifest={'version':VERSION,'bootstrap_replicates':B,'bootstrap_seed':SEED,
        'ordering':'lexical local qid after averaging available replicates within each item',
        'p_values':'exact two-sided paired sign-flip; zeros omitted; half-integer differences enumerated by dynamic programming',
        'guard_holm_family':[f'{ds}:guard-post_equal' for ds in DS],
        'math_sc_holm_family':'no_info, ideal-info, calibration, guard versus Qwen SC k3 and k9 (8 tests)',
        'guard_rule':'pre plus final vote multiset; one vote per agent per phase; archived calib_acc weights; tie by total support, post-phase support, largest individual weight, lexical ascending',
        'calibration':'Original archived calibration accuracies retained as fixed historical weights; no newly generated or retrospectively inferred calibration labels.',
        'grading':'MMLU extract_v3(raw, source options) only; generic parse_response + normalize_answer; GSM numeric thousands separators removed before voting and grading.',
        'baseline':'Main SC k3 and k9 use separate archived runs. Replication SC k9 uses prompt-to-original-ID mapping. GSM baseline compared only on mapped common items.',
        'gsm_source_url':'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl',
        'implementation_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),PAPER/'figure_templates/tug_of_war.tex',CODE/'regrade_mmlu.py',CODE/'lt1/data.py',CODE/'lt1/debate.py']},
        'input_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(INPUTS)}}
    result={'version':VERSION,'meta':meta,'calibration':calib,'accounting':summaries,
            'guard':guard,'guidance':increments,'baselines':baseline}
    return result,all_items,baseline_items,manifest

def pct(v,signed=False):
    """Format a stored rate or absolute difference on the 0--100 scale."""
    n=(Decimal(str(round(float(v),12)))*100).quantize(Decimal('.1'),rounding=ROUND_HALF_UP)
    return ('+' if signed and n>=0 else '')+str(n)
def pval(v):return '<0.001' if v<.001 else f'{v:.3f}'
def pct_ci(s):return '['+','.join(pct(x,True) for x in s['ci'])+']'
def row(cells):return ' & '.join(cells)+r' \\'
def table(caption,label,columns,header,rows,foot='',wide=False,group=''):
    env='table*' if wide else 'table'
    width=r'\textwidth' if wide else r'\columnwidth'
    body='\n'.join([r'\toprule',group,r'\rowcolor{tablehead}',row(header),
                    r'\midrule',*rows,r'\bottomrule'])
    return '\n'.join([r'\begin{'+env+'}[htbp]',r'\centering',r'\caption{'+caption+'}',
        r'\label{'+label+'}',r'\begingroup',r'\footnotesize',
        r'\setlength{\tabcolsep}{2.5pt}',r'\renewcommand{\arraystretch}{1.16}',
        r'\sbox{\resulttablebox}{%',r'\begin{tabular}{'+columns+r'}',
        body,r'\end{tabular}}',r'\ifdim\wd\resulttablebox>'+width,
        r'\resizebox{'+width+r'}{!}{\usebox{\resulttablebox}}',r'\else',
        r'\setlength{\tabcolsep}{\dimexpr 2.5pt+('+width+r'-\wd\resulttablebox)/'+str(2*len(header))+r'\relax}',
        r'\begin{tabular}{'+columns+r'}',body,r'\end{tabular}',r'\fi',r'\endgroup',foot,r'\end{'+env+'}',''])

def generated(R):
    out={}
    rows=[]
    for ds in DS:
        m=R['meta'][ds];c=R['calibration'][ds]
        counts=sorted(set(m['actual_n'].values()))
        n=str(counts[0]) if len(counts)==1 else f'{counts[0]}--{counts[-1]}'
        rows.append(row([NAMES[ds],str(m['calibration_n']),n,
                    *[pct(c['calib_acc'][agent]) for agent in ['A','B','C']]]))
    out['tables/tab_setup.tex']=table(r'Evaluation sizes and calibration accuracies (\%).','tab:setup','lrrrrr',
            ['Setting',r'Cal. $n$',r'Eval. $n$','Qwen','Llama','Mistral'],rows,
            group=r' & \multicolumn{2}{c}{Items} & \multicolumn{3}{c}{Calibration accuracy} \\'+
                  r'\cmidrule(lr){2-3}\cmidrule(l){4-6}')
    for name,datasets in [('tab_tradeoff.tex',DS[:3]),('tab_hard.tex',DS[3:])]:
        rows=[];full_rows=[]
        main=name=='tab_tradeoff.tex'
        for ds in datasets:
            for c,label in zip(CONDS,LABELS):
                m=R['accounting'][ds][c];s=m['test']
                dataset=r'\textbf{'+NAMES[ds]+'}' if c==CONDS[0] else ''
                full_rows.append(row([dataset,label,str(m['n']),
                    '$'+pct(m['dA'],True)+'$ '+pct_ci(s),'$'+pct(m['pool_loss'])+'$',
                    '$'+pct(m['readout_loss'])+'$','$'+pct(m['regret_compression'])+'$', '$'+pval(s['p'])+'$']))
                shortlabel='Calib.' if c=='expert_revealed_calib' else label
                cells=[dataset,shortlabel]+([] if main else [str(m['n'])])
                cells+=['$'+pct(m['pool_loss'])+'$','$'+pct(m['regret_compression'])+'$',
                        '$'+pct(m['dA'],True)+'$ '+pct_ci(s)]
                rows.append(row(cells))
            if ds!=datasets[-1]:rows.append(r'\midrule');full_rows.append(r'\midrule')
        label='tab:tradeoff' if main else 'tab:hard'
        caption='Ceiling--Regret diagnosis on '+('the main evaluation splits' if main else 'disjoint replication splits')+r' (\%).'
        header=['Setting','Protocol']+([] if main else ['$n$'])+['Loss','Comp.',r'$\Delta A$ [95\% CI]']
        out['tables/'+name]=table(caption,label,'llrrr' if main else 'llrrrr',header,rows)
        out['tables/'+name.replace('.tex','_full.tex')]=table(caption,label+'_full','lllrrrrr',
            ['Setting','Protocol','$n$',r'$\Delta A$ [95\% CI]','Loss','Regret','Compression','$p$'],full_rows)
    rows=[]
    for ds in DS:
        g=R['guard'][ds]
        second=max(g[k] for k in ['post','post_calibrated','union_equal'])
        scores=[('$\\underline{'+pct(g[k])+'}$') if g[k]==second else ('$'+pct(g[k])+'$')
                for k in ['post','post_calibrated','union_equal']]
        cells=[NAMES[ds],*scores,
               r'$\mathbf{'+pct(g['guard'])+r'\,(\uparrow '+pct(g['test']['difference'])+r')}$',
               '$'+pct(g['union_ceiling_gain'])+'$','$'+pval(g['holm_p'])+'$']
        rows.append(row(cells))
        if ds=='gsm8k': rows.append(r'\midrule')
    out['tables/tab_guard.tex']=table(r'Readout accuracy and candidate-preservation ablations (\%).','tab:guard',r'lrrr>{\columncolor{guardhighlight}}rrr',
        ['Setting','Post','Post-w','Union',r'\textbf{Guard}',r'$\Delta C_U$',r'$p_H$'],rows,
        group=r' & \multicolumn{4}{c}{Readout accuracy} & \multicolumn{2}{c}{Coverage / test} \\'+
              r'\cmidrule(lr){2-5}\cmidrule(l){6-7}')
    rows=[]
    for ds in DS:
        g=R['guard'][ds]
        rows.append(row([NAMES[ds],*['$'+pct(g[k])+'$' for k in ['union_ceiling_gain','union_gain','weight_gain']],
                         '$'+pct(g['test']['difference'])+'$ '+pct_ci(g['test']),
                         f"{100*g['union_share']:.0f}",'$'+pval(g['holm_p'])+'$']))
    out['tables/tab_attrib.tex']=table(r'Sequential contributions of preservation and calibration (\%).','tab:attrib','lrrrrrr',
        ['Setting',r'$\Delta C_U$','Union','Weights',r'Total [95\% CI]','Share',r'$p_H$'],rows)
    rows=[]
    for ds in DS:
        for i,(key,outcome) in enumerate([('difference_in_changes','Pre/post change'),
                                        ('post_accuracy_difference','Final accuracy')]):
            s=R['guidance'][ds]['expert_revealed_calib'][key]
            rows.append(row([NAMES[ds] if i==0 else '',outcome,str(s['n']),
                             '$'+pct(s['difference'],True)+'$ '+pct_ci(s),
                             '$'+pval(s['p'])+'$']))
        if ds!=DS[-1]:rows.append(r'\midrule')
    out['tables/tab_guidance_details.tex']=table(r'Calibration guidance versus unguided discussion (\%).',
        'tab:guidance_details','llrrr',['Setting','Outcome','$n$',r'Difference [95\% CI]','$p$'],rows)
    rows=[]
    for ds in DS:
        keys=['qwen_k3','qwen_k9'] if ds=='math500' else ['qwen_k9','llama_k9'] if ds=='mmlu_pro' else ['qwen_k9']
        for i,key in enumerate(keys):
            s=R['baselines'][ds][key]['guard_comparison']
            rows.append(row([NAMES[ds] if i==0 else '', 'Qwen' if key.startswith('qwen') else 'Llama',
                             key.rsplit('k',1)[1],str(s['n']),pct(s['left']),pct(s['right']),
                             pct(s['difference'],True)+' '+pct_ci(s),'$'+pval(s['p'])+'$',
                             '$'+pval(s['holm_p'])+'$' if 'holm_p' in s else '---']))
        if ds!=DS[-1]:rows.append(r'\midrule')
    out['tables/tab_sc_details.tex']=table(r'Guard versus self-consistency on paired items (\%).',
        'tab:sc_details','llrrrrrrr',['Setting','SC member','$k$','$n$','Guard','SC',r'Difference [95\% CI]','$p$',r'$p_H$'],rows)
    rows=[]
    for ds in DS:
        s=R['baselines'][ds]['qwen_k9']['guard_comparison']
        rows.append(row([NAMES[ds],str(s['n']),pct(s['left']),pct(s['right']),
                         pct(s['difference'],True)+' '+pct_ci(s)]))
    out['tables/tab_compute.tex']=table(r'Guard versus nine-call Qwen self-consistency (\%).','tab:compute',r'lr>{\columncolor{guardhighlight}}rrr',
        ['Setting','$n$','Guard','SC $k=9$',r'Difference [95\% CI]'],rows)
    # Numerical prose is generated from the same result object as the tables.
    macros={}
    def add(name,value,percent=True):macros[name]=value+(r'\%' if percent else '')
    for ds,prefix in zip(DS,['Math','Mmlu','Gsm','MathHard','MmluHard']):
        for c,suffix in zip(CONDS,['Unguided','Ideal','Calib']):
            m=R['accounting'][ds][c];s=m['test']
            for k,tag in [('pre','Pre'),('post','Post'),('pool_loss','Loss'),('regret_compression','Compression'),('readout_loss','Regret')]:add(prefix+suffix+tag,pct(m[k]))
            add(prefix+suffix+'Delta',pct(m['dA'],True));add(prefix+suffix+'CI',pct_ci(s));add(prefix+suffix+'P',pval(s['p']),False)
        g=R['guard'][ds]
        add(prefix+'Guard',pct(g['guard']));add(prefix+'GuardGain',pct(g['test']['difference'],True));add(prefix+'GuardCI',pct_ci(g['test']));add(prefix+'GuardHolm',pval(g['holm_p']),False)
        for key,tag in [('union_equal','Union'),('union_gain','UnionGain'),('weight_gain','WeightGain'),('union_ceiling_gain','UnionCeiling')]:add(prefix+tag,pct(g[key]))
        for key,tag in [('difference_in_changes','Increment'),('post_accuracy_difference','PostIncrement')]:
            s=R['guidance'][ds]['expert_revealed_calib'][key]
            add(prefix+tag,pct(s['difference'],True));add(prefix+tag+'CI',pct_ci(s));add(prefix+tag+'P',pval(s['p']),False)
            add(prefix+tag+'N',str(s['n']),False)
        for k in [3,9]:
            key=f'qwen_k{k}'
            if key in R['baselines'][ds]:
                v=R['baselines'][ds][key];add(prefix+f'SC{["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"][k]}',pct(v['accuracy']))
        if 'qwen_k9' in R['baselines'][ds]:
            s=R['baselines'][ds]['qwen_k9']['guard_comparison']
            add(prefix+'GuardVsSC',pct(s['difference'],True));add(prefix+'GuardVsSCCI',pct_ci(s));add(prefix+'GuardVsSCN',str(s['n']),False)
            add(prefix+'GuardCommon',pct(s['left']))
    gains=[R['guard'][ds]['test']['difference'] for ds in DS]
    add('GuardMinGain',pct(min(gains)));add('GuardMaxGain',pct(max(gains)))
    add('GuardSignificant',str(sum(R['guard'][ds]['holm_p']<.05 for ds in DS)),False)
    shares=[R['guard'][ds]['union_share']*100 for ds in DS]
    add('UnionShareMin',f'{min(shares):.0f}',False);add('UnionShareMax',f'{max(shares):.0f}',False)
    add('MathLlamaSCNine',pct(R['baselines']['math500']['llama_k9']['accuracy']))
    add('MmluLlamaSCNine',pct(R['baselines']['mmlu_pro']['llama_k9']['accuracy']))
    add('MathGuardVsSCThreeCI',pct_ci(R['baselines']['math500']['qwen_k3']['guard_comparison']))
    num=lambda n:format(n,',').replace(',','{,}')
    add('RecordedRuns',num(sum(sum(m['records'].values()) for m in R['meta'].values())),False)
    add('DiscussionRuns',num(sum(m['discussion_records'] for m in R['meta'].values())),False)
    add('SoloRuns',num(sum(m['solo_records'] for m in R['meta'].values())),False)
    add('MappedBaselineRuns',num(sum(b['n'] for ds in DS for b in R['baselines'][ds].values())),False)
    out['data/generated_numbers.tex']='% Generated by reproduce.py; do not edit.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items())
    # Preserve the original tug-of-war design; derive geometry and labels from v4.
    plot=(PAPER/'figure_templates/tug_of_war.tex').read_text(encoding='utf-8')
    for ds,token,y0 in [('math500','MATH',-1.05),('mmlu_pro','MMLU',-3.20),('gsm8k','GSM',-5.35)]:
        rows=[]
        for i,(c,label) in enumerate(zip(CONDS,LABELS)):
            m=R['accounting'][ds][c]
            compression,loss,net=m['regret_compression'],m['pool_loss'],m['dA']
            assert abs(net-(compression-loss))<1e-12
            color='tugblue' if net>0 else 'tugred' if net<0 else 'tugink'
            netlabel=r'\textcolor{'+color+'}{$'+pct(net,True)+'$}'
            rows.append(r'  \tugrow'+''.join('{'+v+'}' for v in [
                f'{y0-.55*i:.2f}',label,f'{compression:.8f}',f'{loss:.8f}',
                '$'+pct(compression)+'$','$'+pct(loss)+'$',netlabel]))
        plot=plot.replace('@@'+token+'_ROWS@@','\n'.join(rows))
    assert '@@' not in plot
    out['tables/fig_tradeoff.tex']=plot
    return out

def dumps(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    R,items,baselines,manifest=compute()
    outputs=generated(R)
    outputs.update({'data/authoritative_v4.json':dumps(R),'data/manifest_v4.json':dumps(manifest),
        'data/per_item_v4.jsonl':''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in items),
        'data/baseline_items_v4.jsonl':''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in baselines)})
    for name,text in outputs.items():
        p=PAPER/name
        if args.check:assert p.exists() and p.read_text(encoding='utf-8')==text,f'Outdated artifact: {name}'
        else:p.parent.mkdir(exist_ok=True,parents=True);p.write_text(text,encoding='utf-8')
    print(('CHECKED' if args.check else 'GENERATED'),len(outputs),'artifacts;',len(items),'debate records;',len(baselines),'mapped baseline records')
    for ds in DS:
        g=R['guard'][ds];bs=R['baselines'][ds].get('qwen_k9',{})
        print(ds,'guard',pct(g['guard']),'gain',pct(g['test']['difference'],True),'Holm',pval(g['holm_p']),
              'SC9',bs.get('accuracy'),'SC n',bs.get('n'))

if __name__=='__main__':main()
