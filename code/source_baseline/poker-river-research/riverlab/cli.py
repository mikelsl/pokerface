import argparse,json
from pathlib import Path
from .schema import audit,hhmmss

def read(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def write(path,rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))

def synthetic(out):
    import numpy as np
    rng=np.random.default_rng(20261005); rows=[]
    for session in range(20):
        for j in range(10):
            positive=j%2==0; n=session*10+j
            r={'sample_id':f'synthetic_{n:03d}','canonical_source_id':f'synthetic_source_{session}','cutoff_policy':'fixed_bet_window_v1','source_url':f'synthetic://session/{session}','session_id':f'synthetic_s{session:02d}','hand_id':f'synthetic_h{n:03d}','bettor':'synthetic_A','opponent':'synthetic_B','board':['2c','5d','8h','Js','Qc'],'bettor_cards':['3c','4d'] if positive else ['As','Ad'],'opponent_cards':['Kh','Kd'],'source_seconds':20.,'source_hhmmss':hhmmss(20),'requested_start':10.,'requested_end':25.,'actual_start':10.,'actual_end':25.,'bet_time':20.,'model_cutoff':24.,'opponent_response_time':25.,'card_confidence':'high','cards_reviewed':True,'heads_up':True,'street':'river','game_type':'cash','review_screenshot':'synthetic_no_real_image','actor_roi':[0,0,100,100],'mask_rects':[[0,90,100,10]],'rights_basis':'generated synthetic fixture','synthetic':True}
            r=audit(r); r['features']={'dynamic_fixture_noise':float(rng.normal()),'static_fixture_noise':float(rng.normal()),'timing_fixture_noise':float(rng.uniform(1,10))}; rows.append(r)
    write(out,rows)

def main():
    p=argparse.ArgumentParser(description='Private CPU river proxy research. No intent inference.'); sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('synthetic'); s.add_argument('--out',required=True)
    a=sub.add_parser('audit'); a.add_argument('manifest'); a.add_argument('--out',required=True)
    c=sub.add_parser('prepare'); c.add_argument('manifest'); c.add_argument('--outdir',required=True); c.add_argument('--out',required=True)
    f=sub.add_parser('features'); f.add_argument('manifest'); f.add_argument('--face-model',required=True); f.add_argument('--hand-model',required=True); f.add_argument('--out',required=True)
    t=sub.add_parser('train'); t.add_argument('features'); t.add_argument('--outdir',required=True); t.add_argument('--seed',type=int,default=17)
    args=p.parse_args()
    if args.cmd=='synthetic': synthetic(args.out); print('Synthetic engineering fixtures only; no real poker evidence.')
    elif args.cmd=='audit':
        rows=[audit(r) for r in read(args.manifest)]; write(args.out,rows); print(json.dumps({'rows':len(rows),'eligible':sum(r['eligible'] for r in rows)}))
    elif args.cmd=='prepare':
        from .media import prepare
        rows=[]
        for r in read(args.manifest):
            checked=audit(r)
            if checked['eligible']:
                try: checked=prepare(checked,checked['source_path'],args.outdir)
                except Exception as e: checked.update(eligible=False,exclusion_reasons=[f'prepare_failed:{e}'])
            rows.append(checked)
        write(args.out,rows)
    elif args.cmd=='features':
        from .features import extract
        rows=[]
        for r in read(args.manifest):
            if r.get('eligible'):
                try:r=extract(r,args.face_model,args.hand_model)
                except Exception as e:r=dict(r,eligible=False,abstention_reasons=[f'extraction_failed:{e}'])
            rows.append(r)
        write(args.out,rows); print(json.dumps({'rows':len(rows),'eligible':sum(bool(r.get('eligible')) for r in rows)}))
    elif args.cmd=='train':
        from .train import train
        report=train(read(args.features),args.outdir,args.seed); print(json.dumps({'usable_rows':report['usable_rows'],'synthetic_engineering_test':report['synthetic_engineering_test'],'report':str(Path(args.outdir)/'report.json')}))

if __name__=='__main__':main()
