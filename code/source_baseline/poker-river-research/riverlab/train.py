"""Session-held-out final test. Fits imputation/scaling on training only.
Four disjoint session partitions: training, tuning, calibration, final test.
No hyperparameter search occurs on calibration or final-test labels.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score,balanced_accuracy_score,brier_score_loss,log_loss,confusion_matrix,roc_auc_score
import joblib

def split_indices(rows,seed=17):
    groups=np.array([r['session_id'] for r in rows]); idx=np.arange(len(rows))
    if len(set(groups))<8: raise ValueError('Need at least 8 independent sessions; pilot can test extraction without claiming accuracy')
    def take(indices,fraction,s):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=s).split(indices,groups=groups[indices]))
        return indices[a],indices[b]
    dev,test=take(idx,.2,seed); fitdev,cal=take(dev,.25,seed+1); train,tune=take(fitdev,.25,seed+2)
    parts={'train':train,'tune':tune,'calibration':cal,'final_test':test}
    # Same source hand cannot be split even if accidental duplicate session identifiers occur.
    for a,ia in parts.items():
        for b,ib in parts.items():
            if a>=b: continue
            if set(groups[ia]) & set(groups[ib]): raise AssertionError('Session leakage')
            ha={(rows[i]['canonical_source_id'],rows[i]['hand_id']) for i in ia}; hb={(rows[i]['canonical_source_id'],rows[i]['hand_id']) for i in ib}
            if ha&hb: raise ValueError('Source hand appears in multiple partitions')
    return parts

def metrics(y,p):
    pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    bins=[]
    for a,b in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        mask=(p>=a)&(p<(b if b<1 else 1.000001))
        bins.append({'lower':float(a),'upper':float(b),'n':int(mask.sum()),'mean_probability':float(p[mask].mean()) if mask.any() else None,'observed_bluff_fraction':float(y[mask].mean()) if mask.any() else None})
    return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'roc_auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,'false_positive_rate':float(fp/(fp+tn)) if fp+tn else None,'confusion':{'true_value':int(tn),'value_called_bluff':int(fp),'bluff_called_value':int(fn),'true_bluff':int(tp)},'calibration_bins':bins}

def train(rows,outdir,seed=17):
    from .schema import audit
    all_rows=rows
    from .contract import validate_features
    if any('synthetic' in r and not isinstance(r['synthetic'],bool) for r in rows): raise ValueError('Synthetic flag must be boolean')
    if len({r.get('synthetic',False) for r in rows})>1: raise ValueError('Never mix synthetic and actual data')
    checked=[]
    for r in rows:
        if r.get('eligible') is not True or not r.get('features'): continue
        validate_features(r)
        a=audit(r)
        if not a['eligible']: continue
        if not r.get('synthetic') and r.get('model_view_reviewed') is not True: continue
        if r.get('label')!=a.get('label') or r.get('relation')!=a.get('relation'): raise ValueError('Stored label disagrees with evaluated cards')
        checked.append(a)
    rows=checked
    if len(rows)>300: raise ValueError('Maximum 300 usable samples')
    if len({r['sample_id'] for r in rows})!=len(rows): raise ValueError('Duplicate sample IDs')
    if not rows: raise ValueError('No usable feature rows')
    if len({bool(r.get('synthetic',False)) for r in rows})!=1: raise ValueError('Never mix synthetic and actual data')
    parts=split_indices(rows,seed); y=np.array([int(r['label']=='bluff') for r in rows])
    for name,idx in parts.items():
        if len(set(y[idx]))<2: raise ValueError(f'{name} needs both classes; no retries to cherry-pick final test')
    tr,tu,ca,te=(parts[k] for k in ('train','tune','calibration','final_test'))
    report={'synthetic_engineering_test':bool(rows[0].get('synthetic',False)),'label_warning':'Showdown-strength proxy; ties map to bluff. Not intent or a lie detector.','unknown_or_excluded_rate':1-len(rows)/max(len(all_rows),1),'total_rows':len(all_rows),'usable_rows':len(rows),'seed':seed,'partitions':{},'controls':{}}
    for name,idx in parts.items():
        report['partitions'][name]={'n':len(idx),'sessions':sorted({rows[i]['session_id'] for i in idx}),'hands':len({(rows[i]['canonical_source_id'],rows[i]['hand_id']) for i in idx}),'class_counts':{'bluff':int(y[idx].sum()),'value':int((1-y[idx]).sum())},'sample_ids':[rows[i]['sample_id'] for i in idx]}
    p=np.repeat(y[tr].mean(),len(te)); report['controls']['majority']=metrics(y[te],p)
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    for kind,prefixes in [('static',('static_',)),('timing',('timing_',)),('dynamic',('dynamic_',)),('full',('static_','dynamic_','timing_'))]:
        # Training set defines feature vocabulary; no final-test feature inspection.
        names=sorted({k for i in tr for k in rows[i]['features'] if k.startswith(prefixes)})
        if not names: report['controls'][kind]={'skipped':'no_features'}; continue
        X=np.array([[r['features'].get(k,np.nan) for k in names] for r in rows],dtype=float)
        X[~np.isfinite(X)]=np.nan
        candidates=[]
        for c in (.01,.1,1.):
            model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=seed))
            model.fit(X[tr],y[tr]); candidates.append((log_loss(y[tu],model.predict_proba(X[tu]),labels=[0,1]),c,model))
        _,chosen,base=min(candidates,key=lambda z:z[0])
        calibrated=CalibratedClassifierCV(FrozenEstimator(base),method='sigmoid'); calibrated.fit(X[ca],y[ca])
        pred=calibrated.predict_proba(X[te])[:,1]
        report['controls'][kind]=dict(metrics(y[te],pred),selected_C=chosen,feature_count=len(names))
        joblib.dump({'model':calibrated,'feature_names':names,'threshold':.5,'label_definition':report['label_warning'],'synthetic':report['synthetic_engineering_test']},out/f'{kind}.joblib')
    report['limitations']=['Small, selected broadcast samples do not establish generalization.','Repeated players and production/camera artifacts remain confounds.','One fixed final-test report; do not tune from it.','No real-time deployment or behavioral/mental-state claims.']
    (out/'report.json').write_text(json.dumps(report,indent=2)); return report
