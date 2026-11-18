import json
import numpy as np
import pytest
from riverlab.cards import rank5,best7,label
from riverlab.schema import audit
from riverlab.cli import synthetic,read
from riverlab.train import train,split_indices

def test_categories():
    examples=['2c 5d 8h Js Ac','2c 2d 8h Js Ac','2c 2d 8h 8s Ac','2c 2d 2h Js Ac','2c 3d 4h 5s 6c','2c 5c 8c Jc Ac','2c 2d 2h Js Jc','2c 2d 2h 2s Ac','2c 3c 4c 5c 6c']
    scores=[rank5(s.split()) for s in examples]
    assert [s[0] for s in scores]==list(range(9)); assert scores==sorted(scores)

def test_wheel_and_kicker():
    assert rank5('Ac 2d 3h 4s 5c'.split())==(4,5)
    assert rank5('2c 3d 4h 5s 6c'.split())>rank5('Ac 2d 3h 4s 5c'.split())
    assert best7('Ac Ad Kc Kd Qh 2s 3c'.split())>best7('Ah As Kh Ks Jh 4s 5c'.split())

def test_ties_bluff_and_missing():
    r=label('Tc Jc Qc Kc Ac'.split(),['2d','3d'],['4d','5d']); assert r['relation']=='tie' and r['label']=='bluff'
    with pytest.raises(ValueError):label(['Ac'],['2d','3d'],['4d','5d'])
    with pytest.raises(ValueError):label('Tc Jc Qc Kc Ac'.split(),['Ac','3d'],['4d','5d'])

def test_audit_and_cutoff(tmp_path):
    p=tmp_path/'x.jsonl'; synthetic(p); rows=read(p); assert all(r['eligible'] for r in rows)
    r=dict(rows[0],model_cutoff=25); assert not audit(r)['eligible']
    r=dict(rows[0],bettor_cards=[]); assert not audit(r)['eligible']
    r=dict(rows[0],cards_reviewed=False); assert not audit(r)['eligible']
    r=dict(rows[0],source_seconds=float('nan')); assert not audit(r)['eligible']

def test_pipeline(tmp_path):
    p=tmp_path/'x.jsonl';synthetic(p);rows=read(p); parts=split_indices(rows)
    for k,a in parts.items():
        for l,b in parts.items():
            if k!=l:assert not {rows[i]['session_id'] for i in a}&{rows[i]['session_id'] for i in b}
    report=train(rows,tmp_path/'models'); assert report['synthetic_engineering_test'];assert report['usable_rows']==200
    assert set(report['controls'])=={'majority','static','timing','dynamic','full'}
    assert (tmp_path/'models'/'full.joblib').exists()

def test_reject_duplicate_and_mixed(tmp_path):
    p=tmp_path/'x.jsonl';synthetic(p);rows=read(p)
    with pytest.raises(ValueError):train(rows+[rows[0]],tmp_path/'a')
    rows[0]['synthetic']=False
    with pytest.raises(ValueError):train(rows,tmp_path/'a')

def test_fail_closed_flags_and_masks(tmp_path):
    p=tmp_path/'x.jsonl';synthetic(p);r=read(p)[0]
    for patch in ({'cards_reviewed':'false'},{'synthetic':'false'},{'model_view_reviewed':'false'},{'mask_rects':42},{'actor_roi':42}):
        assert not audit(dict(r,**patch))['eligible']

def test_feature_leakage_rejected(tmp_path):
    p=tmp_path/'x.jsonl';synthetic(p);rows=read(p)
    rows[0]['features']['static_label']=1
    with pytest.raises(ValueError,match='Unapproved feature'):train(rows,tmp_path/'m')

def test_synthetic_cannot_be_rebranded_actual(tmp_path):
    p=tmp_path/'x.jsonl';synthetic(p);rows=read(p)
    for r in rows:r.update(synthetic=False,model_view_reviewed=True)
    with pytest.raises(ValueError):train(rows,tmp_path/'m')

def test_cutoff_policy_not_response_selected(tmp_path):
    p=tmp_path/'x.jsonl';synthetic(p);r=read(p)[0]
    assert not audit(dict(r,model_cutoff=23))['eligible']
    assert not audit(dict(r,opponent_response_time=23))['eligible']
    assert not audit(dict(r,cutoff_policy='adaptive_to_opponent'))['eligible']

def test_abstention_counted_not_validated_as_eligible(tmp_path):
    p=tmp_path/'x.jsonl';synthetic(p);rows=read(p)
    excluded=dict(rows[0],sample_id='synthetic_excluded',eligible=False,features={'timing_duration':14.0},quality={'frames':70,'face_fraction':0.0},abstention_reasons=['insufficient_face_coverage'])
    report=train(rows+[excluded],tmp_path/'m')
    assert report['total_rows']==201 and report['usable_rows']==200
    assert report['unknown_or_excluded_rate']==pytest.approx(1/201)

def test_actual_abstention_is_skipped_before_feature_validation(tmp_path):
    row={'sample_id':'excluded_actual_test_fixture','synthetic':False,'eligible':False,'features':{'timing_duration':14.0},'quality':{'frames':70,'face_fraction':0.0},'abstention_reasons':['insufficient_face_coverage']}
    with pytest.raises(ValueError,match='No usable feature rows'):train([row],tmp_path/'m')
