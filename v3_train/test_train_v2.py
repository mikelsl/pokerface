"""Tests for train_v2.py. Real rows are only loaded/validated here; nothing is fit on them."""
import sys, json, copy, hashlib
from pathlib import Path
import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_v2 as T

OUT = HERE / 'test_outputs'


def _rec(sid, session, bettor, y, hand=None):
    return {'sample_id': sid, 'session_id': session, 'bettor': bettor, 'y': y, 'label': 'bluff' if y else 'value',
            'hand_key': hand or f'{session}|{sid}', 'synthetic': True, 'features': {}, 'active_opponents': 1}


# ------------------------------------------------------------ hand key / real loading (no fitting)

def test_hand_key_fallback():
    assert T.hand_key({'duplicate_hand_key': 'abc', 'canonical_source_id': 'c', 'hand_id': 'h'}) == 'abc'
    assert T.hand_key({'canonical_source_id': 'c', 'hand_id': 'h'}) == ('legacy', 'c', 'h')
    assert T.hand_key({'duplicate_hand_key': None, 'canonical_source_id': 'c', 'hand_id': 'h'}) == ('legacy', 'c', 'h')
    with pytest.raises(ValueError):
        T.hand_key({'sample_id': 'x', 'hand_id': 'h'})


def test_real_rows_validate_and_match_grouping(monkeypatch):
    # Guard: loading must never touch any fitting / splitting code.
    def boom(*a, **k):
        raise AssertionError('fitting/splitting called during loading')
    for name in ('split_v2', 'run_primary', 'run_loso', 'nested_loso_predict', 'run'):
        monkeypatch.setattr(T, name, boom)
    recs, meta = T.load_real_rows('all')
    assert meta['total_rows'] == 39 and len(recs) == 39 and meta['excluded_sample_ids'] == []
    legacy = [r for r in recs if isinstance(r['hand_key'], tuple)]
    assert len(legacy) == 4 and all('BV13Dc9eaEd1' in r['hand_key'][1] for r in legacy)
    assert len({r['hand_key'] for r in recs}) == 36
    assert all(not r['synthetic'] for r in recs)
    hu, _ = T.load_real_rows('heads_up')
    assert len(hu) == 33 and all(r['active_opponents'] == 1 for r in hu)


def test_grouping_mismatch_raises():
    recs, _ = T.load_real_rows('all')
    g = json.loads(T.GROUPING.read_text())
    T.cross_check_grouping(recs, g)
    bad = copy.deepcopy(g); bad['same_hand_action_groups'] = bad['same_hand_action_groups'][:-1]
    with pytest.raises(ValueError, match='same_hand_action_groups'):
        T.cross_check_grouping(recs, bad)
    bad = copy.deepcopy(g); bad['samples'][0]['session_group'] = 'HCL_other'
    with pytest.raises(ValueError):
        T.cross_check_grouping(recs, bad)
    recs2 = [dict(r) for r in recs]; recs2[0]['hand_key'] = recs2[1]['hand_key']
    with pytest.raises(ValueError):
        T.cross_check_grouping(recs2, g)


def test_real_loader_rejects_synthetic_markers():
    with pytest.raises(ValueError):
        T._real_marker_check({'sample_id': 'a', 'synthetic': True})
    with pytest.raises(ValueError):
        T._real_marker_check({'sample_id': 'synthetic_1'})


# ------------------------------------------------------------ refusal

def test_refusal_without_flags(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(T, 'load_real_rows', lambda *a, **k: called.append(1))
    for real, ok in ((False, False), (True, False), (False, True)):
        with pytest.raises(PermissionError):
            T.train_real('all', OUT / 'refused', real=real, approved_by_owner=ok)
    assert T.main(['--real', '--outdir', str(OUT / 'refused')]) == 2
    assert 'REFUSED' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        T.main(['--outdir', str(OUT / 'refused')])
    with pytest.raises(SystemExit):
        T.main(['--synthetic', '--approved-by-owner', '--outdir', str(OUT / 'refused')])
    with pytest.raises(ValueError):
        T.train_real('all', OUT / 'refused', real=True, approved_by_owner=True, n_perm=10)
    assert called == [] and not (OUT / 'refused' / 'report.json').exists()


def test_outdir_must_be_inside_v3():
    with pytest.raises(ValueError):
        T._check_outdir('/tmp/pokerface_out')
    with pytest.raises(ValueError):
        T._check_outdir(T.V3)


# ------------------------------------------------------------ leakage

def test_leakage_asserts():
    recs = [_rec('a', 's1', 'A', 0, 'h1'), _rec('b', 's1', 'A', 1, 'h2'), _rec('c', 's2', 'B', 0, 'h3'), _rec('d', 's3', 'B', 1, 'h3')]
    with pytest.raises(AssertionError, match='Session leakage'):
        T.assert_no_leakage(recs, {'train': np.array([0]), 'final_test': np.array([1])})
    with pytest.raises(ValueError, match='multiple partitions'):
        T.assert_no_leakage(recs, {'train': np.array([2]), 'final_test': np.array([3])})
    with pytest.raises(ValueError, match='spans multiple sessions'):
        T.check_hand_keys_nested(recs)
    with pytest.raises(AssertionError):
        T.assert_fold_disjoint(recs, np.array([0]), np.array([1]))


def test_synthetic_split_is_leak_free_and_matches_train_py():
    s, _ = T.load_synthetic_rows(T.make_synthetic_rows())
    parts = T.split_v2(s)
    T.assert_no_leakage(s, parts)
    shim = [{'session_id': r['session_id'], 'canonical_source_id': repr(r['hand_key']), 'hand_id': ''} for r in s]
    ref = T.split_indices_v1(shim, 17)
    assert all(np.array_equal(parts[k], ref[k]) for k in ref)


def test_too_few_sessions_raises():
    s, _ = T.load_synthetic_rows(T.make_synthetic_rows(n_sessions=6))
    with pytest.raises(T.TooFewSessions):
        T.split_v2(s)
    with pytest.raises(ValueError, match='at least 8'):
        T.run_primary(s, OUT / 'unused', True)


# ------------------------------------------------------------ synthetic end-to-end

def test_synthetic_primary_s3():
    out = OUT / 'synthetic_s3'
    rep = T.train_synthetic(out, n_sessions=10)
    assert rep['synthetic_engineering_test'] is True and rep['data_kind'] == 'synthetic' and 'WARNING' in rep
    assert rep['protocol_path'] == 'primary_s3' and rep['primary']['status'] == 'completed'
    assert rep['diagnostic_loso'] is None
    assert rep['primary']['controls']['timing'] == {'skipped': 'degenerate_constant'}
    assert {'majority', 'static', 'dynamic', 'full'} <= set(rep['primary']['controls'])
    assert rep['primary']['controls']['full']['feature_count'] == 168
    assert rep['primary']['controls']['context']['feature_count'] == 3
    assert rep['primary']['controls']['context+full']['feature_count'] == 171
    assert rep['primary']['incremental_test_s7_1']['status'] == 'not_run'
    assert rep['preregistration']['sha256'] == hashlib.sha256(T.PREREG.read_bytes()).hexdigest()
    assert all((out / f'{k}.joblib').exists() for k in ('static', 'dynamic', 'full', 'context', 'context+full'))
    assert json.loads((out / 'report.json').read_text())['protocol_path'] == 'primary_s3'


def _check_loso(rep):
    d = rep['diagnostic_loso']
    assert d['status'] == 'completed' and d['probabilities_calibrated'] is False and d['not_reported_as_accuracy'] is True
    assert {'majority', 'static', 'dynamic', 'full', 'identity'} <= set(d['controls'])
    assert d['controls']['timing'] == {'skipped': 'degenerate_constant'}
    assert set(d['by_bettor']) == {'PLAYER_A', 'non_PLAYER_A'}
    pv = d['permutation_test']['p_value']
    assert 0 < pv['roc_auc'] <= 1 and 0 < pv['balanced_accuracy'] <= 1
    assert 'uncalibrated_probabilities' in d['flags']
    assert ('not_distinguishable_from_identity' in d['flags']) == (d['controls']['full']['roc_auc'] <= d['controls']['identity']['roc_auc'])
    assert {'context', 'context+full'} <= set(d['controls'])
    inc = d['incremental_test_s7_1']
    assert 0 < inc['p_value'] <= 1 and inc['conclusion'] in d['flags']
    assert inc['observed']['delta'] == pytest.approx(d['controls']['context+full']['roc_auc'] - d['controls']['context']['roc_auc'])
    assert rep['audit_exclusion_s7_2']['applied'] is False


def test_synthetic_forced_s4_too_few_sessions():
    rep = T.train_synthetic(OUT / 'synthetic_s4_few_sessions', n_sessions=6, n_perm=20, n_jobs=1)
    assert rep['protocol_path'] == 'diagnostic_s4_loso'
    assert rep['primary']['status'] == 'insufficient_evidence' and 'fewer_than_8' in rep['primary']['reason']
    assert rep['diagnostic_loso']['n_outer_folds'] == 6
    _check_loso(rep)


def test_synthetic_forced_s4_class_missing_partition():
    rep = T.train_synthetic(OUT / 'synthetic_s4_class_missing', n_sessions=10, single_class_sessions=8, n_perm=10, n_jobs=1)
    assert rep['protocol_path'] == 'diagnostic_s4_loso'
    assert rep['primary']['status'] == 'insufficient_evidence' and 'lack a class' in rep['primary']['reason']
    assert 'partitions' in rep['primary']  # failed split is recorded, not retried
    _check_loso(rep)
    assert 'degenerate_single_class_training_folds' in rep['diagnostic_loso']['flags']


# ------------------------------------------------------------ identity / permutation

def test_identity_control():
    recs = [_rec('a', 's1', 'A', 1), _rec('b', 's1', 'A', 0), _rec('c', 's1', 'B', 1),
            _rec('d', 's2', 'A', 1), _rec('e', 's2', 'C', 0),
            _rec('f', 's3', 'B', 0), _rec('g', 's3', 'A', 0)]
    p, unseen = T.identity_predict(recs, T.loso_index(recs))
    # held-out s1: train = d,e,f,g -> A: [1,0]=.5, B: [0]=0
    assert p[0] == p[1] == .5 and p[2] == 0.
    # held-out s2: train = a,b,c,f,g -> A: [1,0,0]=1/3; C unseen -> overall 2/5
    assert p[3] == pytest.approx(1 / 3) and p[4] == pytest.approx(2 / 5)
    # held-out s3: train = a..e -> B: [1]=1, A: [1,0,1]=2/3
    assert p[5] == 1. and p[6] == pytest.approx(2 / 3)
    assert unseen == 1


def test_permutation_determinism():
    s, _ = T.load_synthetic_rows(T.make_synthetic_rows(n_sessions=4, rows_per_session=6))
    y = T.labels(s); sess = np.array([r['session_id'] for r in s])
    a = T.within_session_permutations(y, sess, 50, 17)
    b = T.within_session_permutations(y, sess, 50, 17)
    assert np.array_equal(a, b) and not np.array_equal(a, T.within_session_permutations(y, sess, 50, 18))
    for k in range(50):
        for ss in set(sess):
            m = sess == ss
            assert sorted(a[k][m]) == sorted(y[m])
    folds = T.prepare_loso(s, T.FEATURE_KINDS['full'])
    n1 = T.permutation_null(folds, y, sess, 8, 17, n_jobs=1)
    n2 = T.permutation_null(folds, y, sess, 8, 17, n_jobs=1)
    n3 = T.permutation_null(folds, y, sess, 8, 17, n_jobs=2)
    for x, z in ((n1, n2), (n1, n3)):
        assert np.array_equal(x[0], z[0], equal_nan=True) and np.array_equal(x[1], z[1])


def test_never_mix_synthetic_and_real():
    s, meta = T.load_synthetic_rows(T.make_synthetic_rows(n_sessions=3))
    mixed = s[:3] + [dict(s[3], synthetic=False)]
    with pytest.raises(ValueError, match='mix'):
        T.run(mixed, meta, OUT / 'mixed', synthetic=True, subset='all')
    with pytest.raises(ValueError):
        T.validate_synthetic_row(dict(T.make_synthetic_rows(n_sessions=1)[0], synthetic=False))


def test_bettor_alias_mapping():
    rows, _ = T.load_real_rows('all')
    names = {r['bettor'] for r in rows}
    assert 'PLAYER_B_ALIAS' not in names and 'PLAYER_B' in names
    assert {r['bettor_raw'] for r in rows if r['bettor'] == 'PLAYER_B'} == {'PLAYER_B_ALIAS', 'PLAYER_B'}


# ------------------------------------------------------------ §7.1 betting context

CTX = ['context_is_raise', 'context_multiway', 'context_log_amount_rel']


def _strip_context(recs):
    return [dict(r, features={k: v for k, v in r['features'].items() if not k.startswith('context_')}) for r in recs]


def test_context_features_session_median_label_free():
    rows = T.make_synthetic_rows(n_sessions=3, rows_per_session=6)
    recs, meta = T.load_synthetic_rows(rows)
    assert meta['context_features']['names'] == CTX and meta['context_features']['pot_before_bet_used'] is False
    for r in recs:
        amounts = [q['amount'] for q in recs if q['session_id'] == r['session_id']]
        assert r['features']['context_log_amount_rel'] == pytest.approx(np.log(r['amount'] / np.median(amounts)))
        assert r['features']['context_is_raise'] == float(r['action'] == 'raise')
        assert r['features']['context_multiway'] == float(r['active_opponents'] > 1)
    # Session normalisation removes the per-session stake level (synthetic stakes differ by up to ~30x).
    assert len({round(meta['context_features']['session_median_amount'][s], -1) for s in meta['context_features']['session_median_amount']}) == 3
    # Labels are never read: flipping every label leaves every context value unchanged.
    flipped = [dict(r, y=1 - r['y'], label='value' if r['y'] else 'bluff') for r in _strip_context(recs)]
    T.add_context_features(flipped)
    assert all(a['features'][k] == b['features'][k] for a, b in zip(recs, flipped) for k in CTX)
    # A held-out session is normalised by its own median: dropping other sessions changes nothing for it.
    one = _strip_context([r for r in recs if r['session_id'] == recs[0]['session_id']])
    T.add_context_features(one)
    assert all(o['features']['context_log_amount_rel'] == r['features']['context_log_amount_rel'] for o, r in zip(one, recs))
    # pot_before_bet only when every row has it.
    some = _strip_context(recs); some[0]['pot_before_bet'] = 1000
    assert T.add_context_features(some)['pot_before_bet_used'] is False and T.CONTEXT_POT_FEATURE not in some[0]['features']
    allp = [dict(r, pot_before_bet=2 * r['amount']) for r in _strip_context(recs)]
    assert T.add_context_features(allp)['pot_before_bet_used'] is True
    assert all(r['features'][T.CONTEXT_POT_FEATURE] == pytest.approx(np.log(.5)) for r in allp)
    with pytest.raises(ValueError, match='positive finite amount'):
        T.add_context_features([dict(_strip_context(recs)[0], amount=0)])
    with pytest.raises(ValueError, match='already present'):
        T.add_context_features(recs)


def test_real_rows_have_context_fields():
    recs, meta = T.load_real_rows('all')  # load/validate only; no fitting
    assert meta['context_features']['names'] == CTX and meta['context_features']['pot_before_bet_used'] is False
    assert all(set(CTX) <= set(r['features']) for r in recs)
    hu, hmeta = T.load_real_rows('heads_up')
    by_id = {r['sample_id']: r for r in recs}
    # Session medians come from all usable rows, so heads-up rows keep identical context values.
    assert all(by_id[r['sample_id']]['features'][k] == r['features'][k] for r in hu for k in CTX)


def test_incremental_permutation_determinism():
    s, _ = T.load_synthetic_rows(T.make_synthetic_rows(n_sessions=4, rows_per_session=6))
    y = T.labels(s); sess = np.array([r['session_id'] for r in s])
    a = T.visual_row_permutations(sess, 20, 17)
    assert np.array_equal(a, T.visual_row_permutations(sess, 20, 17))
    perm = a[3]
    assert all(sess[perm[i]] == sess[i] for i in range(len(s))) and sorted(perm) == list(range(len(s)))
    ps = T.permute_visual_features(s, perm)
    for i, r in enumerate(ps):
        assert r['y'] == s[i]['y'] and r['session_id'] == s[i]['session_id'] and r['hand_key'] == s[i]['hand_key']
        assert all(r['features'][k] == s[i]['features'][k] for k in CTX)
        assert r['features']['static_face_spread_mean'] == s[perm[i]]['features']['static_face_spread_mean']
        assert r['features']['dynamic_blend_eyeBlinkLeft_sd'] == s[perm[i]]['features']['dynamic_blend_eyeBlinkLeft_sd']
    n1 = T.incremental_null(s, y, sess, .6, 6, 17, n_jobs=1)
    n2 = T.incremental_null(s, y, sess, .6, 6, 17, n_jobs=1)
    n3 = T.incremental_null(s, y, sess, .6, 6, 17, n_jobs=2)
    assert np.array_equal(n1, n2, equal_nan=True) and np.array_equal(n1, n3, equal_nan=True)
    # Identity permutation reproduces the observed context+full OOF predictions.
    folds = T.prepare_loso(s, T.FEATURE_KINDS['context+full'])
    p_obs, _, _ = T.nested_loso_predict(folds, y)
    folds_id = T.prepare_loso(T.permute_visual_features(s, np.arange(len(s))), T.FEATURE_KINDS['context+full'])
    assert np.array_equal(p_obs, T.nested_loso_predict(folds_id, y)[0])


def test_context_only_signal_gives_large_incremental_p():
    rows = T.make_synthetic_rows(n_sessions=6, rows_per_session=8, signal=0.0, context_signal=2.5)
    s, _ = T.load_synthetic_rows(rows)
    d = T.run_loso(s, n_perm=40, n_jobs=-1)
    inc = d['incremental_test_s7_1']
    assert d['controls']['context']['roc_auc'] > .75
    assert inc['p_value'] > .2 and inc['conclusion'] == 'no_demonstrated_visual_increment'
    assert 'no_demonstrated_visual_increment' in d['flags']


def test_visual_signal_gives_small_incremental_p():
    rows = T.make_synthetic_rows(n_sessions=6, rows_per_session=8, signal=3.0, context_signal=0.0)
    s, _ = T.load_synthetic_rows(rows)
    d = T.run_loso(s, n_perm=40, n_jobs=-1)
    inc = d['incremental_test_s7_1']
    assert inc['observed']['delta'] > 0 and inc['p_value'] < .05
    assert inc['conclusion'] == ('visual_incremental_information' if d['controls']['context+full']['roc_auc'] > d['controls']['identity']['roc_auc']
                                 else 'no_demonstrated_visual_increment')


# ------------------------------------------------------------ §7.2 audit exclusion

def _write_flags(path, ids, flagged, flag='head_down'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({i: {'flags': [flag] if i in flagged else [], 'summary': {}} for i in ids}))
    return path


def test_exclude_audit_flags_path():
    rows = T.make_synthetic_rows(n_sessions=6)
    ids = [r['sample_id'] for r in rows]
    flagged = {i for i in ids if i.endswith('_03')} | {'synthetic_02_05'}
    fpath = _write_flags(OUT / 'audit_flags_synthetic.json', ids, flagged)
    out = OUT / 'synthetic_s4_audit_excluded'
    rep = T.train_synthetic(out, n_sessions=6, n_perm=5, n_jobs=1, exclude_audit_flags=fpath)
    ax = rep['audit_exclusion_s7_2']
    assert ax['applied'] is True and ax['variant'] == 'sensitivity_s7_2_audit_flags_excluded'
    assert sorted(ax['excluded_sample_ids']) == sorted(flagged) and ax['n_kept'] == len(ids) - len(flagged)
    assert ax['sha256'] == hashlib.sha256(fpath.read_bytes()).hexdigest()
    assert rep['usable_rows'] == len(ids) - len(flagged)
    assert rep['diagnostic_loso']['status'] == 'completed'
    # Context medians were computed before exclusion (all usable rows of the session).
    full, meta = T.load_synthetic_rows(rows)
    assert rep['context_features_s7_1']['session_median_amount'] == meta['context_features']['session_median_amount']
    # CLI path records the variant too.
    T.main(['--synthetic', '--synthetic-sessions', '6', '--n-perm', '3', '--n-jobs', '1', '--outdir', str(OUT / 'synthetic_cli_audit_excluded'),
            '--exclude-audit-flags', str(fpath)])
    assert json.loads((OUT / 'synthetic_cli_audit_excluded' / 'report.json').read_text())['audit_exclusion_s7_2']['n_excluded'] == len(flagged)
    # Default run is the main variant.
    assert T.train_synthetic(OUT / 'synthetic_s4_main_variant', n_sessions=6, n_perm=2, n_jobs=1)['audit_exclusion_s7_2'] == {'applied': False, 'variant': 'main'}


def test_exclude_audit_flags_rejects_bad_files():
    s, _ = T.load_synthetic_rows(T.make_synthetic_rows(n_sessions=3))
    ids = [r['sample_id'] for r in s]
    with pytest.raises(ValueError, match='lacks rows'):
        T.apply_audit_exclusion(s, _write_flags(OUT / 'flags_missing.json', ids[1:], set()))
    with pytest.raises(ValueError, match='unknown audit flags'):
        T.apply_audit_exclusion(s, _write_flags(OUT / 'flags_unknown.json', ids, {ids[0]}, flag='blurry'))
    with pytest.raises(ValueError, match='every row'):
        T.apply_audit_exclusion(s, _write_flags(OUT / 'flags_all.json', ids, set(ids)))


def test_real_audit_flags_cover_every_ready_row():
    path = T.V3 / 'audit' / 'audit_flags.json'
    if not path.exists():
        pytest.skip('audit not run yet')
    recs, _ = T.load_real_rows('all')  # load only; no fitting, no label statistics
    kept, info = T.apply_audit_exclusion(recs, path)
    assert info['n_kept'] + info['n_excluded'] == len(recs) == 39
