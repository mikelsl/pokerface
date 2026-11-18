"""pokerface v3 training on the v2 data contract.

Implements v3_train/PREREGISTRATION.md (frozen). Do not change protocol constants here
without adding a dated section to the preregistration first.

* Real rows: v2/analysis/ready_samples.jsonl + v2/expansion/*/ready_samples.jsonl, each
  validated with v2 contract.validate_features_v2 (audit + clip hash). Label = comparison.label.
* Real-data training refuses unless BOTH --real and --approved-by-owner are given.
* --synthetic generates v2-shaped synthetic rows (synthetic=True); only clip-hash/audit
  validation is skipped for them. Synthetic and real rows are never mixed.
* §3 primary: riverlab.train four-way session split (seed 17, no seed retry), C by tune log
  loss, sigmoid calibration, final_test once. Fewer than 8 sessions or a class-missing
  partition -> primary='insufficient_evidence' -> §4.
* §4 diagnostic: nested leave-one-session-out, uncalibrated, pooled OOF metrics,
  within-session label permutation null, identity control, PLAYER_A split.
* §7.1: betting-context features (context_*), controls 'context' and 'context+full' on both
  paths, and in §4 the incremental test AUC(context+full) - AUC(context) with a within-session
  permutation of the visual (static_/dynamic_) feature rows.
* §7.2: --exclude-audit-flags <audit_flags.json> runs the sensitivity variant without flagged rows.
"""
import sys, os, json, glob, hashlib, math, argparse, warnings
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True  # never write .pyc into v2/code or riverlab
V3 = Path(__file__).resolve().parent
ROOT = V3.parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'cache'))
BASELINE = ROOT / 'code/source_baseline/poker-river-research'
for _p in (ROOT / 'v2/code', BASELINE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss, roc_auc_score, balanced_accuracy_score
import joblib

from contract import validate_features_v2, audit as audit_v2, POLICY  # hcl_local/v2/code/contract.py
from riverlab.contract import ALLOWED
from riverlab.train import split_indices as split_indices_v1, metrics as metrics_v1

PREREG = V3 / 'PREREGISTRATION.md'
GROUPING = ROOT / 'analysis/combined_ready_grouping.json'
SEED = 17
C_GRID = (.01, .1, 1.)
N_PERM = 1000
MIN_SESSIONS_PRIMARY = 8
FOCUS_BETTOR = 'PLAYER_A'
# PREREGISTRATION.md §6: same player under two names (BV1XWwsenEPW title says Player B).
BETTOR_ALIASES = {'PLAYER_B_ALIAS': 'PLAYER_B'}
FEATURE_KINDS = {'static': ('static_',), 'dynamic': ('dynamic_',), 'full': ('static_', 'dynamic_'),
                 # PREREGISTRATION.md §7.1
                 'context': ('context_',), 'context+full': ('context_', 'static_', 'dynamic_')}
VISUAL_PREFIXES = ('static_', 'dynamic_')
CONTEXT_POT_FEATURE = 'context_log_amount_pot'
INCREMENT_ALPHA = .05
AUDIT_FLAGS = ('camera_cut_or_zoom', 'multi_face', 'head_down')  # PREREGISTRATION.md §7.2
TIMING_CONTROL = {'skipped': 'degenerate_constant'}
LABEL_WARNING = ('Label = comparison.label, a showdown-strength proxy (best five of seven vs every active '
                 'opponent; weaker or tie = bluff). Not intent and not a lie detector.')
LIMITATIONS = [
    'Small, selected broadcast samples do not establish generalization.',
    'Repeated players and production/camera artifacts remain confounds.',
    'Label is a showdown-strength proxy, not intent.',
    'One fixed final-test report (§3); do not tune from it.',
    '§4 LOSO results are pilot diagnostics only and are not reported as accuracy; probabilities are uncalibrated.',
    'No real-time deployment or behavioral/mental-state claims.',
]
SYNTHETIC_WARNING = 'SYNTHETIC DATA: software smoke test only. These scores are not accuracy and say nothing about real players.'


def ready_files():
    return [ROOT / 'v2/analysis/ready_samples.jsonl'] + [Path(p) for p in sorted(glob.glob(str(ROOT / 'v2/expansion/*/ready_samples.jsonl')))]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TooFewSessions(ValueError):
    pass


# ---------------------------------------------------------------- loading / validation

def hand_key(r):
    """duplicate_hand_key if present, else ('legacy', canonical_source_id, hand_id) (§1)."""
    k = r.get('duplicate_hand_key')
    if k:
        return str(k)
    cs, h = r.get('canonical_source_id'), r.get('hand_id')
    if not cs or not h:
        raise ValueError(f"{r.get('sample_id')}: no duplicate_hand_key and no canonical_source_id/hand_id fallback")
    return ('legacy', cs, h)


def _record(r, synthetic):
    comp = r['comparison']
    label = comp['label']
    if label not in ('bluff', 'value'):
        raise ValueError(f"{r['sample_id']}: unknown label {label!r}")
    return {
        'sample_id': r['sample_id'], 'session_id': r['session_id'],
        'bettor': BETTOR_ALIASES.get(r['bettor'], r['bettor']), 'bettor_raw': r['bettor'],
        'source': r.get('source_id') or str(r.get('source_url', '')).rstrip('/').split('/')[-1],
        'hand_key': hand_key(r), 'label': label, 'y': int(label == 'bluff'),
        'tie': any(p.get('relation') == 'tie' for p in comp.get('pairwise', [])),
        'active_opponents': int(r['active_opponents']), 'action': r['action'],
        'amount': r.get('amount'), 'pot_before_bet': r.get('pot_before_bet'),
        'features': dict(r['features']), 'quality': r.get('quality', {}), 'synthetic': synthetic,
    }


def _positive_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def add_context_features(records):
    """§7.1 betting-context features, written into each record's features as context_*.

    Label-free: uses only action, active_opponents, amount (and pot_before_bet). The session median
    is taken over every record passed in for that session -- callers pass all usable rows before any
    subset / audit exclusion, so a held-out session is normalised by its own median.
    """
    for r in records:
        if not _positive_number(r.get('amount')):
            raise ValueError(f"{r['sample_id']}: §7.1 needs a positive finite amount, got {r.get('amount')!r}")
        if any(k.startswith('context_') for k in r['features']):
            raise ValueError(f"{r['sample_id']}: context features already present")
    by_session = defaultdict(list)
    for r in records:
        by_session[r['session_id']].append(float(r['amount']))
    median = {s: float(np.median(v)) for s, v in sorted(by_session.items())}
    use_pot = bool(records) and all(_positive_number(r.get('pot_before_bet')) for r in records)
    for r in records:
        f = r['features']
        f['context_is_raise'] = float(r['action'] == 'raise')
        f['context_multiway'] = float(r['active_opponents'] > 1)
        f['context_log_amount_rel'] = math.log(float(r['amount']) / median[r['session_id']])
        if use_pot:
            f[CONTEXT_POT_FEATURE] = math.log(float(r['amount']) / float(r['pot_before_bet']))
    names = ['context_is_raise', 'context_multiway', 'context_log_amount_rel'] + ([CONTEXT_POT_FEATURE] if use_pot else [])
    return {'names': names, 'pot_before_bet_used': use_pot,
            'pot_before_bet_rows': sum(_positive_number(r.get('pot_before_bet')) for r in records),
            'session_median_amount': median,
            'median_scope': 'all usable rows of the session, before subset and audit exclusion; labels never read'}


def _real_marker_check(r):
    if 'synthetic' in r and r['synthetic'] is not False:
        raise ValueError(f"{r.get('sample_id')}: synthetic flag on real row")
    if str(r.get('source_url', '')).startswith('synthetic:') or any('synthetic' in str(r.get(k, '')).lower() for k in ('sample_id', 'session_id', 'hand_id')):
        raise ValueError(f"{r.get('sample_id')}: synthetic marker in actual data")


def read_raw_real_rows():
    raw = []
    for f in ready_files():
        for line in f.read_text().splitlines():
            if line.strip():
                raw.append(json.loads(line))
    return raw


def load_real_rows(subset='all', grouping_path=GROUPING):
    """Load + validate real rows. Never fits, splits or scores anything."""
    raw = read_raw_real_rows()
    usable, excluded = [], []
    for r in raw:
        _real_marker_check(r)
        if r.get('status') != 'ready' or r.get('eligible') is not True or r.get('model_view_reviewed') is not True:
            excluded.append(r.get('sample_id')); continue
        validate_features_v2(r)  # raises on audit / feature contract / provenance / clip-hash failure
        a = audit_v2(r)
        if a['comparison']['label'] != r['comparison']['label'] or \
                [p['relation'] for p in a['comparison']['pairwise']] != [p['relation'] for p in r['comparison']['pairwise']]:
            raise ValueError(f"{r['sample_id']}: stored label disagrees with evaluated cards")
        usable.append(_record(r, synthetic=False))
    _basic_checks(usable)
    cross_check_grouping(usable, json.loads(Path(grouping_path).read_text()))
    context = add_context_features(usable)
    meta = {'total_rows': len(raw), 'usable_rows_before_subset': len(usable), 'excluded_sample_ids': excluded,
            'files': [str(f.relative_to(ROOT)) for f in ready_files()], 'context_features': context}
    return apply_subset(usable, subset), meta


def _basic_checks(records):
    if not records:
        raise ValueError('No usable feature rows')
    if len(records) > 300:
        raise ValueError('Maximum 300 usable samples')
    if len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('Duplicate sample IDs')
    if len({r['synthetic'] for r in records}) != 1:
        raise ValueError('Never mix synthetic and actual data')
    check_hand_keys_nested(records)


def check_hand_keys_nested(records):
    """A hand belongs to exactly one session, so session-grouped splits are also hand-disjoint."""
    sess = defaultdict(set)
    for r in records:
        sess[r['hand_key']].add(r['session_id'])
    bad = {str(k): sorted(v) for k, v in sess.items() if len(v) > 1}
    if bad:
        raise ValueError(f'Hand key spans multiple sessions: {bad}')


def cross_check_grouping(records, grouping):
    """Hand keys must reproduce combined_ready_grouping.json exactly; otherwise stop (§1)."""
    gs = {s['sample_id']: s for s in grouping['samples']}
    ids = {r['sample_id'] for r in records}
    if ids != set(gs):
        raise ValueError(f'Grouping sample set mismatch: missing={sorted(set(gs) - ids)} extra={sorted(ids - set(gs))}')
    for r in records:
        g = gs[r['sample_id']]
        if g['session_group'] != r['session_id']:
            raise ValueError(f"{r['sample_id']}: session {r['session_id']} != grouping {g['session_group']}")
        if g['label'] != r['label']:
            raise ValueError(f"{r['sample_id']}: label {r['label']} != grouping {g['label']}")
    groups = defaultdict(list)
    for r in records:
        groups[r['hand_key']].append(r['sample_id'])
    multi = sorted(sorted(v) for v in groups.values() if len(v) > 1)
    expected = sorted(sorted(v) for v in grouping['same_hand_action_groups'])
    if multi != expected:
        raise ValueError(f'same_hand_action_groups mismatch: computed={multi} expected={expected}')
    # Partition induced by hand keys must equal the partition induced by grouping hand_group.
    mine = {frozenset(v) for v in groups.values()}
    theirs = defaultdict(list)
    for sid, g in gs.items():
        theirs[g['hand_group']].append(sid)
    if mine != {frozenset(v) for v in theirs.values()}:
        raise ValueError('Hand-key partition differs from grouping hand_group partition')
    n_expected = grouping.get('distinct_card_roster_hand_fingerprints')
    if n_expected is not None and len(groups) != n_expected:
        raise ValueError(f'{len(groups)} distinct hands != grouping {n_expected}')
    return {'distinct_hands': len(groups), 'same_hand_action_groups': multi}


def apply_subset(records, subset):
    if subset == 'all':
        return list(records)
    if subset == 'heads_up':
        return [r for r in records if r['active_opponents'] == 1]
    raise ValueError(f'unknown subset {subset!r}')


def apply_audit_exclusion(records, path):
    """§7.2 sensitivity variant: drop every row with any audit flag. The flag file must cover all rows."""
    path = Path(path).resolve()
    flags = json.loads(path.read_text())
    missing = sorted(r['sample_id'] for r in records if r['sample_id'] not in flags)
    if missing:
        raise ValueError(f'audit flag file lacks rows: {missing}')
    unknown = sorted({f for r in records for f in flags[r['sample_id']]['flags']} - set(AUDIT_FLAGS))
    if unknown:
        raise ValueError(f'unknown audit flags {unknown}; §7.2 defines {list(AUDIT_FLAGS)}')
    kept = [r for r in records if not flags[r['sample_id']]['flags']]
    dropped = [r['sample_id'] for r in records if flags[r['sample_id']]['flags']]
    if not kept:
        raise ValueError('audit exclusion removed every row')
    info = {'applied': True, 'variant': 'sensitivity_s7_2_audit_flags_excluded',
            'path': str(path.relative_to(ROOT)) if ROOT in path.parents else str(path), 'sha256': sha256_file(path),
            'excluded_sample_ids': dropped, 'n_excluded': len(dropped), 'n_kept': len(kept),
            'flag_counts_among_excluded': {f: sum(f in flags[s]['flags'] for s in dropped) for f in AUDIT_FLAGS}}
    return kept, info


# ---------------------------------------------------------------- synthetic rows

SYN_BETTORS = ['PLAYER_A', 'SYN_ALICE', 'SYN_BOB', 'SYN_CHEN', 'SYN_DANA', 'SYN_EVAN']
SYN_BLUFF_RATE = {'PLAYER_A': .55, 'SYN_ALICE': .2, 'SYN_BOB': .5, 'SYN_CHEN': .35, 'SYN_DANA': .45, 'SYN_EVAN': .3}


def make_synthetic_rows(n_sessions=10, rows_per_session=8, seed=2026, signal=1.0, single_class_sessions=0, context_signal=0.0):
    """v2-shaped synthetic rows. Sessions [0, single_class_sessions) contain only 'value' rows.

    `amount` comes from a separate generator (per-session stake level x lognormal noise, shifted by
    context_signal for bluffs) so features and labels are unchanged by adding it.
    """
    rng = np.random.default_rng(seed)
    arng = np.random.default_rng([seed, 71])
    names = sorted(ALLOWED)
    signal_feats = ['dynamic_blend_eyeBlinkLeft_sd', 'dynamic_blend_mouthPressLeft_delta', 'dynamic_face_spread_sd']
    bettor_off = {b: rng.normal(0, .5, len(names)) for b in SYN_BETTORS}
    rows, tie_done = [], False
    for s in range(n_sessions):
        sid = f'synthetic_session_{s:02d}'
        sess_off = rng.normal(0, .3, len(names))
        stake = 1000. * 10 ** arng.uniform(0, 1.5)
        bettors = list(rng.choice(SYN_BETTORS, rows_per_session, p=[.3, .14, .14, .14, .14, .14]))
        if s >= single_class_sessions:
            ys = [int(rng.random() < SYN_BLUFF_RATE[b]) for b in bettors]
            if len(set(ys)) < 2:
                ys[-1] = 1 - ys[-1]
        else:
            ys = [0] * rows_per_session
        for i, (b, y) in enumerate(zip(bettors, ys)):
            hand_no = max(i, 1)  # rows 0 and 1 share one hand (bet, then raise)
            k = int(rng.choice([1, 1, 1, 1, 1, 1, 2, 3]))
            tie = y == 1 and not tie_done
            tie_done = tie_done or tie
            if y == 0:
                rel = ['stronger'] * k
            else:
                rel = ['stronger'] * (k - 1) + ['tie' if tie else 'weaker']
            x = rng.normal(0, 1, len(names)) + bettor_off[b] + sess_off
            f = {n: float(v) for n, v in zip(names, x)}
            for n in signal_feats:
                f[n] += signal * y
            hand_fraction = float(rng.choice([0., 0., .04, .5, .8, 1.]))
            if hand_fraction == 0:
                for n in names:
                    if 'hand_count' in n:
                        f[n] = 0.
            f['timing_duration'] = 5
            rows.append({
                'synthetic': True, 'sample_id': f'synthetic_{s:02d}_{i:02d}', 'session_id': sid,
                'source_id': f'synthetic_source_{s:02d}', 'source_url': f'synthetic://session/{s:02d}',
                'hand_id': f'synthetic_river_{hand_no:02d}',
                'duplicate_hand_key': hashlib.sha256(f'{sid}|{hand_no}'.encode()).hexdigest(),
                'bettor': b, 'action': 'raise' if (i == 1) else str(rng.choice(['bet', 'bet', 'bet', 'raise'])),
                'amount': max(1, int(round(stake * math.exp(arng.normal(0, .4) + context_signal * y)))),
                'active_opponents': k, 'multiway': k > 1,
                'opponents': [{'name': f'SYN_OPP_{j}', 'cards': [], 'can_respond': True, 'shared_contestable_pot': True} for j in range(k)],
                'comparison': {'label': 'bluff' if y else 'value', 'pairwise': [{'opponent': f'SYN_OPP_{j}', 'relation': r} for j, r in enumerate(rel)],
                               'active_opponents': k, 'multiway': k > 1, 'label_definition': 'showdown-strength proxy; not intent'},
                'cutoff_policy': POLICY, 'status': 'ready', 'eligible': True, 'model_view_reviewed': True,
                'features': f, 'quality': {'frames': 25, 'face_fraction': float(rng.choice([.6, .9, 1.])), 'hand_fraction': hand_fraction},
                'feature_provenance': {'extractor': 'synthetic_generator', 'contract_version': 'v2', 'window_policy': POLICY,
                                       'masked_clip_sha256': None},
            })
    return rows


def validate_synthetic_row(r):
    """validate_features_v2 minus audit and clip hash (synthetic rows have neither cards nor clips)."""
    if r.get('synthetic') is not True:
        raise ValueError('synthetic loader got a non-synthetic row')
    if not str(r.get('source_url', '')).startswith('synthetic:') or 'synthetic' not in r['sample_id']:
        raise ValueError('synthetic row lacks synthetic markers')
    f, p, q = r.get('features', {}), r.get('feature_provenance', {}), r.get('quality', {})
    if not f or set(f) - ALLOWED or f.get('timing_duration') != 5:
        raise ValueError('feature contract mismatch')
    if any(not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v) for v in f.values()):
        raise ValueError('nonfinite feature')
    if p.get('contract_version') != 'v2' or p.get('window_policy') != POLICY or r.get('cutoff_policy') != POLICY:
        raise ValueError('feature provenance mismatch')
    if q.get('frames') != 25 or q.get('face_fraction', 0) < .5:
        raise ValueError('feature quality failed')
    rel = [x['relation'] for x in r['comparison']['pairwise']]
    if len(rel) != r['active_opponents'] or (r['comparison']['label'] == 'value') != all(x == 'stronger' for x in rel):
        raise ValueError('synthetic label inconsistent with pairwise relations')


def load_synthetic_rows(rows, subset='all'):
    records = []
    for r in rows:
        validate_synthetic_row(r)
        records.append(_record(r, synthetic=True))
    _basic_checks(records)
    context = add_context_features(records)
    return apply_subset(records, subset), {'total_rows': len(rows), 'usable_rows_before_subset': len(records), 'excluded_sample_ids': [],
                                           'context_features': context}


# ---------------------------------------------------------------- shared helpers

def labels(records):
    return np.array([r['y'] for r in records], dtype=int)


def feature_matrix(records, names):
    X = np.array([[r['features'].get(k, np.nan) for k in names] for r in records], dtype=float)
    X[~np.isfinite(X)] = np.nan
    return X


def vocabulary(records, idx, prefixes):
    """Training rows define the feature vocabulary (train.py semantics)."""
    return sorted({k for i in idx for k in records[i]['features'] if k.startswith(prefixes)})


def metrics(y, p):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return metrics_v1(np.asarray(y), np.asarray(p, dtype=float))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


# ---------------------------------------------------------------- §3 primary protocol

def assert_no_leakage(records, parts):
    """Same checks as riverlab.train.split_indices, keyed on the v2 hand key."""
    for a, ia in parts.items():
        for b, ib in parts.items():
            if a >= b:
                continue
            if {records[i]['session_id'] for i in ia} & {records[i]['session_id'] for i in ib}:
                raise AssertionError('Session leakage')
            if {records[i]['hand_key'] for i in ia} & {records[i]['hand_key'] for i in ib}:
                raise ValueError('Source hand appears in multiple partitions')


def split_v2(records, seed=SEED):
    """riverlab.train.split_indices on a shim whose (canonical_source_id, hand_id) is the v2 hand key."""
    n_sessions = len({r['session_id'] for r in records})
    if n_sessions < MIN_SESSIONS_PRIMARY:
        raise TooFewSessions(f'Need at least 8 independent sessions (have {n_sessions}); pilot can test extraction without claiming accuracy')
    check_hand_keys_nested(records)
    shim = [{'session_id': r['session_id'], 'canonical_source_id': repr(r['hand_key']), 'hand_id': ''} for r in records]
    parts = split_indices_v1(shim, seed)
    assert_no_leakage(records, parts)
    return parts


def _partition_summary(records, y, parts):
    return {name: {'n': len(idx), 'sessions': sorted({records[i]['session_id'] for i in idx}),
                   'hands': len({records[i]['hand_key'] for i in idx}),
                   'class_counts': {'bluff': int(y[idx].sum()), 'value': int((1 - y[idx]).sum())},
                   'sample_ids': [records[i]['sample_id'] for i in idx]} for name, idx in parts.items()}


def run_primary(records, outdir, synthetic, seed=SEED):
    """§3. Raises TooFewSessions if < 8 sessions. Returns insufficient_evidence if a partition lacks a class."""
    parts = split_v2(records, seed)
    y = labels(records)
    out = {'seed': seed, 'partitions': _partition_summary(records, y, parts)}
    missing = [name for name, idx in parts.items() if len(set(y[idx])) < 2]
    if missing:
        return dict(out, status='insufficient_evidence', reason=f'partition(s) {missing} lack a class; no seed retry (preregistration §3)')
    tr, tu, ca, te = (parts[k] for k in ('train', 'tune', 'calibration', 'final_test'))
    controls = {'majority': metrics(y[te], np.repeat(y[tr].mean(), len(te))), 'timing': dict(TIMING_CONTROL)}
    for kind, prefixes in FEATURE_KINDS.items():
        names = vocabulary(records, tr, prefixes)
        if not names:
            controls[kind] = {'skipped': 'no_features'}; continue
        X = feature_matrix(records, names)
        candidates = []
        for c in C_GRID:
            model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
                                  LogisticRegression(C=c, max_iter=2000, random_state=seed))
            model.fit(X[tr], y[tr])
            candidates.append((log_loss(y[tu], model.predict_proba(X[tu]), labels=[0, 1]), c, model))
        _, chosen, base = min(candidates, key=lambda z: z[0])
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid')
        calibrated.fit(X[ca], y[ca])
        pred = calibrated.predict_proba(X[te])[:, 1]
        controls[kind] = dict(metrics(y[te], pred), selected_C=chosen, feature_count=len(names))
        joblib.dump({'model': calibrated, 'feature_names': names, 'threshold': .5, 'label_definition': LABEL_WARNING,
                     'synthetic': synthetic}, Path(outdir) / f'{kind}.joblib')
    increment = {'status': 'not_run', 'conclusion': 'no_demonstrated_visual_increment',
                 'reason': '§7.1 incremental null is defined by re-running the nested LOSO; only the §4 path runs it'}
    return dict(out, status='completed', controls=controls, probabilities_calibrated=True, final_test_used_once=True,
                incremental_test_s7_1=increment)


# ---------------------------------------------------------------- §4 LOSO diagnostic

def assert_fold_disjoint(records, tr, te):
    if {records[i]['session_id'] for i in tr} & {records[i]['session_id'] for i in te}:
        raise AssertionError('Session leakage')
    if {records[i]['hand_key'] for i in tr} & {records[i]['hand_key'] for i in te}:
        raise ValueError('Source hand appears in multiple partitions')


def _prep(records, prefixes, tr, ap):
    """Median imputation + scaling fitted on the training fold only (y-independent, so cached)."""
    names = vocabulary(records, tr, prefixes)
    if not names:
        return None
    Xtr = feature_matrix([records[i] for i in tr], names)
    Xap = feature_matrix([records[i] for i in ap], names)
    pre = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()).fit(Xtr)
    return pre.transform(Xtr), pre.transform(Xap), len(names)


def prepare_loso(records, prefixes):
    sess = np.array([r['session_id'] for r in records])
    idx = np.arange(len(records))
    folds = []
    for s in sorted(set(sess)):
        tr, te = idx[sess != s], idx[sess == s]
        assert_fold_disjoint(records, tr, te)
        outer = _prep(records, prefixes, tr, te)
        if outer is None:
            return None
        inner = []
        for t in sorted(set(sess[tr])):
            itr, ite = tr[sess[tr] != t], tr[sess[tr] == t]
            assert_fold_disjoint(records, itr, ite)
            Xi, Xa, _ = _prep(records, prefixes, itr, ite)
            inner.append((itr, ite, Xi, Xa))
        folds.append({'held_out': s, 'tr': tr, 'te': te, 'Xtr': outer[0], 'Xte': outer[1], 'n_features': outer[2], 'inner': inner})
    return folds


def _fit_proba(Xtr, ytr, Xap, C, seed):
    """Single-class training fold cannot be fit: predict its bluff rate (flagged as degenerate)."""
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xap), float(ytr.mean())), True
    m = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(Xtr, ytr)
    return m.predict_proba(Xap)[:, 1], False


def nested_loso_predict(folds, y, seed=SEED):
    """Outer LOSO; inner LOSO over training sessions picks C by pooled OOF log loss. No calibration."""
    oof = np.full(len(y), np.nan)
    detail, degenerate = [], 0
    for f in folds:
        cands = []
        for C in C_GRID:
            ys, ps = [], []
            for itr, ite, Xi, Xa in f['inner']:
                p, d = _fit_proba(Xi, y[itr], Xa, C, seed)
                ys.append(y[ite]); ps.append(p); degenerate += d
            cands.append((float(log_loss(np.concatenate(ys), np.concatenate(ps), labels=[0, 1])), C))
        _, C = min(cands, key=lambda z: z[0])
        p, d = _fit_proba(f['Xtr'], y[f['tr']], f['Xte'], C, seed)
        degenerate += d
        oof[f['te']] = p
        detail.append({'held_out_session': f['held_out'], 'n_test': len(f['te']), 'selected_C': C,
                       'inner_pooled_log_loss': {str(c): ll for ll, c in cands}, 'outer_fit_degenerate': d})
    return oof, detail, degenerate


def _auc_bacc(y, p):
    auc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bacc = float(balanced_accuracy_score(y, (p >= .5).astype(int)))
    return auc, bacc


def within_session_permutations(y, sessions, n_perm=N_PERM, seed=SEED):
    """Label permutations restricted to each session (preserves per-session class counts)."""
    rng = np.random.default_rng(seed)
    sessions = np.asarray(sessions)
    groups = [np.where(sessions == s)[0] for s in sorted(set(sessions))]
    out = np.empty((n_perm, len(y)), dtype=int)
    for k in range(n_perm):
        yy = np.array(y, dtype=int).copy()
        for g in groups:
            yy[g] = y[g][rng.permutation(len(g))]
        out[k] = yy
    return out


def _perm_chunk(folds, perms, seed):
    res = []
    for yp in perms:
        p, _, _ = nested_loso_predict(folds, yp, seed)
        res.append(_auc_bacc(yp, p))
    return res


def _map_perm_chunks(fn, perms, n_jobs, *args):
    """fn(first, chunk, *rest) over ordered chunks of perms; serial when n_jobs == 1. Order preserved."""
    first, rest = args[0], args[1:]
    if n_jobs == 1:
        return fn(first, perms, *rest)
    from joblib import Parallel, delayed, cpu_count
    n = cpu_count() if n_jobs in (None, -1) else n_jobs
    chunks = [c for c in np.array_split(perms, max(1, n * 4)) if len(c)]
    return [x for part in Parallel(n_jobs=n)(delayed(fn)(first, c, *rest) for c in chunks) for x in part]


def permutation_null(folds, y, sessions, n_perm=N_PERM, seed=SEED, n_jobs=-1):
    perms = within_session_permutations(y, sessions, n_perm, seed)
    res = _map_perm_chunks(_perm_chunk, perms, n_jobs, folds, seed)
    auc = np.array([np.nan if a is None else a for a, _ in res])
    bacc = np.array([b for _, b in res])
    return auc, bacc


# ---------------------------------------------------------------- §7.1 incremental test

def visual_row_permutations(sessions, n_perm=N_PERM, seed=SEED):
    """Within-session permutations of row indices (same generator as the label null)."""
    return within_session_permutations(np.arange(len(sessions)), sessions, n_perm, seed)


def permute_visual_features(records, perm):
    """Row i keeps label, session, hand key and context_* features; its static_/dynamic_ come from row perm[i]."""
    out = []
    for i, j in enumerate(perm):
        f = {k: v for k, v in records[i]['features'].items() if not k.startswith(VISUAL_PREFIXES)}
        f.update({k: v for k, v in records[j]['features'].items() if k.startswith(VISUAL_PREFIXES)})
        out.append(dict(records[i], features=f))
    return out


def _incremental_chunk(records, perms, y, context_auc, seed):
    res = []
    for perm in perms:
        folds = prepare_loso(permute_visual_features(records, perm), FEATURE_KINDS['context+full'])
        p, _, _ = nested_loso_predict(folds, y, seed)
        auc, _ = _auc_bacc(y, p)
        res.append(np.nan if auc is None or context_auc is None else auc - context_auc)
    return res


def incremental_null(records, y, sessions, context_auc, n_perm=N_PERM, seed=SEED, n_jobs=-1):
    """Null of AUC(context+full) - AUC(context): visual rows shuffled within session, labels and context
    fixed, whole nested LOSO (preprocessing, inner C selection) re-run per permutation. The context
    model sees neither the shuffled columns nor different labels, so AUC(context) stays at its observed value."""
    perms = visual_row_permutations(sessions, n_perm, seed)
    return np.array(_map_perm_chunks(_incremental_chunk, perms, n_jobs, records, y, context_auc, seed), dtype=float)


def incremental_test(records, y, sessions, oof, identity_auc, n_perm=N_PERM, seed=SEED, n_jobs=-1):
    ctx_auc, _ = _auc_bacc(y, oof['context'])
    cf_auc, _ = _auc_bacc(y, oof['context+full'])
    obs = None if ctx_auc is None or cf_auc is None else cf_auc - ctx_auc
    out = {'statistic': 'AUC(context+full) - AUC(context)', 'n_permutations': n_perm, 'seed': seed,
           'scheme': 'within-session shuffle of static_/dynamic_ feature rows; labels and context_* fixed; full nested LOSO re-run per permutation',
           'observed': {'context_roc_auc': ctx_auc, 'context+full_roc_auc': cf_auc, 'delta': obs},
           'identity_roc_auc': identity_auc, 'p_value_formula': '(1 + #{null >= observed}) / (1 + n_permutations)',
           'conclusion_rule': f'visual_incremental_information only if p < {INCREMENT_ALPHA} and AUC(context+full) > AUC(identity) (§7.1)'}
    if n_perm > 0:
        null = incremental_null(records, y, sessions, ctx_auc, n_perm, seed, n_jobs)
        out['p_value'] = _p_value(null, obs)
        finite = null[np.isfinite(null)]
        out['null_summary'] = {'mean': float(finite.mean()), 'p95': float(np.percentile(finite, 95))} if len(finite) else None
    else:
        out['p_value'] = None
    shown = (out['p_value'] is not None and out['p_value'] < INCREMENT_ALPHA and cf_auc is not None
             and identity_auc is not None and cf_auc > identity_auc)
    out['conclusion'] = 'visual_incremental_information' if shown else 'no_demonstrated_visual_increment'
    return out


def _p_value(null, obs):
    if obs is None:
        return None
    null = null[np.isfinite(null)]
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


def identity_predict(records, folds_index):
    """Per-bettor training-fold bluff rate; unseen bettor -> training-fold overall rate."""
    y = labels(records)
    oof = np.full(len(records), np.nan)
    unseen = 0
    for tr, te in folds_index:
        overall = float(y[tr].mean())
        rates = defaultdict(list)
        for i in tr:
            rates[records[i]['bettor']].append(y[i])
        for i in te:
            b = records[i]['bettor']
            if b in rates:
                oof[i] = float(np.mean(rates[b]))
            else:
                oof[i] = overall; unseen += 1
    return oof, unseen


def loso_index(records):
    sess = np.array([r['session_id'] for r in records])
    idx = np.arange(len(records))
    return [(idx[sess != s], idx[sess == s]) for s in sorted(set(sess))]


def run_loso(records, seed=SEED, n_perm=N_PERM, n_jobs=-1):
    """§4. Pilot diagnostic only; never reported as accuracy."""
    sessions = np.array([r['session_id'] for r in records])
    if len(set(sessions)) < 3:
        return {'status': 'insufficient_evidence', 'reason': 'LOSO needs >= 3 sessions (inner LOSO needs >= 2 training sessions)'}
    check_hand_keys_nested(records)
    y = labels(records)
    if len(set(y)) < 2:
        return {'status': 'insufficient_evidence', 'reason': 'only one class present'}
    findex = loso_index(records)
    oof = {}
    controls, fold_detail, degenerate = {}, {}, {}
    maj = np.full(len(y), np.nan)
    for tr, te in findex:
        maj[te] = y[tr].mean()
    oof['majority'] = maj
    oof['identity'], unseen = identity_predict(records, findex)
    folds_full = None
    for kind, prefixes in FEATURE_KINDS.items():
        folds = prepare_loso(records, prefixes)
        if folds is None:
            controls[kind] = {'skipped': 'no_features'}; continue
        p, detail, d = nested_loso_predict(folds, y, seed)
        oof[kind], fold_detail[kind], degenerate[kind] = p, detail, d
        if kind == 'full':
            folds_full = folds
    for kind, p in oof.items():
        controls[kind] = metrics(y, p)
        if kind in fold_detail:
            controls[kind].update(fold_detail=fold_detail[kind], degenerate_single_class_fits=degenerate[kind])
    controls['identity']['unseen_bettor_predictions'] = unseen
    controls['timing'] = dict(TIMING_CONTROL)

    focus = np.array([r['bettor'] == FOCUS_BETTOR for r in records])
    by_bettor = {}
    for name, mask in ((FOCUS_BETTOR, focus), (f'non_{FOCUS_BETTOR}', ~focus)):
        by_bettor[name] = {'n': int(mask.sum()), 'class_counts': {'bluff': int(y[mask].sum()), 'value': int((1 - y[mask]).sum())},
                           'controls': {k: (metrics(y[mask], p[mask]) if mask.any() else None) for k, p in oof.items()}}

    perm = {'n_permutations': n_perm, 'seed': seed, 'scheme': 'within-session label shuffle; full nested LOSO (incl. inner C selection) re-run per permutation',
            'model': 'full'}
    if folds_full is not None and n_perm > 0:
        obs_auc, obs_bacc = _auc_bacc(y, oof['full'])
        null_auc, null_bacc = permutation_null(folds_full, y, sessions, n_perm, seed, n_jobs)
        perm.update({'observed': {'roc_auc': obs_auc, 'balanced_accuracy': obs_bacc},
                     'p_value': {'roc_auc': _p_value(null_auc, obs_auc), 'balanced_accuracy': _p_value(null_bacc, obs_bacc)},
                     'null_summary': {k: {'mean': float(np.nanmean(v)), 'p95': float(np.nanpercentile(v, 95))} for k, v in (('roc_auc', null_auc), ('balanced_accuracy', null_bacc))},
                     'p_value_formula': '(1 + #{null >= observed}) / (1 + n_permutations)'})
    full_auc, id_auc = controls.get('full', {}).get('roc_auc'), controls['identity'].get('roc_auc')
    if 'context' in oof and 'context+full' in oof:
        increment = incremental_test(records, y, sessions, oof, id_auc, n_perm, seed, n_jobs)
    else:
        increment = {'status': 'skipped', 'reason': 'context or context+full control missing', 'conclusion': 'no_demonstrated_visual_increment'}
    flags = ['uncalibrated_probabilities', 'pilot_diagnostic_not_accuracy']
    if full_auc is None or id_auc is None or full_auc <= id_auc:
        flags.append('not_distinguishable_from_identity')
    if any(degenerate.values()):
        flags.append('degenerate_single_class_training_folds')
    flags.append(increment['conclusion'])
    return {'status': 'completed', 'not_reported_as_accuracy': True, 'probabilities_calibrated': False,
            'n_outer_folds': len(findex), 'C_grid': list(C_GRID), 'seed': seed, 'controls': controls,
            'by_bettor': by_bettor, 'permutation_test': perm, 'incremental_test_s7_1': increment, 'flags': flags,
            'identity_comparison': {'full_roc_auc': full_auc, 'identity_roc_auc': id_auc,
                                    'conclusion': 'not_distinguishable_from_identity' if 'not_distinguishable_from_identity' in flags else 'full_auc_exceeds_identity_auc'}}


# ---------------------------------------------------------------- orchestration

def _check_outdir(outdir):
    out = Path(outdir).resolve()
    if out == V3 or V3 not in out.parents:
        raise ValueError(f'--outdir must be a subdirectory of {V3}')
    return out


def run(records, meta, outdir, synthetic, subset, seed=SEED, n_perm=N_PERM, n_jobs=-1):
    out = _check_outdir(outdir)
    if len({r['synthetic'] for r in records}) != 1 or records[0]['synthetic'] != synthetic:
        raise ValueError('Never mix synthetic and actual data')
    if not synthetic and (out / 'report.json').exists():
        raise FileExistsError(f'{out}/report.json exists; final test is one-shot, use a new --outdir')
    out.mkdir(parents=True, exist_ok=True)
    y = labels(records)
    sess = Counter(r['session_id'] for r in records)
    report = {
        'synthetic_engineering_test': synthetic, 'data_kind': 'synthetic' if synthetic else 'real',
        'preregistration': {'path': str(PREREG.relative_to(ROOT)), 'sha256': sha256_file(PREREG)},
        'subset': subset, 'seed': seed, 'label_warning': LABEL_WARNING,
        'total_rows': meta['total_rows'], 'usable_rows': len(records),
        'unknown_or_excluded_rate': 1 - meta['usable_rows_before_subset'] / max(meta['total_rows'], 1),
        'excluded_sample_ids': meta['excluded_sample_ids'],
        'class_counts': {'bluff': int(y.sum()), 'value': int((1 - y).sum())},
        'tie_count_labelled_bluff': sum(r['tie'] for r in records),
        'multiway_rows': sum(r['active_opponents'] > 1 for r in records),
        'n_sessions': len(sess), 'rows_per_session': dict(sorted(sess.items())),
        'distinct_hands': len({r['hand_key'] for r in records}),
        'features': {'kinds': {k: list(v) for k, v in FEATURE_KINDS.items()}, 'timing': dict(TIMING_CONTROL)},
        'context_features_s7_1': meta.get('context_features'),
        'audit_exclusion_s7_2': meta.get('audit_exclusion', {'applied': False, 'variant': 'main'}),
    }
    if synthetic:
        report['WARNING'] = SYNTHETIC_WARNING
    try:
        primary = run_primary(records, out, synthetic, seed)
    except TooFewSessions as e:
        primary = {'status': 'insufficient_evidence', 'reason': f'fewer_than_8_sessions: {e}'}
    report['primary'] = primary
    if primary['status'] == 'completed':
        report['protocol_path'] = 'primary_s3'
        report['diagnostic_loso'] = None
    else:
        report['protocol_path'] = 'diagnostic_s4_loso'
        report['diagnostic_loso'] = run_loso(records, seed, n_perm, n_jobs)
    report['limitations'] = LIMITATIONS
    report['implementation_notes'] = [
        '§3 split = riverlab.train.split_indices on a shim keyed by the v2 hand key, plus identical leakage asserts on the v2 hand key.',
        '§4 permutation null is computed for the full (static+dynamic) model only.',
        '§4 single-class training folds (if any) predict the fold bluff rate and are counted as degenerate.',
        '§4 C ties resolve to the smallest C (same min() ordering as train.py).',
        'Bettor aliases applied before identity control and by-bettor metrics: ' + json.dumps(BETTOR_ALIASES) + ' (§6).',
        '§7.1 context_* features use action, active_opponents and amount only; the per-session amount median is taken over all usable rows '
        'of that session before subset/audit exclusion. log(amount/pot_before_bet) is added only if every usable row has it.',
        '§7.1 context and context+full use the same pipeline and C selection as the other controls.',
        '§7.1 incremental test runs on the §4 path only (its null is defined by re-running the nested LOSO); it reuses the '
        'within-session permutation generator (seed 17) applied to row indices.',
        '§7.2 audit exclusion (if applied) drops rows with any of ' + json.dumps(list(AUDIT_FLAGS)) + ' after loading; context medians are unchanged.',
    ]
    (out / 'report.json').write_text(json.dumps(report, indent=2, default=_json_default))
    return report


def _maybe_exclude(records, meta, exclude_audit_flags):
    if exclude_audit_flags is None:
        return records, dict(meta, audit_exclusion={'applied': False, 'variant': 'main'})
    kept, info = apply_audit_exclusion(records, exclude_audit_flags)
    return kept, dict(meta, audit_exclusion=info)


def train_real(subset, outdir, real=False, approved_by_owner=False, n_perm=N_PERM, n_jobs=-1, exclude_audit_flags=None):
    if not (real and approved_by_owner):
        raise PermissionError('Real-data training refused: requires both --real and --approved-by-owner')
    if n_perm != N_PERM:
        raise ValueError(f'Preregistration fixes {N_PERM} permutations for real data')
    _check_outdir(outdir)
    records, meta = load_real_rows(subset)
    records, meta = _maybe_exclude(records, meta, exclude_audit_flags)
    return run(records, meta, outdir, synthetic=False, subset=subset, n_perm=n_perm, n_jobs=n_jobs)


def train_synthetic(outdir, subset='all', n_sessions=10, rows_per_session=8, synthetic_seed=2026,
                    single_class_sessions=0, n_perm=N_PERM, n_jobs=-1, exclude_audit_flags=None):
    rows = make_synthetic_rows(n_sessions, rows_per_session, synthetic_seed, single_class_sessions=single_class_sessions)
    records, meta = load_synthetic_rows(rows, subset)
    records, meta = _maybe_exclude(records, meta, exclude_audit_flags)
    out = _check_outdir(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'synthetic_rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    report = run(records, meta, out, synthetic=True, subset=subset, n_perm=n_perm, n_jobs=n_jobs)
    report['synthetic_generator'] = {'n_sessions': n_sessions, 'rows_per_session': rows_per_session, 'seed': synthetic_seed,
                                     'single_class_sessions': single_class_sessions}
    (out / 'report.json').write_text(json.dumps(report, indent=2, default=_json_default))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--synthetic', action='store_true')
    mode.add_argument('--real', action='store_true')
    ap.add_argument('--approved-by-owner', action='store_true')
    ap.add_argument('--subset', choices=['all', 'heads_up'], default='all')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--n-perm', type=int, default=N_PERM)
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--synthetic-sessions', type=int, default=10)
    ap.add_argument('--synthetic-rows-per-session', type=int, default=8)
    ap.add_argument('--synthetic-seed', type=int, default=2026)
    ap.add_argument('--synthetic-single-class-sessions', type=int, default=0)
    ap.add_argument('--exclude-audit-flags', default=None, metavar='PATH',
                    help='§7.2 sensitivity variant: drop rows flagged in this audit_flags.json')
    a = ap.parse_args(argv)
    if a.synthetic:
        if a.approved_by_owner:
            ap.error('--approved-by-owner is only meaningful with --real')
        rep = train_synthetic(a.outdir, a.subset, a.synthetic_sessions, a.synthetic_rows_per_session, a.synthetic_seed,
                              a.synthetic_single_class_sessions, a.n_perm, a.n_jobs, a.exclude_audit_flags)
    else:
        if not a.approved_by_owner:
            print('REFUSED: real-data training requires both --real and --approved-by-owner (PREREGISTRATION.md).', file=sys.stderr)
            return 2
        rep = train_real(a.subset, a.outdir, real=True, approved_by_owner=True, n_perm=a.n_perm, n_jobs=a.n_jobs,
                         exclude_audit_flags=a.exclude_audit_flags)
    print(json.dumps({'outdir': str(Path(a.outdir).resolve()), 'data_kind': rep['data_kind'], 'protocol_path': rep['protocol_path'],
                      'primary_status': rep['primary']['status'],
                      'audit_exclusion': rep['audit_exclusion_s7_2'].get('variant')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
