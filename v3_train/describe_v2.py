"""Descriptive statistics of the real v2 ready rows. Counts and quality only:
no feature values are read, and nothing is fit, split or scored."""
import sys, json, statistics
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_v2 as T

OUT = HERE / 'descriptive'


def _lc(rs):
    c = Counter(r['label'] for r in rs)
    return {'n': len(rs), 'bluff': c['bluff'], 'value': c['value']}


def _dist(xs):
    return {'min': min(xs), 'median': statistics.median(xs), 'max': max(xs)}


def describe():
    recs, meta = T.load_real_rows('all')  # validation + grouping cross-check only
    for r in recs:
        r.pop('features')  # make sure no feature value can enter the statistics
    by = lambda key: {k: _lc(v) for k, v in sorted(_group(recs, key).items())}
    sessions = sorted({r['session_id'] for r in recs})
    bettors = sorted({r['bettor'] for r in recs}, key=lambda b: (-sum(r['bettor'] == b for r in recs), b))
    matrix = {b: {s: sum(r['bettor'] == b and r['session_id'] == s for r in recs) for s in sessions} for b in bettors}
    face = [r['quality']['face_fraction'] for r in recs]
    hand = [r['quality']['hand_fraction'] for r in recs]
    groups = _group(recs, 'hand_key')
    mar = [r for r in recs if r['bettor'] == T.FOCUS_BETTOR]
    sess_of_bettor = {b: sorted({r['session_id'] for r in recs if r['bettor'] == b}) for b in bettors}
    stats = {
        'note': 'Real rows; counts and quality only. No feature values, no feature-label analysis, no training/splitting.',
        'total_rows': meta['total_rows'], 'usable_rows': len(recs), 'excluded_sample_ids': meta['excluded_sample_ids'],
        'labels': _lc(recs),
        'n_sessions': len(sessions), 'by_session': by('session_id'),
        'n_sources': len({r['source'] for r in recs}),
        'by_source': {k: dict(_lc(v), session=v[0]['session_id']) for k, v in sorted(_group(recs, 'source').items())},
        'n_bettors': len(bettors),
        'by_bettor': {b: dict(_lc([r for r in recs if r['bettor'] == b]), sessions=sess_of_bettor[b]) for b in bettors},
        'bettors_in_one_session_only': [b for b in bettors if len(sess_of_bettor[b]) == 1],
        'bettor_by_session_matrix': matrix,
        'heads_up_vs_multiway': {'heads_up': _lc([r for r in recs if r['active_opponents'] == 1]),
                                 'multiway': _lc([r for r in recs if r['active_opponents'] > 1]),
                                 'by_active_opponents': {str(k): _lc(v) for k, v in sorted(_group(recs, 'active_opponents').items())}},
        'by_action': by('action'),
        'ties_labelled_bluff': {'count': sum(r['tie'] for r in recs), 'sample_ids': [r['sample_id'] for r in recs if r['tie']]},
        'hands': {'distinct_hand_keys': len(groups),
                  'same_hand_action_groups': sorted(sorted(r['sample_id'] for r in v) for v in groups.values() if len(v) > 1),
                  'legacy_hand_key_rows': [r['sample_id'] for r in recs if isinstance(r['hand_key'], tuple)]},
        'quality': {'frames': dict(Counter(r['quality']['frames'] for r in recs)),
                    'face_fraction': _dist(face), 'hand_fraction': _dist(hand),
                    'zero_hand_rows': {'count': sum(h == 0 for h in hand), 'sample_ids': [r['sample_id'] for r in recs if r['quality']['hand_fraction'] == 0]},
                    'hand_fraction_below_0_1': sum(h < .1 for h in hand),
                    'face_fraction_below_0_75': sum(f < .75 for f in face)},
        'focus_bettor': {'rows': len(mar), 'share': len(mar) / len(recs), 'labels': _lc(mar), 'sessions': sess_of_bettor.get(T.FOCUS_BETTOR, []),
                    'share_by_session': {s: f"{matrix[T.FOCUS_BETTOR][s]}/{by('session_id')[s]['n']}" for s in sessions}},
    }
    return stats


def _group(recs, key):
    g = defaultdict(list)
    for r in recs:
        g[r[key]].append(r)
    return g


def to_markdown(s):
    L = ['# pokerface v2 真实 ready 样本描述性统计', '',
         '> 只统计计数和画面质量。没有读取任何特征值，没有做特征与标签的关联分析，也没有训练、划分或打分。',
         '> 标签是摊牌牌力代理（`comparison.label`），不是意图；平局记为 bluff。', '',
         '## 总览', '',
         f"- 样本：{s['usable_rows']} 条（读入 {s['total_rows']} 条，全部通过 v2 契约校验与分组交叉核对，排除 {len(s['excluded_sample_ids'])} 条）",
         f"- 标签：bluff {s['labels']['bluff']} / value {s['labels']['value']}",
         f"- 场次 {s['n_sessions']} 个，来源视频 {s['n_sources']} 个，下注者 {s['n_bettors']} 人",
         f"- 不同手牌 {s['hands']['distinct_hand_keys']} 手；同一手多动作组：{s['hands']['same_hand_action_groups']}",
         f"- 用 legacy 手牌键（缺 `duplicate_hand_key`）的行：{s['hands']['legacy_hand_key_rows']}",
         f"- 平局（记为 bluff）：{s['ties_labelled_bluff']['count']} 条 {s['ties_labelled_bluff']['sample_ids']}",
         f"- 独立场次 {s['n_sessions']} < 8：按预先登记，§3 主协议不会运行，真实训练（需 项目负责人 批准）将直接走 §4 LOSO 诊断。", '',
         '## 按场次', '', '| 场次 | 条数 | bluff | value |', '|---|---|---|---|']
    L += [f"| {k} | {v['n']} | {v['bluff']} | {v['value']} |" for k, v in s['by_session'].items()]
    L += ['', '## 按来源视频', '', '| 来源 | 场次 | 条数 | bluff | value |', '|---|---|---|---|---|']
    L += [f"| {k} | {v['session']} | {v['n']} | {v['bluff']} | {v['value']} |" for k, v in s['by_source'].items()]
    L += ['', '## 按下注者', '', '| 下注者 | 条数 | bluff | value | 出现场次数 |', '|---|---|---|---|---|']
    L += [f"| {k} | {v['n']} | {v['bluff']} | {v['value']} | {len(v['sessions'])} |" for k, v in s['by_bettor'].items()]
    L += ['', f"只在一个场次出现的下注者（LOSO 留出该场次时，identity 对照会退回总体比例）：{', '.join(s['bettors_in_one_session_only'])}", '']
    sess = list(s['by_session'])
    L += ['## 下注者 × 场次（条数）', '', '| 下注者 | ' + ' | '.join(x.replace('HCL_', '') for x in sess) + ' |', '|---' * (len(sess) + 1) + '|']
    L += [f"| {b} | " + ' | '.join(str(row[x] or '·') for x in sess) + ' |' for b, row in s['bettor_by_session_matrix'].items()]
    h = s['heads_up_vs_multiway']
    L += ['', '## 单挑 vs 多人', '', '| 类型 | 条数 | bluff | value |', '|---|---|---|---|',
          f"| 单挑 | {h['heads_up']['n']} | {h['heads_up']['bluff']} | {h['heads_up']['value']} |",
          f"| 多人 | {h['multiway']['n']} | {h['multiway']['bluff']} | {h['multiway']['value']} |"]
    L += [f"| 其中 {k} 个对手 | {v['n']} | {v['bluff']} | {v['value']} |" for k, v in h['by_active_opponents'].items()]
    L += ['', '## 动作', '', '| 动作 | 条数 | bluff | value |', '|---|---|---|---|']
    L += [f"| {k} | {v['n']} | {v['bluff']} | {v['value']} |" for k, v in s['by_action'].items()]
    q = s['quality']
    fmt = lambda d: f"{d['min']:.2f} / {d['median']:.2f} / {d['max']:.2f}"
    L += ['', '## 画面质量（不分标签）', '',
          f"- 帧数：{q['frames']}",
          f"- face_fraction 最小/中位/最大：{fmt(q['face_fraction'])}；低于 0.75 的 {q['face_fraction_below_0_75']} 条",
          f"- hand_fraction 最小/中位/最大：{fmt(q['hand_fraction'])}；低于 0.1 的 {q['hand_fraction_below_0_1']} 条",
          f"- 手部检出率为 0 的 {q['zero_hand_rows']['count']} 条：{', '.join(q['zero_hand_rows']['sample_ids'])}（没检出手可能只是手不在画面里，不能当行为线索）"]
    m = s['focus_bettor']
    L += ['', '## PLAYER_A 占比', '',
          f"- {m['rows']}/{s['usable_rows']} 条（{m['share']:.1%}），bluff {m['labels']['bluff']} / value {m['labels']['value']}，出现在 {len(m['sessions'])} 个场次",
          '- 各场次中 PLAYER_A 条数：' + '，'.join(f'{k} {v}' for k, v in m['share_by_session'].items()),
          '- 一个选手占比这么高，模型很可能学到"认人"而不是表情；§4 的 identity 对照和 PLAYER_A/非 PLAYER_A 分开报告就是为此。', '']
    return '\n'.join(L)


if __name__ == '__main__':
    s = describe()
    OUT.mkdir(exist_ok=True)
    (OUT / 'stats.json').write_text(json.dumps(s, indent=2, ensure_ascii=False, default=T._json_default))
    (OUT / 'STATS.md').write_text(to_markdown(s))
    print(OUT / 'stats.json'); print(OUT / 'STATS.md')
