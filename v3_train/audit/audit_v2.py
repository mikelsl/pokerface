"""PREREGISTRATION.md §7.2 feature-provenance audit. Label-blind.

For every ready v2 row (same file list as train_v2.ready_files()) read its masked_clip and run the
frozen MediaPipe models with the riverlab/features.py settings (CPU delegate, VIDEO mode, same
timestamps):

* face1 = FaceLandmarker num_faces=1 + blendshapes (exact frozen-extractor settings) plus the
  facial transformation matrix output -> the face the features were computed from: bbox, pitch.
* face2 = FaceLandmarker num_faces=2 -> used ONLY to count faces (multi_face flag).
* hands = HandLandmarker num_hands=2 (frozen settings) -> hand presence per frame.

Flags (§7.2, thresholds fixed in the preregistration, not tuned here):
* camera_cut_or_zoom: any adjacent frame pair (both with a face) with bbox area ratio > 1.5 or
  < 1/1.5, or centre shift > 0.5 x bbox width (width of the earlier frame of the pair).
* multi_face: >= 2 faces in any frame (face2).
* head_down: > 30 % of face-detected frames with head pitch > 25 deg (downward, camera-relative).

Pitch: R = orthonormalised 3x3 block of the face1 facial transformation matrix (canonical face
model -> camera space; x right, y up, z towards the camera). forward = R @ (0, 0, 1) is the
direction the face points; pitch_down = atan2(-forward_y, forward_z) in degrees (> 0 = chin
down / face turned towards the floor). Sign is sanity-checked visually (see AUDIT.md) and against
a landmark proxy (nose tip position between eye line and chin).

Never reads labels: rows are reduced to a whitelist immediately after json parsing and the
label-bearing fields are asserted absent.
"""
import sys, os, json, math, hashlib, argparse, time
from pathlib import Path

sys.dont_write_bytecode = True
AUDIT = Path(__file__).resolve().parent
V3 = AUDIT.parent
ROOT = V3.parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'cache'))
BASELINE = ROOT / 'code/source_baseline/poker-river-research'
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

import numpy as np
from riverlab.contract import FACE_SHA256, HAND_SHA256

FACE_MODEL = ROOT / 'models/face_landmarker.task'
HAND_MODEL = ROOT / 'models/hand_landmarker.task'
KEEP = ('sample_id', 'session_id', 'status', 'eligible', 'model_view_reviewed', 'masked_clip', 'feature_provenance', 'features')
FORBIDDEN = ('comparison', 'label', 'bettor_cards', 'opponents', 'opponent_cards', 'bettor_best5', 'opponent_best5',
             'bettor_rank', 'opponent_rank', 'relation', 'label_definition', 'opponent', 'board')
AREA_RATIO = 1.5
SHIFT_OVER_WIDTH = .5
PITCH_DEG = 25.
HEAD_DOWN_FRACTION = .30
FLAG_NAMES = ('camera_cut_or_zoom', 'multi_face', 'head_down')
# Landmark indices of the 468-point face mesh used for the pitch proxy.
NOSE_TIP, CHIN, EYE_L, EYE_R = 1, 152, 33, 263


def ready_files():
    """Same list as train_v2.ready_files() (duplicated so this script never imports the label loader)."""
    import glob
    return [ROOT / 'v2/analysis/ready_samples.jsonl'] + [Path(p) for p in sorted(glob.glob(str(ROOT / 'v2/expansion/*/ready_samples.jsonl')))]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows():
    """Ready rows reduced to a label-free whitelist right after parsing."""
    rows = []
    for f in ready_files():
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            r = {k: raw[k] for k in KEEP if k in raw}
            del raw
            assert not any(k in r for k in FORBIDDEN), 'label-bearing field survived the whitelist'
            if r.get('status') != 'ready' or r.get('eligible') is not True or r.get('model_view_reviewed') is not True:
                continue
            if sha256(r['masked_clip']) != r['feature_provenance']['masked_clip_sha256']:
                raise ValueError(f"{r['sample_id']}: masked clip hash mismatch")
            rows.append(r)
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate sample ids')
    return rows


def _landmarkers():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    cpu = python.BaseOptions.Delegate.CPU
    vid = vision.RunningMode.VIDEO
    face1 = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL), delegate=cpu), running_mode=vid, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    face2 = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL), delegate=cpu), running_mode=vid, num_faces=2))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL), delegate=cpu), running_mode=vid, num_hands=2))
    return face1, face2, hands


def head_angles(matrix):
    """(pitch_down, yaw, roll) in degrees from a 4x4 facial transformation matrix."""
    M = np.asarray(matrix, dtype=float).reshape(4, 4)
    U, _, Vt = np.linalg.svd(M[:3, :3])
    R = U @ Vt
    fwd = R @ np.array([0., 0., 1.])
    up = R @ np.array([0., 1., 0.])
    pitch = math.degrees(math.atan2(-fwd[1], fwd[2]))
    yaw = math.degrees(math.atan2(fwd[0], fwd[2]))
    roll = math.degrees(math.atan2(-up[0], up[1]))
    return pitch, yaw, roll


def nose_proxy(pts_px):
    """(nose_y - eye_line_y) / (chin_y - eye_line_y): grows when the head pitches down (image y down)."""
    eye_y = (pts_px[EYE_L, 1] + pts_px[EYE_R, 1]) / 2
    den = pts_px[CHIN, 1] - eye_y
    return float((pts_px[NOSE_TIP, 1] - eye_y) / den) if den > 1e-6 else None


def analyse_clip(path):
    import cv2
    import mediapipe as mp
    face1, face2, hands = _landmarkers()
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, n = [], 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            im = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ts = round(n * 1000 / max(fps, 1))  # identical timestamps to riverlab/features.py
            f1, f2, h = face1.detect_for_video(im, ts), face2.detect_for_video(im, ts), hands.detect_for_video(im, ts)
            fr = {'i': n, 'face': bool(f1.face_landmarks), 'n_faces': len(f2.face_landmarks), 'n_hands': len(h.hand_landmarks)}
            if f1.face_landmarks:
                pts = np.array([[p.x, p.y, p.z] for p in f1.face_landmarks[0]])
                px = pts[:, :2] * [W, H]
                x0, y0 = px.min(0); x1, y1 = px.max(0)
                fr.update(bbox=[round(float(v), 2) for v in (x0, y0, x1, y1)], w=float(x1 - x0), h=float(y1 - y0),
                          area=float((x1 - x0) * (y1 - y0)), cx=float((x0 + x1) / 2), cy=float((y0 + y1) / 2),
                          center_x=float(pts[:, 0].mean()), center_y=float(pts[:, 1].mean()),
                          spread=float(np.linalg.norm(pts.max(0) - pts.min(0))), nose_proxy=nose_proxy(px))
                if f1.facial_transformation_matrixes:
                    fr['pitch'], fr['yaw'], fr['roll'] = head_angles(f1.facial_transformation_matrixes[0])
            frames.append(fr); n += 1
    finally:
        cap.release(); face1.close(); face2.close(); hands.close()
    return frames, {'fps': fps, 'width': W, 'height': H}


def flags_for(frames):
    """Apply the §7.2 thresholds. Returns (flags, summary)."""
    pairs = []
    for a, b in zip(frames, frames[1:]):
        if not (a['face'] and b['face']):
            continue
        ratio = b['area'] / a['area'] if a['area'] > 0 else math.inf
        shift = math.hypot(b['cx'] - a['cx'], b['cy'] - a['cy']) / a['w'] if a['w'] > 0 else math.inf
        pairs.append({'pair': [a['i'], b['i']], 'area_ratio': ratio, 'shift_over_width': shift,
                      'cut': ratio > AREA_RATIO or ratio < 1 / AREA_RATIO or shift > SHIFT_OVER_WIDTH})
    face_frames = [f for f in frames if f['face']]
    pitched = [f for f in face_frames if 'pitch' in f]
    down = [f for f in pitched if f['pitch'] > PITCH_DEG]
    down_frac = len(down) / len(pitched) if pitched else None
    face_seq = [int(f['face']) for f in frames]
    hand_seq = [int(f['n_hands'] > 0) for f in frames]

    def runs(seq, v):
        best = cur = 0
        for s in seq:
            cur = cur + 1 if s == v else 0
            best = max(best, cur)
        return best

    flags = []
    if any(p['cut'] for p in pairs):
        flags.append('camera_cut_or_zoom')
    if any(f['n_faces'] >= 2 for f in frames):
        flags.append('multi_face')
    if down_frac is not None and down_frac > HEAD_DOWN_FRACTION:
        flags.append('head_down')
    pitches = [f['pitch'] for f in pitched]
    summary = {
        'frames': len(frames), 'face_frames': len(face_frames), 'compared_pairs': len(pairs),
        'cut_pairs': [p['pair'] for p in pairs if p['cut']],
        'max_area_ratio': max([max(p['area_ratio'], 1 / p['area_ratio']) for p in pairs if p['area_ratio'] > 0], default=None),
        'max_shift_over_width': max([p['shift_over_width'] for p in pairs], default=None),
        'face_count_max': max(f['n_faces'] for f in frames) if frames else 0,
        'multi_face_frames': [f['i'] for f in frames if f['n_faces'] >= 2],
        'face_presence_changes': int(np.sum(np.abs(np.diff(face_seq)))) if len(face_seq) > 1 else 0,
        'head_down_frames': [f['i'] for f in down], 'head_down_fraction': down_frac,
        'pitch_median': float(np.median(pitches)) if pitches else None,
        'pitch_min': min(pitches, default=None), 'pitch_max': max(pitches, default=None),
        'hand_frames': sum(hand_seq), 'hand_presence_changes': int(np.sum(np.abs(np.diff(hand_seq)))) if len(hand_seq) > 1 else 0,
        'longest_hand_gap': runs(hand_seq, 0),
    }
    return flags, summary


def reproduction_check(row, frames):
    """Recompute a few stored static_* features from face1 to prove we audited the same face track."""
    face = [f for f in frames if f['face']]
    if not face:
        return {'checked': 0}
    mine = {'static_face_center_x_mean': np.mean([f['center_x'] for f in face]),
            'static_face_center_y_mean': np.mean([f['center_y'] for f in face]),
            'static_face_spread_mean': np.mean([f['spread'] for f in face]),
            'static_hand_count_mean': np.mean([f['n_hands'] for f in face])}
    diffs = {k: abs(float(v) - float(row['features'][k])) for k, v in mine.items() if k in row['features']}
    return {'checked': len(diffs), 'max_abs_diff': max(diffs.values()) if diffs else None, 'diffs': diffs}


def _round(fr):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in fr.items()}


def contact_sheet(clip, frames, title, out_png, cols=5, cell_h=200):
    import cv2
    cap = cv2.VideoCapture(str(clip)); imgs = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        imgs.append(bgr)
    cap.release()
    cells = []
    for img, fr in zip(imgs, frames):
        img = img.copy()
        if fr['face']:
            x0, y0, x1, y1 = (int(round(v)) for v in fr['bbox'])
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        s = cell_h / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * s), cell_h))
        pitch = f"p{fr['pitch']:+.0f}" if 'pitch' in fr else 'p--'
        txt = f"#{fr['i']} f{fr['n_faces']} h{fr['n_hands']} {pitch}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
        cv2.putText(img, txt, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(img)
    cw = max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, 0, 0, cw - c.shape[1], cv2.BORDER_CONSTANT) for c in cells]
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    grid = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
    head = np.zeros((26, grid.shape[1], 3), np.uint8)
    cv2.putText(head, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_png), np.vstack([head, grid]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', help='sample ids (debug); default all ready rows')
    ap.add_argument('--out', default=str(AUDIT))
    ap.add_argument('--sheets', nargs='*', default=None, help='extra sample ids to render contact sheets for')
    a = ap.parse_args(argv)
    if sha256(FACE_MODEL) != FACE_SHA256 or sha256(HAND_MODEL) != HAND_SHA256:
        raise ValueError('Frozen model hash mismatch')
    out = Path(a.out); (out / 'review').mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    if a.only:
        rows = [r for r in rows if r['sample_id'] in set(a.only)]
    t0 = time.time(); result, repro, clips = {}, {}, {}
    for k, r in enumerate(rows):
        frames, video = analyse_clip(r['masked_clip'])
        flags, summary = flags_for(frames)
        repro[r['sample_id']] = reproduction_check(r, frames)
        result[r['sample_id']] = {'flags': flags, 'session_id': r['session_id'], 'video': video, 'summary': summary,
                                  'per_frame': [_round(f) for f in frames]}
        clips[r['sample_id']] = r['masked_clip']
        print(json.dumps({'k': k + 1, 'n': len(rows), 'sample': r['sample_id'], 'flags': flags, 's': round(time.time() - t0, 1)}), flush=True)
    (out / 'audit_flags.json').write_text(json.dumps(result, indent=1))
    counts = {f: sum(f in v['flags'] for v in result.values()) for f in FLAG_NAMES}
    run_meta = {'n_rows': len(result), 'flag_counts': counts, 'any_flag': sum(bool(v['flags']) for v in result.values()),
                'thresholds': {'area_ratio': AREA_RATIO, 'shift_over_width': SHIFT_OVER_WIDTH, 'pitch_deg': PITCH_DEG,
                               'head_down_fraction': HEAD_DOWN_FRACTION},
                'models': {'face_sha256': FACE_SHA256, 'hand_sha256': HAND_SHA256}, 'files': [str(f.relative_to(ROOT)) for f in ready_files()],
                'reproduction_check': repro, 'label_fields_read': False, 'seconds': round(time.time() - t0, 1)}
    (out / 'audit_run.json').write_text(json.dumps(run_meta, indent=1))
    sheet_ids = [s for s, v in result.items() if v['flags']] + [s for s in (a.sheets or []) if s in result]
    for sid in dict.fromkeys(sheet_ids):
        v = result[sid]
        title = f"{sid}  flags={','.join(v['flags']) or 'none'}  cut_pairs={v['summary']['cut_pairs']}"
        contact_sheet(clips[sid], v['per_frame'], title, out / 'review' / f'{sid}.png')
    print(json.dumps({'done': True, 'flag_counts': counts, 'any_flag': run_meta['any_flag'], 'seconds': run_meta['seconds']}), flush=True)


if __name__ == '__main__':
    main()
