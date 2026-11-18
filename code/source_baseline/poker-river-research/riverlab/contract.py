"""Versioned extractor-owned feature and temporal contract (set before actual data)."""
import hashlib, math
from pathlib import Path
PRE_BET_SECONDS=10.0
POST_BET_SECONDS=4.0
POLICY='fixed_bet_window_v1'
FACE_SHA256='64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
HAND_SHA256='fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
BLENDS='''_neutral browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight cheekPuff cheekSquintLeft cheekSquintRight eyeBlinkLeft eyeBlinkRight eyeLookDownLeft eyeLookDownRight eyeLookInLeft eyeLookInRight eyeLookOutLeft eyeLookOutRight eyeLookUpLeft eyeLookUpRight eyeSquintLeft eyeSquintRight eyeWideLeft eyeWideRight jawForward jawLeft jawOpen jawRight mouthClose mouthDimpleLeft mouthDimpleRight mouthFrownLeft mouthFrownRight mouthFunnel mouthLeft mouthLowerDownLeft mouthLowerDownRight mouthPressLeft mouthPressRight mouthPucker mouthRight mouthRollLower mouthRollUpper mouthShrugLower mouthShrugUpper mouthSmileLeft mouthSmileRight mouthStretchLeft mouthStretchRight mouthUpperUpLeft mouthUpperUpRight noseSneerLeft noseSneerRight'''.split()
BASE=['face_center_x','face_center_y','face_spread','hand_count']+['blend_'+x for x in BLENDS]
ALLOWED={f'{prefix}_{base}_{stat}' for base in BASE for prefix,stat in [('static','mean'),('dynamic','sd'),('dynamic','delta')]}|{'timing_duration'}
SYNTHETIC_ALLOWED={'dynamic_fixture_noise','static_fixture_noise','timing_fixture_noise'}
def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate_features(row):
    features=row.get('features',{})
    if not isinstance(features,dict) or not features:raise ValueError('No numeric feature mapping')
    synthetic=row.get('synthetic') is True
    allowed=SYNTHETIC_ALLOWED if synthetic else ALLOWED
    bad=set(features)-allowed
    if bad:raise ValueError(f'Unapproved feature keys: {sorted(bad)}')
    if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in features.values()):raise ValueError('Nonfinite or nonnumeric feature')
    if synthetic:return
    if str(row.get('source_url','')).startswith('synthetic:') or any('synthetic' in str(row.get(k,'')).lower() for k in ('sample_id','session_id','hand_id')):raise ValueError('Synthetic marker in actual data')
    p=row.get('feature_provenance',{})
    if p.get('extractor')!='mediapipe_tasks_cpu_frozen' or p.get('contract_version')!='v1' or p.get('face_sha256')!=FACE_SHA256 or p.get('hand_sha256')!=HAND_SHA256:raise ValueError('Unverified extractor/model provenance')
    clip=Path(row.get('masked_clip',''))
    if not clip.is_file() or p.get('masked_clip_sha256')!=sha256(clip):raise ValueError('Masked clip integrity failure')
    if features.get('timing_duration')!=PRE_BET_SECONDS+POST_BET_SECONDS:raise ValueError('Unexpected timing feature')
    q=row.get('quality',{})
    if q.get('frames',0)<10 or q.get('face_fraction',0)<.5:raise ValueError('Insufficient feature quality')
