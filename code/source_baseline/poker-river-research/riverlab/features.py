"""Frozen official MediaPipe Tasks CPU extraction. No training of vision weights."""
import hashlib
from pathlib import Path
import numpy as np

def extract(row, face_model, hand_model, min_frames=10, min_face_fraction=.5):
    if row.get("synthetic") is not True and row.get("model_view_reviewed") is not True:
        raise ValueError("Masked model view must be manually reviewed before extraction")
    from .contract import FACE_SHA256,HAND_SHA256,sha256
    if sha256(face_model)!=FACE_SHA256 or sha256(hand_model)!=HAND_SHA256: raise ValueError('Frozen model hash mismatch')
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    face=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(face_model),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.VIDEO,num_faces=1,output_face_blendshapes=True))
    hands=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(hand_model),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.VIDEO,num_hands=2))
    cap=cv2.VideoCapture(row['masked_clip']); fps=cap.get(cv2.CAP_PROP_FPS)
    values=[]; frames=0; face_count=0; hand_count=0
    try:
        while True:
            ok,bgr=cap.read()
            if not ok: break
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); im=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            ts=round(frames*1000/max(fps,1)); f=face.detect_for_video(im,ts); h=hands.detect_for_video(im,ts); frames+=1
            face_count+=bool(f.face_landmarks); hand_count+=bool(h.hand_landmarks)
            if f.face_landmarks:
                pts=np.array([[p.x,p.y,p.z] for p in f.face_landmarks[0]])
                # Fixed geometry/blendshape summaries; all model weights frozen.
                blend={b.category_name:float(b.score) for b in (f.face_blendshapes[0] if f.face_blendshapes else [])}
                v={'face_center_x':float(pts[:,0].mean()),'face_center_y':float(pts[:,1].mean()),'face_spread':float(np.linalg.norm(pts.max(0)-pts.min(0)))}
                v.update({'blend_'+k:x for k,x in blend.items()})
                v['hand_count']=float(len(h.hand_landmarks)); values.append(v)
    finally:
        cap.release(); face.close(); hands.close()
    reasons=[]
    if frames<min_frames: reasons.append('too_few_frames')
    if face_count/max(frames,1)<min_face_fraction: reasons.append('insufficient_face_coverage')
    feats={}
    if values:
        for k in sorted(set.intersection(*(set(v) for v in values))):
            x=np.array([v[k] for v in values]); feats['static_'+k+'_mean']=float(x.mean()); feats['dynamic_'+k+'_sd']=float(x.std())
            feats['dynamic_'+k+'_delta']=float(np.abs(np.diff(x)).mean()) if len(x)>1 else 0.
    feats['timing_duration']=float(row['model_duration'])
    return dict(row,features=feats,eligible=bool(row.get('eligible')) and not reasons,abstention_reasons=reasons,quality={'frames':frames,'face_fraction':face_count/max(frames,1),'hand_fraction':hand_count/max(frames,1)},feature_provenance={'extractor':'mediapipe_tasks_cpu_frozen','contract_version':'v1','masked_clip_sha256':sha256(row['masked_clip']),'face_sha256':hashlib.sha256(Path(face_model).read_bytes()).hexdigest(),'hand_sha256':hashlib.sha256(Path(hand_model).read_bytes()).hexdigest()})
