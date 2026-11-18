import shutil
import pytest
from riverlab.cli import synthetic,read
from riverlab.media import prepare,run,probe

@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='FFmpeg unavailable')
def test_masking_and_duration(tmp_path):
    import cv2
    path=tmp_path/'source.mp4'
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i','color=c=white:s=160x120:r=10:d=30','-c:v','libx264','-pix_fmt','yuv420p',str(path)])
    manifest=tmp_path/'s.jsonl';synthetic(manifest);r=read(manifest)[0]
    r.update(actor_roi=[0,0,160,120],mask_rects=[[0,90,160,30]])
    result=prepare(r,path,tmp_path/'outputs')
    assert abs(float(result['clip_probe']['format']['duration'])-14)<.21
    original=cv2.imread(result['review_screenshot']);cap=cv2.VideoCapture(result['masked_clip']);ok,masked=cap.read();cap.release()
    assert ok and original[105,80].mean()>245 and masked[105,80].mean()<10
    assert masked[20,80].mean()>245
    assert all(s['codec_type']!='audio' for s in result['clip_probe']['streams'])

@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='FFmpeg unavailable')
def test_bounds_and_eof_rejected(tmp_path):
    path=tmp_path/'source.mp4'
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i','color=c=white:s=160x120:r=10:d=30','-c:v','libx264','-pix_fmt','yuv420p',str(path)])
    manifest=tmp_path/'s.jsonl';synthetic(manifest);r=read(manifest)[0]
    with pytest.raises(ValueError,match='outside source'):prepare(dict(r,actor_roi=[0,0,1000,1000]),path,tmp_path/'out')
    with pytest.raises(ValueError,match='EOF'):prepare(dict(r,actual_end=99),path,tmp_path/'out')
