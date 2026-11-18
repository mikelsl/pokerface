"""Review stills retain cards; model clips mask overlays then crop bettor ROI."""
import json, subprocess
from pathlib import Path

def run(args):
    return subprocess.run(args,check=True,capture_output=True,text=True)

def probe(path):
    return json.loads(run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)]).stdout)

def prepare(row,source,outdir):
    from .schema import audit
    row=audit(row)
    if not row['eligible']: raise ValueError(row['exclusion_reasons'])
    info=probe(source)
    video=next((s for s in info['streams'] if s.get('codec_type')=='video'),None)
    if not video: raise ValueError('Source has no video stream')
    duration_source=float(info['format']['duration'])
    if row['actual_end']>duration_source+.001: raise ValueError('Requested source window exceeds EOF')
    for x,y,w,h in [row['actor_roi']]+row['mask_rects']:
        if any(int(v)!=v for v in (x,y,w,h)) or x+w>video['width'] or y+h>video['height']: raise ValueError('ROI or mask outside source dimensions')
    out=Path(outdir); (out/'review').mkdir(parents=True,exist_ok=True); (out/'masked').mkdir(parents=True,exist_ok=True)
    sid=row['sample_id']
    if not sid.replace('_','').replace('-','').isalnum(): raise ValueError('Unsafe sample_id')
    review=out/'review'/f'{sid}.png'; clip=out/'masked'/f'{sid}.mp4'
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(row['bet_time']),'-i',str(source),'-frames:v','1',str(review)])
    # Crop and masking coordinates are source pixels. Keep only pre-response frames, no audio.
    filters=[f'drawbox=x={x}:y={y}:w={w}:h={h}:color=black:t=fill' for x,y,w,h in row['mask_rects']]
    x,y,w,h=row['actor_roi']; filters += [f'crop={w}:{h}:{x}:{y}','fps=5:eof_action=pass']
    duration=row['model_cutoff']-row['actual_start']
    if duration<=0: raise ValueError('Empty model window')
    filters = [f'trim=start={row["actual_start"]}:end={row["model_cutoff"]}', 'setpts=PTS-STARTPTS'] + filters
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(source),'-vf',','.join(filters),'-an','-c:v','libx264','-pix_fmt','yuv420p',str(clip)])
    encoded=probe(clip)
    if abs(float(encoded['format']['duration'])-duration)>.21: raise ValueError('Encoded duration differs from prespecified window')
    row.update(review_screenshot=str(review),masked_clip=str(clip),model_view_reviewed=False,model_duration=duration,clip_probe=encoded,masked_source_start=row['actual_start'],masked_source_end=row['model_cutoff'])
    return row
