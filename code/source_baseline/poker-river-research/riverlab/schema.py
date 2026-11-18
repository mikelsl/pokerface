"""Fail-closed source audit; times are absolute source seconds."""
import math
from .cards import label

REQUIRED=('sample_id','canonical_source_id','cutoff_policy','source_url','session_id','hand_id','bettor','opponent','board','bettor_cards','opponent_cards','source_hhmmss','source_seconds','requested_start','requested_end','actual_start','actual_end','bet_time','model_cutoff','opponent_response_time','card_confidence','cards_reviewed','heads_up','street','game_type','review_screenshot','actor_roi','mask_rects','rights_basis')

def hhmmss(seconds):
    h=int(seconds//3600); m=int(seconds%3600//60); s=seconds%60
    return f'{h:02d}:{m:02d}:{s:06.3f}'

def audit(row):
    errors=[f'missing:{k}' for k in REQUIRED if k not in row or row[k] is None]
    result=dict(row)
    if errors: return dict(result,eligible=False,exclusion_reasons=errors)
    from .contract import POLICY,PRE_BET_SECONDS,POST_BET_SECONDS
    if row['cutoff_policy']!=POLICY: errors.append('unapproved_cutoff_policy')
    if row['street']!='river' or row['game_type']!='cash' or row['heads_up'] is not True: errors.append('outside_cash_heads_up_river_scope')
    if row['cards_reviewed'] is not True or row['card_confidence']!='high': errors.append('cards_unverified')
    if not row['rights_basis']: errors.append('rights_unresolved')
    if 'synthetic' in row and not isinstance(row['synthetic'],bool): errors.append('invalid_synthetic_flag')
    if 'model_view_reviewed' in row and not isinstance(row['model_view_reviewed'],bool): errors.append('invalid_model_view_reviewed_flag')
    if row['bettor']==row['opponent']: errors.append('actor_opponent_identical')
    try:
        keys=('source_seconds','requested_start','requested_end','actual_start','actual_end','bet_time','model_cutoff','opponent_response_time')
        if not all(isinstance(row[k],(int,float)) and not isinstance(row[k],bool) and math.isfinite(row[k]) and row[k]>=0 for k in keys): raise ValueError('nonfinite or negative time')
        a,b,t,c,r=map(float,(row['actual_start'],row['actual_end'],row['bet_time'],row['model_cutoff'],row['opponent_response_time']))
        if abs(a-(t-PRE_BET_SECONDS))>.001 or abs(c-(t+POST_BET_SECONDS))>.001: errors.append('not_prespecified_bet_relative_window')
        if not a<=t<=c<r or c>b or b<=a: errors.append('invalid_or_response_leaking_time_window')
        if row['requested_end']<=row['requested_start']: errors.append('invalid_requested_window')
        h,m,s=map(float,row['source_hhmmss'].split(':'))
        if h<0 or h!=int(h) or not 0<=m<60 or m!=int(m) or not 0<=s<60: errors.append('invalid_hhmmss')
        if abs(h*3600+m*60+s-row['source_seconds'])>.02: errors.append('source_timestamp_mismatch')
        if abs(row['source_seconds']-row['bet_time'])>.02: errors.append('source_timestamp_not_bet_time')
        result.update(label(row['board'],row['bettor_cards'],row['opponent_cards']))
    except (ValueError,TypeError,KeyError) as e: errors.append(str(e))
    for key in ('actor_roi','mask_rects'):
        rects=[row[key]] if key=='actor_roi' else row[key]
        if key=='mask_rects' and not rects: errors.append('mask_rects_empty')
        if not isinstance(rects,(list,tuple)):
            errors.append(f'invalid_{key}'); continue
        for rect in rects:
            if not isinstance(rect,(list,tuple)) or len(rect)!=4 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in rect): errors.append(f'invalid_{key}')
            elif rect[0]<0 or rect[1]<0 or rect[2]<=0 or rect[3]<=0: errors.append(f'invalid_{key}')
    result.update(eligible=not errors,exclusion_reasons=errors)
    return result
