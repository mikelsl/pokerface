"""V2 frozen before extraction: five seconds before visible bet, endpoint excluded."""
import sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code/source_baseline/poker-river-research'))
from riverlab.cards import best7,normalize
POLICY='prebet_5s_v2';PRE=5.;POST=0.
def label_multi(board,bettor,opponents):
 if len(board)!=5 or len(bettor)!=2 or not opponents:raise ValueError('missing board/bettor/opponents')
 if any(len(o.get('cards',[]))!=2 for o in opponents):raise ValueError('missing active opponent cards')
 cards=[normalize(c) for c in board+bettor+sum([o['cards'] for o in opponents],[])]
 if len(set(cards))!=len(cards):raise ValueError('duplicate card')
 a=best7(board+bettor);pairs=[]
 for o in opponents:
  b=best7(board+o['cards']);pairs.append({'opponent':o['name'],'rank':list(b),'relation':'stronger' if a>b else 'weaker' if a<b else 'tie'})
 return {'label':'value' if all(p['relation']=='stronger' for p in pairs) else 'bluff','bettor_rank':list(a),'pairwise':pairs,'active_opponents':len(opponents),'multiway':len(opponents)>1,'label_definition':'showdown-strength proxy; not intent'}
def audit(r):
 errors=[]
 for k in ['sample_id','session_id','hand_id','source_url','bet_time','bet_onset_interval_seconds','opponents','board','bettor_cards','cards_reviewed','actor_roi','model_view_reviewed','cutoff_policy']:
  if r.get(k) is None:errors.append('missing:'+k)
 if errors:return dict(r,eligible=False,exclusion_reasons=errors)
 try:
  t=r['bet_time'];a=r['actual_start'];b=r['model_cutoff']
  if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in [t,a,b]):raise ValueError('nonfinite time')
  if a<0 or abs(a-(t-5))>1e-6 or b!=t or r['cutoff_policy']!=POLICY:errors.append('window_not_frozen_5s')
  if r.get('active_opponents')!=len(r['opponents']):errors.append('active_opponent_count_mismatch')
  if r.get('side_pot_status') not in ('none observed','resolved'):errors.append('side_pot_unresolved')
  if any(o.get('shared_contestable_pot') is not True or not isinstance(o.get('can_respond'),bool) for o in r['opponents']):errors.append('opponent_pot_or_response_status_unresolved')
  if r.get('action') not in ('bet','raise'):errors.append('not_bet_or_raise')
  if r['cards_reviewed'] is not True:errors.append('cards_unreviewed')
  if r['model_view_reviewed'] is not True:errors.append('model_view_unreviewed')
  if r.get('street')!='river' or r.get('game_type')!='cash':errors.append('outside_scope')
  comp=label_multi(r['board'],r['bettor_cards'],r['opponents'])
 except (KeyError,ValueError,TypeError) as e:errors.append(str(e));comp={}
 return dict(r,comparison=comp,eligible=not errors,exclusion_reasons=errors)

def opponents_before(roster,bettor,t):
 """Explicit event-time membership; all-in is not a fold."""
 return [dict(p,can_respond=not (p.get('all_in_time') is not None and p['all_in_time']<t)) for p in roster if p['name']!=bettor and (p.get('fold_time') is None or p['fold_time']>=t)]

def validate_features_v2(r):
 from riverlab.contract import ALLOWED,FACE_SHA256,HAND_SHA256,sha256
 if not audit(r)['eligible']:raise ValueError('v2 audit failed')
 f=r.get('features',{});p=r.get('feature_provenance',{});q=r.get('quality',{})
 if not f or set(f)-ALLOWED or f.get('timing_duration')!=5:raise ValueError('feature contract mismatch')
 if any(not isinstance(v,(float,int)) or not math.isfinite(v) for v in f.values()):raise ValueError('nonfinite feature')
 if p.get('contract_version')!='v2' or p.get('window_policy')!=POLICY or p.get('face_sha256')!=FACE_SHA256 or p.get('hand_sha256')!=HAND_SHA256 or p.get('masked_clip_sha256')!=sha256(r['masked_clip']):raise ValueError('feature provenance mismatch')
 if q.get('frames')!=25 or q.get('face_fraction',0)<.5:raise ValueError('feature quality failed')
