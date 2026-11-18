"""Exhaustive, deterministic best-five-of-seven evaluator; larger tuple wins."""
from collections import Counter
from itertools import combinations
RANKS = '23456789TJQKA'
SUITS = 'cdhs'

def normalize(card):
    c = str(card).strip().replace('10', 'T')
    c = c[:1].upper() + c[1:].lower()
    if len(c) != 2 or c[0] not in RANKS or c[1] not in SUITS:
        raise ValueError(f'Invalid card: {card!r}')
    return c

def rank5(cards):
    cards = [normalize(c) for c in cards]
    if len(cards) != 5 or len(set(cards)) != 5:
        raise ValueError('Five distinct cards required')
    ranks = sorted((RANKS.index(c[0]) + 2 for c in cards), reverse=True)
    count = Counter(ranks)
    groups = sorted(((n, r) for r, n in count.items()), reverse=True)
    flush = len({c[1] for c in cards}) == 1
    unique = sorted(count)
    straight = 5 if unique == [2,3,4,5,14] else (max(unique) if len(unique)==5 and max(unique)-min(unique)==4 else 0)
    if straight and flush: return (8, straight)
    if groups[0][0]==4: return (7, groups[0][1], groups[1][1])
    if [g[0] for g in groups]==[3,2]: return (6, groups[0][1], groups[1][1])
    if flush: return (5, *ranks)
    if straight: return (4, straight)
    if groups[0][0]==3: return (3, groups[0][1], *sorted((r for r in ranks if r != groups[0][1]),reverse=True))
    pairs = sorted((r for r,n in count.items() if n==2),reverse=True)
    if len(pairs)==2: return (2,*pairs,next(r for r,n in count.items() if n==1))
    if pairs: return (1,*pairs,*sorted((r for r,n in count.items() if n==1),reverse=True))
    return (0,*ranks)

def best7(cards):
    cards = [normalize(c) for c in cards]
    if len(cards)!=7 or len(set(cards))!=7: raise ValueError('Seven distinct cards required')
    return max(rank5(c) for c in combinations(cards,5))

def label(board, bettor, opponent):
    if len(board)!=5 or len(bettor)!=2 or len(opponent)!=2: raise ValueError('Need all nine cards')
    cards=[normalize(c) for c in board+bettor+opponent]
    if len(set(cards))!=9: raise ValueError('Duplicate physical card')
    a,b=best7(cards[:5]+cards[5:7]),best7(cards[:5]+cards[7:])
    relation='stronger' if a>b else 'weaker' if a<b else 'tie'
    return {'relation':relation,'label':'value' if a>b else 'bluff','bettor_rank':list(a),'opponent_rank':list(b),'label_definition':'showdown-strength proxy, not betting intent'}
