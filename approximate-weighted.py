import sys
import random
import time
from collections import deque

if len(sys.argv) != 9:
    print('Usage: python3 faster.py <region_graph> <trips_graph> <s_a> <s_r> <S_O> <S_D> <S_T> <M>')
    exit()

region_graph = sys.argv[1]
trips_graph = sys.argv[2]
s_a = float(sys.argv[3])
s_r = float(sys.argv[4])
S_O = int(sys.argv[5])
S_D = int(sys.argv[6])
S_T = int(sys.argv[7])
M = int(sys.argv[8])

print(f'Parameters: s_a={s_a}, s_r={s_r}, S_O={S_O}, S_D={S_D}, S_T={S_T}, M={M}')

# Start timing
start_time = time.time()

# Load trips
atomic = {}
flows = []
timeslots_set = set()

with open(trips_graph) as f:
    for l in f:
        l = [int(x) for x in l.split()]
        if len(l) < 4:
            o, d, t = l[0], l[1], l[2]
            flow = 1
        else:
            o, d, t, flow = l[0], l[1], l[2], l[3]
        
        if o == d: continue
        flows.append(flow)
        timeslots_set.add(t)
        
        if o in atomic:
            if d in atomic[o]:
                atomic[o][d][t] = flow
            else:
                atomic[o][d] = {t: flow}
        else:
            atomic[o] = {d: {t: flow}}

timeslots = sorted(timeslots_set)
thresh = sorted(flows, reverse=True)[min(int(s_a * len(flows)), len(flows) - 1)]

# Keep atomic patterns above threshold
atomic_patterns = set()
for o in atomic:
    for d in atomic[o]:
        for t in atomic[o][d]:
            if atomic[o][d][t] >= thresh:
                atomic_patterns.add((o, d, t))

print(f'Atomic patterns: {len(atomic_patterns)}')

# Load graph
neighbor = {}
with open(region_graph) as f:
    for l in f:
        l = [int(x) for x in l.split()]
        if len(l) < 2: continue
        a, b = l[0], l[1]
        if a == b: continue
        
        if a in neighbor:
            neighbor[a].append(b)
        else:
            neighbor[a] = [b]
        
        if b in neighbor:
            neighbor[b].append(a)
        else:
            neighbor[b] = [a]

V = list(neighbor.keys())
print(f'Graph nodes: {len(V)}')
print(f'Timeslots: {len(timeslots)}\n')

if S_O > S_D:
    S_O, S_D = S_D, S_O
    swap_later = True
else:
    swap_later = False

S_O_candidates = []
S_D_candidates = []

print('Generating O and D candidates...')

# For each node v in V
for v in V:
    if v not in neighbor:
        continue
    
    S = {v}
    
    # Add neighbors of v in a FIFO queue Q
    Q = deque()
    for n in neighbor[v]:
        Q.append(n)
    
    # While Q not empty
    while Q:
        u = Q.popleft()  # u = dequeue(Q)
        
        if u in S:  # if u in S: continue while-loop
            continue
        
        S.add(u)  # Add u to S
        
        # If S.size == S_O, add S to O_candidates and break while-loop
        if len(S) == S_O:
            S_O_candidates.append(frozenset(S))
            break
        
        # Else add neighbors of u to Q
        for n in neighbor[u]:
            if n not in S:
                Q.append(n)
    
    # If S_O == S_D, add S to D_candidates and continue for-loop
    if S_O == S_D:
        if len(S) == S_D:
            S_D_candidates.append(frozenset(S))
        continue
    
    # While Q not empty // if we are here, this means S_O < S_D
    while Q:
        u = Q.popleft()
        
        if u in S:
            continue
        
        S.add(u)
        
        # If S.size == S_D, add S to D_candidates and break while-loop
        if len(S) == S_D:
            S_D_candidates.append(frozenset(S))
            break
        
        # Else add neighbors of u to Q
        for n in neighbor[u]:
            if n not in S:
                Q.append(n)

if swap_later:
    S_O_candidates, S_D_candidates = S_D_candidates, S_O_candidates
    S_O, S_D = S_D, S_O

print(f'O-candidates: {len(S_O_candidates)}')
print(f'D-candidates: {len(S_D_candidates)}')

print('Generating T candidates...')

S_T_candidates = []
if len(timeslots) >= S_T:
    for i in range(len(timeslots) - S_T + 1):
        S_T_candidates.append(tuple(timeslots[i:i+S_T]))
else:
    S_T_candidates = [tuple(timeslots)]

print(f'T-candidates: {len(S_T_candidates)}\n')
print(f'Sampling {M} random ODT triples...\n')

result = {}

progress_interval = max(1, M // 10)

card = S_O * S_D * S_T
for i in range(M):
    if M >= 10 and (i+1) % progress_interval == 0:
        print(f'{i+1}/{M} ({len(result)} patterns)')
    
    # Pick random O from S_O_candidates
    O = S_O_candidates[random.randint(0, len(S_O_candidates)-1)]
    
    # Pick random D from S_D_candidates
    D = S_D_candidates[random.randint(0, len(S_D_candidates)-1)]
    
    if O & D:
        continue
    
    # Pick random T from S_T_candidates
    T = S_T_candidates[random.randint(0, len(S_T_candidates)-1)]
    
    cnt = 0
    for o in O:
        for d in D:
            for t in T:
                if (o, d, t) in atomic_patterns:
                    cnt += 1
    
    # Check if triple satisfies s_r
    ratio = cnt / card
    if ratio >= s_r:
        key = (O, D, T)
        if key not in result or result[key][0] < ratio:
            result[key] = (ratio, cnt, card)

print(f'\nFound {len(result)} patterns')

# Calculate and print success ratio
# Success ratio = patterns found / total possible ODT combinations
total_possible_patterns = len(S_O_candidates) * len(S_D_candidates) * len(S_T_candidates)
success_ratio = (len(result) / total_possible_patterns) * 100 if total_possible_patterns > 0 else 0
print(f'Success ratio: {success_ratio:.4f}% ({len(result)}/{total_possible_patterns} possible ODT combinations)')

# Print average ratio of found patterns
if result:
    avg_ratio = sum(ratio for ratio, cnt, card in result.values()) / len(result)
    print(f'Average pattern ratio: {avg_ratio:.4f}')

with open('odt_patterns.tsv', 'w') as f:
    f.write('O\tD\tT\tratio\tcnt\tcard\n')
    for key, (ratio, cnt, card) in result.items():
        O, D, T = key
        o_str = ','.join(map(str, sorted(O)))
        d_str = ','.join(map(str, sorted(D)))
        t_str = ','.join(map(str, T))
        f.write(f'{o_str}\t{d_str}\t{t_str}\t{ratio:.4f}\t{cnt}\t{card}\n')

# End timing
end_time = time.time()
execution_time = end_time - start_time
print(f'\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)')