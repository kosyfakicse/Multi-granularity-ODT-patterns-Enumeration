import sys
import random
import time
from collections import deque

if len(sys.argv) != 10:
    print('Usage: python3 faster.py <region_graph> <trips_graph> <s_a> <s_r> <S_O> <S_D> <S_T> <M> <N>')
    exit()

region_graph = sys.argv[1]
trips_graph = sys.argv[2]
s_a = float(sys.argv[3])
s_r = float(sys.argv[4])
S_O = int(sys.argv[5])
S_D = int(sys.argv[6])
S_T = int(sys.argv[7])
M = int(sys.argv[8])
N = int(sys.argv[9])

print(f'Parameters: s_a={s_a}, s_r={s_r}, S_O={S_O}, S_D={S_D}, S_T={S_T}, M={M}, N={N}')

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

# Get all nodes from atomic patterns too (some may not be in neighbor graph)
all_nodes = set(V)
for (o, d, t) in atomic_patterns:
    all_nodes.add(o)
    all_nodes.add(d)

# Initialize counts
ocount = {v: 0 for v in all_nodes}
dcount = {v: 0 for v in all_nodes}
tcount = {t: 0 for t in timeslots}

# Count atomic patterns
for (o, d, t) in atomic_patterns:
    ocount[o] += 1
    dcount[d] += 1
    tcount[t] += 1

if S_O > S_D:
    S_O, S_D = S_D, S_O
    swap_later = True
    ocount, dcount = dcount, ocount
else:
    swap_later = False

S_O_candidates = []
S_D_candidates = []
O_weights = []
D_weights = []

print('Generating O and D candidates with weights...')

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
        u = Q.popleft()
        
        if u in S:
            continue
        
        S.add(u)
        
        # If S.size == S_O, add S to O_candidates
        if len(S) == S_O:
            S_O_candidates.append(frozenset(S))
            # Weight is sum of ocount for nodes in S
            weight_o = sum(ocount[node] for node in S)
            O_weights.append(weight_o)
            break
        
        # Else add neighbors of u to Q
        for n in neighbor[u]:
            if n not in S:
                Q.append(n)
    
    # If S_O == S_D, add S to D_candidates and continue
    if S_O == S_D:
        if len(S) == S_D:
            S_D_candidates.append(frozenset(S))
            weight_d = sum(dcount[node] for node in S)
            D_weights.append(weight_d)
        continue
    
    # While Q not empty
    while Q:
        u = Q.popleft()
        
        if u in S:
            continue
        
        S.add(u)
        
        # If S.size == S_D, add S to D_candidates
        if len(S) == S_D:
            S_D_candidates.append(frozenset(S))
            weight_d = sum(dcount[node] for node in S)
            D_weights.append(weight_d)
            break
        
        # Else add neighbors of u to Q
        for n in neighbor[u]:
            if n not in S:
                Q.append(n)

if swap_later:
    S_O_candidates, S_D_candidates = S_D_candidates, S_O_candidates
    O_weights, D_weights = D_weights, O_weights
    S_O, S_D = S_D, S_O

print(f'O-candidates: {len(S_O_candidates)}')
print(f'D-candidates: {len(S_D_candidates)}')

print('Generating T candidates with weights...')

S_T_candidates = []
T_weights = []

if len(timeslots) >= S_T:
    for i in range(len(timeslots) - S_T + 1):
        T_cand = tuple(timeslots[i:i+S_T])
        S_T_candidates.append(T_cand)
        # Weight is sum of tcount for timeslots in T
        weight_t = sum(tcount[t] for t in T_cand)
        T_weights.append(weight_t)
else:
    S_T_candidates = [tuple(timeslots)]
    T_weights = [sum(tcount[t] for t in timeslots)]

print(f'T-candidates: {len(S_T_candidates)}\n')

# Sort all ODT-candidates by weight product
print('Sorting ODT-candidates by weight...')
odt_candidates = []

for i, O in enumerate(S_O_candidates):
    for j, D in enumerate(S_D_candidates):
        if O & D:  # Skip if overlap
            continue
        for k, T in enumerate(S_T_candidates):
            weight = O_weights[i] * D_weights[j] * T_weights[k]
            odt_candidates.append((weight, O, D, T))

# Sort in decreasing order of weight
odt_candidates.sort(key=lambda x: x[0], reverse=True)

print(f'Total ODT-candidates: {len(odt_candidates)}')
print(f'Considering top N={N} candidates\n')

# Consider first N ODT-candidates
N_to_check = min(N, len(odt_candidates))

result = {}
card = S_O * S_D * S_T

print(f'Checking {N_to_check} ODT triples...\n')
progress_interval = max(1, N_to_check // 10)

for idx in range(N_to_check):
    if N_to_check >= 10 and (idx+1) % progress_interval == 0:
        print(f'{idx+1}/{N_to_check} ({len(result)} patterns)')
    
    weight, O, D, T = odt_candidates[idx]
    
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
            result[key] = (ratio, cnt, card, weight)

print(f'\nFound {len(result)} patterns')

# Calculate and print success ratio
# Success ratio = patterns found / total possible ODT combinations (excluding overlapping O&D)
# Note: total_possible_patterns is the number of non-overlapping ODT candidates generated
total_possible_patterns = len(odt_candidates)
success_ratio = (len(result) / total_possible_patterns) * 100 if total_possible_patterns > 0 else 0
print(f'Success ratio: {success_ratio:.4f}% ({len(result)}/{total_possible_patterns} possible ODT combinations)')

# Print average ratio of found patterns
if result:
    avg_ratio = sum(ratio for ratio, cnt, card, weight in result.values()) / len(result)
    print(f'Average pattern ratio: {avg_ratio:.4f}')

with open('odt_patterns.tsv', 'w') as f:
    f.write('O\tD\tT\tratio\tcnt\tcard\tweight\n')
    for key, (ratio, cnt, card, weight) in result.items():
        O, D, T = key
        o_str = ','.join(map(str, sorted(O)))
        d_str = ','.join(map(str, sorted(D)))
        t_str = ','.join(map(str, T))
        f.write(f'{o_str}\t{d_str}\t{t_str}\t{ratio:.4f}\t{cnt}\t{card}\t{weight}\n')

# End timing
end_time = time.time()
execution_time = end_time - start_time
print(f'\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)')