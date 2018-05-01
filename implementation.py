from heapq import heappush, heappop
from itertools import count
from itertools import permutations
from collections import defaultdict
from itertools import chain, combinations
from functools import reduce
import copy
from collections import deque
import signal
class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True


killer = GracefulKiller()

input()
n = int(input().split()[1])
e = int(input().split()[1])

graph = defaultdict(dict)
max_cost = 0
is_terminal = dict()
for x in range(e):
    s,u,v,w = input().split()
    u,v,w = map(int,(u,v,w))
    graph[u][v] = w
    graph[v][u] = w
    is_terminal[u] = False
    is_terminal[v] = False
    max_cost += w

input() # end
check = input()
if 'SECTION' not in check:input()

t = int(input().split()[1])
terminals = []
for x in range(t):
    _, ter = input().split()
    terminals.append(int(ter))
    is_terminal[int(ter)] = True

terminals = sorted(terminals)
#terminals = sorted(terminals,key = lambda x: (len(graph[x]),x),reverse = True)  
steinertree = defaultdict(dict)
steinercost = copy.deepcopy(graph)

def add_edges(path):
    global steinertree, steinercost
    if isinstance(path,int):return
    if len(path)<= 1:return
    for a,b in zip(path[:-1],path[1:]):
        w = graph[a][b]
        steinertree[a][b] = w
        steinertree[b][a] = w
        steinercost[a][b] = 0
        steinercost[b][a] = 0
        
def remove_edges(edges):
    global steinertrees
    reduced = set((min(a,b),max(a,b)) for a,b in edges)
    for a,b in reduced:
        steinertree[a].pop(b, None)
        steinertree[b].pop(a, None)
        if len(steinertree[a]) == 0:
            steinertree.pop(a, None)
        if len(steinertree[b]) == 0:
            steinertree.pop(b, None)


def get_all_subsets(source):
    return chain(*[combinations(source, i) for i in range(1,len(source)+1)])


def optimize(relevant_nodes, keep,dist,score_to_improve):
    global graph
    keep = sorted(keep)
    root = keep[0]
    sources = keep[1:]
    missing_sources = {repr(list(I)): [x for x in sources if x  not in I] for I in get_all_subsets(sources)}
    l = defaultdict(dict)
    b = defaultdict(dict)
    for v in relevant_nodes:
        for I in get_all_subsets(sources):
            l[v][repr(list(I))] = float('inf')
            b[v][repr(list(I))] = [] # empty set

    for v in relevant_nodes:
        if v in sources:
            l[v][repr([v])] = 0
        l[v][repr([])] = 0

    # N = [(s,[s]) for s in sources]
    P =   [(v,[]) for v in relevant_nodes]




    N = []
    for s in sources:
        push(N, (l[s][repr([s])], s, [s]))

    while (root, sources) not in P:
        if killer.kill_now:
            return score_to_improve,None
        min_cost, v, I = pop(N)
        P.append((v,I))
        for w in graph[v]:
            if killer.kill_now:
                return score_to_improve,None
            # only search relevant nodes
            if w not in relevant_nodes:continue
            if l[v][repr(I)] + steinercost[v][w] < l[w][repr(I)] and (w,I) not in P:
                l[w][repr(I)] = l[v][repr(I)] + steinercost[v][w]
                b[w][repr(I)] = [(v,I)]
                # l(v,I) + L(v,R\I)    -> L could also be 0 to be valid
                extra_cost = 0
                missing = missing_sources[repr(I)]
                if len(missing) == 0:
                    extra_cost = 0
                elif len(missing) == 1:
                    extra_cost = dist[missing[0]][w]
                else:
                    extra_cost = min(0.5*(dist[m1][w]+dist[m2][w]) for m1, m2 in combinations(missing,2))
                push(N,(l[w][repr(I)] + extra_cost,w,I))

        for J in get_all_subsets(sources):
            if killer.kill_now:
                return score_to_improve,None
            J = list(J)
            if J == I:continue
            if (v,J) not in P:continue

            combined = list(sorted(set(I+J)))
            if l[v][repr(I)] + l[v][repr(J)] <= l[v][repr(combined)] and (v,combined) not in P:
                l[v][repr(combined)] = l[v][repr(I)] + l[v][repr(J)]
                b[v][repr(combined)] = [(v,I),(v,J)]
                
                extra_cost = 0
                missing = missing_sources[repr(combined)]
                if len(missing) == 0:
                    extra_cost = 0
                elif len(missing) == 1:
                    extra_cost = dist[missing[0]][v]
                else:
                    extra_cost = min(0.5*(dist[m1][v]+dist[m2][v]) for m1, m2 in combinations(missing,2))
                push(N,(l[v][repr(combined)] + extra_cost,v,combined))
                
    if l[root][repr(sources)] < score_to_improve:
        return l[root][repr(sources)], reconstruct(root,sources,b)
    return l[root][repr(sources)], None

def add_edge_list(edges):
    global steinertree
    for a,b in edges:
        w = graph[a][b]
        steinertree[a][b] = w
        steinertree[b][a] = w

def tree_cost():
    global steinertree
    steinertree_edges = set()
    for a in steinertree.keys():
        steinertree_edges.update((min(a,b),max(a,b)) for b in steinertree[a].keys())
    tree_cost = sum(steinertree[a][b] for a,b in steinertree_edges)
    return tree_cost

def reduce_nodes(keep, current_cost):
    push = heappush
    pop = heappop
    c = count()
    
    dist = defaultdict(dict)  # dictionary of final distances
    occurences = defaultdict(set)
    #cumulated_dist = defaultdict(list)
    cumulated_dist = dict()
    min_max_keep_dist = current_cost
    #paths = defaultdict(dict)
    for node in keep:
        max_keep_dist_per_node = 0
        cutoff = current_cost
        #paths[node][node] = [node]
        seen = {}
        fringe = []   
        for source in [node]:
            seen[source] = 0
            push(fringe, (0, next(c), source))
        while fringe:
            (d, _, v) = pop(fringe)
            if v in dist[node]:
                continue  # already searched this node.
            dist[node][v] = d
            for u, cost in steinercost[v].items():
                vu_dist = dist[node][v] + cost
        #         if cutoff is not None:
                if vu_dist > cutoff:
                    continue
                if u in keep and u!=node:
                    max_keep_dist_per_node = max(vu_dist,max_keep_dist_per_node)
                    #cutoff = min(cutoff, vu_dist)
                    #cutoff = min(current_cost, cutoff-vu_dist)
                if u in dist[node]:
                    if vu_dist < dist[node][u]:
                        seen[u] = vu_dist
                        push(fringe, (vu_dist, next(c), u))
                        occurences[u].add(node)
                        #paths[node][u] = paths[node][v] + [u]
                elif u not in seen or vu_dist < seen[u]:
                    seen[u] = vu_dist
                    push(fringe, (vu_dist, next(c), u))
                    occurences[u].add(node)
                    #paths[node][u] = paths[node][v] + [u]
        min_max_keep_dist = min(min_max_keep_dist,max_keep_dist_per_node)
        for k,v in dist[node].items():
            #cumulated_dist[k].append(v)
            cumulated_dist[k] = cumulated_dist.get(k,0) + v
                    
    if len(keep) == 3:
        nodes_relevant = keep + [k for k,v in occurences.items() if len(v)==len(keep) and cumulated_dist[k]<=(min_max_keep_dist+current_cost)]
        #nodes_relevant = keep + [k for k,v in occurences.items() if len(v)==len(keep) and cumulated_dist[k]<=(current_cost/3+current_cost)]
    elif len(keep) == 4:
        nodes_relevant = keep + [k for k,v in occurences.items() if len(v)==len(keep) and cumulated_dist[k]<=(2*min_max_keep_dist+current_cost)]
        #nodes_relevant = keep + [k for k,v in occurences.items() if len(v)==len(keep)]
    else:
        nodes_relevant = keep + [k for k,v in occurences.items() if len(v)==len(keep)]
    return nodes_relevant, dist

edge_list = set()

def reconstruct(v,X,b):
    global edge_list
    edge_list = set()
    trace = traceback(v,X,b)
    all_edges(trace)
    return edge_list
    
def traceback(v, X,b):
    edge_list = set()
    res = b[v][repr(X)]
    if res == []:
        return [traceback(v, [x]) for x in X if x!=v]
    else:
        if len(res)>1:
            return [traceback(v_i, X_i,b) for v_i,X_i in res] 
        else:
            res_u, res_x =res[0]
            return [(v,res_u)] + traceback(res_u,X,b)
            #if isinstance(res_u, int) and X == res_x:
# trace = traceback(root, sources)

# edge_list = set()
def all_edges(l):
    global edge_list
    for x in l:
        if x == []:continue
        if isinstance(x,tuple):
            a,b = x
            edge_list.add((min(a,b),max(a,b)))
        else:
            all_edges(x)

def get_shorter(path,path_length,max_size = 4):
    global steinertree, steinercost,graph
    start = path[0]
    intersection = path[-1]
    if is_terminal[intersection]:return None,None
    
    nodes_to_keep = [start]
    current_dist = path_length
    
    
    paths_to_intersections = []
    nodes_in_path = set()
    base_node = intersection
    for v in steinertree[base_node].keys():
        if v in nodes_in_path:continue # don't go back to path already taken
        prev = base_node
        current_dist += graph[v][prev]
        paths_to_intersections.append((v,prev))
        nodes_in_path.add(v)
        nodes_in_path.add(prev)
        while not v in terminals and len(steinertree[v]) == 2:
            v,prev = [k for k in steinertree[v].keys() if k!= prev][0],v
            current_dist += graph[v][prev]
            paths_to_intersections.append((v,prev))
            nodes_in_path.add(v)
        nodes_to_keep.append(v)   

        if len(nodes_to_keep)>max_size:return None,None
    if len(nodes_to_keep)<3:
        return None,None
    for a,b in paths_to_intersections:
        w = graph[a][b]
        steinercost[a][b] = w
        steinercost[b][a] = w
    relevant, dijkstra_keep = reduce_nodes(nodes_to_keep,current_dist) 
    try:
        opt_result,new_edges = optimize(relevant, nodes_to_keep,dijkstra_keep,current_dist) # all_nodes
    except:
        return None,None

    if opt_result<current_dist:
        return new_edges, paths_to_intersections
        remove_edges(paths_to_intersections)
        add_edge_list(new_edges)
        for a,b in new_edges:
            steinercost[a][b] = 0
            steinercost[b][a] = 0
    else:
        return None,None
#         for a,b in paths_to_intersections:
#             steinercost[a][b] = 0
#             steinercost[b][a] = 0
    return None, None



best_solution = max_cost + 1
best_tree = None
best_terminal = 0
limit = min(t,200)
check_terminals = terminals[:limit]
terminals_checked = 0
while check_terminals:
    if killer.kill_now:
        break
    ttt = check_terminals.pop(0)
    #sources = terminals
    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    paths = {source: [source] for source in terminals}
    start = ttt
    
    sources = [start]
    searched = set()
    for source in sources:
        seen[source] = 0
        push(fringe, (0, next(c), source))
        paths[source] = [source]
        #closest_terminal[source] = (0, [source], source) 
    while fringe:
        (d, _, v) = pop(fringe)
        if v in searched:
            continue  # already searched this node.
        dist[v] = d
        searched.add(v)
        for u, cost in graph[v].items():
            vu_dist = dist[v] + cost
    #         if cutoff is not None:
    #             if vu_dist > cutoff:
    #                 continue
            if u in dist:
                if vu_dist < dist[u]:
                    seen[u] = vu_dist
                    push(fringe, (vu_dist, next(c), u))
                    paths[u] = paths[v] + [u]
                    closest_terminal[u] = (vu_dist, paths[u], source) 
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                paths[u] = paths[v] + [u]
                
    steinertree = defaultdict(dict)
    steinercost = copy.deepcopy(graph)
    other_terminals = sorted([_ for _ in terminals if _!=ttt],key = lambda x: (dist[x],x),reverse=False)
    closest = other_terminals[0]
    closest_path = paths[closest]
    
    add_edges(closest_path)
    
    if killer.kill_now:
        break
    for ter in other_terminals[1:]:
        if killer.kill_now:
            break
        dist = {}  # dictionary of final distances
        seen = {}
        # fringe is heapq with 3-tuples (distance,c,node)
        # use the count c to avoid comparing nodes (may not be able to)
        c = count()
        fringe = []
        paths = {ter: [ter]}
        closest_dist = 2**32
        closest_path = None
        sources = [ter]
        searched = set()

        for source in sources:
            seen[source] = 0
            push(fringe, (0, next(c), source))
            paths[source] = [source]
            #closest_terminal[source] = (0, [source], source) 
        while fringe:
            (d, _, v) = pop(fringe)
            if v in searched:
                continue  # already searched this node.
            dist[v] = d
            searched.add(v)
            for u, cost in graph[v].items():
                vu_dist = dist[v] + cost
                if vu_dist > closest_dist:
                    continue
                if u in dist:
                    if vu_dist < dist[u]:
                        seen[u] = vu_dist
                        push(fringe, (vu_dist, next(c), u))
                        paths[u] = paths[v] + [u]
                elif u not in seen or vu_dist < seen[u]:
                    seen[u] = vu_dist
                    push(fringe, (vu_dist, next(c), u))
                    paths[u] = paths[v] + [u]
                if u in steinertree:
                    closest_dist = vu_dist
                    closest_path = paths[u]
        if terminals_checked>=limit:
            shorter_add, shorter_remove = get_shorter(closest_path,closest_dist)
            if shorter_add is None:
                add_edges(closest_path)
            else:
                remove_edges(shorter_remove)
                add_edge_list(shorter_add)
                for a,b in shorter_add:
                    steinercost[a][b] = 0
                    steinercost[b][a] = 0            
        else:
            add_edges(closest_path)            

    if killer.kill_now:
        break
        
    steinertree_edges = set()
    for a in steinertree.keys():
        steinertree_edges.update((min(a,b),max(a,b)) for b in steinertree[a].keys())
    total_cost = sum(steinertree[a][b] for a,b in steinertree_edges)
    if total_cost < best_solution:
        best_solution = total_cost
        best_tree = steinertree_edges
        best_terminal = ttt
        if terminals_checked<limit:check_terminals.append(ttt)

steinertree = defaultdict(dict)
steinercost = copy.deepcopy(graph)
for a,b in best_tree:
    w = graph[a][b]
    steinertree[a][b] = w
    steinertree[b][a] = w
    
    steinercost[a][b] = 0
    steinercost[b][a] = 0


changed = True
target_size = 3
#while target_size<5:
while True:
    if killer.kill_now:
        break
    if not changed:
        target_size += 1
    if target_size >= 8:break
    changed = False
    for pot in list(steinertree):
        if killer.kill_now:
            break
        if pot not in steinertree:continue
        if len(steinertree[pot])==2 and pot not in terminals: continue
        if len(steinertree[pot])<2 and pot in terminals: continue
        nodes_to_keep = []
        if pot in terminals:
            nodes_to_keep.append(pot)

        current_dist = 0
        paths_to_intersections = []
        nodes_in_path = set()

        node_queue = [pot]
        while len(nodes_to_keep) < target_size and node_queue:
            base_node = node_queue.pop(0)
            for v in steinertree[base_node].keys():
                if v in nodes_in_path:continue # don't go back to path already taken
                prev = base_node
                current_dist += graph[v][prev]
                paths_to_intersections.append((v,prev))
                nodes_in_path.add(v)
                nodes_in_path.add(prev)
                while not v in terminals and len(steinertree[v]) == 2:
                    v,prev = [k for k in steinertree[v].keys() if k!= prev][0],v
                    current_dist += graph[v][prev]
                    paths_to_intersections.append((v,prev))
                    nodes_in_path.add(v)
                nodes_to_keep.append(v)   
                node_queue.append(v)

                if len(nodes_to_keep) == target_size and base_node in terminals:break
        if killer.kill_now:
            break
        if len(nodes_to_keep) != target_size:continue
        for a,b in paths_to_intersections:
            w = graph[a][b]
            steinercost[a][b] = w
            steinercost[b][a] = w

        if killer.kill_now:
            break
        relevant, dijkstra_keep = reduce_nodes(nodes_to_keep,current_dist) 

        if killer.kill_now:
            break
        try:
            opt_result,new_edges = optimize(relevant, nodes_to_keep,dijkstra_keep,current_dist) # all_nodes
        except:
            continue
        if killer.kill_now:
            break
        if opt_result<current_dist:
            changed = True
            remove_edges(paths_to_intersections)
            add_edge_list(new_edges)
            for a,b in new_edges:
                steinercost[a][b] = 0
                steinercost[b][a] = 0
        else:
            for a,b in paths_to_intersections:
                steinercost[a][b] = 0
                steinercost[b][a] = 0

final_cost = tree_cost()
final_edges = set()
for a in steinertree:
    for b in steinertree[a]:
        final_edges.add((min(a,b),max(a,b)))
print ('VALUE {}'.format(final_cost))
print ('\n'.join('{} {}'.format(a,b) for a,b in final_edges))