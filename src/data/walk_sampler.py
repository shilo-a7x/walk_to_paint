import random

def sample_walks(edges, num_walks=100, walk_length=16):
    from collections import defaultdict
    graph = defaultdict(list)
    for u, v, label in edges:
        graph[u].append((v, label))

    walks = []
    for _ in range(num_walks):
        start = random.choice(list(graph.keys()))
        walk = []
        node = start
        for _ in range(walk_length):
            if not graph[node]:
                break
            node, edge_label = random.choice(graph[node])
            walk.append(f"N{node}")
            walk.append(f"E{edge_label}")
        walks.append(walk)
    return walks