import random
from collections import defaultdict


def sample_random_walks(edges, num_walks=100, max_walk_length=16):
    graph = defaultdict(list)
    for u, v, label in edges:
        graph[u].append((v, label))

    walks = []
    for _ in range(num_walks):
        start = random.choice(list(graph.keys()))
        walk = [f"N_{start}"]  # Start with a node
        node = start
        for _ in range(max_walk_length):
            if not graph[node]:
                break
            next_node, edge_label = random.choice(graph[node])
            walk.append(f"E_{edge_label}")  # Append edge first
            walk.append(f"N_{next_node}")  # Then the destination node
            node = next_node
        walks.append(walk)
    return walks
