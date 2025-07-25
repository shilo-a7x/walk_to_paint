def load_edges(path):
    edges = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            u, v = int(parts[0]), int(parts[1])
            label = int(parts[2].split('\t')[0])
            edges.append((u, v, label))
    return edges