#!/usr/bin/env python3
"""
Generate a very small toy dataset for testing and debugging.
"""

import os
import json
import random
from collections import defaultdict


def create_toy_graph(num_nodes=10, num_edges=20, edge_labels=[-1, 0, 1]):
    """Create a small connected graph."""
    print(f"🎯 Creating toy graph: {num_nodes} nodes, {num_edges} edges")

    edges = []

    # First, create a connected path to ensure connectivity
    for i in range(1, num_nodes):
        label = random.choice(edge_labels)
        edges.append((i, i + 1, label))

    # Add random edges to reach target count
    while len(edges) < num_edges:
        u = random.randint(1, num_nodes)
        v = random.randint(1, num_nodes)
        if u != v:  # No self-loops
            label = random.choice(edge_labels)
            edge = (u, v, label)
            if edge not in edges:  # No duplicates
                edges.append(edge)

    # Show statistics
    label_counts = defaultdict(int)
    for _, _, label in edges:
        label_counts[label] += 1

    print(f"  Edges created: {len(edges)}")
    print(f"  Label distribution: {dict(label_counts)}")

    return edges


def save_toy_dataset(edges, data_dir="data/toy"):
    """Save toy dataset in the expected format."""
    print(f"💾 Saving toy dataset to {data_dir}/")

    # Create directory
    os.makedirs(data_dir, exist_ok=True)

    # Save edge list file (same format as out.chess)
    edge_file = os.path.join(data_dir, "out.toy")
    with open(edge_file, "w") as f:
        f.write("% toy dataset\n")
        for u, v, label in edges:
            f.write(f"{u} {v} {label}\t1.0\n")  # Add dummy timestamp

    print(f"  ✅ Saved edge list: {edge_file}")
    return data_dir


def add_toy_dataset_loader():
    """Add toy dataset loader to datasets.py"""
    print(f"🔧 Adding toy dataset loader...")

    # Read current datasets.py
    datasets_file = "src/data/datasets.py"
    with open(datasets_file, "r") as f:
        content = f.read()

    # Check if toy loader already exists
    if "load_toy" in content:
        print(f"  ⚠️ Toy loader already exists in {datasets_file}")
        return

    # Add toy loader function after load_bitcoin
    toy_loader = '''

def load_toy(cfg):
    """Load the toy dataset from an edge list file."""
    path = os.path.join(cfg.dataset.data_dir, cfg.dataset.edge_list_file)
    edges = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("%"):
                continue  # Skip comments
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # Invalid line
            u, v, label = int(parts[0]), int(parts[1]), int(parts[2])
            edges.append((u, v, label))
    return edges'''

    # Find insertion point after load_bitcoin function
    bitcoin_end = content.find("    return edges", content.find("def load_bitcoin"))
    if bitcoin_end == -1:
        print(f"  ❌ Could not find load_bitcoin function end")
        return

    # Find the end of the line
    insert_pos = content.find("\n", bitcoin_end) + 1

    # Insert toy loader
    new_content = content[:insert_pos] + toy_loader + content[insert_pos:]

    # Update registry - find the DATASET_LOADERS dict
    registry_start = new_content.find("DATASET_LOADERS = {")
    registry_end = new_content.find("}", registry_start)

    if registry_start != -1 and registry_end != -1:
        # Add toy to registry
        registry_content = new_content[registry_start:registry_end]
        if '"toy"' not in registry_content:
            # Insert before the closing brace
            new_content = (
                new_content[:registry_end]
                + '    "toy": load_toy,\n'
                + new_content[registry_end:]
            )

    # Write back
    with open(datasets_file, "w") as f:
        f.write(new_content)

    print(f"  ✅ Added toy loader to {datasets_file}")


def create_toy_config(base_config="config.yaml", toy_config="config_toy.yaml"):
    """Create a toy configuration file."""
    print(f"📝 Creating toy config: {toy_config}")

    # Load base config
    import yaml

    with open(base_config, "r") as f:
        config = yaml.safe_load(f)

    # Modify for toy dataset
    config["dataset"]["name"] = "toy"
    config["dataset"]["data_dir"] = "data/toy"
    config["dataset"]["edge_list_file"] = "out.toy"
    config["dataset"]["max_walk_length"] = 3  # Very short walks (3 steps = 7 tokens)
    config["dataset"]["num_walks"] = 1500  # Very few walks for super fast testing

    # Tiny model for toy dataset
    config["model"]["embedding_dim"] = 16
    config["model"]["hidden_dim"] = 32
    config["model"]["nhead"] = 2
    config["model"]["nlayers"] = 1

    # Tiny training
    config["training"]["batch_size"] = 2
    config["training"]["epochs"] = 3

    # Save toy config
    with open(toy_config, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"  ✅ Saved toy config: {toy_config}")


def test_toy_dataset():
    """Test the toy dataset with a quick data loading."""
    print(f"🧪 Testing toy dataset...")

    try:
        import yaml
        from omegaconf import OmegaConf
        from src.data.datasets import get_loader

        # Load toy config
        with open("config_toy.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        cfg = OmegaConf.create(cfg)

        # Test loading
        loader = get_loader(cfg.dataset.name)
        edges = loader(cfg)

        print(f"  ✅ Successfully loaded {len(edges)} edges")
        print(f"  Sample edges: {edges[:3]}")

        return True

    except Exception as e:
        print(f"  ❌ Error testing: {e}")
        return False


def main():
    """Create complete toy dataset setup."""
    print("🎮 CREATING TOY DATASET")
    print("=" * 40)

    # Set seed for reproducibility
    random.seed(42)

    # Create toy graph
    edges = create_toy_graph(
        num_nodes=10,  # Even smaller - just 6 nodes
        num_edges=20,  # Just 10 edges
        edge_labels=[-1, 0, 1],
    )

    # Save dataset
    data_dir = save_toy_dataset(edges)

    # Add loader to datasets.py
    add_toy_dataset_loader()

    # Create toy config
    create_toy_config()

    # Test loading
    success = test_toy_dataset()

    print(f"\n" + "=" * 40)
    if success:
        print("✅ TOY DATASET CREATED SUCCESSFULLY!")
        print(f"\n💡 How to use:")
        print(f"   1. Use config_toy.yaml for experiments")
        print(f"   2. Much faster testing and debugging")
        print(f"   3. Data in data/toy/ directory")
        print(f"\n🚀 Try: python run.py --config config_toy.yaml")
    else:
        print("❌ TOY DATASET CREATION FAILED!")
    print("=" * 40)


if __name__ == "__main__":
    main()
