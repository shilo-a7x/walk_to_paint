#!/usr/bin/env python3
"""
Test the consistency of deduplication approaches.
"""

import random
from collections import OrderedDict


def test_dedup_consistency():
    """Test different deduplication approaches for consistency."""
    print("🔍 TESTING DEDUPLICATION CONSISTENCY")
    print("=" * 50)

    # Sample data with duplicates
    edges = [
        (1, 2, 0),
        (2, 3, 1),
        (1, 2, 0),  # duplicate
        (3, 4, -1),
        (2, 3, 1),  # duplicate
        (4, 5, 0),
        (1, 2, 0),  # duplicate again
    ]

    print(f"Original edges: {edges}")
    print(f"Length: {len(edges)}")

    # Method 1: Using set() - NOT DETERMINISTIC
    print(f"\n❌ Method 1: set() - NOT DETERMINISTIC")
    for run in range(3):
        edge_tuples = [tuple(edge) for edge in edges]
        unique_set = set(edge_tuples)
        unique_list1 = list(unique_set)
        print(f"  Run {run+1}: {unique_list1}")

    # Method 2: Using OrderedDict - DETERMINISTIC
    print(f"\n✅ Method 2: OrderedDict.fromkeys() - DETERMINISTIC")
    for run in range(3):
        edge_tuples = [tuple(edge) for edge in edges]
        unique_list2 = list(OrderedDict.fromkeys(edge_tuples))
        print(f"  Run {run+1}: {unique_list2}")

    # Method 3: Manual dedup - DETERMINISTIC
    print(f"\n✅ Method 3: Manual seen set - DETERMINISTIC")
    for run in range(3):
        seen = set()
        unique_list3 = []
        for edge in edges:
            edge_tuple = tuple(edge)
            if edge_tuple not in seen:
                seen.add(edge_tuple)
                unique_list3.append(edge_tuple)
        print(f"  Run {run+1}: {unique_list3}")


def create_deterministic_fix():
    """Create a deterministic version of the deduplication fix."""
    print(f"\n🔧 DETERMINISTIC DEDUPLICATION FIX")
    print("=" * 50)

    fix_code = '''
def deterministic_dedup(edges):
    """Deterministic deduplication that preserves order."""
    seen = set()
    unique_edges = []
    
    for edge in edges:
        edge_tuple = tuple(edge)
        if edge_tuple not in seen:
            seen.add(edge_tuple)
            unique_edges.append(edge_tuple)
    
    return unique_edges

# Usage in fix_data_splits.py:
# Replace this line:
# unique_edges = list(set(edge_tuples))
# 
# With this:
# unique_edges = deterministic_dedup(edges)
'''

    print(fix_code)

    # Test it works
    edges = [(1, 2, 0), (2, 3, 1), (1, 2, 0), (3, 4, -1)]

    def deterministic_dedup(edges):
        seen = set()
        unique_edges = []
        for edge in edges:
            edge_tuple = tuple(edge)
            if edge_tuple not in seen:
                seen.add(edge_tuple)
                unique_edges.append(edge_tuple)
        return unique_edges

    result = deterministic_dedup(edges)
    print(f"Example: {edges} → {result}")
    print(f"✅ Maintains order: {result == [(1, 2, 0), (2, 3, 1), (3, 4, -1)]}")


def main():
    test_dedup_consistency()
    create_deterministic_fix()

    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Use deterministic deduplication to ensure reproducible splits")
    print(f"   2. Always set random.seed() before any randomization")
    print(f"   3. Consider sorting edges by (u, v, label) for extra consistency")


if __name__ == "__main__":
    main()
