"""
Approximate ODT Pattern Mining
Based on Section 6 of the paper (Alternative approximate solution)
"""

import argparse
import random
import json
from collections import defaultdict
import os
import sys


class ApproximateODTPatternMiner:
    def __init__(self, atomic_patterns, neighborhood_graph, timeslots, s_r=0.5):
        """
        Initialize the approximate ODT pattern miner.
        
        Args:
            atomic_patterns: List of dicts with 'origin', 'destination', 'timeslot', 'flow'
            neighborhood_graph: Dict where keys are regions and values are lists of neighbors
            timeslots: Total number of atomic timeslots
            s_r: Support ratio threshold
        """
        self.atomic_patterns = atomic_patterns
        self.G = neighborhood_graph
        self.timeslots = timeslots
        self.s_r = s_r
        
        # Build set for O(1) lookup
        self.atomic_pattern_set = set()
        for pattern in atomic_patterns:
            self.atomic_pattern_set.add((
                pattern['origin'], 
                pattern['destination'], 
                pattern['timeslot']
            ))
        
        print(f"Loaded {len(atomic_patterns)} atomic patterns")
        print(f"Graph has {len(neighborhood_graph)} regions")
    
    def wilson_spanning_tree(self, graph, num_trees=10):
        """Generate random spanning trees using Wilson's algorithm."""
        vertices = list(graph.keys())
        spanning_trees = []
        
        print(f"Generating {num_trees} spanning trees...")
        for i in range(num_trees):
            in_tree = set()
            tree = {}
            
            root = random.choice(vertices)
            in_tree.add(root)
            tree[root] = None
            
            remaining = set(vertices) - in_tree
            
            while remaining:
                current = random.choice(list(remaining))
                path = [current]
                
                while current not in in_tree:
                    neighbors = graph.get(current, [])
                    if len(neighbors) == 0:
                        break
                    
                    next_node = random.choice(neighbors)
                    
                    if next_node in path:
                        idx = path.index(next_node)
                        path = path[:idx+1]
                    else:
                        path.append(next_node)
                    
                    current = next_node
                
                for j in range(len(path)-1):
                    tree[path[j]] = path[j+1]
                    in_tree.add(path[j])
                    remaining.discard(path[j])
            
            spanning_trees.append(tree)
            
            # Progress indicator
            if (i + 1) % max(1, num_trees // 10) == 0:
                print(f"  Generated {i + 1}/{num_trees} trees")
        
        print(f"✓ Generated {len(spanning_trees)} spanning trees")
        return spanning_trees
    
    def bfs_subtree(self, tree, seed, size):
        """Extract a connected subtree of given size using BFS."""
        if seed not in tree:
            return {seed}
        
        subtree = {seed}
        queue = [seed]
        
        adjacency = defaultdict(set)
        for node, parent in tree.items():
            if parent is not None:
                adjacency[parent].add(node)
                adjacency[node].add(parent)
        
        while queue and len(subtree) < size:
            current = queue.pop(0)
            for neighbor in adjacency[current]:
                if neighbor not in subtree:
                    subtree.add(neighbor)
                    queue.append(neighbor)
                    if len(subtree) >= size:
                        break
        
        return subtree
    
    def generate_timeslot_range(self, start_idx, size):
        """Generate consecutive timeslots."""
        end_idx = min(start_idx + size, self.timeslots)
        return list(range(start_idx, end_idx))
    
    def check_pattern_support(self, O, D, T):
        """Check if pattern has enough atomic patterns."""
        total = len(O) * len(D) * len(T)
        if total == 0:
            return False
        
        count = 0
        for o in O:
            for d in D:
                for t in T:
                    if (o, d, t) in self.atomic_pattern_set:
                        count += 1
        
        support_ratio = count / total
        return support_ratio >= self.s_r
    
    def mine_patterns(self, S_O, S_D, S_T, num_trees=10, K=100):
        """Main mining algorithm."""
        print(f"\n{'='*60}")
        print(f"Mining Approximate ODT Patterns")
        print(f"{'='*60}")
        print(f"Target sizes: Origin={S_O}, Destination={S_D}, Timeslots={S_T}")
        print(f"Parameters: trees={num_trees}, trials_per_tree={K}, support_threshold={self.s_r}")
        
        print("\nStep 1: Generating spanning trees...")
        spanning_trees = self.wilson_spanning_tree(self.G, num_trees)
        
        patterns = []
        total_trials = num_trees * K
        
        print(f"\nStep 2: Testing {total_trials} random candidates...")
        
        trial_count = 0
        progress_interval = max(1, total_trials // 20)  # Show progress 20 times
        
        for tree_idx, tree in enumerate(spanning_trees):
            for trial in range(K):
                trial_count += 1
                
                # Progress indicator
                if trial_count % progress_interval == 0:
                    percentage = (trial_count / total_trials) * 100
                    print(f"  Progress: {trial_count}/{total_trials} ({percentage:.1f}%) - Found {len(patterns)} patterns so far")
                
                vertices = list(self.G.keys())
                
                seed_o = random.choice(vertices)
                O = self.bfs_subtree(tree, seed_o, S_O)
                
                seed_d = random.choice(vertices)
                D = self.bfs_subtree(tree, seed_d, S_D)
                
                if O & D:
                    continue
                
                max_start = max(0, self.timeslots - S_T)
                start_t = random.randint(0, max_start)
                T = self.generate_timeslot_range(start_t, S_T)
                
                if self.check_pattern_support(O, D, T):
                    pattern = {
                        'origin': sorted(list(O)),
                        'destination': sorted(list(D)),
                        'timeslots': T,
                        'sizes': {
                            'O': len(O),
                            'D': len(D),
                            'T': len(T)
                        }
                    }
                    patterns.append(pattern)
        
        print(f"  Progress: {total_trials}/{total_trials} (100.0%)")
        print(f"\n{'='*60}")
        print(f"✓ Found {len(patterns)} valid patterns")
        print(f"{'='*60}\n")
        
        return patterns


def load_atomic_patterns(filepath):
    """Load atomic patterns from .txt or .json file."""
    print(f"Loading atomic patterns from: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)
    
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} atomic patterns")
        return data
    
    # Handle .txt file
    patterns = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            if i == 0 and ('origin' in line.lower() or 'destination' in line.lower()):
                continue
            
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    pattern = {
                        'origin': parts[0].strip(),
                        'destination': parts[1].strip(),
                        'timeslot': int(parts[2].strip()),
                        'flow': float(parts[3].strip())
                    }
                    patterns.append(pattern)
                except ValueError:
                    print(f"Warning: Skipping invalid line {i+1}")
                    continue
    
    if len(patterns) == 0:
        print("ERROR: No valid patterns found in file!")
        sys.exit(1)
    
    print(f"✓ Loaded {len(patterns)} atomic patterns")
    return patterns


def load_neighborhood_graph(filepath):
    """Load neighborhood graph from .txt or .json file."""
    print(f"Loading neighborhood graph from: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)
    
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"✓ Loaded graph with {len(data)} nodes")
        return data
    
    graph = defaultdict(list)
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            if ':' in line:
                node, neighbors = line.split(':', 1)
                node = node.strip()
                neighbor_list = [n.strip() for n in neighbors.split(',')]
                graph[node] = neighbor_list
            
            elif ',' in line:
                parts = line.split(',')
                if len(parts) == 2:
                    node1 = parts[0].strip()
                    node2 = parts[1].strip()
                    
                    if node2 not in graph[node1]:
                        graph[node1].append(node2)
                    if node1 not in graph[node2]:
                        graph[node2].append(node1)
    
    graph = dict(graph)
    
    if len(graph) == 0:
        print("ERROR: No valid graph data found in file!")
        sys.exit(1)
    
    print(f"✓ Loaded graph with {len(graph)} nodes")
    return graph


def save_patterns(patterns, filepath):
    """Save discovered patterns to .txt or .json file."""
    print(f"\nSaving {len(patterns)} patterns to: {filepath}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    if filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(patterns, f, indent=2)
    else:
        with open(filepath, 'w') as f:
            f.write("# Discovered ODT Patterns\n")
            f.write("# Format: origin_regions | destination_regions | timeslots\n\n")
            
            for i, pattern in enumerate(patterns, 1):
                origins = ','.join(pattern['origin'])
                destinations = ','.join(pattern['destination'])
                timeslots = ','.join(map(str, pattern['timeslots']))
                
                f.write(f"Pattern {i}:\n")
                f.write(f"  Origins: {origins}\n")
                f.write(f"  Destinations: {destinations}\n")
                f.write(f"  Timeslots: {timeslots}\n")
                f.write(f"  Sizes: O={pattern['sizes']['O']}, D={pattern['sizes']['D']}, T={pattern['sizes']['T']}\n")
                f.write("\n")
    
    print(f"✓ Results saved successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Approximate ODT Pattern Mining (No external dependencies)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--atomic_patterns', type=str, required=True,
                       help='Path to atomic patterns file (.txt or .json)')
    parser.add_argument('--graph', type=str, required=True,
                       help='Path to neighborhood graph file (.txt or .json)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output file (.txt or .json)')
    parser.add_argument('--S_O', type=int, default=3,
                       help='Target size for origin regions (default: 3)')
    parser.add_argument('--S_D', type=int, default=3,
                       help='Target size for destination regions (default: 3)')
    parser.add_argument('--S_T', type=int, default=4,
                       help='Target size for timeslots (default: 4)')
    parser.add_argument('--num_trees', type=int, default=10,
                       help='Number of spanning trees (default: 10)')
    parser.add_argument('--K', type=int, default=100,
                       help='Trials per tree (default: 100)')
    parser.add_argument('--s_r', type=float, default=0.5,
                       help='Support ratio threshold (default: 0.5)')
    parser.add_argument('--timeslots', type=int, default=48,
                       help='Total timeslots (default: 48)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("APPROXIMATE ODT PATTERN MINING")
    print("="*60 + "\n")
    
    # Load data
    atomic_patterns = load_atomic_patterns(args.atomic_patterns)
    neighborhood_graph = load_neighborhood_graph(args.graph)
    
    # Initialize miner
    miner = ApproximateODTPatternMiner(
        atomic_patterns=atomic_patterns,
        neighborhood_graph=neighborhood_graph,
        timeslots=args.timeslots,
        s_r=args.s_r
    )
    
    # Mine patterns
    patterns = miner.mine_patterns(
        S_O=args.S_O,
        S_D=args.S_D,
        S_T=args.S_T,
        num_trees=args.num_trees,
        K=args.K
    )
    
    # Save results
    save_patterns(patterns, args.output)
    
    print("\n" + "="*60)
    print("✓ DONE!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()