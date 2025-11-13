import networkx as nx
from typing import Dict, List, Tuple, Union

# --- Configuration ---
NODE_PREFIX = "Room"
START_NODE = f"{NODE_PREFIX} 1"

# --- Graph Creation Functions ---

def create_line_graph(n_nodes: int = 7) -> nx.Graph:
    """
    Creates a Simple Linear (Chain) Graph (n-line).
    Topology: 1 -- 2 -- 3 -- ... -- N
    """
    G = nx.path_graph(n_nodes)
    # Relabel nodes to start from 1 and use the prefix (e.g., 'Room 1')
    mapping = {i: f"{NODE_PREFIX} {i+1}" for i in range(n_nodes)}
    G = nx.relabel_nodes(G, mapping)
    
    # Assign rewards: High reward at the end (N), medium reward in the middle (N-2)
    rewards = {node: 0 for node in G.nodes()}
    rewards[f"{NODE_PREFIX} {n_nodes}"] = 50
    rewards[f"{NODE_PREFIX} {n_nodes-2}"] = 20
    nx.set_node_attributes(G, rewards, 'reward')
    
    return G

def create_tree_graph(n_nodes: int = 7) -> nx.Graph:
    """
    Creates a Hierarchical (Balanced Binary Tree) Graph (n-tree).
    Note: NetworkX doesn't have a direct 'balanced tree' generator for custom node count,
    so we construct a small one manually for 7 nodes (depth 3).
    """
    G = nx.Graph()
    nodes = [f"{NODE_PREFIX} {i}" for i in range(1, n_nodes + 1)]
    G.add_nodes_from(nodes)
    
    # Edges: 1->(2,3), 2->(4,5), 3->(6,7)
    edges = [
        ("Room 1", "Room 2"), ("Room 1", "Room 3"),
        ("Room 2", "Room 4"), ("Room 2", "Room 5"),
        ("Room 3", "Room 6"), ("Room 3", "Room 7"),
    ]
    G.add_edges_from(edges)
    
    # Assign rewards: Highest reward deep on one side (e.g., Room 7)
    rewards = {node: 0 for node in nodes}
    rewards["Room 7"] = 50
    rewards["Room 4"] = 15
    nx.set_node_attributes(G, rewards, 'reward')
    
    return G

def create_clustered_graph(n_clusters: int = 3, cluster_size: int = 5) -> nx.Graph:
    """
    Creates a Clustered/Dense graph by linking three complete graphs (cliques).
    Total nodes: n_clusters * cluster_size (e.g., 15 nodes)
    """
    G = nx.Graph()
    # Create and connect clusters
    nodes = []
    for i in range(n_clusters):
        # Create a complete graph for the cluster
        cluster = nx.complete_graph(cluster_size)
        mapping = {j: f"{NODE_PREFIX} {i*cluster_size + j + 1}" for j in range(cluster_size)}
        cluster = nx.relabel_nodes(cluster, mapping)
        
        G = nx.union(G, cluster, rename=(None, None))
        nodes.extend(cluster.nodes())
        
        # Connect clusters sequentially (Bridge node)
        if i > 0:
            # Connect the last node of the previous cluster to the first node of the current cluster
            G.add_edge(nodes[i*cluster_size - 1], nodes[i*cluster_size])

    # Assign rewards: High reward in the final cluster
    rewards = {node: 0 for node in G.nodes()}
    # Target in the last cluster
    rewards[nodes[-1]] = 75 
    # Distractor reward in the first cluster
    rewards[nodes[3]] = 30 
    nx.set_node_attributes(G, rewards, 'reward')

    return G

# --- Ground Truth and Prompt Generation ---

def get_optimal_path(G: nx.Graph, start_node: str) -> List[str]:
    """
    (Symbolic Planner Core) Finds the shortest path to the node with the maximum reward.
    Assumes shortest path is optimal for planning speed/efficiency.
    """
    # Find the node with the maximum reward
    max_reward_node = max(G.nodes(data='reward'), key=lambda x: x[1])[0]
    
    try:
        # Use BFS to find the shortest path in an unweighted graph
        optimal_path = nx.shortest_path(G, source=start_node, target=max_reward_node)
        return optimal_path
    except nx.NetworkXNoPath:
        return [f"No path found to {max_reward_node}"]

def generate_stimuli_text(G: nx.Graph, task: str) -> str:
    """
    Generates the text description and specific task instruction.
    """
    node_list = list(G.nodes())
    start_node = node_list[0] # Always start at the first room for consistency
    
    # Part 1: Graph Description
    desc_lines = [f"Imagine a facility with {G.number_of_nodes()} rooms, starting at {start_node}."]
    
    # Describe connections (simplification: only list the full set of edges)
    connections = []
    for u, v in G.edges():
        # Only list connection once since it's an undirected graph
        if v.startswith(NODE_PREFIX) and u < v:
             connections.append(f"{u} and {v}")
    
    desc_lines.append(f"The facility is connected as follows: {'; '.join(connections)}.")
    
    # Part 2: Rewards Description (Always needed for Value-Based Planning)
    reward_desc = "The known rewards are: "
    rewards = nx.get_node_attributes(G, 'reward')
    # List all non-zero rewards
    reward_desc += ", ".join([f"{node} has a reward of ${rewards[node]}" for node in G.nodes() if rewards[node] > 0])
    desc_lines.append(reward_desc + ".")
    
    # Part 3: Task-Specific Instruction (Scratchpad/ValuePath/Reval)
    if task == 'valuePath':
        task_instruction = f"TASK: Find the shortest path from {start_node} to the room with the absolute highest reward."
    elif task == 'rewardReval':
        # The key for Reval is a two-part prompt structure
        initial_goal_node = max(G.nodes(data='reward'), key=lambda x: x[1])[0]
        # Temporarily change a reward to create a new optimal path
        new_rewards = rewards.copy()
        # Find a different node (e.g., the second-highest reward node) and increase its reward
        second_best = sorted(G.nodes(data='reward'), key=lambda x: x[1], reverse=True)[1][0]
        new_rewards[second_best] = new_rewards[initial_goal_node] + 10 # Make it the new max
        
        task_instruction = (
            f"INITIAL SCENARIO: The optimal path leads to {initial_goal_node}. "
            f"DYNAMIC UPDATE: Suddenly, the reward at {second_best} changes to ${new_rewards[second_best]}! "
            f"TASK: Re-evaluate and find the *new* optimal shortest path from {start_node} to the room with the absolute highest reward."
        )
        # NOTE: The LLM must reason with the new reward, but the ground truth calculation will use the original graph structure with the NEW rewards.
        G_reval = G.copy()
        nx.set_node_attributes(G_reval, new_rewards, 'reward')
        return "\n".join(desc_lines) + "\n" + task_instruction, G_reval # Return the modified graph for ground truth

    return "\n".join(desc_lines) + "\n" + task_instruction, G # Return the original graph

def generate_scratchpad_instruction(base_prompt: str) -> str:
    """Appends the required Scratchpad Method instructions."""
    scratchpad_instruction = (
        "\n\n***SCRATCHPAD METHOD***: Before your final answer, list your full reasoning. "
        "For each step, state the current room, the available choices, the chosen next room, and the rationale based on the goal. "
        "Final answer MUST be the path only (e.g., Room 1 -> Room 2 -> ...)."
    )
    return base_prompt + scratchpad_instruction

# --- Main Execution for Data Generation ---

def generate_all_stimuli():
    """Generates a structured dictionary for all experimental stimuli."""
    graph_configs = {
        "n7line": create_line_graph(7),
        "n7tree": create_tree_graph(7),
        "n15clustered": create_clustered_graph(3, 5) 
    }
    tasks = ['valuePath', 'rewardReval']
    
    stimuli = {}
    
    for graph_name, G in graph_configs.items():
        for task in tasks:
            # 1. Generate the base prompt and get the graph structure (might be modified for Reval)
            base_prompt, G_task = generate_stimuli_text(G, task)
            
            # 2. Get the ground truth path based on the final reward structure
            ground_truth = get_optimal_path(G_task, START_NODE)
            
            # 3. Apply the Scratchpad instruction
            scratchpad_prompt = generate_scratchpad_instruction(base_prompt)
            
            key_base = f"{graph_name}_{task}"
            
            stimuli[key_base] = {
                "graph_type": graph_name,
                "task_type": task,
                "prompt": base_prompt, # Used for Dynamic Analysis (without explicit scratchpad in input)
                "scratchpad_prompt": scratchpad_prompt, # Used for Scratchpad condition
                "ground_truth_path": ground_truth,
                "raw_graph_data": G_task # Use this to feed the symbolic planner
            }
            
    return stimuli

if __name__ == '__main__':
    all_stimuli = generate_all_stimuli()
    
    print("--- Sample Stimulus: n7tree_rewardReval ---")
    sample_key = "n7tree_rewardReval"
    sample = all_stimuli[sample_key]
    
    print(f"Graph Type: {sample['graph_type']}")
    print(f"Task Type: {sample['task_type']}")
    print(f"Ground Truth Path: {' -> '.join(sample['ground_truth_path'])}")
    print("\nScratchpad Prompt:")
    print(sample['scratchpad_prompt'])
    
    print("\n--- Example: Visualizing a Clustered Graph ---")
    clustered_G = all_stimuli['n15clustered_valuePath']['raw_graph_data']
    
    import matplotlib.pyplot as plt

    for key, data in all_stimuli.items():
        G = data['raw_graph_data']
        rewards = nx.get_node_attributes(G, 'reward')
        node_colors = [rewards[node] for node in G.nodes()]

        plt.figure(figsize=(7, 5))
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        nx.draw(
            G, pos,
            with_labels=True,
            node_size=1000,
            node_color=node_colors,
            cmap=plt.cm.YlOrRd,
            font_weight='bold'
        )
        plt.title(f"{data['graph_type']} ({data['task_type']})")
        plt.show()
