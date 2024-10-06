import logging.handlers
import multiprocessing.pool
import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
import spatialgeometry as sg
import swift
from sklearn.neighbors import KDTree
from typing import List, Tuple
from dataclasses import dataclass
import heapq
import os
import matplotlib.pyplot as plt
import logging
import pickle
import multiprocessing
from tqdm import tqdm


logger = logging.getLogger('PRM')


@dataclass
class Node:
    """
    Convenience dataclass to store the node information.
    """
    q: List[float]
    ee_pose: sm.SE3
    
    def __hash__(self):
        # Create a hash based on the tuple of joint positions only.
        # This allows a node to be used as key in a dictionary
        return hash((tuple(self.q)))
    
    def __eq__(self, other):
        # Create an equality check based on the joint positions only
        if isinstance(other, Node):
            return np.array_equal(self.q, other.q)
        return False     
        

def create_nodes(robot: rtb.Robot, obstacles: List[sg.CollisionShape], n_nodes: int) -> List[Node]:
    """
    Returns a list of n_nodes randomly generated nodes that are collision-free.
    """
    logger.info(f'Creating {n_nodes} nodes')
    if not isinstance(obstacles, list):
        obstacles = [obstacles]
        
    nodes: List[Node] = []
    i = 0
    while i < n_nodes:
        q_node = robot.qlim[0, :] + np.random.rand(robot.n) * (robot.qlim[1, :] - robot.qlim[0, :])
        is_collided = False
        for obstacle in obstacles:
            if robot.iscollided(q_node, obstacle):
                is_collided = True
                break
        if not is_collided:
            node = Node(q_node, robot.fkine(q_node))
            nodes.append(node)
            i += 1
    return nodes


def has_collision_along_linear_path(
    robot: rtb.Robot, 
    obstacles: List[sg.CollisionShape], 
    q1: List[float], 
    q2: List[float], 
    threshold: float = 0.05
) -> bool:
    """
    Returns True if there is a collision along the path between q1 and q2 by
    recursively checking the midpoint and subdividing until the segment length
    is smaller than the threshold.
    """
    def recursive_check(q1, q2):
        midpoint = (np.array(q1) + np.array(q2)) / 2        
        for obstacle in obstacles:
            if robot.iscollided(midpoint, obstacle):
                return True

        # If the segment length is above the threshold, recursively check further
        diff = np.array(q2) - np.array(q1)
        segment_length_squared = np.dot(diff, diff)
        if segment_length_squared > threshold**2:
            if recursive_check(q1, midpoint) or recursive_check(midpoint, q2):
                return True
        return False

    # Check the endpoints first for early exits
    for obstacle in obstacles:
        if robot.iscollided(q1, obstacle) or robot.iscollided(q2, obstacle):
            return True
            
    return recursive_check(q1, q2)


def worker_init(robot: rtb.Robot, obstacles: List[sg.CollisionShape]):
    """
    Initialize local copies of robot and obstacles in each worker process.
    This ensures that each process has its own instance of the robot and obstacles.
    """
    # These are "global" within the worker process
    global robot_instance, obstacles_instance
    robot_instance = robot.copy()
    obstacles_instance = [obstacle.copy() for obstacle in obstacles]


def check_edges_in_batch(edge_batch: List[Tuple[Node, Node]]) -> List[Tuple[Node, Node]]:
    """
    Check multiple edges in a batch. Each worker processes a batch of edges.
    Returns only the valid edges without collisions.
    """
    global robot_instance, obstacles_instance
    valid_edges = []
    
    for edge in edge_batch:
        node_1, node_2 = edge
        if not has_collision_along_linear_path(robot_instance, obstacles_instance, node_1.q, node_2.q):
            valid_edges.append(edge)
    
    return valid_edges


def parallel_edge_checking(
    edge_candidates: List[Tuple[Node, Node]], 
    robot: rtb.Robot, 
    obstacles: List[sg.CollisionShape], 
    batch_size: int = 50
) -> List[Tuple[Node, Node]]:
    """
    Parallel edge checking using multiprocessing with progress tracking and small batch size.
    """
    # Split edge candidates into smaller batches for frequent progress updates
    edge_batches = [edge_candidates[i:i + batch_size] for i in range(0, len(edge_candidates), batch_size)]

    with multiprocessing.Pool(
        processes=multiprocessing.cpu_count(), 
        initializer=worker_init, 
        initargs=(robot, obstacles)
    ) as pool:
        results = list(tqdm(pool.imap(check_edges_in_batch, edge_batches), total=len(edge_batches)))

    # Flatten the results to get all valid edges
    valid_edges = [edge for edges in results for edge in edges]
    return valid_edges


def compute_edges(nodes: List[Node], robot: rtb.Robot, obstacles: List[sg.CollisionShape], k_neighbours=10):
    """
    Finds valid neighbours using KD-Tree and checks for collisions
    before path finding.
    """
    logger.info('Building KD-Tree for efficient neighbor search')
    node_positions = np.array([node.q for node in nodes])
    kd_tree = KDTree(node_positions)

    logger.info('Performing K-nearest neighbors search')
    distances, indices = kd_tree.query(node_positions, k=k_neighbours)
    
    edge_candidates = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # Skip the first one since it is the node itself
            node_1 = nodes[i]
            node_2 = nodes[neighbor]
            edge_candidates.append((node_1, node_2))

    logger.info(f'Checking {len(edge_candidates)} edge candidates for collisions')
    valid_edges = parallel_edge_checking(edge_candidates, robot, obstacles)  # Full collision check

    logger.info(f'Found {len(valid_edges)} valid collision-free edges')
    return valid_edges


def generate_prm_graph(
    robot: rtb.Robot, 
    obstacles: List[sg.CollisionShape], 
    n_nodes: int, 
    k_neighbours: int
):
    nodes = create_nodes(robot, obstacles, n_nodes)
    edges = compute_edges(nodes, robot, obstacles, k_neighbours)
    
    # Convert nodes and edges into a graph structure
    graph = {node: [] for node in nodes}    
    for node_1, node_2 in edges:
        graph[node_1].append(node_2)
        graph[node_2].append(node_1)
    
    return graph


def distance_function(node_1: Node, node_2: Node):
    """
    Computes the distance between two nodes.
    You can experiment with different distance metrics here.
    """
    JOINT_WEIGHT = 1
    POSE_WEIGHT = 10
    theta, axis = node_1.ee_pose.angvec(node_2.ee_pose)
    omega = theta * axis
    t = node_1.ee_pose.t - node_2.ee_pose.t
    pose_diff = np.hstack((t, omega))    
    pose_distance = np.linalg.norm(pose_diff)
    q_diff = node_1.q - node_2.q
    q_distance = np.linalg.norm(q_diff)
    return JOINT_WEIGHT*q_distance + POSE_WEIGHT*pose_distance


def find_path_astar(start_node: Node, goal_node: Node, graph: dict) -> List[Node]:
    """
    A* algorithm for finding the shortest path after full collision checking.
    """
    logger.info('Finding path using A* algorithm')
    open_set = []
    heapq.heappush(open_set, (0, start_node))
    
    g_score = {start_node: 0}
    f_score = {start_node: distance_function(start_node, goal_node)}
    
    came_from = {}
    closed_set = set()
    while open_set:
        current_f, current_node = heapq.heappop(open_set)
        
        if current_node == goal_node:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start_node)
            path.reverse()
            return path
        
        closed_set.add(current_node)
        
        for neighbor in graph[current_node]:
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current_node] + distance_function(current_node, neighbor)
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + distance_function(neighbor, goal_node)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None


def add_temporary_node(
    q: List[float],
    graph: dict, 
    robot: rtb.Robot, 
    obstacles: List[sg.CollisionShape], 
    k_neighbours: int
) -> Node:
    """
    Add a temporary node to the graph and connect it to its neighbors. Returns Node if successful.
    MODIFIES the graph in place!
    """
    node = Node(q=q, ee_pose=robot.fkine(q))
    nodes = list(graph.keys())
    
    # Check if the node is in collision
    for obstacle in obstacles:
        if robot.iscollided(q, obstacle):
            logger.warning("The temporary node is in collision.")
            return None
    
    # Find neighbors and check for collisions
    graph[node] = []
    kd_tree = KDTree(np.array([n.q for n in nodes]))
    _, indices = kd_tree.query([q], k=k_neighbours)
    for idx in indices[0]:
        neighbor_node = nodes[idx]
        if not has_collision_along_linear_path(robot, obstacles, node.q, neighbor_node.q):
            graph[node].append(neighbor_node)
            graph[neighbor_node].append(node)
    
    if len(graph[node]) == 0:
        logger.warning("No neighbors found for the temporary node.")
        return None
    
    return node


def plot_path(graph: dict, path: List[Node]=None, save_to_file: str=None):
    """
    Plot the nodes, edges, and path in the configuration space.
    ONLY FOR 2 DOF ROBOTS!
    """
    if len(next(iter(graph)).q) != 2:
        raise RuntimeError('The c-space plotting only works for 2 DOF!')
    
    # Extract nodes and edges
    nodes: List[Node] = list(graph.keys())
    edges: List[Tuple[Node, Node]] = []
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            edges.append((node, neighbor))
    q1 = [node.q[0] for node in nodes]
    q2 = [node.q[1] for node in nodes]
    
    # Plot nodes
    plt.scatter(q1, q2, c='blue', label='Nodes')

    # Plot edges
    for edge in edges:
        q1_start, q2_start = edge[0].q[0], edge[0].q[1]
        q1_end, q2_end = edge[1].q[0], edge[1].q[1]
        plt.plot([q1_start, q1_end], [q2_start, q2_end], 'k-', alpha=0.5)
    
    # Plot path if it exists
    if path is not None:
        path_q1 = [node.q[0] for node in path]
        path_q2 = [node.q[1] for node in path]
        plt.plot(path_q1, path_q2, 'r-', linewidth=2, label='Path')
        plt.scatter(path_q1, path_q2, c='red', label='Path Nodes')
    
    plt.xlabel('q1')
    plt.ylabel('q2')
    plt.title('Nodes, edges, and path in configuration space')
    plt.legend()
    if save_to_file is not None:
        plt.savefig(save_to_file)
    plt.show()


def simulate_path(
    robot: rtb.Robot, 
    path: List[Node], 
    obstacles: List[sg.CollisionShape], 
    graph: dict = None):
    """
    Simulate the robot moving along the path in the environment.
    """
    if not isinstance(obstacles, list):
        obstacles = [obstacles]
    env = swift.Swift()
    env.launch(realtime=True)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('websockets.server').setLevel(logging.WARNING)
        
    env.add(robot)
    for obstacle in obstacles:
        env.add(obstacle)
    env.add(sg.Axes(length=0.1, pose=path[0].ee_pose))
    env.add(sg.Axes(length=0.1, pose=path[-1].ee_pose))
    if graph is not None:
        nodes: List[Node] = list(graph.keys())
        for node in nodes:
            env.add(sg.Sphere(0.01, color='red', pose=node.ee_pose))
                        
    STEP_SIZE = 0.1
    for i in range(len(path)-1):
        start = path[i].q
        end = path[i+1].q
        n_interpolations = int(np.linalg.norm(np.array(end) - np.array(start)) / STEP_SIZE)
        for j in range(n_interpolations):
            q = start + j/n_interpolations * (end - start)
            robot.q = q
            env.step(0.1)
    env.close()


def save_graph_and_nodes(filename: str, graph: dict):
    """
    Save the graph to a file using pickle.
    """
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)
    logger.info(f"Saved to {filename}")


def load_graph_and_nodes(filename: str) -> dict:
    """
    Load the graph from a file using pickle.
    """
    with open(filename, 'rb') as f:
        graph = pickle.load(f)
    logger.info(f"Loaded from {filename}")
    return graph



##########################################
# EXAMPLE USAGE
##########################################
if __name__ == '__main__':
    import time
    import argparse
    
    # Configure logger
    format = '%(asctime)s: %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=format)
    logger.setLevel(logging.INFO)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PRM Path Planning Example')
    parser.add_argument('--n_nodes', type=int, default=1000, help='Number of nodes to generate')
    parser.add_argument('--k_neighbours', type=int, default=5, help='Number of neighbors to connect')
    parser.add_argument('--robot', type=str, default='kinova_gen3', help='Robot name', choices=['kinova_gen3', 'planar_2dof'])
    parser.add_argument('--load_from_file', type=str, default=None, help='Load graph from file if provided')
    args = parser.parse_args()

    # Create robot and obstacles
    if args.robot == 'kinova_gen3':
        robot = rtb.models.KinovaGen3()
    elif args.robot == 'planar_2dof':
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        robot = rtb.Robot.URDF(os.path.join(this_file_dir, 'robots', 'planar_2dof.urdf'))           
    obstacles = []
    obstacles.append(sg.Cuboid([1, 1, 1], pose = sm.SE3(0.7, 0.7, 0)))
    obstacles.append(sg.Cuboid([0.2, 0.2, 0.2], pose = sm.SE3(-0.3, 0.4, 0)))
    obstacles.append(sg.Sphere(0.4, pose = sm.SE3(0, -0.5, 0.5)))
    obstacles.append(sg.Sphere(0.2, pose = sm.SE3(-0.3, 0.3, 0.5)))

    # Generate PRM graph
    print(args.load_from_file)
    if args.load_from_file is None:
        logger.info('Generating PRM graph')
        start_time = time.perf_counter()
        graph = generate_prm_graph(robot, obstacles, args.n_nodes, args.k_neighbours)
        logger.info(f'Graph creation took {time.perf_counter() - start_time:.2f} seconds')
        save_graph_and_nodes('prm_map.pkl', graph)
    else:
        graph = load_graph_and_nodes('prm_map.pkl')
    
    # Generate start and goal nodes
    logger.info('Randomly selecting start and goal nodes')
    while True:
        q0 = robot.qlim[0, :] + np.random.rand(robot.n) * (robot.qlim[1, :] - robot.qlim[0, :]) 
        qf = robot.qlim[0, :] + np.random.rand(robot.n) * (robot.qlim[1, :] - robot.qlim[0, :])
        start_node = add_temporary_node(q0, graph, robot, obstacles, args.k_neighbours)
        goal_node = add_temporary_node(qf, graph, robot, obstacles, args.k_neighbours)
        if start_node is not None and goal_node is not None:
            # Iterate until both start and goal nodes are valid
            break
        
    # Find path
    start_time = time.perf_counter()
    path = find_path_astar(start_node, goal_node, graph)
    logger.info(f'Path finding took {time.perf_counter() - start_time:.2f} seconds')
    if path is None:
        logger.info('No path found!')
        exit()
    logger.info(f'Path length: {len(path)}')
    
    # Plot
    if robot.n == 2:
        plot_path(graph, path)
    simulate_path(robot, path, obstacles)
