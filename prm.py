import logging.handlers
import multiprocessing.pool
import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
import spatialgeometry as sg
import swift
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple
from dataclasses import dataclass
import heapq
import os
import matplotlib.pyplot as plt
import logging
import pickle
import multiprocessing


# TODO: Speed up collision checking
# TODO: Speed up path finding with additional temporary nodes
# TODO: Can the goal be specified as a pose instead of a joint configuration?
# TODO: Overall cleanup


logger = logging.getLogger('PRM')


@dataclass
class Node:
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


def distance_function(node_1: Node, node_2: Node):
    _, omega = node_1.ee_pose.angvec(node_2.ee_pose)
    t = node_1.ee_pose.t - node_2.ee_pose.t
    twist = np.hstack((t, omega))    
    q_diff = node_1.q - node_2.q
    return 1*np.dot(q_diff, q_diff) + 0*np.dot(t, t)


def has_collision_along_linear_path(
    robot: rtb.Robot, 
    obstacles: List[sg.CollisionShape], 
    q1: List[float], 
    q2: List[float]
) -> bool:
    # TODO: incrementally subdivide path for faster collision checking
    n_steps = 10
    for i in range(n_steps):
        alpha = i / (n_steps - 1)
        q = alpha * np.array(q1) + (1 - alpha) * np.array(q2)
        for obstacle in obstacles:
            if robot.iscollided(q, obstacle):
                return True
    return False

def compute_distances_chunk(chunk: Tuple[int, int, List[Node]]) -> np.ndarray:
    """
    Compute a chunk of the distance matrix for a given range of indices.
    """
    start, end, nodes = chunk
    dist_chunk = np.zeros((end - start, len(nodes)))
    
    for i, node_1 in enumerate(nodes[start:end]):
        for j, node_2 in enumerate(nodes):
            dist_chunk[i, j] = distance_function(node_1, node_2)
    
    return start, dist_chunk

def parallel_distance_computation(nodes: List[Node], num_workers: int = None) -> np.ndarray:
    """
    Parallel computation of the distance matrix for a list of nodes.
    """
    num_nodes = len(nodes)
    distances = np.zeros((num_nodes, num_nodes))

    # Define the chunk size and number of workers
    chunk_size = num_nodes // (num_workers if num_workers else multiprocessing.cpu_count())
    chunks = [(i, min(i + chunk_size, num_nodes), nodes) for i in range(0, num_nodes, chunk_size)]

    # Perform multiprocessing to compute distance chunks
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(compute_distances_chunk, chunks)

    # Recombine the results
    for start, dist_chunk in results:
        distances[start:start + dist_chunk.shape[0], :] = dist_chunk

    # Ensure the distance matrix is symmetric
    distances = np.maximum(distances, distances.T)

    return distances


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

def parallel_edge_checking(edge_candidates: List[Tuple[Node, Node]], robot: rtb.Robot, obstacles: List[sg.CollisionShape]) -> List[Tuple[Node, Node]]:
    """
    Parallel edge checking using multiprocessing.
    """
    # Split edge candidates into batches
    batch_size = len(edge_candidates) // multiprocessing.cpu_count()
    edge_batches = [edge_candidates[i:i + batch_size] for i in range(0, len(edge_candidates), batch_size)]

    # Perform multiprocessing with each worker checking a batch of edges
    logger.info('Starting parallel collision checking for edges')
    with multiprocessing.Pool(
        processes=multiprocessing.cpu_count(), 
        initializer=worker_init, 
        initargs=(robot, obstacles)
    ) as pool:
        results = pool.map(check_edges_in_batch, edge_batches)

    # Flatten the results to get all valid edges
    valid_edges = [edge for edges in results for edge in edges]
    return valid_edges


def find_edges(nodes: List[Node], robot: rtb.Robot, obstacles: List[sg.CollisionShape], k_neighbours=10):
    """
    Main function that finds valid neighbours and handles multiprocessing.
    Each process checks a batch of edges for collision in parallel.
    """
    logger.info('Calculating distances between nodes')
    distances = parallel_distance_computation(nodes)

    logger.info('Performing K-nearest neighbours search')
    nbrs = NearestNeighbors(n_neighbors=k_neighbours, algorithm='auto', metric='precomputed', n_jobs=-1)
    nbrs.fit(distances)
    _, indices = nbrs.kneighbors(distances)

    edge_candidates = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # Skip the first one since it is the node itself
            node_1 = nodes[i]
            node_2 = nodes[neighbor]    
            edge_candidates.append((node_1, node_2))

    logger.info(f'Checking {len(edge_candidates)} edge candidates for collisions')
    valid_edges = parallel_edge_checking(edge_candidates, robot, obstacles)
    
    logger.info(f'Found {len(valid_edges)} valid collision-free edges')
    return valid_edges



def create_prm_graph(
    robot: rtb.Robot, 
    obstacles: List[sg.CollisionShape], 
    n_nodes: int, 
    k_neighbours: int
):
    nodes = create_nodes(robot, obstacles, n_nodes)
    edges = find_edges(nodes, robot, obstacles, k_neighbours)
    
    # Convert nodes and edges into a graph structure
    graph = {node: [] for node in nodes}    
    for node_1, node_2 in edges:
        graph[node_1].append(node_2)
        graph[node_2].append(node_1)
    
    return graph, nodes, edges


def find_path_dijkstra(start_node: Node, goal_node: Node, graph) -> List[Node]:
    # Priority queue (min-heap) to store the open set of nodes to explore
    open_set = []
    heapq.heappush(open_set, (0, start_node))
    
    # Dictionary to store the cost from the start node to a given node
    g_score = {start_node: 0}
    
    # Dictionary to reconstruct the path
    came_from = {}
    
    # Closed set to keep track of explored nodes
    closed_set = set()
    
    while open_set:
        current_g, current_node = heapq.heappop(open_set)
        
        # If we reach the goal, reconstruct and return the path
        if current_node == goal_node:
            path: List[Node] = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start_node)
            path.reverse()
            return path
        
        closed_set.add(current_node)
        
        # Explore neighbors of the current node
        for neighbor in graph[current_node]:
            if neighbor in closed_set:
                continue
            
            # Calculate tentative g_score (cost to reach this neighbor from the previous node)
            tentative_g_score = g_score[current_node] + distance_function(current_node, neighbor)
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # This is a better path, record it
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (tentative_g_score, neighbor))
    
    return None


def add_temporary_node(
    q0: List[float],
    graph: dict,
    nodes: List[Node],
    robot: rtb.Robot,
    obstacles: List[sg.CollisionShape],
    k_neighbours: int
) -> Node:
    logger.info("Adding temporary  node and connecting to k nearest neighbors.")
    node = Node(q=q0, ee_pose=robot.fkine(q0))
    # Find k nearest neighbors to the start configuration q0
    k_nearest_neighbors = NearestNeighbors(n_neighbors=k_neighbours).fit([node.q for node in nodes])
    distances, indices = k_nearest_neighbors.kneighbors([q0])

    # Add the temporary start node to the graph
    graph[node] = []
    for idx in indices[0]:
        neighbor_node = nodes[idx]
        # Check for collision-free path
        if not has_collision_along_linear_path(robot, obstacles, node.q, neighbor_node.q):
            graph[node].append(neighbor_node)
            graph[neighbor_node].append(node)
    if len(graph[node]) == 0:
        logger.info("No neighbors found for the temporary node.")
        return None

    logger.info(f"Temporary node connected to {len(graph[node])} neighbors.")
    
    return node






##########################################
# PLOT NODES, EDGES, AND PATH
##########################################
def plot_path(nodes: List[Node], edges, path=None):
    if len(nodes[0].q) != 2:
        raise RuntimeError('The c-space plotting only works for 2 DOF!')
    
    # Extract q1 and q2 values from the Node objects
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
    plt.show()


def simulate_path(robot: rtb.Robot, path: List[Node], obstacles, nodes: List[Node] = None):
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
    if nodes is not None:
        for node in nodes:
            env.add(sg.Sphere(0.01, color='red', pose=node.ee_pose))
        
        
    for i in range(len(path)-1):
        n_interpolations = 20
        start = path[i].q
        end = path[i+1].q
        for j in range(n_interpolations):
            q = start + j/n_interpolations * (end - start)
            robot.q = q
            env.step(0.2)
    env.close()




def save_graph_and_nodes(filename: str, graph: dict, nodes: List[Node]):
    """Save the graph and nodes to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump((graph, nodes), f)
    logger.info(f"Saved to {filename}")

def load_graph_and_nodes(filename: str) -> Tuple[dict, List[Node]]:
    """Load the graph and nodes from a file using pickle."""
    with open(filename, 'rb') as f:
        graph, nodes = pickle.load(f)
    logger.info(f"Loaded from {filename}")
    return graph, nodes



##########################################
# EXAMPLE
##########################################
if __name__ == '__main__':
    import time
    
    # Also show the logger name in the output
    format = '%(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=format)
    logger.setLevel(logging.INFO)



    # this_file_path = os.path.dirname(os.path.realpath(__file__))
    # robot = rtb.Robot.URDF(f'{this_file_path}/robots/acrobot.urdf')
    robot = rtb.models.KinovaGen3()
    obstacles = []
    obstacles.append(sg.Cuboid([1, 1, 1], pose = sm.SE3(0.7, 0.7, 0)))
    obstacles.append(sg.Cuboid([0.2, 0.2, 0.2], pose = sm.SE3(-0.3, 0.4, 0)))
    obstacles.append(sg.Sphere(0.4, pose = sm.SE3(0, -0.5, 0.5)))
    obstacles.append(sg.Sphere(0.2, pose = sm.SE3(-0.3, 0.3, 0.5)))


    # It is preferable to have many nodes but few edges per node
    N_NODES = 1000
    K_NEIGHBOURS = 5

    start_time = time.perf_counter()
    graph, nodes, edges = create_prm_graph(robot, obstacles, N_NODES, K_NEIGHBOURS)
    logger.info(f'Graph creation took {time.perf_counter() - start_time:.2f} seconds')
    save_graph_and_nodes('prm_map.pkl', graph, nodes)  
      
    graph, nodes = load_graph_and_nodes('prm_map.pkl')
    
    while True:
        q0 = robot.qlim[0, :] + np.random.rand(robot.n) * (robot.qlim[1, :] - robot.qlim[0, :]) 
        qf = robot.qlim[0, :] + np.random.rand(robot.n) * (robot.qlim[1, :] - robot.qlim[0, :])
        start_node = add_temporary_node(q0, graph, nodes, robot, obstacles, 10)
        goal_node = add_temporary_node(qf, graph, nodes, robot, obstacles, 10)
        if start_node is not None and goal_node is not None:
            break
        
    start_time = time.perf_counter()
    path = find_path_dijkstra(start_node, goal_node, graph)
    logger.info(f'Path finding took {time.perf_counter() - start_time:.2f} seconds')
    if path is None:
        logger.info('No path found!')
        exit()
    logger.info(f'Path length: {len(path)}')
    
    # plot_path(nodes, edges, path)
    simulate_path(robot, path, obstacles)
