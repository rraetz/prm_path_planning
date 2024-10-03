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

# TODO: Parallelize map generation




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
    if not isinstance(obstacles, list):
        obstacles = [obstacles]
        
    nodes: List[Node] = []
    i = 0
    while i < n_nodes:
        q_min = robot.qlim[0, :]
        q_max = robot.qlim[1, :]
        q_node = q_min + np.random.rand(robot.n) * (q_max - q_min)
        # TODO: make this shorter and cleaner
        is_collided = False
        for obstacle in obstacles:
            if robot.iscollided(q_node, obstacle):
                is_collided = True
                break
        if not is_collided:
            node = Node(q_node, robot.fkine(q_node))
            nodes.append(node)
            i += 1
        if i % 100 == 0:
            logging.debug(f'Generated {i} nodes')
    return nodes



def distance_function(node_1: Node, node_2: Node):
    cartesian_diff = node_1.ee_pose.t - node_2.ee_pose.t
    q_diff = node_1.q - node_2.q
    return np.dot(q_diff, q_diff) + np.dot(cartesian_diff, cartesian_diff)


def find_neighbours(nodes: List[Node], robot: rtb.Robot, obstacles: List[sg.CollisionShape], k_neighbours=10):
    if not isinstance(obstacles, list):
        obstacles = [obstacles]
    
    logging.debug('Performing K-nearest neighbours search')
    distances = np.zeros((len(nodes), len(nodes)))
    for i, node_1 in enumerate(nodes):
        for j, node_2 in enumerate(nodes):
            if j < i: # Copy upper triangle to lower triangle
                distances[i, j] = distances[j, i]
            distances[i, j] = distance_function(node_1, node_2)
        
    nbrs = NearestNeighbors(n_neighbors=k_neighbours, algorithm='auto', metric='precomputed')
    nbrs.fit(distances)
    _, indices = nbrs.kneighbors(distances)

    def has_collision_along_linear_path(node_1: Node, node_2: Node) -> bool:
        # TODO: incrementally subdivide path for faster collision checking
        n_steps = 10
        for i in range(n_steps):
            alpha = i / (n_steps - 1)
            q = alpha * node_1.q + (1 - alpha) * node_2.q
            for obstacle in obstacles:
                if robot.iscollided(q, obstacle):
                    return True
        return False

    # TODO: pre-allocate memory
    edges = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            node_1 = nodes[i]
            node_2 = nodes[neighbor]        
            if not has_collision_along_linear_path(node_1, node_2):
                edges.append((node_1, node_2))
        if i % 100 == 0:
            logging.debug(f'Found neighbours for {i} nodes')
    return edges



def create_prm_graph(
    robot: rtb.Robot, 
    obstacles: List[sg.CollisionShape], 
    n_nodes: int, 
    k_neighbours: int
):
    nodes = create_nodes(robot, obstacles, n_nodes)
    edges = find_neighbours(nodes, robot, obstacles, k_neighbours)
    
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
            
            # Calculate tentative g_score (cost to reach this neighbor from the start)
            tentative_g_score = g_score[current_node] + distance_function(current_node, neighbor)
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # This is a better path, record it
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (tentative_g_score, neighbor))
    
    return None




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
        plt.plot([q1_start, q1_end], [q2_start, q2_end], 'k-', alpha=0.5)  # Plotting edges in joint space
    
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


def simulate_path(robot: rtb.Robot, path: List[Node], obstacles):
    if not isinstance(obstacles, list):
        obstacles = [obstacles]
    env = swift.Swift()
    env.launch(realtime=True)
    env.add(robot)
    for obstacle in obstacles:
        env.add(obstacle)
    for i in range(len(path)-1):
        n_interpolations = 30
        start = path[i].q
        end = path[i+1].q
        for j in range(n_interpolations):
            q = start + j/n_interpolations * (end - start)
            robot.q = q
            env.step(0.2)
    env.close()




##########################################
# EXAMPLE
##########################################
if __name__ == '__main__':
    import time
    
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=format)

    # this_file_path = os.path.dirname(os.path.realpath(__file__))
    # robot = rtb.Robot.URDF(f'{this_file_path}/robots/acrobot.urdf')
    robot = rtb.models.KinovaGen3()
    obstacles = []
    obstacles.append(sg.Cuboid([1, 1, 1], pose = sm.SE3(0.7, 0.7, 0)))
    obstacles.append(sg.Cuboid([0.2, 0.2, 0.2], pose = sm.SE3(-0.3, 0.4, 0)))
    obstacles.append(sg.Sphere(0.4, pose = sm.SE3(0, -0.5, 0.5)))
    obstacles.append(sg.Sphere(0.2, pose = sm.SE3(-0.3, 0.3, 0.5)))


    N_NODES = 100
    K_NEIGHBOURS = 5

    start_time = time.perf_counter()
    graph, nodes, edges = create_prm_graph(robot, obstacles, N_NODES, K_NEIGHBOURS)
    logging.debug(f'Graph creation took {time.perf_counter() - start_time:.2f} seconds')
    
    start = nodes[np.random.randint(0, len(nodes))]  
    goal = nodes[np.random.randint(0, len(nodes))]
    start_time = time.perf_counter()
    path = find_path_dijkstra(start, goal, graph)
    logging.debug(f'Path finding took {time.perf_counter() - start_time:.2f} seconds')
    if path is None:
        logging.debug('No path found!')
        exit()
    logging.debug(f'Path length: {len(path)}')
    
    # plot_path(nodes, edges, path)
    simulate_path(robot, path, obstacles)
