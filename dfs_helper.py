
from madeye import get_all_neighbors_with_rotation
from madeye_utils import extract_pan, extract_tilt, extract_zoom

def get_next_pan(input):
    p = extract_pan(input)
    t = extract_tilt(input)
    z = extract_zoom(input)

path_and_cost = (None, None)
def dfs(graph, node, cost, visited, path, node_to_neighbors):
    global path_and_cost
    visited.add(node)
    path.append(node)

    if len(path) == len(graph):
        path_and_cost = (path, cost)
        return

    for neigh in node_to_neighbors[node]:
        neighbor = neigh[0]
        next_cost = neigh[1]
        if neighbor not in visited:
            dfs(graph, neighbor, cost + next_cost, visited, path, node_to_neighbors)

    path.pop()
    visited.remove(node)

def find_least_cost_path(graph):
    global path_and_cost
    graph_as_set = set(graph)
    node_to_neighbors = {}
    for n in graph:
        neighbors = get_all_neighbors_with_rotation(n)
        for neighbor in neighbors:
            node = neighbor[0]
            if node not in graph_as_set:
                continue
            if node not in node_to_neighbors:
                node_to_neighbors[node] = set()
            if n not in node_to_neighbors:
                node_to_neighbors[n] = set()
            node_to_neighbors[n].add((node, neighbor[1]))
            node_to_neighbors[node].add((n, neighbor[1]))

    least_cost = -1
    lowest_cost_path = None
    for element in graph:
        dfs(graph, element, 0, set(), [], node_to_neighbors)
        if path_and_cost[1] is None:
            continue
        if path_and_cost[1] < least_cost or least_cost < 0:
            least_cost = path_and_cost[1]
            lowest_cost_path = path_and_cost[0]

    path_and_cost = (None, None)
    return lowest_cost_path, least_cost

if __name__ == "__main__":
    graph = ['270-0-1', '270--15-1', '240-0-1', '240--15-1']
    print(find_least_cost_path(graph))
