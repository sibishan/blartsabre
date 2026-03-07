import json
from architecture import multi_core_grid

arch = multi_core_grid(4, 3, 2, 2)
arch.draw()

node_to_core = {}
for core_idx, group in enumerate(arch.core_node_groups):
    for node in group:
        node_to_core[node] = core_idx

intra_core_edges = []
inter_core_edges = []

for u, v in arch.graph.edges():
    edge = [int(u), int(v)]
    if node_to_core[u] == node_to_core[v]:
        intra_core_edges.append(edge)
    else:
        inter_core_edges.append(edge)

intra_core_edges.sort()
inter_core_edges.sort()

result = {
    "intra_core_edges": intra_core_edges,
    "inter_core_edges": inter_core_edges,
}

print(json.dumps(result, indent=2))

with open("tests/edges.json", "w") as f:
    json.dump(result, f, indent=2)