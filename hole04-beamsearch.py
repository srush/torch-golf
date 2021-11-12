# ## Hole #4 - Beam Search

fig, ax = plt.subplots()
camera = celluloid.Camera(fig)
# Make graph
BEAM, NODES, N, INF = 10000, 30, 30, 1e5
I, Z = t.eye(NODES).bool(), t.zeros(NODES)
base_edges = t.where(I, t.tensor([0.0]), t.rand(NODES, NODES))

# Graph
G = nx.from_numpy_matrix(base_edges.numpy())  
pos = nx.spring_layout(G)

# Setup
edges = t.where(I, t.tensor([INF]), base_edges)
edges.requires_grad_(True)
visit = t.zeros(BEAM, NODES).bool()
cur = t.arange(BEAM).long()
score = t.zeros(BEAM)
cur[NODES:] = -1
score[NODES:] = INF

# Beam Search
for i in range(N):
    expand = score[:, None] + edges[cur] + (visit * INF)
    score, indices = t.topk(expand.view(-1), BEAM, largest=False)
    visit = (visit + I[cur])[(indices / NODES).long()]
    cur = indices % NODES

    # Graph
    edge_g = t.autograd.grad(score[0], edges, retain_graph=True)[0]
    nx.draw(G, pos, width=0.2, ax=ax)
    nx.draw_networkx_edges(G, pos, 
                       edgelist=edge_g.nonzero().tolist(), 
                       edge_color='r', width=2, ax=ax)
    camera.snap()


HTML(camera.animate(repeat_delay=2000).to_jshtml())
__st.write(camera.animate(repeat_delay=2000).to_html5_video(), unsafe_allow_html=True)
