
# ## Hole #2: Gaussian Mixture Model

fig, ax = plt.subplots()
camera = celluloid.Camera(fig)
# Data create
BATCH, DIM, CLASSES = 100, 2, 4
I = t.eye(DIM)
N = t.distributions.MultivariateNormal
y = t.randint(0, CLASSES, (BATCH,))
means = t.tensor([[2, 2.], [-2, 2.], [2, -2], [-2, -2.]])
X = N(means, I[None]).sample((BATCH,))[t.arange(BATCH), y]


# Fit the model
mu = t.rand(CLASSES, DIM) * 0.1
for epoch in range(15):

    # Model
    prior = t.distributions.Categorical(logits = t.zeros(CLASSES))
    dis = N(mu[:, :], I[None, :, :])
    class_ind = t.ones(BATCH, CLASSES, requires_grad=True)
    p_z = class_ind * prior.probs[None]
    log_p_x = dis.log_prob(X[:, None]).add(p_z.log()).logsumexp(-1).sum()
    
    # E
    log_p_x.backward()
    q = class_ind.grad


    # Plot
    ax.scatter(X[:, 0], X[:, 1], c=q.argmax(1))
    ax.scatter(mu[:, 0],  mu[:, 1], s= 200, marker="X", color="black")
    camera.snap()

    # M
    mu = (q[:, :, None] * X[:, None, :]).sum(0) / q.sum(0)[:, None]

HTML(camera.animate(interval=500, repeat_delay=1500).to_jshtml())
__st.write(camera.animate(interval=500, repeat_delay=1500).to_html5_video(), unsafe_allow_html=True)

