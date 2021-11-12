# ## Hole #3: Bayesian Linear Regression

fig, ax = plt.subplots()
ax.set_ylim(-50000, 200000)
camera = celluloid.Camera(fig)
# Data
N = t.distributions.Normal
n, sigma_2_o = 5, 1000
beta, m_0, S_0 = t.tensor(1e-4), t.zeros(1), t.ones(1, 1)
basis = lambda x: x * x

pts = []
for points in range(1, n+5):
    pts += [3*(t.rand(1)*10 +20), -3*(t.rand(1)*10 +20)]
    orig = t.hstack(pts)
    phi = basis(orig)[:, None]
    T = N(phi, sigma_2_o).sample()

    # Posterior
    S_0_inv = S_0.inverse()
    S_N = (S_0_inv + beta * phi.t() @ phi).inverse()
    m_N = S_N @ (S_0_inv @ m_0 + beta * phi.t() @ T)

    # Posterior Predictive
    x = t.arange(-175, 157).float()[:, None]
    phi_x = basis(x)
    mu = phi_x @ m_N  
    sigma_2 = 1 / beta + (phi_x[:, None, :] @ S_N @ phi_x[:, :, None]).view(-1)
    out = N(mu, sigma_2[:, None]).sample()

    # Plot
    ax.plot(x, mu, color="black")
    ax.plot(x, out, alpha=0.5, color="orange")
    ax.fill_between(x.view(-1), mu.view(-1) - sigma_2, mu.view(-1) + sigma_2,
                    color='gray', alpha=0.4)
    ax.scatter(orig, T,  zorder=10, color="blue")
    camera.snap()



HTML(camera.animate(interval=500, repeat_delay=1500).to_jshtml())
__st.write(camera.animate(interval=500, repeat_delay=1500).to_html5_video(), unsafe_allow_html=True)

