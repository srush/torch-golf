# ## Hole #5 - Hidden Markov Model

fig, ax = plt.subplots(nrows=1, ncols=1)
camera = celluloid.Camera(fig)
def HMM(O, H, T, E, P):
    p = 1.0
    for l in range(O.shape[0]):
        P = ((H[l] * P)[:, None] * E) @ O[l] @ T
        p = p * P.sum()
        P = P / P.sum()
    return (p * P.sum()).log()

# Generate simple HMM with circulant transitions
STATES, OBS = 500, 500
E, T = t.eye(STATES), t.zeros(STATES, STATES), 
P = t.ones(STATES) / STATES
kernel = t.arange(-6, 7)[:, None]
s = t.arange(STATES)
T[s, (s + kernel).remainder(STATES)] = 1. / kernel.shape[0]

# Posterior inference over states with some known observations
obs = lambda x, N=OBS: \
         t.nn.functional.one_hot(x, N)[None].float()
hidden = lambda s, N=OBS: t.ones(s, N, requires_grad=True)

start = hidden(1000).detach()
start.requires_grad_(False)
for i in range(5):
    
    start[t.randint(1000, (1,))[0], :] = obs(t.randint(STATES, (1,))[0])
    states = hidden(start.shape[0], STATES)

    # Run and plot...
    HMM(start, states, T, E, P).backward()
    ax.imshow(states.grad.transpose(1, 0), vmax=0.02)
    camera.snap()



HTML(camera.animate(interval=300, repeat_delay=2000).to_jshtml())
__st.write(camera.animate(interval=300, repeat_delay=2000).to_html5_video(), unsafe_allow_html=True)
