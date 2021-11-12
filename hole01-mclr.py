# ## Hole #1: Multiclass Logistic Regression

# Plot Setup
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
camera = celluloid.Camera(fig)


# Data creation 
BATCH, CLASS, EPOCHS, FEATURES = 100, 10, 10, 2
r = t.arange(BATCH).long()
X = t.rand(BATCH, FEATURES + 1)
X[:, -1] = 1
y = (X.sum(1) > 2).long()

# Training
w = t.zeros(FEATURES + 1, CLASS, requires_grad=True)
opt = t.optim.LBFGS([w])
for _ in range(EPOCHS):
    def loss():
        out = -((X @ w).log_softmax(1)[r, y]).mean()
        (w.grad,) = t.autograd.grad(out, w)

        # Plotting
        ax.scatter(X[:, 0], X[:, 1], c=y)
        w2 = (w[:,  0] - w[:, 1]).detach()
        xs = t.linspace(0, 1.0, 100)
        ax.plot(xs, (-xs * (w2[0] /w2[1]) - (w2[2]/w2[1])), color="blue")
        camera.snap()
        return out
    opt.step(loss)


# Plot    
HTML(camera.animate().to_jshtml())
__st.write(camera.animate().to_html5_video(), unsafe_allow_html=True)



# [Discussion](https://twitter.com/srush_nlp/status/1450474508675174400)
