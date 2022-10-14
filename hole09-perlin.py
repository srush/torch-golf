from torch import *
def perlin(M, N):
    x, y = tensor([0, 0, 1, 1]), tensor([0, 1, 0, 1])
    grad = (rand(2, N // M + 1, N // M + 1) * 2 - 1) 
    grid = stack(meshgrid(arange(N) / M, arange(N) / M))
    corners = stack([grid // 1, grid // 1 + 1]).long()
    offset = stack([grid % 1.0, grid - (1.0 + grid // 1)])
    g = grad[:, corners[x, 0, :], corners[y, 1, :]]
    f = stack([offset[x, 0], offset[y, 1]])
    x = (g * f).sum(0).view(2, 2, N, -1)
    def interp(t, a, b):
        return a + (3 * t ** 2 - 2 * t ** 3) * (b - a) 
    d = [interp(offset[0, 0], x[0, i], x[1, i]) 
         for i in [0,1]]
    return interp(offset[0, 1], d[0], d[1])
octave_weights = tensor([3, 2, 2, 1, 0.5, 0.2, 0.2])
octaves = stack([perlin(i, 512) 
                for i in [256, 128, 64, 32, 16, 8, 4]])
cur = (octave_weights[:, None, None] * octaves).sum(0)

from celluloid import Camera
fig = plt.figure(dpi = 100)
camera = Camera(fig)
for i in arange(300):
    cur[1:301] = cur[arange(300)]
    cur[301:, 1:-1] = cur[arange(300, 511)[:, None, None], 
                          arange(1, 511)[None, :] + arange(3)[:, None] - 1].mean(1)
    plt.imshow(cur[300:].flip(0), vmax=1, vmin=-1, cmap="hot")
    camera.snap()
