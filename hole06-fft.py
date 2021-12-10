from torch import pi, exp, arange, empty, cfloat

# FFT 
def fft(x, N, s):
    B = x.shape[0]
    if N == 1: return x[:, :1]
    X = empty(B, N, dtype=cfloat).view(B, 2, -1)
    X[:, 0], X[:, 1] = fft(x, N // 2, 2 * s), fft(x[:, s:], N // 2, 2 * s)
    q = exp(-((2j * pi) / N) * arange(N // 2)) * X[:, 1]
    X[:, 0], X[:, 1] = X[:, 0] + q, X[:, 0] - q
    return X.view(B, N)

# Sound Processing
import torchaudio
sound = torchaudio.load("regal-allmine.mp3")[0][0,10000:]
STEP = 1000; WINDOW = 512; TIME = 800
i = arange(WINDOW)[None, :] +  STEP * arange(TIME)[:, None] 
out = (10 * fft(sound[i], WINDOW, 1).abs().square().log10())
out = out.view(-1, 16, 32).mean(-1).view(TIME // 2, 2, 16).mean(1)

# Make Video
import matplotlib.pyplot as plt
import celluloid
fig, ax = plt.subplots(); camera = celluloid.Camera(fig)
for i in range(TIME // 2):
    ax.bar(arange(16), (-out[i]).relu(), color="purple")
    camera.snap()
    
from IPython.display import HTML
time_step = STEP / 48000 * 2
animation = camera.animate(interval=time_step*1000) # animation ready
HTML(animation.to_html5_video())
