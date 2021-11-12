# # Torch Golf  🔥 ⛳
# - [Sasha Rush](http://rush-nlp.com/) / [@srush_nlp](https://twitter.com/srush_nlp)

# ## Introduction

# Being a professor is a fun job. You get to teach courses, write grants, mentor students.
# However, sometimes you have a day when you look at your schedule and it is just 7 hours of
# meetings straight. When this happens there is a part of me that lashes out and just wants to
# write some %$*% code. 


# This semester I have been playing Torch Golf. The game is to pick an extremely common
# algorithm in ML or AI and tor try to write it in <20 lines of Torch.  

# ### Rules

# 1. Keep the code short and focused.
# 1. Default to indexing tricks, avoid Torch specific functions.
# 1. Abuse the heck out of autodiff when possible.
# 1. Every example needs to have a cool graph.

# ### Prelude


import torch as t
import networkx as nx
import seaborn
import celluloid
import matplotlib.pyplot as plt
from IPython.core.display import HTML
seaborn.set_context("talk", font_scale=0.5)
plt.ioff()
None

