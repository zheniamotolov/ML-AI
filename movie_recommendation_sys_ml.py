import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating=5)
print(repr(data['train']))
print(repr(data['test']))
model = LightFM(loss='warp')
