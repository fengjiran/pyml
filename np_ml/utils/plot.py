import matplotlib as plt
from matplotlib.colors import ListedColormap
import numpy as np


def plot(x, y, **kwargs):
    """can only do 2D plot right now"""
    assert (x.shape[-1] == 2)
    color = (y + 2) / 5
