from __future__ import division
import torch
import copy
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import random
import sys
#import radam
import torch.nn.functional as F
import torchvision
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.autograd import Variable
from scipy.spatial import ConvexHull
from numpy.linalg import matrix_rank
from scipy.special import comb
from torch import nn
from sympy.geometry import *
from torchvision.transforms import ToTensor, Compose, Normalize, Resize,RandomVerticalFlip,ToPILImage
from torch.utils.data import DataLoader
from torchvision.models import vgg16,alexnet
from torchvision import datasets, models, transforms
from time import time
from torch import optim
from torch.optim import lr_scheduler
from collections import OrderedDict
from sklearn.datasets import make_classification,make_blobs,make_gaussian_quantiles

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

    
