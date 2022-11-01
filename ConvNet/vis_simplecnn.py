from models.data_utils import *
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from models.simpleCNN import *
from vis_util import *
import numpy as np
import torch.nn as nn
import torch

def visualize(numLayerToVis=1):
    num_epochs = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = 'cifar10'
    for i in tqdm(range(num_epochs), desc="training"):
          
          model = Net(dataset).to(device)
          model, modelName, dataset, cfg, acc = load_model(model, './%s-%s/models/%d.pth.tar' % ('simplecnn', dataset, i))
          
          base_path = './%s-%s/' % (modelName, dataset)
      
          trainset, testset = get_data(dataset)
      
          N = 10
          canvas = getCanvas(modelName, dataset, testset, N)
          grid = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
          act_storage = defaultdict(list)          
          hooks = set_hooks(model, modelName, numLayerToVis, act_storage)
          model.eval()
          with torch.no_grad():
              model(canvas)
          for h in hooks:
              h.remove()
          for key, acts in act_storage.items():                
              drawConv(acts[0].cpu(), grid, N, key, i, acc, take=(1,1,1), path=base_path)

if __name__ == '__main__':
    visualize()