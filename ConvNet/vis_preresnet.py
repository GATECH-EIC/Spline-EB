from models.data_utils import *
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from models import preresnet
from vis_util import *
from experiments import *

def drawConv(activ, grid, N, title, iteration, acc, take=None, impose=False, color="b", datasetName='Mnist', modelName='simpleCNN', path=None, samples=None):
    num_samples, num_channels, h, w = activ.shape
    if not impose:
        plt.figure(figsize=(6, 4))

    if path is None:
        path = './%s-%s/' % (modelName, datasetName)
        if not os.path.exists(path + 'conv/'):
            os.makedirs(path + 'conv/')
    else:
        if not os.path.exists(path):
            os.makedirs(path)

    if take is None:
        take = (h,w)
    if samples == None:
        samples = [i for i in range(num_channels)]

    # visualize "take" number of rows and cols in a channel
    for k in tqdm(samples, desc='channel'):
        for i in range(min(take[0], h)):
            for j in range(min(take[1], w)):
                color_to_use = color
                plt.contour(grid[0], grid[1], activ[:, k, i, j].reshape((N,N)), levels=[0],colors=color_to_use)

    if not impose:
        plt.xticks([])
        plt.yticks([])
        plt.title(title + 'Iter = {0}, Acc = {1}%'.format(iteration, acc))

        name = path + title + "_" + str(iteration) + '.png'
        if take is not None:
            name = path + title + "_" + str(iteration) + '_%s.png' % str(take)
        plt.savefig(name)
        plt.close("all")

def test(model, testset, epoch):
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            model.eval()
            outputs = model(images)
            _, predicted = torch.max(outputs.data.to('cpu'), 1)
            total += labels.to('cpu').size(0)
            correct += (predicted == labels.to('cpu')).sum().item()
    acc = 100 * correct / total
    print('Accuracy at %d: %d %%' % (epoch, acc))
    return acc

if __name__ =='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoints = [0,3,4]
    datasetName = 'cifar10'
    modelName = 'preresnet'

    try:
        trainset, testset = get_data(datasetName)
    except Exception as err:
        print(err)
    else:
        x = getCanvas(modelName, datasetName,testset,N=10,toSave=True)
        for i in tqdm(checkpoints,desc='figures'):

            checkpoint = torch.load('./preresnet-cifar10/models/%d.pth.tar' % (i))
            model = get_model(modelName, datasetName).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            with torch.no_grad():
                acc = test(model,testset,i)

            visualizeBatch(model.to(device), i, [1,2], acc,datasetName,modelName,x,N=10)