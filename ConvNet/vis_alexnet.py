from models.data_utils import *
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from models.alexnet import *
import numpy as np
import torch.nn as nn
import torch

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

def visualizeBatch(model, iteration, num_class, testset, acc, datasetName, modelName, path=None):
    color = ['deepskyblue', 'dodgerblue', 'cornflowerblue', 'blue', 'navy', 'darkslateblue','black',]
    activations_fc = []
    acti_conv = []
    base_path = './%s-%s/' % (modelName, datasetName)
    if path is not None:
        base_path += path
    if not os.path.exists(base_path):
        os.makedirs(base_path)


    def get_activation_fc():
        def hook(model, input, output):
            activations_fc.append(output.detach())
        return hook
    hooks = []
    hooks.append(model.classifier.register_forward_hook(get_activation_fc()))
    for m in model.feature.children():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_hook(lambda model, input, output : acti_conv.append(output.detach())))
    N = 100
    grid = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    alpha = torch.from_numpy(grid[0].reshape(-1)[:,None,None, None]).float().to(device)
    beta = torch.from_numpy(grid[1].reshape(-1)[:,None,None, None]).float().to(device)
    data = testset[0]
    image0, label0 = data[0].to(device), data[1]
    i = 1
    while True:
        if testset[i][1] != label0:
            break
        i += 1
    image1, label1 = testset[i][0].to(device), testset[i][1]
    x = alpha * image0 + beta * image1
    to_save = torch.squeeze(x).cpu().numpy()
    to_save = np.concatenate([np.concatenate(to_save[i*N:i*N+N,:,:], axis=1) for i in range(N)][::-1])

    if iteration == 0 and datasetName == 'Mnist':
        plt.imsave(base_path+str(iteration)+"_gt.png", to_save)

    model.eval()
    with torch.no_grad():
        outputs = model(x)

    #conv
    samples = [7, 91, 24, 4, 10, 82, 27, 30, 35, 11]
    plt.figure(figsize=(6, 4))
    for i, acti in enumerate(acti_conv):
        drawConv(acti.cpu(), grid, N, "conv_%d" % (i), iteration, acc, color=color[i],
                 take=(1, 1),samples=samples, datasetName='Cifar10', modelName='alexnet',
                 path='./%s-%s/conv-%d/' % (modelName, datasetName,i))

    # fc layer
    activ3 = activations_fc[0]
    for i in range(1,len(activations_fc)):
        activ = torch.cat([activ,activations_fc[i]], dim=0)
    activ3 = activ3.cpu()
    plt.figure(figsize=(6, 4))


    for i in [label0, label1]:
        plt.contour(grid[0], grid[1], activ3[:, i].reshape((N, N)), levels=[0], colors='r')
    plt.xticks([])
    plt.yticks([])
    plt.title('Iter = {0}, Acc = {1}%%'.format(iteration, acc))
    if not os.path.exists(base_path + "fc/"):
        os.makedirs(base_path + "fc/")
    plt.savefig(base_path + "fc/" + str(iteration) + '_final.png')
    plt.close()

    del activations_fc, acti_conv
    for i in hooks:
        i.remove()

if __name__ =='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 30
    checkpoints = [1, 2, 3, 4, 5, 6] + [i for i in range(10, epochs, 10)] + [epochs - 1]
    datasetName = 'cifar10'
    try:
        trainset, testset = get_data(datasetName)
    except Exception as err:
        print(err)
    else:
        for i in tqdm(checkpoints, desc='checkpoint'):
            model = AlexNet(datasetName).to(device)
            loaded_model, modelName, datasetName, cfg, acc = load_model(model, './alexnet-cifar10/models/%s.pth.tar' % (i))
            visualizeBatch(loaded_model, i, 10, testset, acc, datasetName, modelName)