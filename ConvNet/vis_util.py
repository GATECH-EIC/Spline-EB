import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from  models.data_utils import *
from experiments import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getFinal(activ):
    return np.argmax(nn.Softmax(dim=1)(activ), axis=1).reshape(-1,1)

def drawConv(activ, grid, N, title, iteration, acc, take=None, impose=False, color="b", datasetName='Mnist', modelName='simpleCNN', path=None):
    rands = [398, 238, 32, 335, 435, 112, 78, 429, 140, 120, 249, 403, 72, 234, 316, 175, 386, 286, 276, 129, 420, 504, 392, 139, 242, 499, 212, 171, 196, 122, 369, 227, 121, 13, 117, 124, 452, 36, 304, 131, 163, 262, 343, 455, 329, 93, 373, 183, 347, 101, 127, 404, 25, 364, 170, 208, 8, 490, 103, 198, 284, 145, 338, 458, 47, 462, 104, 178, 406, 30, 407, 114, 409, 293, 477, 16, 205, 248, 305, 29, 176, 501, 69, 395, 461, 289, 508, 10, 443, 80, 215, 45, 200, 370, 213, 19, 211, 486, 428, 340]

    num_samples, num_channels, h, w = activ.shape
    if not impose:
        plt.figure(figsize=(6, 4))

    if path is None:
        path = './%s-%s/' % (modelName, datasetName)
    if not os.path.exists(path + 'conv/'):
        os.makedirs(path + 'conv/')

    if take is None:
        take = (num_channels, h,w)

    # visualize "take" number of rows and cols in a channel
    for k in tqdm(range(min(num_channels,take[0])), desc='channel'):
        for i in range(min(take[1], h)):
            for j in range(min(take[2], w)):
                color_to_use = color
                plt.contour(grid[0], grid[1], activ[:, k, i, j].reshape((N,N)), levels=[0],colors=color_to_use)

    if not impose:
        plt.xticks([])
        plt.yticks([])
        plt.title(title + 'Iter = {0}, Acc = {1}%'.format(iteration, acc))

        name = path + 'conv/' + title + "_" + str(iteration) + '.png'
        if take is not None:
            name = path + 'conv/' + title + "_" + str(iteration) + '_%s.png' % str(take)
        plt.savefig(name)
        plt.close("all")

def visualizeBatch(model, iteration, num_class, acc, datasetName, modelName, vis_image, N=100, path=None):
    color = ['grey', 'g', 'r', 'c', 'y', 'm', 'w', 'k', 'violet', 'pink']
    grid = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    activations_fc = []
    acti_conv = []
    base_path = './%s-%s/' % (modelName, datasetName)
    if path is not None:
        base_path += path
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    take = (10,1,1)

    def get_activation_fc():
        def hook(model, input, output):
            activations_fc.append(output.detach().cpu())
        return hook
    hooks = []
    hooks.append(model.classifier.register_forward_hook(get_activation_fc()))
    store = defaultdict(list)
    hooks = set_hooks(model, modelName,1,store)

    model.eval()
    with torch.no_grad():
        outputs = model(vis_image)

    for key,acti in store.items():
        print(key)
        if 'bn' in key:
            acti_conv.append((key,acti[0]))
    acti_conv = [acti[1] for acti in sorted(acti_conv, key=lambda x:x[0])]

    #conv
    for i in range(len(acti_conv)):
        data_act = acti_conv[i]
        drawConv(acti_conv[i].cpu(), grid, N, "conv_%d"%i, iteration, acc, take=take, path=base_path, color=color[i])

    # fc layer
    activ3 = activations_fc[0]
    for i in range(1,len(activations_fc)):
        activ3 = torch.cat([activ3,activations_fc[i]], dim=0)
    activ3 = activ3.cpu()
    plt.figure(figsize=(6, 4))

    # 0 visualization
    for i in num_class:
        plt.contour(grid[0], grid[1], activ3[:, i].reshape((N, N)), levels=[0], colors='r')
    plt.xticks([])
    plt.yticks([])
    plt.title('Iter = {0}, Acc = {1}%%'.format(iteration, acc))
    if not os.path.exists(base_path + "fc/"):
        os.makedirs(base_path + "fc/")
    plt.savefig(base_path + "fc/" + str(iteration) + '_final.png')
    plt.close()

    # combine visualization
    plt.figure(figsize=(6, 4))
    drawConv(acti_conv[0].cpu(), grid, N, "conv_6", iteration, acc, take=take, impose=True, color="grey")
    drawConv(acti_conv[1].cpu(), grid, N, "conv_7", iteration, acc, take=take, impose=True, color="g")
    for i in num_class:
        plt.contour(grid[0], grid[1], activ3[:, i].reshape((N, N)), levels=[0], colors='r')
    # plt.xticks(np.linspace(0, 1, N))
    # plt.yticks(np.linspace(0, 1, N))
    plt.xticks([])
    plt.yticks([])
    plt.title('Iter = {0}, Acc = {1}%'.format(iteration, acc))
    if not os.path.exists(base_path + "com/"):
        os.makedirs(base_path + "com/")
    plt.savefig(base_path + "com/" + str(iteration) + '_combine.png')
    plt.close("all")

    del activations_fc, acti_conv
    for i in hooks:
        i.remove()

def getDiff(acti, prevActi):
    acti1 = (acti > 0).type(torch.FloatTensor)
    acti2 = (prevActi > 0).type(torch.FloatTensor)
    diff = torch.abs(acti1 - acti2)
    return torch.sum(diff) / torch.numel(diff)

def visualizeStorage(storage, accuracies, std_storage, std_accuracies, path):
    fig, ax1 = plt.subplots(figsize=(10,6))
    color = 'tab:blue'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('difference', color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('accuracy', color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    x = [i for i in range(5,len(accuracies)+5)]
    # count = 10
    # for i in range(len(x)):
    #     if i >=5:
    #         x[i] = count
    #         count += 5

    for name, differences in storage.items():
        ax1.plot(x, differences, '-', label=name)
        ax1.fill_between(x,differences-std_storage[name],differences+std_storage[name], alpha=0.2)
    # print(x, accuracies)
    ax2.plot(x, accuracies, "--", color="r", label="accuracy")
    ax2.fill_between(x, accuracies-std_accuracies, accuracies+std_accuracies, color='r', alpha=0.2)
    ax1.legend()
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + "EB_Plot_avg.png")
    plt.close("all")

def drawAverageVisualizeStore(storages,accuracies,path):
    act_arr = defaultdict(list)

    for key in storages[0]:
        for s in storages:
            act_arr[key].append(s[key])
    std_act_arr = {}
    for key in act_arr:
        act_arr[key] = np.array(act_arr[key])
        print(act_arr[key].shape)
        std_act_arr[key] = np.std(act_arr[key], axis=0)
        print(std_act_arr[key].shape)
        act_arr[key] = np.mean(act_arr[key], axis=0)

    acc_arr = np.mean(np.array(accuracies),axis=0)
    std_acc_arr = np.std(np.array(accuracies),axis=0)
    print(len(acc_arr))
    visualizeStorage(act_arr, acc_arr, std_act_arr, std_acc_arr, path)


def getCanvas(modelName, datasetName, testset, N=100, toSave=True):
    grid = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    alpha = torch.from_numpy(grid[0].reshape(-1)[:, None, None, None]).float().to(device)
    beta = torch.from_numpy(grid[1].reshape(-1)[:, None, None, None]).float().to(device)
    data = testset[4690]
    image0, label0 = data[0].to(device), data[1]
    i = 1    
    while True:
        if testset[i][1] != label0 and testset[i][1] == 1:
            break
        i += 1
    image1, label1 = testset[i][0].to(device), testset[i][1]
    print("Two chosen label classes: ", label0, label1)
    vis_image = alpha * image0 + beta * image1
    if toSave:
        to_save = vis_image.cpu().numpy()
        to_save = np.concatenate([np.concatenate(to_save[i * N:i * N + N, :, :, :], axis=2) for i in range(N)][::-1],
                                 axis=1)
        to_save = np.moveaxis(to_save, 0, -1)
        print("Canvas shape: ", to_save.shape)
        to_save = (to_save - np.min(to_save)) / (np.max(to_save) - np.min(to_save))

        # Save the reference image
        path = './%s-%s/' % (modelName, datasetName)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.imsave(path + "1_gt.png", to_save.squeeze())
    return vis_image

if __name__ == '__main__':
    base_path = "alexnet-cifar10/"
    storage_names = ["storage.pickle","storage1.pickle","storage2.pickle"]
    accuracy_names = ["accuracies.pickle","accuracies1.pickle","accuracies2.pickle"]
    storage_list = [load_pickle(base_path + name) for name in storage_names]
    accuracy_list = [load_pickle(base_path + name) for name in accuracy_names]
    drawAverageVisualizeStore(storage_list,accuracy_list,base_path)






