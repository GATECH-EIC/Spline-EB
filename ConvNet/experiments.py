import torch
import time
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models.preresnet import resnet
from models.alexnet import AlexNet
from models.simpleCNN import Net
from models.vgg import vgg
from models.Experiment_mobilenetv2 import MobileNetV2
from models.data_utils import CIFAR10, MNIST, CIFAR100
from models.resnet_imagenet import resnet18
from vis_util import *
from EB_utils import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from copy import deepcopy

PRERESNET = "preresnet"
PRERESNET101 = "preresnet101"
ALEXNET = "alexnet"
SIMPLECNN = "simplecnn"
VGG = "vgg"
RESNET18 = "resnet18"
MBV2 = "mbv2"
NETWORK_CHOICES = [PRERESNET, PRERESNET101, ALEXNET, SIMPLECNN, VGG, RESNET18, MBV2]
DATASET_CHOICES = [CIFAR10, MNIST, CIFAR100]

# Experiment settings
batchSize = 128
cuda = torch.cuda.is_available()
lr = 0.1
momentum = 0.9
weight_decay = 1e-4
num_epochs = 30
milestones = [80,120]
gamma = 0.1
sparsity_reg = 0.0001
prevs = []


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getOptimizer(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    return optimizer, lr_policy

def get_model(modelName, dataset, cfg=None):
    channel, size, classes = get_dataset_setting(dataset)
    if modelName == PRERESNET:
        model = resnet(depth=20,dataset=dataset,cfg=cfg)
    elif modelName == ALEXNET:
        model = AlexNet(datasetName=dataset)
    elif modelName == VGG:
        model = vgg(dataset=dataset,depth=16, cfg=cfg)
    elif modelName == PRERESNET101:
        model = resnet(depth=101,dataset=dataset,cfg=cfg)
    elif modelName == RESNET18:
        model = resnet18(dataset,cfg=cfg)
    elif modelName == MBV2:
        model = MobileNetV2(cfg=cfg)
    else:
        model = Net(dataset,cfg)
    return model

def set_global_hooks(model,store):
    hooks = []
    fc_hook = lambda model, input, output: store['fc'].append(output.detach().cpu())
    layers = []
    i = 0
    for name,m in model.named_modules():

        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            layers.append(name)
            def bn_hook(model, input, output):
                nonlocal i
                store[layers[i]].append(output.detach().cpu())
                i += 1

            hooks.append(m.register_forward_hook(bn_hook))
    return hooks

# num_layers: number of bn to visualize in each block
# store: defaultdict to store everything
def set_hooks(model, modelName, num_layers, store):
    hooks = []
    fc_hook = lambda model, input, output: store['fc'].append(output.detach().cpu())

    # hooks.append(model.classifier.register_forward_hook(fc_hook))
    if modelName == PRERESNET or modelName == PRERESNET101:
        i = 0
        layer_num = 0

        for layer in [model.layer1, model.layer2, model.layer3]:
            bns = []
            for block in layer.children():
                for child in block.children():
                    if isinstance(child, nn.BatchNorm1d) or isinstance(child, nn.BatchNorm2d):
                        bns.append(child)

            def bn_hook(model, input, output):
                nonlocal layer_num, i
                store[str(layer_num) + '_bn' + str(i)].append(output.detach().cpu())
                print(i)
                i += 1
                if i == num_layers:
                    i = 0
                    layer_num += 1

            for ind in range(num_layers):
                child = bns[ind]
                hooks.append(child.register_forward_hook(bn_hook))

    elif modelName == SIMPLECNN or modelName == ALEXNET or modelName == VGG:
        i = 0
        start = 0
        for m in model.feature.children():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                def bn_hook(model, input, output):
                    nonlocal i
                    store['bn' + str(i)].append(output.detach().cpu())
                    i += 1
                start += 1
                hooks.append(m.register_forward_hook(bn_hook))
                if start == num_layers:
                    break                

    return hooks

def test(model, testset, epoch):
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=True)
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
    acc = 100.0 * correct / total
    print('\nEpoch {}, Test set: Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, correct, total, acc))
    return acc

"""
Detect eb for the current training epoch. 
"""
def detect_eb(epoch, model, canvas, thres=0.16, lookback=3):  
        global prevs
        flag = False
        i_storage = defaultdict(list)
        hooks_i = set_global_hooks(model, i_storage)
        model.eval()
        with torch.no_grad():
            model(canvas)
        for h1 in hooks_i:
            h1.remove()
        if epoch >= lookback:
            maxi = 0
            for j_storage in prevs:
                total = 0
                for key in i_storage:
                    acti_i = i_storage[key][0]
                    acti_j = j_storage[key][0]
                    diff = getDiff(acti_i, acti_j)
                    total += diff
                total /= (len(i_storage) * 1.0)
                maxi = max(maxi, total)
            if maxi < thres:
                print(epoch, " Find EB !!!!")
                flag = True
            prevs = prevs[1:]
        prevs.append(deepcopy(i_storage))
        return flag

"""
Train the model on dataset with EB detection.
The EB ticket detected will be saved to EB_epoch.pth.tar.
"""
def EB_experiment(modelName, dataset, num_epochs, numLayerToVis=1, toVis=False):
    save_path = './%s-%s/' % (modelName, dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = get_model(modelName, dataset).to(device)
    trainset, testset = get_data(dataset)
    optimizer, lr_policy = getOptimizer(model)
    canvas = getCanvas(modelName, dataset, testset, N=100)

    # train the model and visualzie
    accuracies = []
    act_storage = defaultdict(list)
    diff_storage = defaultdict(list)
    trainLoader = torch.utils.data.DataLoader(trainset, batchSize, shuffle=True)
    flag = False
    for i in tqdm(range(num_epochs), desc="training"):
        model.train()
        for batch_idx, (data, target) in enumerate(trainLoader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        lr_policy.step()
        acc = test(model, testset, i)
        if flag == False:
            flag = detect_eb(i, model, canvas)
            if flag == True:
                save_model(model, modelName, dataset, acc, '%smodels/EB_%d.pth.tar' % (save_path, i))

        if toVis:
            # obtain activations
            if (i < 5 or i % 5 == 0):
                hooks = set_hooks(model, modelName, numLayerToVis, act_storage)
                model.eval()
                with torch.no_grad():
                    model(canvas)
                for h in hooks:
                    h.remove()
                if i > 0:
                    for key, acts in act_storage.items():
                        prevacti = acts[-2]
                        acti = acts[-1]
                        diff = getDiff(acti, prevacti)
                        diff_storage[key].append(diff)
                        act_storage[key] = acts[1:]
                    accuracies.append(acc)
        save_model(model, modelName, dataset, acc, '%smodels/%d.pth.tar' % (save_path, i))
    if toVis:
        # Save and load the storage dict
        save_pickle(diff_storage, base_path, "storage.pickle")
        save_pickle(accuracies, base_path, 'accuracies.pickle')

        storage = load_pickle(base_path + "storage.pickle")
        accuracies = load_pickle(base_path + "accuracies.pickle")
        visualizeStorage(storage,accuracies,path=save_path)

"""
Draw the epoch-to-epoch distance comparison during a model's training process.
"""
def EB_experiment2d(modelName, dataset, numLayerToVis=1, model_load_path="", load_path=""):
    if load_path == "":
        base_path = './%s-%s/' % (modelName, dataset)
        if model_load_path == "":
            model_load_path = './%s-%s/models/' % (modelName, dataset)
        if not os.path.exists(model_load_path):
            os.makedirs(model_load_path)

        mat_store = {}
        # models = []
        trainset, testset = get_data(dataset)
        canvas = getCanvas(modelName, dataset, testset, N=100)

        for i in tqdm(range(num_epochs), desc="training"):
            model_i = get_model(modelName, dataset).to(device)
            load_model(model_i, model_load_path + '%d.pth.tar' % (i))
            i_storage = defaultdict(list)
            hooks_i = set_hooks(model_i, modelName, numLayerToVis, i_storage)
            model_i.eval()
            with torch.no_grad():
                model_i(canvas)
            for h1 in hooks_i:
                h1.remove()

            if len(mat_store) == 0:
                for key in i_storage:
                    mat_store[key] = np.zeros((num_epochs,num_epochs))

            for j in tqdm(range(i, num_epochs), desc='comparing'):
                model_j = get_model(modelName, dataset).to(device)
                load_model(model_j, model_load_path + '%d.pth.tar' % (j))

                j_storage = defaultdict(list)
                hooks_j = set_hooks(model_j, modelName, numLayerToVis, j_storage)

                model_j.eval()
                with torch.no_grad():
                    model_j(canvas)
                for h2 in hooks_j:
                    h2.remove()

                for key in i_storage:
                    mat = mat_store[key]
                    acti_i = i_storage[key][0]
                    acti_j = j_storage[key][0]
                    diff = getDiff(acti_i, acti_j)
                    mat[i, j] = diff
                    mat[j, i] = diff
            save_pickle(i_storage,base_path,"distance.pickle")


        save_pickle(mat_store, model_load_path,"distance_mat")
    else:
        mat_store = load_pickle(load_path)
    global_mat = np.zeros((num_epochs,num_epochs))
    for key,mat in mat_store.items():
        # mat = mat_store
        mat = np.array([np.array(l) for l in mat])
        global_mat += mat
    min_m = np.min(global_mat)
    max_m = np.max(global_mat)
    global_mat = (global_mat - min_m) / (max_m - min_m)

    print(global_mat)

    fig = plt.figure(figsize=(70, 32))

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks([])
    plt.yticks([])

    ax1 = fig.add_subplot(1, 1, 1)
    overlap = ax1.imshow(global_mat, cmap=plt.cm.viridis.reversed(), interpolation='none', vmin=0, vmax=1)
    cb = plt.colorbar(overlap, fraction=0.046, pad=0.09)

    ax1.set_title('%s-%s' %(modelName, dataset), fontsize=60, fontweight='bold', y=1.04)
    ax1.set_yticks([25, 50, 75, 100, 125, 150])

    ax1.set_xticks([])

    # plt.savefig(key+'_overlap.jpg', bbox_inches='tight')
    plt.savefig('%s-%s_eb_2d.jpg'%(modelName, dataset), bbox_inches='tight')


def get_channel_scores(loaded_model, modelName):

    sorted_channels = []
    channel_scores = []

    if modelName == SIMPLECNN or modelName == ALEXNET:
        for m in loaded_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                w = m.weight.data.abs().clone()
                y, i = torch.sort(w)
                sorted_channels.append(i)
                channel_scores.append(y)
    elif modelName == PRERESNET:
        layers = [loaded_model.layer1, loaded_model.layer2, loaded_model.layer3]
        for layer in layers:
            last_block = list(layer.children())[-1]
            for m in last_block.modules():
                if isinstance(m, nn.BatchNorm2d):
                    w = m.weight.data.abs().clone()
                    y, i = torch.sort(w)
                    # print("y,i", y,i)
                    sorted_channels.append(i)
                    channel_scores.append(w)
                    break
    return sorted_channels, channel_scores

# Specify which batchNorm to look at
def  channel_diff(model, modelName, trainset, sorted_channels, channel_scores, path, filter_to_see=2):
    acti_conv = []

    hooks = []

    # Set up the hooks
    if modelName == SIMPLECNN or modelName == ALEXNET:
        for m in model.feature.children():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                hooks.append(m.register_forward_hook(lambda model, input, output : acti_conv.append(output.detach())))
    elif modelName == PRERESNET:
        layers = [model.layer1, model.layer2, model.layer3]
        for layer in layers:
            last_block = list(layer.children())[-1]
            for m in last_block.modules():
                if isinstance(m, nn.BatchNorm2d):
                    hooks.append(m.register_forward_hook(lambda model, input, output : acti_conv.append(output.detach())))
                    break

    testloader = torch.utils.data.DataLoader(trainset, batch_size=10000, shuffle=False)

    # For two classes, for all channels belonging to a specific filter
    stats = [[0 for c in range(len(sorted_channels[filter_to_see]))] for i in range(2)]
    diff = np.zeros((len(sorted_channels[filter_to_see])))
    model.eval()
    with torch.no_grad():
        counter = 0
        for data in testloader:
            counter += 1
            images, labels = data[0].to(device), data[1].to(device)

            for l0 in range(10):
                for l1 in range(l0+1,10):
                    zeros_ind = labels == l0
                    ones_ind = labels == l1
                    zero_images, zero_labels = images[zeros_ind,:,:,:], labels[zeros_ind]
                    ones_images, ones_labels = images[ones_ind,:,:,:], labels[ones_ind]
                    class_images = [zero_images, ones_images]

                    model.eval()

                    for cl in range(2):
                        outputs = model(class_images[cl])
                        channels = sorted_channels[filter_to_see]
                        data_act = acti_conv[filter_to_see]
                        for i,c in enumerate(channels):
                            mean = torch.mean(data_act[:,c,:,:])
                            stats[cl][i]  += mean.item()
                        acti_conv = []
                    st0 = np.array(stats[0])
                    st1 = np.array(stats[1])
                    diff += np.abs(st0 - st1)
            break

        plt.figure(figsize=(6, 4))
        avg_diff = diff
        x = [i for i in range(len(avg_diff))]

        fig, ax = plt.subplots()
        ax.set_xlabel('importance ranking(higher is more important)')
        ax.set_ylabel('differentiation score')
        ax.set_title('%s' % (modelName))
        ax.scatter(x,avg_diff)
        z = np.polyfit(x, avg_diff, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--")
        plt.savefig(path + "trend_line1.png")
        plt.close("all")
        x = channel_scores[filter_to_see].cpu().numpy()
        fig, ax = plt.subplots()
        ax.set_xlabel('BN Scaling Factor')
        ax.set_ylabel('differentiation score')
        ax.set_title('%s' % (modelName))
        ax.scatter(x,avg_diff)
        z = np.polyfit(x, avg_diff, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--")
        plt.savefig(path + "trend_line2.png")
        plt.close()

def difference_channel_experiment(modelName, dataset, filter_to_see):
    base_path = './%s-%s/' % (modelName, dataset)

    model = get_model(modelName, dataset).to(device)
    trainset, testset = get_data(dataset)

    loaded_model, _, datasetName, cfg = load_model(model, base_path +'models/159.pth.tar')

    sorted_channels, channel_scores = get_channel_scores(loaded_model,modelName)
    channel_diff(model, modelName, trainset, sorted_channels, channel_scores, base_path, filter_to_see=filter_to_see)



# Define what experiment you want to run here
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Trainning')
    parser.add_argument('--network', choices=NETWORK_CHOICES, default=SIMPLECNN)
    parser.add_argument('--dataset', choices=DATASET_CHOICES, default=CIFAR10)
    parser.add_argument('--epochs', type=int, default=160)
    args = parser.parse_args()
    
    EB_experiment(args.network, args.dataset, args.epochs)