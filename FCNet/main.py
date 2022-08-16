from Libraries import *
from Geometry_Plottings import *
from Geometry_utilities import *
from Lottery_Utilities import *
# device = 'cpu'
# if torch.cuda.is_available():
#     device = 'cuda'
import math, os, shutil, time
import numpy as np
# %matplotlib inline

def Data_Generation (Dataset_Size,Condition1,Condition2,isCircle,centers = {'c1' : (0, 0), 'c2': (0, 0)} ):
    if(not isCircle):
        fig, ax = plt.subplots(1, 1)
        x = np.random.rand(Dataset_Size,1)*10-5
        y = np.random.rand(Dataset_Size,1)*10-5
        nx1 = []
        ny1 = []
        nx2 = []
        ny2 = []
        count = 0
        Evaluation1 = eval(Condition1)
        Evaluation2 = eval(Condition2)
        nx1.append(x[np.argwhere(Evaluation1==True)[:,0]])
        ny1.append(y[np.argwhere(Evaluation1==True)[:,0]])
        plt.scatter(nx1,ny1,color = 'red')
        nx2.append(x[np.argwhere(Evaluation2==True)[:,0]])
        ny2.append(y[np.argwhere(Evaluation2==True)[:,0]])
        plt.scatter(nx2,ny2,color = 'blue')
        nx1=np.array(nx1)
        ny1=np.array(ny1)
        nx2=np.array(nx2)
        ny2=np.array(ny2)
        loc =np.argwhere(Evaluation1==True)[:,0]
        nz= np.zeros([nx1.shape[1]+nx2.shape[1],1])
        nz[0:nx1.shape[1]]=1
        plt.grid()
        nx = np.concatenate((nx1.squeeze(),nx2.squeeze()))
        ny = np.concatenate((ny1.squeeze(),ny2.squeeze()))
        zt = np.array(nz)
        xt = np.concatenate((nx.reshape(-1,1),ny.reshape(-1,1)),1)
        x = Variable(torch.from_numpy(xt).type(torch.FloatTensor),requires_grad=False).to(device)
        y = Variable(torch.from_numpy(zt).type(torch.LongTensor),requires_grad=False).view(-1,1).to(device)

        ax.spines['bottom'].set_linewidth(2)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(2)
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_linewidth(2)
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_linewidth(2)
        ax.spines['right'].set_color('black')

        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)

        plt.tight_layout()
        plt.savefig('gen_data.svg')
        return x,y
    else:
            # size of plot
        plt.figure(figsize=(7, 6))
        num_samples = Dataset_Size // len(centers)            #Integer Division To get Equal Number of samples for all classes
        data = np.zeros((Dataset_Size, 2))                    # Generate a matrix of size    Size of Data Setx2
        labels = np.zeros(Dataset_Size)                       # Generate labels of size      Size of Data Set
        label_count = 0                                          # We start by labeling the class 0 entries
        for _, c in centers.items():                             # We loop over all the dictionary values ignoring the label of the centers by using '_'
            # extract the center of the circle
            c_x = c[0]                                           # X value of Center
            c_y = c[1]                                           # Y value of Center
            # generate the angles
            theta = np.linspace(0, 2 * np.pi, num_samples)       # Generate linearly spaced values from 0 to2pi
            r = np.random.rand(num_samples)                      # Generate a set of uniform distributed values
            # generate the points
            x, y = r * np.cos(theta) + c_x, r * np.sin(theta) + c_y
         # Add points to data and assign labels
            data[label_count*num_samples: (label_count+1)*num_samples, 0] = x                 # Setting Rows     0 to num_samples  in the first column in data  to the calculated x value
            data[label_count * num_samples: (label_count + 1) * num_samples, 1] = y           # Setting Rows     num_samples to 2*numsamples  in the second column in data  to the calculated y value
            labels[label_count * num_samples: (label_count + 1) * num_samples] = label_count  # Setting the labels of the data
            # plots
            plt.scatter(x, y)
            # increase label count
            label_count += 1
        plt.grid()
        plt.legend(['class {}'.format(i) for i in np.arange(len(centers) + 1)])
        plt.show(block=True)
        plt.savefig('gen_data.svg')
        data = Variable(torch.from_numpy(data).type(torch.FloatTensor),requires_grad=False).to(device)
        labels = Variable(torch.from_numpy(labels).type(torch.LongTensor),requires_grad=False).view(-1,1).to(device)
        return data, labels
########################################################################################################################################
def dataloader(x,y):
    x=x.float()
    y=y.float()
    data_train=torch.cat((x,y.reshape(-1,1)),1)
    #data_train_loader = DataLoader(data_train, batch_size=100, shuffle=True)####

    train_size = int(0.8 * len(data_train))
    test_size = len(data_train) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data_train, [train_size, test_size])

    train_dataset=data_train[train_dataset.indices,:]
    test_dataset= data_train[test_dataset.indices,:]
    trainx= train_dataset [:,0:train_dataset.shape[1]-1]
    trainy=train_dataset [:,train_dataset.shape[1]-1]
    testx=test_dataset [:,0:test_dataset.shape[1]-1]
    testy=test_dataset [:,test_dataset.shape[1]-1]

    return trainx, trainy.reshape(-1,1).long(), testx,testy.reshape(-1,1).long()


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
###########################################################Networks definitions##############################################################
class Net(nn.Module): #Single hidden layer NN
    def __init__(self,net_size,input_size,output_size,Bias):
        super(Net, self).__init__()
        self.net_size = net_size
        self.fc1 = nn.Linear(input_size, net_size,bias = Bias)
        self.fc2 = nn.Linear(net_size,output_size,bias = Bias)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc2(self.relu(self.fc1(x)))
        return x

def create_path(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

def train(model,criterion, x, y, alpha ,epochs,batchsize, num_class, init='', thre=''):
    save_p = './net_size' + str(model.net_size) + '/' + init + str(thre)
    create_path(save_p + '/spline_fc1')
    create_path(save_p + '/spline_fc2')
    create_path(save_p + '/spline_fc1_fc2')
    costs = []
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha)
    trainx, trainy, testx,testy= dataloader(x,y)
    x=trainx.float()
    y=trainy.float()
    data_train=torch.cat((x,y.reshape(-1,1)),1)
    data_train_loader = DataLoader(data_train, batch_size=batchsize, shuffle=True)
    model.train()
    j = 0
    for i in range(epochs):
        for index,samples in enumerate(data_train_loader):
            j += 1
            x1=samples[:,0:2]
            y1=samples[:,2].long().reshape(-1,1)
            if (j%50 == 0):
                model.eval()
#                 print(testx.shape)
                acc = accuracy(model,testx,testy)
                print(f'Test accuracy is #{acc:.2f} , Iter = {j}')
                ### add by haoran ###
                visualizeBatch(model, j, init, thre, num_class, mode='both')
                #####################################
                model.train()
            cost = criterion(model(x1), y1.squeeze())
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            costs.append(float(cost))
    return model, acc

def getFinal(activ):
    # print(activ.shape)
    return activ
#     return np.argmax(nn.Softmax(dim=1)(activ), axis=1).reshape(-1,1)

def visualizeBatch(model, iteration, init, thre, num_class, mode='both'):
    activations_fc1 = []
    activations_fc2 = []
    def get_activation_fc1():
        def hook(model, input, output):
            activations_fc1.append(output.detach())
        return hook
    def get_activation_fc2():
        def hook(model, input, output):
            activations_fc2.append(output.detach())
        return hook
    model.fc1.register_forward_hook(get_activation_fc1())
    model.fc2.register_forward_hook(get_activation_fc2())

    N = 20; L = 2
    grid = np.meshgrid(np.linspace(-L, L, N), np.linspace(- L, L, N))
    dummy_input = torch.Tensor(np.c_[grid[0].ravel(), grid[1].ravel()])
#     print(dummy_input.shape)
    pred = model(dummy_input.cuda()).detach().cpu().numpy()

    if mode == 'fc1':
        batchActivation = activations_fc1
        firstActiv = batchActivation[0].cpu()
        final = firstActiv
        print(final.shape)

        plt.figure(figsize=(6, 4))
        for k in range(final.shape[1]):
            plt.contour(grid[0], grid[1], final[:, k].reshape((N, N)), levels=[0])
        plt.xticks([])
        plt.yticks([])
#         plt.title('Iter = {}'.format(iteration))
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(2)
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_linewidth(2)
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_linewidth(2)
        ax.spines['right'].set_color('black')
        plt.tight_layout()
        if init != '' or thre != '':
            save_path = './net_size' + str(model.net_size) + '/' + init + str(thre) + '/spline_fc1/'
        else:
            save_path = './net_size' + str(model.net_size) + '/spline_fc1/'
        plt.savefig(save_path + str(iteration) + '.png')
    elif mode == 'fc2':
        batchActivation = activations_fc2
        firstActiv = batchActivation[0].cpu()
        final = getFinal(firstActiv)
#         final = np.argmax(firstActiv,axis=1).reshape(-1,1)

        plt.figure(figsize=(6, 4))
        for k in range(num_classes):
            plt.contour(grid[0], grid[1], final[:, k].reshape((N, N)), levels=[0], colors='b')
        plt.xticks([])
        plt.yticks([])
#         plt.title('Iter = {}'.format(iteration))
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(2)
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_linewidth(2)
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_linewidth(2)
        ax.spines['right'].set_color('black')
        plt.tight_layout()
        if init != '' or thre != '':
            save_path = './net_size' + str(model.net_size) + '/' + init + str(thre) + '/spline_fc2/'
        else:
            save_path = './net_size' + str(model.net_size) + '/spline_fc2/'
        plt.savefig(save_path + str(iteration) + '.png')
        plt.savefig(save_path + str(iteration) + '.svg')
    elif mode == 'both':
        batchActivation = activations_fc1
        firstActiv = batchActivation[0].cpu()
        final_1 = firstActiv

        batchActivation = activations_fc2
        firstActiv = batchActivation[0].cpu()
#         final_2 = np.argmax(firstActiv,axis=1).reshape(-1,1)
        final_2 = getFinal(firstActiv)
        print(final_2.shape)

        plt.figure(figsize=(6, 4))
        for k in range(final_1.shape[1]):
            plt.contour(grid[0], grid[1], final_1[:, k].reshape((N, N)), levels=[0], colors='b')
        plt.xticks([])
        plt.yticks([])
#         plt.title('Iter = {}'.format(iteration))
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(2)
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_linewidth(2)
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_linewidth(2)
        ax.spines['right'].set_color('black')
        plt.tight_layout()

        save_path = './net_size' + str(model.net_size) + '/' + init + str(thre) + '/spline_fc1/'
        plt.savefig(save_path + str(iteration) + '.png')
        plt.savefig(save_path + str(iteration) + '.svg')

        plt.figure(figsize=(6, 4))
        for k in range(1):
            plt.contour(grid[0], grid[1], final_2[:, k].reshape((N, N)), levels=[0], colors='r')
        plt.xticks([])
        plt.yticks([])
#         plt.title('Iter = {}'.format(iteration))
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(2)
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_linewidth(2)
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_linewidth(2)
        ax.spines['right'].set_color('black')
        plt.tight_layout()

        save_path = './net_size' + str(model.net_size) + '/' + init + str(thre) + '/spline_fc2/'
        plt.savefig(save_path + str(iteration) + '.png')
        plt.savefig(save_path + str(iteration) + '.svg')

        plt.figure(figsize=(6, 4))
        for k in range(final_1.shape[1]):
            plt.contour(grid[0], grid[1], final_1[:, k].reshape((N, N)), levels=[0], colors='b', linewidths=5)
#         for k in range(num_classes):
#             plt.contour(grid[0], grid[1], firstActiv[:, k].reshape((N,N)), levels=[0], colors='g')
        for k in range(1):
            plt.contour(grid[0], grid[1], final_2[:, k].reshape((N, N)), levels=[0], colors='r', linewidths=5)

        plt.xticks([])
        plt.yticks([])
#         plt.title('Iter = {}'.format(iteration))
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(5)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(5)
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_linewidth(5)
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_linewidth(5)
        ax.spines['right'].set_color('black')
        plt.tight_layout()

        if init != '' or thre != '':
            save_path = './net_size' + str(model.net_size) + '/' + init + str(thre) + '/spline_fc1_fc2/'
        else:
            save_path = './net_size' + str(model.net_size) + '/spline_fc1_fc2/'
        plt.savefig(save_path + str(iteration) + '.png')
        plt.savefig(save_path + str(iteration) + '.svg')
    elif mode == 'return':
        return activations_fc1, activations_fc2


def accuracy(model,x,y):
    prediction = model(x).detach().cpu().numpy()
    return (np.sum(np.argmax(prediction,axis=1).reshape(-1,1)==np.array(y.cpu().numpy()).reshape(-1,1))/len(y.cpu().numpy()))*100
#########################################################################################
#########################################################################################


def gen_data():
    num_classes =2
    setting = 5

    ''' Run this cell to generate synthetic datasets for experimentation.

        Option 1 : Simple Semi-Random Data
        Option 2 : Simple Blobs
        Option 3 : Gaussian Quantiles (Circles)
        Option 4 : X shaped data biased
        Option 5 : X shaped data non-biased
        Option 6 : Circular Data (2 Blobs)

    '''
    if setting==1 :
        X1, Y1 = make_classification(n_samples=4000,n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1, n_classes=num_classes)
        plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')
        x = torch.from_numpy(X1).to(device,dtype=torch.float)
        y = torch.from_numpy(Y1).to(device,dtype=torch.double)

    if setting==2 :
        X1, Y1 = make_blobs(n_samples=5000,n_features=2, centers = num_classes)
        plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,s=30, edgecolor='k')
        x = torch.from_numpy(X1).to(device,dtype=torch.float)
        y = torch.from_numpy(Y1).to(device,dtype=torch.double)

    if setting==3:
        X1, Y1 = make_gaussian_quantiles(n_samples=2000,n_features=2, n_classes=num_classes)
        plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,s=25, edgecolor='k')
        x = torch.from_numpy(X1).to(device,dtype=torch.float)
        y = torch.from_numpy(Y1).to(device,dtype=torch.double)

    if setting==4:
        centers = {'c1' : (1.05, 2.55), 'c2': (1.85,8.5)}
        Condition1='np.logical_or(np.logical_and(y>x ,y>-x), np.logical_and(y<-x+3,y<x-3))'
        Condition2='np.logical_or(np.logical_and(y<x ,y>-x), np.logical_and(y>x-3,y<-x+3))'
        x,y=Data_Generation(2000,Condition1, Condition2, False, centers)

    if setting==5:
        centers = {'c1' : (1.05, 2.55), 'c2': (1.85,8.5)}
        Condition1='np.logical_or(np.logical_and(y>x ,y>-x), np.logical_and(y<-x,y<x))'
        Condition2='np.logical_or(np.logical_and(y<x ,y>-x), np.logical_and(y>x,y<-x))'
        x,y=Data_Generation(2000,Condition1, Condition2, False, centers)


    if setting==6:
        centers = {'c1' : (1.05, 2.55), 'c2': (1.85,8.5)}
        Condition1='np.logical_or(np.logical_and(y>x ,y>-x), np.logical_and(y<-x,y<x))'
        Condition2='np.logical_or(np.logical_and(y<x ,y>-x), np.logical_and(y>x,y<-x))'
        x,y=Data_Generation(2000,Condition1, Condition2, True, centers)

    return x, y


def run(size_net, x, y, iters=10):
    biases = True
    size_net = size_net
    criterion = nn.CrossEntropyLoss()
    epochs = 400
    alpha = 1e-2
    batch_size = 256
    num_classes = 2
    model = Net(size_net,x.size(1),num_classes,biases).to(device)
    torch.save(model.state_dict(),'lottery_initialization.pth') #The initializtion of the original model

    # model, acc=train(model.to(device),criterion,x,y,alpha,epochs,batch_size,num_classes)

    lat_list = []
    acc_list = []
    for i in range(iters):
        start_time = time.time()
        _, acc=train(model.to(device),criterion,x,y,alpha,epochs,batch_size,num_classes)
        acc_list.append(acc)
        end_time = time.time()
        lat_list.append(end_time - start_time)
    print(acc_list)
    print(lat_list)
    print('mean acc: ', np.mean(acc_list))
    print('var acc: ', np.var(acc_list))
    print('std acc: ', np.std(acc_list, ddof=1))
    print('mean latency: ', np.mean([400 / item  for item in lat_list]), ' epochs/s')
    print('var latency: ', np.var([400 / item  for item in lat_list]), ' epochs/s')
    print('std latency: ', np.std([400 / item  for item in lat_list], ddof=1), ' epochs/s')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Trainning')
    parser.add_argument('--net_size', type=int, default=20, metavar='N', help='net_size')
    parser.add_argument('--iters', type=int, default=10, metavar='N', help='net_size')
    args = parser.parse_args()

    x, y = gen_data()
    run(args.net_size, x, y, args.iters)