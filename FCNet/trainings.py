from Libraries import *
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
###########################################################Networks definitions##############################################################
class Net(nn.Module): #Single hidden layer NN
    def __init__(self,net_size,input_size,output_size,Bias):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, net_size,bias = Bias)
        self.fc2 = nn.Linear(net_size,output_size,bias = Bias)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc2(self.relu(self.fc1(x)))    
        return x
def train(model,criterion, x, y, alpha ,epochs,batchsize):
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
                #### add by haoran ###
                activations = []
                def get_activation():
                    def hook(model, input, output):
                        activations.append(output.detach())
                    return hook
                model.f1.register_forward_hook(get_activation())
#                 model.f2.register_forward_hook(get_activation())
                print('hhh')
                #####################################
                acc = accuracy(model,testx,testy)
                print(f'Test accuracy is #{acc:.2f} , Iter = {j}')
                ### add by haoran ###
                visualizeBatch(len(testx), activations, iteration=j)
                #####################################
                model.train()
            cost = criterion(model(x1), y1.squeeze())
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()        
            costs.append(float(cost))
    return model

def visualizeBatch(batchSize, batchActivation, iteration):
    firstActiv = batchActivation[0].cpu()
    N = int(math.sqrt(firstActiv.shape[0]))
    print(firstActiv.shape)
    L = 2
    grid = np.meshgrid(np.linspace(-L, L, N),
                     np.linspace(- L, L, N))
    plt.figure(figsize=(12, 4))
    for k in range(1):
        plt.contour(grid[0], grid[1], firstActiv[:, k].reshape((N, N)), levels=[0], colors='b')
    plt.xticks([])
    plt.yticks([])
    plt.title('layer1')
    save_path = './spline/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + 'Iter_'+ str(iteration) + '.png')
    
def accuracy(model,x,y):
    prediction = model(x).detach().cpu().numpy()
    return (np.sum(np.argmax(prediction,axis=1).reshape(-1,1)==np.array(y.cpu().numpy()).reshape(-1,1))/len(y.cpu().numpy()))*100
#########################################################################################
#########################################################################################
def Data_Generation (Dataset_Size,Condition1,Condition2,isCircle,centers = {'c1' : (0, 0), 'c2': (0, 0)} ):
    if(not isCircle):
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