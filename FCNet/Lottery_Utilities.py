from Libraries import *
from trainings import *
from Geometry_utilities import *
def class_blind_rows(model,percentage):
    new_model = deepcopy(model)
    A = new_model.state_dict()['fc1.weight'].cpu().numpy()
    norm_a = np.linalg.norm(A,axis = 1)
    num_com = int(percentage * len(norm_a) / 100)
    for i in range(num_com):
        min_ind = np.argmin(norm_a)
        A[min_ind,:] = 0
        norm_a[min_ind] = 100
    new_model.state_dict()['fc1.weight'][:,:] = torch.from_numpy(A).to(device)
    return new_model

def lottery_experiment(model,x,y,initialization,size_net = 20,threshold = [20,40,60,80]):
    num_classes = 2
    biases = False
    criterion = nn.CrossEntropyLoss()
    cmap =plt.get_cmap('Reds') 
    Z = [[0,0],[0,0]]
    levels = np.linspace(0,80,50,endpoint=True)
    CS3 = plt.contourf(Z, levels, cmap=cmap)
    plt.clf()
    batch_size = 100
    color_index=570
    colormap = 600
    acc=[]
    import matplotlib
    countt=7
    threshold = [20,40,60,80]
    counter1=1
    XavierMu,XavierSig1,XavierSig2 = 0 , 1/np.sqrt(2) , 1/np.sqrt(40)
    GaussianMu,GuassianSigma= 0,0.1
    if initialization == 'standard gaussian':
        GuassianSigma = 1
    newmodel=deepcopy(model)
    model_lot_init = deepcopy(model)
    for thresh in threshold:
        newmodel = class_blind_rows(newmodel,thresh)
        #print('Threshold : ' , thresh)
        #print('Accuracy No Retrain : ' , accuracy(newmodel.to(device),x,y))
        #Implementing Lottery_Ticket
        Temp = []
        for alpha in range(size_net):
            if alpha not in np.argwhere(newmodel.state_dict()['fc1.weight'].cpu().numpy()  == np.array([0,0]))[:,0]:
                Temp.append(alpha)
        if initialization == 'lottery':
            model_lot_init.load_state_dict(torch.load('lottery_initialization.pth'))
            model_lot_init.eval()
            lottery_a = model_lot_init.fc1.weight.data.clone()
            lottery_b = model_lot_init.fc2.weight.data.clone()
            model_lot = Net(len(Temp),2,num_classes,biases)
            model_lot.fc1.weight.data.copy_(lottery_a[Temp,:])
            model_lot.fc2.weight.data.copy_(lottery_b[:,Temp])
            model_lot = train(model_lot.to(device),criterion,x,y,0.01,30,batch_size)
            plotted_model = model_lot
        
        if initialization == 'xavier':
            Xavier = Net(len(Temp),2,num_classes,biases)
            Xavier.fc1.weight.data= torch.from_numpy(np.random.normal(XavierMu,1/np.sqrt(x.shape[1])  ,(Xavier.fc1.weight.data.shape[0],Xavier.fc1.weight.data.shape[1])).astype('f'))
            Xavier.fc2.weight.data= torch.from_numpy(np.random.normal(XavierMu,1/np.sqrt(len(Temp)),(Xavier.fc2.weight.data.shape[0],Xavier.fc2.weight.data.shape[1])).astype('f'))
            if biases:
                Xavier.fc1.bias.data= torch.from_numpy(np.zeros((Xavier.fc1.bias.data.shape[0] , 1)).astype('f')).squeeze()
                Xavier.fc2.bias.data= torch.from_numpy(np.zeros((Xavier.fc2.bias.data.shape[0] , 1)).astype('f')).squeeze()
            Xavier = train(Xavier.to(device),criterion,x,y,0.01,25,batch_size)
            plotted_model = Xavier

        if 'gaussian' in initialization:
            Gaussian = Net(len(Temp),2,num_classes,biases)
            Gaussian.fc1.weight.data= torch.from_numpy(np.random.normal(GaussianMu,GuassianSigma,(Gaussian.fc1.weight.data.shape[0],Gaussian.fc1.weight.data.shape[1])).astype('f'))
            Gaussian.fc2.weight.data= torch.from_numpy(np.random.normal(GaussianMu,GuassianSigma,(Gaussian.fc2.weight.data.shape[0],Gaussian.fc2.weight.data.shape[1])).astype('f'))
            if biases:
                Gaussian.fc1.bias.data= torch.from_numpy(np.random.normal(GaussianMu,GuassianSigma,(Gaussian.fc1.bias.data.shape[0] , 1)).astype('f')).squeeze()
                Gaussian.fc2.bias.data= torch.from_numpy(np.random.normal(GaussianMu,GuassianSigma,(Gaussian.fc2.bias.data.shape[0] , 1)).astype('f')).squeeze()
            Gaussian = train(Gaussian.to(device),criterion,x,y,0.01,30,batch_size)
            plotted_model = Gaussian


        ver,bound,_,_,_,_,_,_ = get_model_polytopes(plotted_model,False,True,False)
        zonotope1_vertices=ver
        hull1=bound
        n_lines=1
        i=1
        legend=''
        
        if zonotope1_vertices.shape[0] > 3:
            xp = zonotope1_vertices[hull1.vertices,0].tolist()
            yp = zonotope1_vertices[hull1.vertices,1].tolist()
            xp += [zonotope1_vertices[hull1.vertices,0][-1], zonotope1_vertices[hull1.vertices,0][0]]
            yp += [zonotope1_vertices[hull1.vertices,1][-1], zonotope1_vertices[hull1.vertices,1][0]]
            plt.plot(xp, yp, c=cmap(7/650 * thresh +6/65 ), lw=2 , zorder=1)
            plt.grid()
            for simplex in hull1.simplices:#Before Compression
                plt.scatter(zonotope1_vertices[simplex, 0], zonotope1_vertices[simplex, 1], marker='o' ,color='black', s=10 ,zorder=2)
        else:
            plt.plot(zonotope1_vertices[0],zonotope1_vertices[1],c=cmap.to_rgba(i + 1))
        
        color_index=color_index-20
        countt-=1
    matplotlib.rcParams.update({'font.size': 15})
    plt.xticks([])
    plt.yticks([])
    plt.title(initialization + ' Initialization')
    plt.colorbar(CS3, ticks = np.linspace(80,0,5,endpoint=True))
    plt.show()    
    plt.clf()
