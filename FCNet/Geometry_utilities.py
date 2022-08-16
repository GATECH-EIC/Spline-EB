from Libraries import *
from scipy.spatial import ConvexHull
from itertools import combinations
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
def get_model_polytopes(model,bias,boundary = True, All_generators_only = True,classes = [0,1]):
    #Extracting the generators of the zonotopes
    A = model.state_dict()['fc1.weight'].cpu().numpy()
    B = model.state_dict()['fc2.weight'][classes,:].cpu().numpy()
    A_pos = 0.5*(np.abs(A) + A)
    A_neg = 0.5*(np.abs(A) - A)
    gen_pos = get_generators(A_pos,B) 
    gen_neg = get_generators(A_neg,B) 

    G1pos_pos, G1pos_neg = gen_pos[0]
    G2pos_pos, G2pos_neg = gen_pos[1]
    G1neg_pos, G1neg_neg = gen_neg[0]
    G2neg_pos, G2neg_neg = gen_neg[1]
    if All_generators_only:
        return G1pos_pos-G1neg_pos, G1pos_neg-G1neg_neg, G2pos_pos-G2neg_pos, G2pos_neg-G2neg_neg

    
    
    if not bias:
        #The shift for each zonotopes:
        shifting_points = shift_zonotope(A_neg,B)
        shift_z1,shift_z2 = shifting_points[0]
        shift_z3,shift_z4 = shifting_points[1]

        if boundary: #Extracting the decision boundary's polytope
        #These Temps are the generators for the decision boundary zonotopes
            Temp1 = np.concatenate((G1pos_pos,G2pos_neg), axis = 0)
            Temp2 = np.concatenate((G1neg_pos,G2neg_neg), axis = 0)
            Temp3 = np.concatenate((G1pos_neg,G2pos_pos), axis = 0)
            Temp4 = np.concatenate((G1neg_neg,G2neg_pos), axis = 0)

            shift_tot1 = shift_z1 + shift_z4
            shift_tot2 = shift_z2 + shift_z3

            zon1_decisionboundary,conv1,_ = get_zonotope(Temp1, Temp2, shift_tot1)
            zon2_decisionboundary,conv2,_ = get_zonotope(Temp3, Temp4, shift_tot2)
            boundary_vertices = np.concatenate((zon1_decisionboundary, zon2_decisionboundary),axis=0)
            decision_boundary = ConvexHull(boundary_vertices)
            return boundary_vertices,decision_boundary,Temp1-Temp2,Temp3-Temp4,zon1_decisionboundary,conv1,zon2_decisionboundary,conv2
        else:
        #Extracting The zonotope vertices using Polynomial time algorithm
            zon1,hull1,count1 = get_zonotope(G1pos_pos , G1neg_pos , shift_z1)
            zon2,hull2,count2 = get_zonotope(G1pos_neg , G1neg_neg , shift_z2)
            zon3,hull3,count3 = get_zonotope(G2pos_pos , G2neg_pos , shift_z3)
            zon4,hull4,count4 = get_zonotope(G2pos_neg , G2neg_neg , shift_z4)
            return zon1,hull1,zon2,hull2,zon3,hull3,zon4,hull4
        
    else: #If we have biases
        bias2 = model.state_dict()['fc2.bias']
        size_n = model.state_dict()['fc1.bias'].size()[0]
        A = model.state_dict()['fc1.weight']
        B = model.state_dict()['fc2.weight'][classes,:]
        A_pos = 0.5*(torch.abs(A) + A)
        A_neg = 0.5*(torch.abs(A) - A)
        A_pos = torch.cat((A_pos,model.state_dict()['fc1.bias'].reshape(-1,1)),1)
        A_neg = torch.cat((A_neg,torch.zeros((size_n,1)).to(device)),1)
        gen_pos = get_generators(A_pos,B) 
        gen_neg = get_generators(A_neg,B) 

        G1pos_pos, G1pos_neg = gen_pos[0]
        G2pos_pos, G2pos_neg = gen_pos[1]
        G1neg_pos, G1neg_neg = gen_neg[0]
        G2neg_pos, G2neg_neg = gen_neg[1]

        #The shift for each zonotopes:
        shifting_points = shift_zonotope(A_neg,B)
        shift_z1,shift_z2 = shifting_points[0]
        shift_z3,shift_z4 = shifting_points[1]
        sift_z1 = np.concatenate((shift_z1,np.array([bias2[0].item()])))
        sift_z2 = np.concatenate((shift_z2,np.array([0])))
        sift_z3 = np.concatenate((shift_z3,np.array([bias2[1].item()])))
        sift_z4 = np.concatenate((shift_z4,np.array([0])))

        if boundary:
        #These Temps are the generators for the decision boundary zonotopes
            Temp1 = np.concatenate((G1pos_pos,G2pos_neg), axis = 0)
            Temp2 = np.concatenate((G1neg_pos,G2neg_neg), axis = 0)
            Temp3 = np.concatenate((G1pos_neg,G2pos_pos), axis = 0)
            Temp4 = np.concatenate((G1neg_neg,G2neg_pos), axis = 0)

            shift_tot1 = shift_z1 + shift_z4
            shift_tot2 = shift_z2 + shift_z3

            zon1_decisionboundary,_,_ = get_zonotope(Temp1, Temp2, shift_tot1)
            zon2_decisionboundary,_,_ = get_zonotope(Temp3, Temp4, shift_tot2)
            boundary_vertices = np.concatenate((zon1_decisionboundary, zon2_decisionboundary),axis=0)
            boundary_vertices = boundary_vertices[:,0:2]
            decision_boundary = ConvexHull(boundary_vertices)
            return boundary_vertices,decision_boundary
        else:
        #Extracting The zonotope vertices using Polynomial time algorithm
            zon1,hull1,count1 = get_zonotope(G1pos_pos , G1neg_pos , shift_z1)
            zon2,hull2,count2 = get_zonotope(G1pos_neg , G1neg_neg , shift_z2)
            zon3,hull3,count3 = get_zonotope(G2pos_pos , G2neg_pos , shift_z3)
            zon4,hull4,count4 = get_zonotope(G2pos_neg , G2neg_neg , shift_z4)
            return zon1,hull1,zon2,hull2,zon3,hull3,zon4,hull4

#########################################Zonotope library###################################
def shift_zonotope(A_neg,b):
    out = []
    for label in range(b.shape[0]):
        b_pos = b[label,:] > 0
        b_neg = b[label,:] < 0
        shift1 = (A_neg[b_neg,:].T * np.abs(b[label,b_neg] )).sum(1)
        shift2 = (A_neg[b_pos,:].T * np.abs(b[label,b_pos] )).sum(1)
        out.append((shift1,shift2))
    return out
def get_generators(A,B):
    out = []
    for label in range(B.shape[0]):
        b_pos = np.maximum(0,B[label, :])
        b_neg = np.maximum(0,-1*B[label, :])
        positive = A.T * b_pos
        negative = A.T * b_neg
        out.append((positive.T, negative.T))
    return out
def get_zonotope(G1pos_pos,G1neg_pos,shift):
    #Using the polynomial time algorithm to generrate the zonotope
    set_of_generators = (G1pos_pos - G1neg_pos).T
    sum_of_generators = set_of_generators.sum(1) 
    num = nov(set_of_generators) #Computing the number of vertices
    zon1,count = zonotope(set_of_generators,num,len(set_of_generators[:,0]))
    zon1 = np.asarray(zon1) 
    zon1 = zon1.squeeze()
    # Applying the modification on the generated zonotope
    zon1 = 0.5*zon1 + 0.5*sum_of_generators + G1neg_pos.sum(0) + shift  #The shift is the second part in the equations of H_2, and G_2 that was ignored before
    if zon1.shape[0] > 3:
        hull = ConvexHull(zon1)
    else:
        hull = []
    return zon1,hull,count
#This function returns the number of vertices of the zonotope given the set of generators of that zonotope.
def nov(G):
    m,n = G.shape
    total_sum = 0
    for i in range(m):
        total_sum = total_sum + comb(n-1, i,exact=True)
    return (total_sum*2)

########################################################################################################################################
#This function is an implementation of the algorithm 1 in "Enumerating zonotope vertices using randomized algorithm"
#The set of generators "Gen" are the columns of Gen
#k is the expected number of vertices
#dim is the dimension of a generator
def zonotope(Gen,k,dim):
    V = [] #Set of vertices
    count = 0
    while (len(V) < k) and (count < 10000 ) :
        count += 1
        x = np.random.randn(dim, 1)
        GG = Gen.T.dot (x)
        mask = np.sign(GG)
        v_pos = Gen.dot(mask)
        v_pos = v_pos.T
        temp3 = any((v_pos == z).all() for z in V)
        if not temp3:
            V.append(v_pos)
            V.append(-1*v_pos)
    return V,count

