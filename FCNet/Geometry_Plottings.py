from Libraries import *
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'     
    
#This function samples a square grid between -7.5 and 7.5 and classifies all the points to get the decision boundary
def plot_2d(model, device='cpu'):
    x = torch.linspace(-7.5,7.5,1000,device=device)
    y = torch.linspace(-7.5,7.5,1000,device=device)
    inputs = torch.stack(torch.meshgrid(x, y)).view(2, -1).t()
    outputs = model(inputs) #classifiying the points
    red = inputs[outputs[:,0] <= outputs[:,1]]
    blue = inputs[outputs[:,0] > outputs[:,1]]
    plt.scatter(red[:,0].cpu(), red[:,1].cpu(), color='red')
    plt.scatter(blue[:,0].cpu(), blue[:,1].cpu(), color='blue')           
########################################################################################################################################


def plot_polytope(zonotope1_vertices,hull1,legend = 'legend'):
    if zonotope1_vertices.shape[0] > 3:
        xp = zonotope1_vertices[hull1.vertices,0].tolist()
        yp = zonotope1_vertices[hull1.vertices,1].tolist()

        xp += [zonotope1_vertices[hull1.vertices,0][-1], zonotope1_vertices[hull1.vertices,0][0]]
        yp += [zonotope1_vertices[hull1.vertices,1][-1], zonotope1_vertices[hull1.vertices,1][0]]
        plt.plot(xp, yp, 'k--', lw=2)
        plt.legend([legend])
        plt.grid()
        for simplex in hull1.simplices:#Before Compression
            plt.plot(zonotope1_vertices[simplex, 0], zonotope1_vertices[simplex, 1], 'ro')
    else:
        plt.plot(zonotope1_vertices[0],zonotope1_vertices[1],'k--')
      
#plotting the bisectors on the edges of the boundary with thier magnitude
def plot_bisectors(boundary_vertices,decision_boundary):
    xp = boundary_vertices[decision_boundary.vertices,0].tolist()
    yp = boundary_vertices[decision_boundary.vertices,1].tolist()

    xp += [boundary_vertices[decision_boundary.vertices,0][-1], boundary_vertices[decision_boundary.vertices,0][0]]
    yp += [boundary_vertices[decision_boundary.vertices,1][-1], boundary_vertices[decision_boundary.vertices,1][0]]
    temp = len(xp)
    temp2 = np.zeros((temp,2))
    lines = []
    for i in range(temp-1):
        p1 = Point(xp[i],yp[i])
        p2 = Point(xp[i+1],yp[i+1])
        if p1 != p2:
            l1 = Segment(p1,p2)
            bisec = l1.perpendicular_bisector()
            lines.append(np.array(bisec.points))                       
    for line in lines:
        p1 = line[0]
        p2 = 2*line[0] - line[1] #To make the dirction of the line to the out of the zonotope
        xb = [0,p2[0]-p1[0]] #To make the lines centered around the origin
        yb = [0,p2[1]-p1[1]] #To make the lines centered around the origin
        plt.plot(xb,yb,'w-')

def plot_bisectors_on_polytope(boundary_vertices,decision_boundary):
    xp = boundary_vertices[decision_boundary.vertices,0].tolist()
    yp = boundary_vertices[decision_boundary.vertices,1].tolist()

    xp += [boundary_vertices[decision_boundary.vertices,0][-1], boundary_vertices[decision_boundary.vertices,0][0]]
    yp += [boundary_vertices[decision_boundary.vertices,1][-1], boundary_vertices[decision_boundary.vertices,1][0]]

    temp = len(xp)
    temp2 = np.zeros((temp,2))
    lines = []

    for i in range(temp-1):
        p1 = Point(xp[i],yp[i])
        p2 = Point(xp[i+1],yp[i+1])
        if p1 != p2:
            l1 = Segment(p1,p2)
            bisec = l1.perpendicular_bisector()
            lines.append(np.array(bisec.points))
                        
    for line in lines:
        p1 = line[0]
        p2 = 2*line[0] - line[1] #To make the dirction of the line to the out of the zonotope
        xb = [p1[0],p2[0]] #To make the lines centered around the origin
        yb = [p1[1],p2[1]] #To make the lines centered around the origin
        plt.plot(xb,yb,'b-')

