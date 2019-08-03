import projection as pro
import lownerjohn_ellipsoid as lj
import numpy as np
import convex as c
import itertools
import random

def LJQ(dimension, edges, density):
    if(edges > dimension):
        print("Requested amount of edges is not possible")
    else: 
        polytope = randomPolytope(dimension, edges)
            
        projection = pro.projection(dimension, polytope, density)
                
        convexPolytop = c.convex(projection)
        
        points = convexPolytop
        
        A,b = inequalities(points)
            
        C1,d1 = lj.lownerjohn_inner(A,b)
                
        C2,d2 = lj.lownerjohn_outer(points)
        
        C2= np.linalg.inv(C2)
        
        d2 = np.matmul(C2,d2)
                    
        Q = np.linalg.det(C1)/np.linalg.det(C2)
        
        print('C1: '+ str(C1))
        print('d1: '+ str(d1))
        print('C2: '+ str(C2))
        print('d2: '+ str(d2))
        print('LJQ: '+ str(Q))        
    
def randomPolytope(dimension, edges):
    x=2**edges
    return random.sample(list(itertools.product(range(2),repeat=dimension)),x)
    
    
def inequalities(points):
    points.append(points[0])
    middle = (points[0]+points[1]+points[2])/3
    A=[]
    b=[]
    for i in range(len(points)-1):
        x1 = points[i][0]
        x2 = points[i][1]
        y1 = points[i+1][0]
        y2 = points[i+1][1]
        if y1 == x1:
            A.append([(np.sign(y1-middle[0]))*1,0])
            b.append(abs(y1))
        else:
            x = ((y2-x2)/(y1-x1))
            r = -x1*x + x2
            z = [x1+1, x2+x]
            l = c.position(points[i],z,middle)
            A.append([l*x,l*-1])
            b.append(abs(r))  
    return [A,b]