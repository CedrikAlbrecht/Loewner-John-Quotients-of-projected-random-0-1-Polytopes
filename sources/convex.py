import numpy as np
    
def convex(points):
    points = np.unique(points, axis=0)
    
    points = sort(points)
    
    points = grahamScan(points)

    return points

'''
sort the points counter-clockwise
'''
def sort(points):
    sortedPoints = [0]* len(points)
    Q = points[0]
      
    angles = []
    
    for p in points:
        t=p-Q
        alpha = np.arctan2(np.dot(t, [1,0]), np.linalg.det([t,[1,0]] ) )
        angles.append(alpha)
        
    angles = np.array(angles)
    order = np.argsort(angles)   
    
    for i in range(len(order)):
        j = order[i]
        sortedPoints[i] = points[j]
        
    return sortedPoints
    
def grahamScan(points):
    convex = [points[0],points[1]]
    points.append(points[0])
    for i in range(2,len(points)):
      while len(convex) >1 and position(convex[len(convex)-2],convex[len(convex)-1],points[i]) < 1:
          convex.pop()
      convex.append(points[i])
    return convex
        
'''
position(x,y,z)=-1 <=> z is to the right of the line through xy
position(x,y,z)=0 <=> z is on the line through xy
position(x,y,z)=1 <=> z is to the left of the line through xy
'''
def position(x,y,z):
    xy = np.subtract(y,x)
    xz = np.subtract(z,x)
    m = [xy, xz]
    det = np.linalg.det(m)
    return np.sign(det)


