import random
import numpy as np

def projection(dimension, polytope, density):
    matrix = projectionMatrix(dimension, density)
    
    projection = []
    
    polytope = np.unique(np.array(polytope),axis=0)
    
    for p in polytope:
        np.transpose(p)
        projection.append(np.matmul(matrix, p))
            
    return projection


def projectionMatrix(dimension, density):
    matrix = []
    
    notNull = int((1 + (dimension-1)*density))
    
    indexes = []
    
    for i in range(2):
        indexes.append(random.sample(range(dimension),notNull))
    
    for i in range(2):
        matrix.append(projectionVector(dimension, indexes[i]))
        
    
    while(np.array_equal(matrix[0], matrix[1])):
        matrix=[]
        for i in range(2):
            matrix.append(projectionVector(dimension, indexes[i]))
    
    return matrix


def projectionVector (dimension, indexes):
    vector = []
    normedVector = []
    for i in range (dimension):
        if i in indexes:
            vector.append(np.random.normal(0,1))
        else:
            vector.append(0)
    
    norm = np.linalg.norm(vector)
    for v in vector:
        i = v/norm
        normedVector.append(i)
    return normedVector