import numpy as np
from numpy.linalg import matrix_power
from numpy.linalg import norm
import matplotlib.pyplot as plt

phi = (1.0 + np.sqrt(5.0))/2.0
theta = np.pi/10.0
vec0 = np.zeros((2,1))
e1 = np.array([[1],[0]])
e2 = np.array([[0],[1]])
m = np.array([[np.cos(theta),-np.sin(theta)],
              [np.sin(theta),np.cos(theta)]])

class Tile:
    # translate, scale, and rotate methods are inherited by Kite and Dart from Tile class
    def translate(self,v):
        self.tail += v
        self.right += v
        self.head += v
        self.left += v
        self.centroid = (self.head + self.tail + self.right + self.left)/4
        return self

    def rescale(self):
        self.tail *= 1/self.scale
        self.right *= 1/self.scale
        self.head *= 1/self.scale
        self.left *= 1/self.scale
        self.centroid = (self.head + self.tail + self.right + self.left)/4
        self.scale = 1
        return self

    def rotate(self,n):
        self.tail = matrix_power(m,n) @ self.tail
        self.right = matrix_power(m,n) @ self.right
        self.head = matrix_power(m,n) @ self.head
        self.left = matrix_power(m,n) @ self.left
        self.centroid = (self.head + self.tail + self.right + self.left)/4
        self.orient = (self.orient + n) % 20
        return self

    def distance(self,vec):
        return norm(self.centroid - vec)

    def draw(self):
        plt.plot([self.tail[0],self.right[0],self.head[0],self.left[0],self.tail[0]],
                 [self.tail[1],self.right[1],self.head[1],self.left[1],self.tail[1]],
                 '-k')

    def fill(self,**kwargs):
        plt.fill([self.tail[0],self.right[0],self.head[0],self.left[0],self.tail[0]],
                 [self.tail[1],self.right[1],self.head[1],self.left[1],self.tail[1]],
                 **kwargs)

    def __eq__(self,other):
        if isinstance(other,type(self)):
            return ((norm(self.tail - other.tail)/self.scale < 1e-2) and
                    (norm(self.head - other.head)/self.scale < 1e-2))
        else:
            return NotImplemented

class Kite(Tile):
    
    def __init__(self,tail=vec0,orient=0,scale=1):
        
        self.tail = tail
        self.right = self.tail + scale * e1
        self.head = self.tail + scale * matrix_power(m,2) @ e1
        self.left = self.tail + scale * matrix_power(m,4) @ e1
        
        self.tail = matrix_power(m,orient) @ self.tail
        self.right = matrix_power(m,orient) @ self.right
        self.head = matrix_power(m,orient) @ self.head
        self.left = matrix_power(m,orient) @ self.left 
        self.centroid = (self.head + self.tail + self.right + self.left)/4
        self.orient = orient
        self.scale = scale

    def inflate(self,fill_ace=False):
        list_of_tiles = []
        list_of_tiles.append(Kite(scale=self.scale/phi,
                                  orient=((self.orient + 14)%20)).translate(self.left))
        list_of_tiles.append(Kite(scale=self.scale/phi,
                                  orient=((self.orient + 6)%20)).translate(self.right))
        list_of_tiles.append(Dart(scale=self.scale/phi,
                                  orient=((self.orient + 18)%20)).translate(self.tail))
        list_of_tiles.append(Dart(scale=self.scale/phi,
                                  orient=((self.orient + 2)%20)).translate(self.tail))
        if fill_ace:
            list_of_tiles.append(Kite(scale=self.scale/phi,
                                      orient=((self.orient + 10)%20)).translate(self.left))
            list_of_tiles.append(Kite(scale=self.scale/phi,
                                      orient=((self.orient + 10)%20)).translate(self.right))
        list_of_tiles = [t.rescale() for t in list_of_tiles]
        return list_of_tiles

    # draw method overrides the one inherited from the Tile class        
    def draw(self,arcs=False,fill=False):
        if fill:
            plt.fill([self.tail[0],self.right[0],self.head[0],self.left[0],self.tail[0]],
                     [self.tail[1],self.right[1],self.head[1],self.left[1],self.tail[1]],
                     facecolor='0.7', edgecolor='black')
        else:
            plt.plot([self.tail[0],self.right[0],self.head[0],self.left[0],self.tail[0]],
                     [self.tail[1],self.right[1],self.head[1],self.left[1],self.tail[1]],
                     '-k')
        if arcs:
            t = np.linspace(0,np.pi/2,30)

            x = self.right - self.tail
            y = self.left - self.tail
            norm = np.sqrt((x[0]*np.cos(t) + y[0]*np.sin(t))**2 
                           + (x[1]*np.cos(t) + y[1]*np.sin(t))**2)
            z1 = self.tail[0] + (1/phi)*(x[0]*np.cos(t) + y[0]*np.sin(t))/norm
            z2 = self.tail[1] + (1/phi)*(x[1]*np.cos(t) + y[1]*np.sin(t))/norm
            plt.plot(z1,z2,color='tab:orange',linestyle='-',alpha=0.6)

            x = self.left - self.head
            y = self.right - self.head
            norm = np.sqrt((x[0]*np.cos(t) + y[0]*np.sin(t))**2 
                           + (x[1]*np.cos(t) + y[1]*np.sin(t))**2)
            z1 = self.head[0] + (1/phi**2)*(x[0]*np.cos(t) + y[0]*np.sin(t))/norm
            z2 = self.head[1] + (1/phi**2)*(x[1]*np.cos(t) + y[1]*np.sin(t))/norm
            plt.plot(z1,z2,color='tab:blue',linestyle='-', alpha=0.6)
        

class Dart(Tile):
    
    def __init__(self,head=vec0, orient=0,scale=1):
        
        self.head = head
        self.left = self.head + scale * e1
        self.tail = self.head + (scale/phi) * matrix_power(m,2) @ e1
        self.right = self.head + scale * matrix_power(m,4) @ e1
        
        self.tail = matrix_power(m,orient) @ self.tail
        self.right = matrix_power(m,orient) @ self.right
        self.head = matrix_power(m,orient) @ self.head
        self.left = matrix_power(m,orient) @ self.left
        self.centroid = (self.head + self.tail + self.right + self.left)/4
        self.orient = orient
        self.scale = scale

    
    def inflate(self,fill_ace=False):
        list_of_tiles = []
        list_of_tiles.append(Kite(scale=self.scale/phi,
                                  orient=self.orient).translate(self.head))
        list_of_tiles.append(Dart(scale=self.scale/phi,
                                  orient=((self.orient + 12)%20)).translate(self.right))
        list_of_tiles.append(Dart(scale=self.scale/phi,
                                  orient=((self.orient + 8)%20)).translate(self.left))
        if fill_ace:
            list_of_tiles.append(Kite(scale=self.scale/phi,
                                      orient=((self.orient + 4)%20)).translate(self.head))
            list_of_tiles.append(Kite(scale=self.scale/phi,
                                      orient=((self.orient + 16)%20)).translate(self.head))
        list_of_tiles = [t.rescale() for t in list_of_tiles]
        return list_of_tiles
        
    def draw(self,arcs=False,fill=False):
        if fill:
            plt.fill([self.tail[0],self.right[0],self.head[0],self.left[0],self.tail[0]],
                     [self.tail[1],self.right[1],self.head[1],self.left[1],self.tail[1]],
                    facecolor='0.85', edgecolor='black')
        else:
            plt.plot([self.tail[0],self.right[0],self.head[0],self.left[0],self.tail[0]],
                     [self.tail[1],self.right[1],self.head[1],self.left[1],self.tail[1]],
                     '-k')
        if arcs:
            t = np.linspace(0,np.pi/2,30)
            t2 = np.linspace(np.pi/2,2*np.pi,60)

            x = self.left - self.tail
            y = self.right - self.tail
            norm = np.sqrt((x[0]*np.cos(t2) + y[0]*np.sin(t2))**2 
                           + (x[1]*np.cos(t2) + y[1]*np.sin(t2))**2)
            z1 = self.tail[0] + (1/phi**3)*(x[0]*np.cos(t2) + y[0]*np.sin(t2))/norm
            z2 = self.tail[1] + (1/phi**3)*(x[1]*np.cos(t2) + y[1]*np.sin(t2))/norm
            plt.plot(z1,z2,color='tab:blue',linestyle='-', alpha=0.6)

            x = self.right - self.head
            y = self.left - self.head
            norm = np.sqrt((x[0]*np.cos(t) + y[0]*np.sin(t))**2 
                           + (x[1]*np.cos(t) + y[1]*np.sin(t))**2)
            z1 = self.head[0] + (1/phi**2)*(x[0]*np.cos(t) + y[0]*np.sin(t))/norm
            z2 = self.head[1] + (1/phi**2)*(x[1]*np.cos(t) + y[1]*np.sin(t))/norm
            plt.plot(z1,z2,color='tab:orange',linestyle='-', alpha=0.6)


ace = [Kite(orient=15).translate(e2),
       Kite(orient=11).translate(e2),
       Dart(orient=3).translate(-(1/phi)*e2)]

deuce = [Kite(orient=19).translate(matrix_power(m,11)@e1),
         Kite(orient=7).translate(matrix_power(m,19)@e1),
         Dart(orient=11).translate(e2),
         Dart(orient=15).translate(e2)]

sun = [Kite(orient=3),
       Kite(orient=7),
       Kite(orient=11),
       Kite(orient=15),
       Kite(orient=19)]

star = [Dart(orient=3),
        Dart(orient=7),
        Dart(orient=11),
        Dart(orient=15),
        Dart(orient=19)]

jack = [Kite(orient=3).translate(matrix_power(m,15)@e1),
        Dart(orient=11).translate(matrix_power(m,1)@e1),
        Dart(orient=15).translate(matrix_power(m,9)@e1),
        Kite(orient=1),
        Kite(orient=5)]

queen = [Dart(orient=3),
         Kite(orient=13).translate(matrix_power(m,7)@e1),
         Kite(orient=5).translate(matrix_power(m,15)@e1),
         Kite(orient=1).translate(matrix_power(m,15)@e1),
         Kite(orient=13).translate(matrix_power(m,3)@e1)]

king = [Dart(orient=3),
        Dart(orient=7),
        Kite(orient=17).translate(-matrix_power(m,1)@e1),
        Kite(orient=9).translate(-matrix_power(m,9)@e1),
        Dart(orient=19)]
