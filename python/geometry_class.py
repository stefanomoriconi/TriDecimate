# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:57:25 2025

@author: OGSMORIC
"""

# Edit on uvect (eps)
# Edit on readObj -> flip Matrix Rigid


import numpy as np
import matplotlib.pyplot as plt # Default 3D plot (maybe not optimal for 3D renderings in Python...)
import time
import point_cloud_utils as pcu # << you can install this from: https://fwilliams.info/point-cloud-utils/ ## THIS IS NEEDED ONLY FOR DECIMATING A MESH MODEL
import inspect 

# Ancillary Functions
def uvect(v3):
    # retrieve unit-vector: v3 expected array [3,]
    uv3 = np.array(v3)
    if len(v3.shape) == 1:
        uv3 = uv3/np.linalg.norm(uv3)
    else:
        dd = np.linalg.norm(uv3, axis=1)
        uv3 = np.divide(uv3, dd.reshape(dd.shape[0], 1))
    return uv3

def projct(a3, b3):
    # projection of vector a3 to vector b3: a3 and b3 expected arrays [3,]
    c = np.dot(a3, b3)
    if len(np.shape(a3)) > 1:
        c = np.reshape(c, [np.shape(c)[0], 1])
    return c

    
def rotM3D(a, b, c):
    # Rotation Matrix as (improper) Euler angles:
    # a -> around X, b -> around Y, c -> around Z
    Rz = np.array([[np.cos(c),    -np.sin(c), 0],
                   [np.sin(c),     np.cos(c), 0],
                   [        0,             0, 1]])

    Ry = np.array([[ np.cos(b),  0, np.sin(b)],
                   [         0,  1,         0],
                   [-np.sin(b),  0, np.cos(b)]])

    Rx = np.array([[1,          0,          0],
                   [0,  np.cos(a), -np.sin(a)],
                   [0,  np.sin(a),  np.cos(a)]])

    # Rotation Matrix Mutliplication 
    R = np.matmul(Rz, np.matmul(Ry, Rx))
    return R



def rotM3Da2b(a0, a1, b0, b1):
    # WIP: estimate the rotation matrix mapping two sets of orthonormal bases:
    # a0, a1: source orthonormal base
    # b0, b1: target orthonormal base
    # all input vectors are arrays: 1x3
    # Output rotation matrix R is 3x3 so that:
    #   b0 = a0*R
    #   b1 = a1*R
    a0 = np.array(a0)
    a1 = np.array(a1)
    b0 = np.array(b0)
    b1 = np.array(b1)
    
    # Estimate partial rotation matrix (R1)
    R1 =  _rot3DMap_a2b(a0, b0)
        
    # Rotate a1 with estimated partial rotation matrix R1
    a1r = np.matmul(a1, R1)
    
    # Estimate partial rotation matrix (R2)
    R2 =  _rot3DMap_a2b(a1r, b1)
        
    # Composition of partial rotations (R1, R2)
    R = np.matmul(R1, R2)
    
    return R.transpose()

def _rot3DMap_a2b(a3, b3, tol=0.001):
    
    if np.any(np.abs(a3-b3) > tol):
        if np.any(np.abs(np.abs(a3) - np.abs(b3)) > tol):
            
            v3 = np.cross(b3, a3) # Cross product, taking into account the target FIRST, the source AFTER
            s = np.linalg.norm(v3) # Norm of the resulting vector (sin)
            c = np.dot(a3, b3) # (cos) between vectors
        
            vmat = np.array([[   0  , -v3[2],   v3[1] ],
                             [ v3[2],    0  ,  -v3[0] ],
                             [-v3[1],  v3[0],      0  ]])
        
            R = np.eye(3) + vmat + np.matmul(vmat, vmat)*((1-c)/s**2)
                  
        else: # case a0 = -b0
            R = np.eye(3)
            for i in range(0,len(a3)):
                if a3[i] * b3[i] < 0:
                    R[i, i] = -1
            
    else: # case a0 = b0 
        R = np.eye(3)
    
    return R

# Define Material Object (Class)
class Material():
    # Material defined with the Blinn-Phong Model
    def __init__(self, ambRGB=[0.1, 0.0, 0.0], dffRGB=[0.7, 0.0, 0.0], spcRGB=[1.0, 1.0, 1.0], shnC=16.0, rflC=0.0, alpha=1.0, MaxDepth=1):
        self.ambRGB = ambRGB # Ambient Colour RGB
        self.dffRGB = dffRGB # Diffuse Colour RGB
        self.spcRGB = spcRGB # Specular Colour RGB (of incident Light)
        self.shnC = shnC # Glossiness/Shininess Coefficient [1, 100]
        self.rflC = rflC # Reflection Coefficient (Matte:0.0, Mirror -> 1.0)
        self.MaxDepth = MaxDepth  # Number of light bounces (Matte:[0,3], Mirror:[10,15])
        self.alpha = alpha # Transparency Index [0, 1] with 0: Transparent, 1: Fully Solid Color
        
        self._iniMat()
        # A Mirror Material usually has:
            # Ambient RGB: [51, 51, 51]/255 = [.2, .2, .2]
            # Diffuse RGB: [15, 15, 15]/255 = [.02, .02, .02]
            # Specular RGB = [215, 215, 215]/255 = [.85, .85, .85] 
            # Shine Coef: 0.996
            # Reflect Coef: 1.0
            # MaxDepth: 10
            
    def _iniMat(self):
        self.ambRGB = np.array(self.ambRGB) 
        self.dffRGB = np.array(self.dffRGB) 
        self.spcRGB = np.array(self.spcRGB) 
        self.shnC = np.array(self.shnC)
        self.rflC = np.array(self.rflC) 
        self.MaxDepth = np.array(self.MaxDepth)

    def resetMat(self):
        self.ambRGB = [0.1, 0.0, 0.0]
        self.dffRGB = [0.7, 0.0, 0.0] 
        self.spcRGB = [1.0, 1.0, 1.0] 
        self.shnC = 16.0 
        self.rflC = 0.0 
        self.MaxDepth = 1
        
        self._iniMat()
                
# Define Geometry Object (Class)
class Geometry():
    # Geometry defined as any raster triangulated mesh (vertices, faces)
    def __init__(self, vts=None, fcs=None, label='Geometry', visibleFlag=True):
        self.vts = vts # Vertices list (3D Points)
        self.fcs = fcs # Faces list (Triangles)
        self.mat = Material() # Material of the Geometry
        self.label = label
        self.visibleFlag=visibleFlag
        
        self.fcs_ctr = None
        self.fn = None
        self.vn = None
    
        self.CoM = [0, 0, 0]
        self.BBox = [[0, 0, 0],[0, 0, 0]]
        self.size = 0
        self.trisize = 0
        
        # Initialising Geometry Normals
        self._getFeatures()
        
        
    def _getFeatures(self):# OK - optimised for PERFORMANCE 
    # Initialising mesh features: size, Face Centres, Face & Vertex Normals
        if self.fcs is None or self.vts is None:
            self.fcs_ctr = None
            self.fn = None
            self.vn = None
            self.size = 0
            self.trisize = 0
            self.CoM = [0, 0, 0]
            self.BBox = [[0, 0, 0],[0, 0, 0]]
            return

        # Store the Size of the Geometry
        self.size = self._getSize()     
        self.trisize = self._gettrisize()
        self.CoM = np.mean(self.vts, axis=0) # XYZ coords of the CoM
        self.BBox = np.vstack((np.min(self.vts, axis=0), 
                               np.max(self.vts, axis=0))) # Bounding Box [2x3]
        self.fcs_ctr = self._getFaceCenters() # Face Centers Coordinates
        self.fn = self._getFaceNormals()   # Faces Normals of the Geometry
        self.vn = self._getVertexNormals()   # Vertex Normals of the Geometry
        
    def _getSize(self):# OK
        # Estimate Centre
        c3 = np.mean(self.vts, axis=0)
        # Size: average distance of vertices wrt their centre
        size = np.mean(np.linalg.norm(self.vts - c3, axis=1))
        return size
        
    def _gettrisize(self):#WIP
        e0 = np.linalg.norm(self.vts[self.fcs[:,0],:] - self.vts[self.fcs[:,1],:], axis=1)
        e1 = np.linalg.norm(self.vts[self.fcs[:,1],:] - self.vts[self.fcs[:,2],:], axis=1)
        e2 = np.linalg.norm(self.vts[self.fcs[:,2],:] - self.vts[self.fcs[:,0],:], axis=1)
        trisize = np.mean(np.concatenate((e0, e1, e2)))
        return trisize
    
    def _getFaceCenters(self):# OK
        fcs_vts = np.stack((self.vts[self.fcs[:,0],:],
                            self.vts[self.fcs[:,1],:],
                            self.vts[self.fcs[:,2],:]), axis=2)
        fcs_ctr = np.mean(fcs_vts, axis=2)
        return fcs_ctr
    
    def _getFaceNormals(self):# OK
        # Computing Face Normals (ONLY Triangles)
        v1s = self.vts[self.fcs[:,1],:] - self.vts[self.fcs[:,0],:]
        v2s = self.vts[self.fcs[:,2],:] - self.vts[self.fcs[:,1],:]
        fn = uvect(np.cross(v1s, v2s))
        return fn
    
    def _getVertexNormals(self):# OK
        # Computing Vertex Normals (ONLY Triangles)
        idx = np.argsort(self.fcs.flatten()).astype(np.float64)
        idx = np.floor(idx/np.shape(self.fcs)[1]).astype(np.int64)
        _, cts = np.unique(self.fcs, return_counts=1)
        fns_vts = self.fn[idx, :]
        fns_vts = np.split(fns_vts, np.cumsum(cts)[:-1])
        vn = vn = np.array([np.mean(i, axis=0) for i in fns_vts])
        return vn
    
    def flipNormals(self):# OK
        # Filling the sign of Normals (or 3D unit vectors)
        self.fn = -1*self.fn
        self.vn = -1*self.vn
    
    def flipFacesOrder(self):
        self.fcs = np.fliplr(self.fcs)
        self._getFeatures()
    
    def loadBunny(self):# OK
        self.readOBJ('./Geometry/Bunny.obj')
        self.tformRigid(np.pi/2, 0, np.pi, 5, -3, 10)
        self.scale(10)
        self.tformRigid(0,0,0,-50,29,-103)
    
    def genPlane(self, ptA=[0,0,0], ptB=[1,0,0], ptC=[1,1,0], ptD=[0,1,0]):
        # Generates a polygonal plane given 4 corners. The triangulation is:
        # ABC, CDA, in that order for normals.
        self.vts = np.vstack((ptA, ptB, ptC, ptD))
        ptM = np.mean(self.vts, axis=0)
        self.vts = np.vstack((self.vts, ptM))
        self.fcs = np.array([[0, 1, 4],
                             [1, 2, 4],
                             [2, 3, 4],
                             [3, 0, 4]])
        
        # Initialising Features
        self._getFeatures()
        
    def genTriangle(self, ptA=[0,1,0], ptB=[-np.sqrt(3)/2,-1/2,0], ptC=[np.sqrt(3)/2,-1/2,0]):
        # Generates a polygonal plane given 4 corners. The triangulation is:
        # ABC, CDA, in that order for normals.
        self.vts = np.vstack((ptA, ptB, ptC))
        self.fcs = np.array([[0, 1, 2]])
        
        # Initialising Features
        self._getFeatures()
        
    def genCircle(self, C=[0,0,0], rad=1, e1=[1,0,0], nn=[0,0,1], sbdv=3):
        self.genTriangle()

        # Refine with Subdivision
        for ss in range(0, sbdv):
            edg_bdr = self._getBoundaryEdges()
            vts = self.vts.copy()
            fcs = self.fcs.copy()
            for ee in range(0, edg_bdr.shape[0]):
                m_mod = np.mean([np.linalg.norm(vts[edg_bdr[ee, 0],:]),
                                 np.linalg.norm(vts[edg_bdr[ee, 1],:])])
                m_vec = uvect(np.mean([vts[edg_bdr[ee, 0],:],
                                       vts[edg_bdr[ee, 1],:]], axis=0))
                m = np.reshape(m_vec, [1, 3])*m_mod
                vts = np.vstack((vts, m))
                f = np.array([edg_bdr[ee, 1],
                              edg_bdr[ee, 0],
                              vts.shape[0]-1],
                             dtype=np.uint32)
                fcs = np.vstack((fcs, f))
                
            self.vts = vts
            self.fcs = fcs
            # return self.vts, self.fcs
            
        self.scale(rad)
        self.tformRigid(x_offset=C[0], y_offset=C[1], z_offset=C[2])
        R = rotM3Da2b([1,0,0],[0,0,1],e1,nn)
        M = np.eye(4)
        M[0:3,0:3] = R
        self.tformRigidM(M)
        self._getFeatures()
            
    
    def genIcoSphere(self, c3=[0.0, 0.0, 0.0], rad=1.0, nsub=0):# OK
        vts, fcs = self._genIcosahedron(nsub=nsub)
        self.vts = rad*vts + np.array(c3) # Broadcasting!
        self.fcs = fcs
        
        # Initialising Features
        self._getFeatures()
        
    def _genIcosahedron(self, nsub=0): # OK
        # Creating a unit regular icosahedron
        t = (1 + np.sqrt(5.0)) / 2
        # Define Vertices
        vts = np.array([[-1, t, 0], # v1
                        [ 1, t, 0], # v2
                        [-1,-t, 0], # v3
                        [ 1,-t, 0], # v4
                        [ 0,-1, t], # v5
                        [ 0, 1, t], # v6
                        [ 0,-1,-t], # v7
                        [ 0, 1,-t], # v8
                        [ t, 0,-1], # v9
                        [ t, 0, 1], # v10
                        [-t, 0,-1], # v11
                        [-t, 0, 1]]) # v12
        
        # Normalising Vertices to unit vectors        
        vts = vts/np.linalg.norm(vts, axis=1, keepdims=1)
        
        # Define Faces (Triangles)
        fcs = np.array([[ 0, 11,  5], # f1
                        [ 0,  5,  1], # f2
                        [ 0,  1,  7], # f3
                        [ 0,  7, 10], # f4
                        [ 0, 10, 11], # f5
                        [ 1,  5,  9], # f6
                        [ 5, 11,  4], # f7
                        [11, 10,  2], # f8
                        [10,  7,  6], # f9
                        [ 7,  1,  8], # f10
                        [ 3,  9,  4], # f11
                        [ 3,  4,  2], # f12
                        [ 3,  2,  6], # f13
                        [ 3,  6,  8], # f14
                        [ 3,  8,  9], # f15
                        [ 4,  9,  5], # f16
                        [ 2,  4, 11], # f17
                        [ 6,  2, 10], # f18
                        [ 8,  6,  7], # f19
                        [ 9,  8,  1]], dtype=np.uint64)# f20
        
        # Subdivision
        nsub = int(nsub)
        if nsub > 0:
            vts, fcs = self._subdTriFaces(vts, fcs, nsub=nsub)
        
        return vts, fcs
    
    def subdivide(self, nsub=1, nrmFlag=True):
        nsub = int(nsub)
        if nsub > 0:
            vts, fcs = self._subdTriFaces(self.vts, self.fcs, 
                                          nsub=nsub, nrmFlag=nrmFlag)
            
        # Assignment
        self.vts = vts
        self.fcs = fcs
        # Initialising Features
        self._getFeatures()
    
    def _subdTriFaces(self, vts, fcs, nsub, nrmFlag=True):# OK
        # Recursively subdivision of triangular faces
        if nsub == 0:
            return vts, fcs
        
        for sbd in range(0, nsub):
            # Initialising subdivided Faces (ONLY Triangles)
            sbd_fcs = np.zeros([fcs.shape[0]*4, 3])
            
            for ff in range(0, fcs.shape[0]): # for each triangular face    
                # Select the i-th Trianglular face
                fc = fcs[ff, :]

                # Calculate the mid points (add new points to v)
                a, vts = self._appendNormMidPoint(fc[0], fc[1], vts, nrmFlag=nrmFlag)
                b, vts = self._appendNormMidPoint(fc[1], fc[2], vts, nrmFlag=nrmFlag)
                c, vts = self._appendNormMidPoint(fc[2], fc[0], vts, nrmFlag=nrmFlag)
                    
                # Generating new subdivision triangles
                nfc = np.array([[fc[0], a, c],
                                [fc[1], b, a],
                                [fc[2], c, b],
                                [    a, b, c]])
                    
                # Replacing Triangle with subdivision
                idx = list(range((4*ff), (4*(ff+1))))
                sbd_fcs[idx, :] = nfc
                
            # Updating Faces with the Subdivided Faces
            fcs = np.array(sbd_fcs, dtype=np.uint64)  
        
        # Removing duplicate vertices
        # NEED for ROBUST UNIQUE? (tolerance) [Potential source of BUG?]
        vts_unq, vts_idx = np.unique(vts.round(decimals=6), axis=0, return_inverse=1) # << Unique with Real values! (BUG?)
        # Re-assigning faces to trimmed vertex list and remove duplicate faces
        fcs_unq = np.zeros(fcs.shape, dtype=np.uint64)
        for vidx in range(0, len(vts_idx)):
            fcs_unq[fcs == vidx] = vts_idx[vidx]
        fcs_unq = np.unique(fcs_unq, axis=0) # << Unique with integers (OK)
        
        return vts_unq, fcs_unq
        
    def _appendNormMidPoint(self, idx0, idx1, vts, nrmFlag=True):# OK
        # Retrieve vertices 
        v0 = vts[idx0, :]
        v1 = vts[idx1, :]
        if nrmFlag:
            # New length-normalised Mid-Point
            v2mod = np.mean([np.linalg.norm(v0), np.linalg.norm(v1)])
            v2vec = uvect(np.mean([v0, v1], axis=0))
            v2 = np.reshape(v2vec, [1, 3])*v2mod
        else:
            v2 = np.mean([v0, v1], axis=0).reshape([1, 3])
        # Concatenate Mid-Point
        vts = np.concatenate((vts, v2), axis=0)
        
        idx = vts.shape[0] - 1
        return idx, vts
        
    def _getTriEdges(self):
        
        edg = []
        fcs = self.fcs
        for ff in range(0, fcs.shape[0]):
            edg.append([fcs[ff,0], fcs[ff,1]])
            edg.append([fcs[ff,1], fcs[ff,2]])
            edg.append([fcs[ff,2], fcs[ff,0]])
        edg = np.vstack(edg).astype(np.uint64)
        _, eidx = np.unique(np.sort(edg, axis=1), axis=0, return_index=1)
        edg = edg[eidx,:]
        
        return edg
    
    def _getBoundaryEdges(self):
        
        edg = self._getTriEdges()
        chk = []
        for ee in range(0, edg.shape[0]):
            cc = np.sum(np.sum(np.hstack((self.fcs == edg[ee,0],
                                          self.fcs == edg[ee,1])),
                               axis=1) == 2,
                        axis=0)
            chk.append(cc)
        chk = np.vstack(chk) == 1
        edg_bdr = edg[chk.squeeze(),:]
        
        return edg_bdr
    
    def _getBoundaryTriangles(self):
        
        edg_bdr = self._getBoundaryEdges()
        fcs_idx = []
        for ee in range(0, edg_bdr.shape[0]):
            chk = np.sum(np.bitwise_or(self.fcs == edg_bdr[ee,0], 
                                       self.fcs == edg_bdr[ee,1]), axis=1) == 2
            fcs_idx.append(np.argwhere(chk).squeeze())
        fcs_idx = np.hstack(fcs_idx)
        
        fcs_idx = np.unique(fcs_idx)
        return fcs_idx
    
    def decimate(self, fcsRate=0.5):
        
        maxFcs = int(fcsRate * self.fcs.shape[0])
        vts_red, fcs_red, _, _ = pcu.decimate_triangle_mesh(self.vts, 
                                                            self.fcs,
                                                            max_faces=maxFcs)
        # Assignment
        self.vts = vts_red
        self.fcs = fcs_red
        # Initialising Features
        self._getFeatures()
    
    def _showGeo3D(self, fig=None, fnFlag=False, vnFlag=False, fnIdx=None, vnIdx=None, fcsAlpha=0.5, edgeAlpha=0.25, linewidth=0.5, supTitle=None, frmFileName=None, shadeFlag=False, antiAliasFlag=True):# OK
        # Enable interactive mode
        plt.ion()
        
        hs = [] # handles to Geometries plotted
    
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            # to flush the GUI events
            fig.canvas.flush_events()
            time.sleep(0.001)
            ax = fig.axes[0]

        # Displaying Geometry as Triangualr Mesh (Patch)
        h0 = ax.plot_trisurf(self.vts[:, 0],
                        self.vts[:, 1],
                        self.vts[:, 2],
                        triangles = self.fcs,
                        color=self.mat.dffRGB,
                        edgecolor=[[self.mat.ambRGB[0],self.mat.ambRGB[1],self.mat.ambRGB[2],edgeAlpha]],
                        linewidth=linewidth,
                        alpha=fcsAlpha,
                        shade=shadeFlag,
                        antialiased=antiAliasFlag)
        hs.append(h0)
        
        if fnFlag:
            if fnIdx is None:
                h1 = ax.quiver(self.fcs_ctr[:, 0], self.fcs_ctr[:, 1], self.fcs_ctr[:, 2],
                          self.fn[:, 0],  self.fn[:, 1],  self.fn[:, 2],
                          length=self.size/5, normalize=True, color='k')
            else:
                h1 = ax.quiver(self.fcs_ctr[fnIdx, 0], self.fcs_ctr[fnIdx, 1], self.fcs_ctr[fnIdx, 2],
                          self.fn[fnIdx, 0],  self.fn[fnIdx, 1],  self.fn[fnIdx, 2],
                          length=self.size/5, normalize=True, color='k')
            hs.append(h1)
        
        if vnFlag:
            if vnIdx is None:
                h2 = ax.quiver(self.vts[:, 0], self.vts[:, 1], self.vts[:, 2],
                          self.vn[:, 0],  self.vn[:, 1],  self.vn[:, 2],
                          length=self.size/5, normalize=True, color='b')
            else:
                h2 = ax.quiver(self.vts[vnIdx, 0], self.vts[vnIdx, 1], self.vts[vnIdx, 2],
                           self.vn[vnIdx, 0],  self.vn[vnIdx, 1],  self.vn[vnIdx, 2],
                           length=self.size/5, normalize=True, color='b')  
            hs.append(h2)
        
        ax.set_aspect('equal')
        ax.axes.set_xlabel('X-axis')
        ax.axes.set_ylabel('Y-axis')
        ax.axes.set_zlabel('Z-axis')
        
        if supTitle is not None:
            fig.suptitle(supTitle)
        
        # Re-drawing the figure
        fig.canvas.draw()
        
        if frmFileName is not None:
            fig.savefig(frmFileName, bbox_inches='tight')
        
        return fig, hs
    
    def _removeALLgeoHandles(self, geoHandles):
        for ii in range(0, len(geoHandles)):
            if isinstance(geoHandles[ii], list):
                geoHandles[ii][0].remove()
            else:
                geoHandles[ii].remove()
        
    def readOBJ(self, OBJFileName=None, flip = 0):# OK
        try:
            vts = []
            fcs = []
            clr = []
            with open(OBJFileName) as file:
                for line in file:
                    if line[0:2] == "v ":
                        vv = list(map(float, line[2:].strip().split()))
                        if len(vv) <= 3:
                            vts.append(vv)
                        else:
                            vts.append(vv[0:3])
                            clr = vv[3:]
                    elif line[0:2] == "f ":
                        ff = list(map(int, line[2:].strip().replace('/',' ').split()))
                        if len(ff) <=3:
                            fcs.append(ff)
                        else:
                            fcs.append(ff[0::2])
                     
            if len(clr) == 3:
                self.mat.dffRGB = np.array(clr)
                self.mat.ambRGB = np.divide(np.array(clr), 5.0)
                     
            vts = np.array(vts)
            fcs = np.array(fcs, dtype=np.uint64)
            
            try:
                v0 = vts[fcs[:,0],:]
                v1 = vts[fcs[:,1],:]
                v2 = vts[fcs[:,2],:]
            except:
                try:
                    fcs = fcs - 1
                    v0 = vts[fcs[:,0],:]
                    v1 = vts[fcs[:,1],:]
                    v2 = vts[fcs[:,2],:]
                except:
                    print('Cannot Map Faces!')
            
            self.vts = vts
            self.fcs = fcs # Indices start from 0 (not from 1)
            if flip == 1 :
                M = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]])
                self.tformRigidM(M = M, centredFlag=1)

            
            print('>> {}: Succesfully read data from: {}'.format(
                inspect.stack()[0][3], str(OBJFileName)))
            
        except:
            print('<!> {}: Unable to read data from: {}'.format(
                inspect.stack()[0][3], str(OBJFileName)))
            self.vts = None
            self.fcs = None
        
        # Initialising Features
        self._getFeatures()
        
    def writeOBJ(self, OBJFileName=None, matExport=False): #OK
        try:
            hdr = ['# OBJ File Generated with Python\n',
                   '# Vertices: {:d}\n'.format(np.shape(self.vts)[0]),
                   '# Faces: {:d}\n'.format(np.shape(self.fcs)[0])]
            
            vts = self.vts
            fcs = self.fcs + 1 # Faces indices must start form 1 (not from 0!)
            clrRGB = self.mat.dffRGB
            
            if matExport:
                hdr.append('# Vertex Colour: Enabled\n\n')
            else:
                hdr.append('#\n')
            
            with open(OBJFileName, 'w') as file:
                for hh in range(0, len(hdr)):
                    file.write(hdr[hh])
                
                for vv in range(0, vts.shape[0]):
                    if matExport:
                        file.write('v {:.7f} {:.7f} {:.7f} {:.3f} {:.3f} {:.3f}\n'.format(
                            vts[vv,0], vts[vv,1], vts[vv,2], 
                            clrRGB[0], clrRGB[1], clrRGB[2]))
                    else:
                        file.write('v {:.7f} {:.7f} {:.7f}\n'.format(
                            vts[vv,0], vts[vv,1], vts[vv,2]))
                        
                file.write('\n')
                    
                for ff in range(0, self.fcs.shape[0]):
                    file.write('f {:d} {:d} {:d}\n'.format(
                        fcs[ff,0], fcs[ff,1], fcs[ff,2])) 

                file.write('\n# END of FILE\n')                 
                
            file.close()
            
        except:
            print('<!> {}: Unable to write data to: {}'.format(
                inspect.stack()[0][3], str(OBJFileName)))

        return None
        
    def scale(self, scale=1.0):
        scale = np.array(scale)
        self.vts = np.multiply(self.vts, scale) # multiply by scalar (ISOTROPIC SCALING)
        
        # Initialising Features
        self._getFeatures()

    def scaleAniso(self, scale=[1.0, 1.0, 1.0]):
        scale = np.array(scale)
        self.vts[:, 0] = self.vts[:, 0]*scale[0]
        self.vts[:, 1] = self.vts[:, 1]*scale[1]
        self.vts[:, 2] = self.vts[:, 2]*scale[2]
        
        # Initialising Features
        self._getFeatures()
    
    def tformRigid(self, a_euler=0.0, b_euler=0.0, c_euler=0.0, x_offset=0.0, y_offset = 0.0, z_offset=0.0, centredFlag=1): # OK
        # E.g for Bunny:
        #   a_euler = np.pi/2,
        #   b_euler = 0,
        #   c_euler = np.pi
        
        # A 4x4 transformation matrix will be generated from the input parameters
        M = self._getRigidMatrix(a_euler, b_euler, c_euler, x_offset, y_offset, z_offset)
        
        # If centred: Center to Origin -> Transform -> ReOffset
        if centredFlag:
            CoM = np.nanmean(self.vts, axis=0)
            vts = self.vts - CoM
        else:
            vts = self.vts
            
        # Transformation
        vts = np.concatenate((vts.transpose(), np.ones([1, vts.shape[0]])), axis=0)
        vts = np.matmul(M, vts).transpose()
        vts = vts[:, 0:3]
            
        if centredFlag: 
            vts = vts + CoM
            
        self.vts = vts
        # Initialising Features
        self._getFeatures()
    
    def _getRigidMatrix(self, a_euler=0.0, b_euler=0.0, c_euler=0.0, x_offset=0.0, y_offset = 0.0, z_offset=0.0): #OK
        
        M = np.eye(4)
        M[0:3, 0:3] = rotM3D(a_euler, b_euler, c_euler)
        M[0,3] = x_offset
        M[1,3] = y_offset
        M[2,3] = z_offset
        
        return M
        
    def tformRigidM(self, M=np.eye(4), centredFlag=1):
        # De-offset
        if centredFlag:
            CoM = np.nanmean(self.vts, axis=0)
            vts = self.vts - CoM
        else:
            vts = self.vts
            
        # Transformation
        vts = np.concatenate((vts.transpose(), np.ones([1, vts.shape[0]])), axis=0)
        vts = np.matmul(M, vts).transpose()
        vts = vts[:, 0:3]
           
        # Re-offset
        if centredFlag: 
            vts = vts + CoM
            
        self.vts = vts
        # Initialising Features
        self._getFeatures()
        
    def tformCenterOrig(self):
        CoM = np.nanmean(self.vts, axis=0)
        self.vts = self.vts - CoM
        # Initialising Features
        self._getFeatures()
        return CoM
        
    def getTFormRigidInv(self, M=np.eye(4)):
        Minv = M.copy()
        Minv[:3,:3] = M[:3,:3].transpose()
        Minv[:3,3] = -M[:3,3]
        return Minv
    
    def cullFaces(self, v3=None, tol=1e-1):
        if v3 is None:
            return None
        v3 = uvect(v3)
        fcs_cull = projct(self.fn, v3) <= 0.0 + tol
        return fcs_cull  
    
    def perimFaces(self, v3=None, tol=1e-1):
        if v3 is None:
            return None
        v3 = uvect(v3)
        fcs_perim = np.bitwise_and(projct(self.fn, v3) >= - tol, 
                                   projct(self.fn, v3) <=   tol)
        return fcs_perim 

    def shdFaces(self, sel_fn=None, v3=None):
        if v3 is None:
            return None
        v3 = uvect(v3)
        if sel_fn is None:
            fcs_shd = projct(uvect(self.fn), v3)
        else:
            fcs_shd = projct(uvect(sel_fn), v3)
        return fcs_shd
    
    def dstFaces(self, v3=None, Nd=np.array([0.0])):
        if v3 is None:
            return None
        v3 = np.array(v3).flatten() #v3 expected a single 3D point coordinate (reference point to consider for Euclidean distance - Usually, v3=CamPos)
        fcs_dst = np.linalg.norm(self.fcs_ctr - v3, axis=1, keepdims=True)
        if len(Nd) == 1:
            if Nd[0] == 0.0:
                return fcs_dst # RETURN RAW VALUES, NOT-NORMALISED 
            elif Nd[0] < 0.0:
                # NORMALISE within fcs_dst RANGE [0,1]
                fcs_dst = (fcs_dst - fcs_dst.min())/(fcs_dst.max() - fcs_dst.min())
        else:
            # NORMALISE upon EXTERNAL Normalising-distance (Nd) reference value (min, max)
            fcs_dst = (fcs_dst - Nd.min())/(Nd.max() - Nd.min()) # NOTE: No Control over Nd values! (They MUST be Positive)
            # Make Sure:
                # Nd.min() <= fcs_dst.min()     AND    Nd.max() >= fcs_dst.max()
            
        return fcs_dst # RETURN NORMALISED VALUES
    
    def stats(self, fullFlag=1):
        print(' ')
        print(' * Geometry Stats*')
        print(' - Label: {:s}'.format(self.label))
        print(' - Vertices: {:d} x {:d}'.format(self.vts.shape[0], self.vts.shape[1]))
        print(' - Faces: {:d} x {:d}'.format(self.fcs.shape[0], self.fcs.shape[1]))
        print(' - Face Size: {:.3f}'.format(self.trisize))
        if fullFlag:
            print(' - CoM coords XYZ: [{:.3f},{:.3f},{:.3f}]'.format(
                self.CoM[0], self.CoM[1], self.CoM[2]))
            print(' - CoM-Vertices Avg Dist: {:.3f}'.format(self.size))
            print(' - Bounding Box XYZ: [[{:.3f},{:.3f},{:.3f}],[{:.3f},{:.3f},{:.3f}]]'.format(
                self.BBox[0,0], self.BBox[0,1], self.BBox[0,2],
                self.BBox[1,0], self.BBox[1,1], self.BBox[1,2]))
            print(' - Material (Diffuse RGB): [{:.3f},{:.3f},{:.3f}]'.format(
                self.mat.dffRGB[0], self.mat.dffRGB[1], self.mat.dffRGB[2]))
            print(' - Material (Ambient RGB): [{:.3f},{:.3f},{:.3f}]'.format(
                self.mat.ambRGB[0], self.mat.ambRGB[1], self.mat.ambRGB[2]))
            print(' - Material (Specular RGB): [{:.3f},{:.3f},{:.3f}]'.format(
                self.mat.spcRGB[0], self.mat.spcRGB[1], self.mat.spcRGB[2]))
            print(' - Material (Gloss coeff): {:.3f}'.format(self.mat.shnC))
            print(' - Material (Reflective coeff): {:.3f}'.format(self.mat.rflC))
            print(' - Material (Max Depth - Refl. Bounces): {:d}'.format(self.mat.MaxDepth))
            print(' ')
        else:
            print(' ')
        return None
        
    def _getFidxsFromVidxs(self, vidx):
        vidx = list(vidx)
        fidx = np.empty([1,1], dtype=np.uint64)
        map_idx = np.empty([1,1], dtype=np.uint64)
        for ii in range(0, len(vidx)):
            fidx_tmp = np.argwhere(np.any(self.fcs == vidx[ii], axis=1))
            fidx = np.vstack((fidx, fidx_tmp.reshape([len(fidx_tmp), 1])))
            map_idx = np.vstack((map_idx, ii*np.ones([len(fidx_tmp), 1])))

        return fidx[1:], map_idx[1:]