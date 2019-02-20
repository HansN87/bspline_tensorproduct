#!/usr/bin/env python

import numpy as np
import sys

class bspline_basis(object):
    def __init__(self, t, p):
        '''
        t (numpy.array): knot sequence without any padding (can be non-uniform)
        p (integer): degree of b-spline (not order!) 

		notation follows
		"A Practical Guide to Splines" by C. De Boor

        use Cox-de-Boor alogrithm, i.e. explicit evaluation of spline basis functions
        '''

        self.t_orig = t # original knot sequence
        self.p = p # degree of spline
        self.k = self.p + 1 # order of spline
        self.q = len(t) # number of original knots

        # padding to guarantee proper boundary behavior
        t0 = np.ones(p) * self.t_orig[0]
        t1 = np.ones(p) * self.t_orig[-1]
        self.t = np.concatenate((t0, self.t_orig, t1), axis=None)

        

    def _w(self, i, k, x):
        ii=i-1 # re-index to account for python array indexing
    
        if not self.t[ii]==self.t[ii+k-1]:
            return (x-self.t[ii]) / (self.t[ii+k-1]-self.t[ii])
        else:
            return 0.0

    def _bspline(self, i, k, x):
	    '''
	    B_{i,k} (x) on extended knot sequence t (t1, ...., tq)
	    where k is the order of the spline
	    i is the index of the basis (1...q + p -1)
	    where q is len(original_knot sequence)
	    p = k-1 (degree of spline)
	    
	    in order to cope with python indexing will internally shift all indices by 1
	    '''
	    ii = i-1 # re-index to account for python array indexing
	    
	    if k==1:
	        # piecewise constant
	        if self.t[ii]<=x and x<self.t[ii+1]:
	            return 1.
	        else:
	            return 0.
	        
	    else:
	        c1 = self._w(i, k, x)
	        c2 = 1.-self._w(i+1, k, x)
	        return c1 * self._bspline(i, k-1, x) + c2 * self._bspline(i+1, k-1, x)

    def find_closest_knot(self, x):
        found = False
        for j in range(len(self.t)-1):
            if self.t[j+1]>x:
                found = True
                break

        if not found:
            j=len(self.t)-1     

        # go back to spline index
        return j+1
        
    def eval_element(self, i, x):
        closest_j = self.find_closest_knot(x)
        #return self._bspline(i, self.k, x)
        if i<=closest_j-self.k:
            return 0.0
        else:
            return self._bspline(i, self.k, x)



class bspline_func_1d(object):
    # constructs 1D B-spline function from knots and coefficients
    def __init__(self, t, p, c):

        '''
        t (numpy.array): knot sequence without any padding (can be non-uniform)
        p (integer): degree of b-spline (not order!) 
        c (numpy.array): spline coefficients

        notation follows
        "A Practical Guide to Splines" by C. De Boor

        use Cox-de-Boor alogrithm, i.e. explicit evaluation of spline basis functions
        '''

        self.t = t
        self.p = p
        self.c = c	
        self.q = len(t)

        # ensure that choice of t, p, c is sensible
        self.N = self.q+self.p-1
        if len(self.c) != self.N:
            print "length of knots, degree of spline and number of coefficients do not match. exiting ..."
            sys.exit(1)

        self.basis = bspline_basis(self.t, self.p)


    def eval(self, x):
        # beware of indexing. c[i-1] is coefficient for B_{i,k} since
        # splines are indexed starting at index i=1
        evals = [self.c[i] * self.basis.eval_element(i+1, x) for i in range(self.N)]
        return np.sum(evals)


class bspline_func_2d(object):
    # constructs 2D Tensor-Product B-spline function from knots and coefficients
    def __init__(self, tx, ty, p, c):
        '''
        tx (numpy.array): knot sequence without any padding (can be non-uniform)                                                                
        ty (numpy.array): knot sequence without any padding (can be non-uniform) 
        p (integer): degree of b-spline (not order!) 
        c (numpy.array): matrix of spline coefficients of shape(len(Nx), len(Ny))

        notation follows
        "A Practical Guide to Splines" by C. De Boor

        use Cox-de-Boor alogrithm, i.e. explicit evaluation of spline basis functions
        '''

        self.tx = tx
        self.ty = ty
        self.p = p
        self.c = c
        self.qx = len(tx)
        self.qy = len(ty)

        # ensure that choice of t, p, c is sensible
        self.Nx = self.qx+self.p-1
        self.Ny = self.qy+self.p-1
        self.N = self.Nx * self.Ny
        if c.shape != (self.Nx, self.Ny):
            print "length of knots, degree of spline and number of coefficients do not match. exiting ..."
            sys.exit(1)

        self.basisx = bspline_basis(self.tx, self.p)
        self.basisy = bspline_basis(self.ty, self.p)

    def eval(self, x, y):
        # beware of indexing. c[i-1] is coefficient for B_{i,k} since
        # splines are indexed starting at index i=1
        
        # build vector of evaluations along y
        bspline_along_x = np.array([[self.basisx.eval_element(i+1, x) for i in range(self.Nx)]])
        bspline_along_y = np.array([[self.basisy.eval_element(i+1, y) for i in range(self.Ny)]]).T
        return np.dot(bspline_along_x, np.dot(self.c, bspline_along_y))[0][0] 
