# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:36:40 2019

@author: ofn77899
"""
#from ccpi.optimisation.ops import Operator
import numpy
from numbers import Number
import functools
from ccpi.framework import AcquisitionData, ImageData, BlockDataContainer
from ccpi.optimisation.operators import Operator, LinearOperator


       
class BlockOperator(Operator):
    '''Class to hold a block operator

    Class to hold a number of Operators in a block. 
    User may specify the shape of the block, by default is a row vector
    '''
    def __init__(self, *args, **kwargs):
        '''
        Class creator

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            vararg (Operator): Operators in the block.
            shape (:obj:`tuple`, optional): If shape is passed the Operators in 
                  vararg are considered input in a row-by-row fashion. 
                  Shape and number of Operators must match.
                  
        Example:
            BlockOperator(op0,op1) results in a row block
            BlockOperator(op0,op1,shape=(1,2)) results in a column block
        '''
        self.operators = args
        shape = kwargs.get('shape', None)
        if shape is None:
            shape = (len(args),1)
        self.shape = shape
        n_elements = functools.reduce(lambda x,y: x*y, shape, 1)
        if len(args) != n_elements:
            raise ValueError(
                    'Dimension and size do not match: expected {} got {}'
                    .format(n_elements,len(args)))
    def get_item(self, row, col):
        if row > self.shape[0]:
            raise ValueError('Requested row {} > max {}'.format(row, self.shape[0]))
        if col > self.shape[1]:
            raise ValueError('Requested col {} > max {}'.format(col, self.shape[1]))
        
        index = row*self.shape[1]+col
        return self.operators[index]
    
    def norm(self):
        norm = [op.norm() for op in self.operators]
        b = []
        for i in range(self.shape[0]):
            b.append([])
            for j in range(self.shape[1]):
                b[-1].append(norm[i*self.shape[1]+j])
        return numpy.asarray(b)
    
    def direct(self, x, out=None):
        shape = self.get_output_shape(x.shape)
        res = []
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if col == 0:
                    prod = self.get_item(row,col).direct(x.get_item(col))
                else:
                    prod += self.get_item(row,col).direct(x.get_item(col))
            res.append(prod)
        return BlockDataContainer(*res, shape=shape)
    
    def adjoint(self, x, out=None):
        shape = self.get_output_shape(x.shape, adjoint=True)
        res = []
        for row in range(self.shape[1]):
            for col in range(self.shape[0]):
                if col == 0:
                    prod = self.get_item(col,row).adjoint(x.get_item(row))
                else:
                    prod += self.get_item(col,row).adjoint(x.get_item(row))
            res.append(prod)
        return BlockDataContainer(*res, shape=shape)
    
    def get_output_shape(self, xshape, adjoint=False):
        sshape = self.shape[1]
        oshape = self.shape[0]
        if adjoint:
            sshape = self.shape[0]
            oshape = self.shape[1]
        if sshape != xshape[0]:
            raise ValueError('Incompatible shapes {} {}'.format(self.shape, xshape))
        return (oshape, xshape[-1])
    
    
    
class BlockLinearOperator(BlockOperator):
    '''Class to hold a block operator

    Class to hold a number of Operators in a block. 
    User may specify the shape of the block, by default is a row vector
    '''
    def __init__(self, *args, **kwargs):
        '''
        Class creator

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            vararg (Operator): LinearOperators in the block.
            shape (:obj:`tuple`, optional): If shape is passed the Operators in 
                  vararg are considered input in a row-by-row fashion. 
                  Shape and number of Operators must match.
                  
        Example:
            BlockLinearOperator(op0,op1) results in a row block
            BlockLinearOperator(op0,op1,shape=(1,2)) results in a column block
        '''
        for i,op in enumerate(args):
            if not op.is_linear():
                raise ValueError('Operator {} must be LinearOperator'.format(i))
        super(BlockLinearOperator, self).__init__(*args, **kwargs)
    
    def adjoint(self, x, out=None):
        '''Adjoint operation for the BlockOperator
        
        only available on BlockLinearOperator
        '''
        shape = self.get_output_shape(x.shape, adjoint=True)
        res = []
        for row in range(self.shape[1]):
            for col in range(self.shape[0]):
                if col == 0:
                    prod = self.get_item(col,row).adjoint(x.get_item(row))
                else:
                    prod += self.get_item(col,row).adjoint(x.get_item(row))
            res.append(prod)
        return BlockDataContainer(*res, shape=shape)







    
if __name__ == '__main__':
    #from ccpi.optimisation.Algorithms import GradientDescent
    from ccpi.optimisation.algorithms import CGLS

    from ccpi.plugins.ops import CCPiProjectorSimple
    from ccpi.optimisation.ops import PowerMethodNonsquare
    from ccpi.optimisation.ops import TomoIdentity
    from ccpi.optimisation.funcs import Norm2sq, Norm1
    from ccpi.framework import ImageGeometry, AcquisitionGeometry
    from ccpi.optimisation.Algorithms import GradientDescent
    #from ccpi.optimisation.Algorithms import CGLS
    import matplotlib.pyplot as plt

    
    # Set up phantom size N x N x vert by creating ImageGeometry, initialising the 
    # ImageData object with this geometry and empty array and finally put some
    # data into its array, and display one slice as image.
    
    # Image parameters
    N = 128
    vert = 4
    
    # Set up image geometry
    ig = ImageGeometry(voxel_num_x=N,
                       voxel_num_y=N, 
                       voxel_num_z=vert)
    
    # Set up empty image data
    Phantom = ImageData(geometry=ig,
                        dimension_labels=['horizontal_x',
                                          'horizontal_y',
                                          'vertical'])
    Phantom += 0.05
    # Populate image data by looping over and filling slices
    i = 0
    while i < vert:
        if vert > 1:
            x = Phantom.subset(vertical=i).array
        else:
            x = Phantom.array
        x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
        x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 0.94
        if vert > 1 :
            Phantom.fill(x, vertical=i)
        i += 1
    
    
    perc = 0.02
    # Set up empty image data
    noise = ImageData(numpy.random.normal(loc = 0.04 ,
                             scale = perc , 
                             size = Phantom.shape), geometry=ig,
                        dimension_labels=['horizontal_x',
                                          'horizontal_y',
                                          'vertical'])
    Phantom += noise
    
    # Set up AcquisitionGeometry object to hold the parameters of the measurement
    # setup geometry: # Number of angles, the actual angles from 0 to 
    # pi for parallel beam, set the width of a detector 
    # pixel relative to an object pixe and the number of detector pixels.
    angles_num = 20
    det_w = 1.0
    det_num = N
    
    angles = numpy.linspace(0,numpy.pi,angles_num,endpoint=False,dtype=numpy.float32)*\
                 180/numpy.pi
    
    # Inputs: Geometry, 2D or 3D, angles, horz detector pixel count, 
    #         horz detector pixel size, vert detector pixel count, 
    #         vert detector pixel size.
    ag = AcquisitionGeometry('parallel',
                             '3D',
                             angles,
                             N, 
                             det_w,
                             vert,
                             det_w)
    
    # Set up Operator object combining the ImageGeometry and AcquisitionGeometry
    # wrapping calls to CCPi projector.
    A = CCPiProjectorSimple(ig, ag)
    
    # Forward and backprojection are available as methods direct and adjoint. Here 
    # generate test data b and some noise
    
    b = A.direct(Phantom)
    
    
    #z = A.adjoint(b)
    
    
    # Using the test data b, different reconstruction methods can now be set up as
    # demonstrated in the rest of this file. In general all methods need an initial 
    # guess and some algorithm options to be set. Note that 100 iterations for 
    # some of the methods is a very low number and 1000 or 10000 iterations may be
    # needed if one wants to obtain a converged solution.
    x_init = ImageData(geometry=ig, 
                       dimension_labels=['horizontal_x','horizontal_y','vertical'])
    X_init = BlockDataContainer(x_init)
    B = BlockDataContainer(b, 
                               ImageData(geometry=ig, dimension_labels=['horizontal_x','horizontal_y','vertical']))
    
    # setup a tomo identity
    Ibig = 1e5 * TomoIdentity(geometry=ig)
    Ismall = 1e-5 * TomoIdentity(geometry=ig)
    
    # composite operator
    Kbig = BlockOperator(A, Ibig, shape=(2,1))
    Ksmall = BlockOperator(A, Ismall, shape=(2,1))
    
    #out = K.direct(X_init)
    
    f = Norm2sq(Kbig,B)
    f.L = 0.00003
    
    fsmall = Norm2sq(Ksmall,B)
    f.L = 0.00003
    
    simplef = Norm2sq(A, b)
    simplef.L = 0.00003
    
    gd = GradientDescent( x_init=x_init, objective_function=simplef,
                         rate=simplef.L)
    gd.max_iteration = 10
    
    cg = CGLS()
    cg.set_up(X_init, Kbig, B )
    cg.max_iteration = 1
    
    cgsmall = CGLS()
    cgsmall.set_up(X_init, Ksmall, B )
    cgsmall.max_iteration = 1
    
    
    cgs = CGLS()
    cgs.set_up(x_init, A, b )
    cgs.max_iteration = 6
#    
    #out.__isub__(B)
    #out2 = K.adjoint(out)
    
    #(2.0*self.c)*self.A.adjoint( self.A.direct(x) - self.b )
    
    for _ in gd:
        print ("iteration {} {}".format(gd.iteration, gd.get_current_loss()))
    
    cg.run(10, lambda it,val: print ("iteration {} objective {}".format(it,val)))
    
    cgs.run(10, lambda it,val: print ("iteration {} objective {}".format(it,val)))
    
    cgsmall.run(10, lambda it,val: print ("iteration {} objective {}".format(it,val)))
    cgsmall.run(10, lambda it,val: print ("iteration {} objective {}".format(it,val)))
#    for _ in cg:
#        print ("iteration {} {}".format(cg.iteration, cg.get_current_loss()))
#    
#    fig = plt.figure()
#    plt.imshow(cg.get_output().get_item(0,0).subset(vertical=0).as_array())
#    plt.title('Composite CGLS')
#    plt.show()
#    
#    for _ in cgs:
#        print ("iteration {} {}".format(cgs.iteration, cgs.get_current_loss()))
#    
    fig = plt.figure()
    plt.subplot(1,5,1)
    plt.imshow(Phantom.subset(vertical=0).as_array())
    plt.title('Simulated Phantom')
    plt.subplot(1,5,2)
    plt.imshow(gd.get_output().subset(vertical=0).as_array())
    plt.title('Simple Gradient Descent')
    plt.subplot(1,5,3)
    plt.imshow(cgs.get_output().subset(vertical=0).as_array())
    plt.title('Simple CGLS')
    plt.subplot(1,5,4)
    plt.imshow(cg.get_output().get_item(0,0).subset(vertical=0).as_array())
    plt.title('Composite CGLS\nbig lambda')
    plt.subplot(1,5,5)
    plt.imshow(cgsmall.get_output().get_item(0,0).subset(vertical=0).as_array())
    plt.title('Composite CGLS\nsmall lambda')
    plt.show()
