# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from cil.framework import AcquisitionData, ImageGeometry, DataOrder
import importlib
import weakref
import logging

class Reconstructor(object):
    
    """ Abstract class representing a reconstructor 
    """

    supported_backends = ['tigre']
    
    #_input is a weakreference object
    @property
    def input(self):
        if self._input() is None:
            raise ValueError("Input has been deallocated")
        else:
            return self._input()


    @property
    def acquisition_geometry(self):
        return self._acquisition_geometry


    @property
    def image_geometry(self):
        return self._image_geometry


    @property
    def backend(self):
        return self._backend


    def __init__(self, input, image_geometry=None, backend='tigre'):


        if not issubclass(type(input), AcquisitionData):
            raise TypeError("Input type mismatch: got {0} expecting {1}"
                            .format(type(input), AcquisitionData))

        self._acquisition_geometry = input.geometry.copy()
        self._configure_for_backend(backend)
        self.set_image_geometry(image_geometry)
        self.set_input(input)


    def set_input(self, input):
        """
        Update the input data to run the reconstructor on. The geometry of the dataset must be compatible with the reconstructor.

        Parameters
        ----------
        input : AcquisitionData
            A dataset with a compatible geometry

        """
        if input.geometry != self.acquisition_geometry:
            raise ValueError ("Input not compatible with configured reconstructor. Initialise a new reconstructor with this geometry")
        else:
            self._input = weakref.ref(input)


    def set_image_geometry(self, image_geometry=None):
        """
        Sets a custom image geometry to be used by the reconstructor

        Parameters
        ----------
        image_geometry : ImageGeometry, default used if None
            A description of the area/volume to reconstruct
        """
        if image_geometry is None:
            self._image_geometry = self.acquisition_geometry.get_ImageGeometry()
        elif issubclass(type(image_geometry), ImageGeometry):
            self._image_geometry = image_geometry.copy()
        else:
            raise TypeError("ImageGeometry type mismatch: got {0} expecting {1}"\
                                .format(type(input), ImageGeometry))   
           

    def _configure_for_backend(self, backend='tigre'):
        """
        Configures the class for the right engine. Checks the dataorder.
        """        
        if backend not in self.supported_backends:
            raise ValueError("Backend unsupported. Supported backends: {}".format(self.supported_backends))

        if not DataOrder.check_order_for_engine(backend, self.acquisition_geometry):
            raise ValueError("Input data must be reordered for use with selected backend. Use input.reorder{'{0}')".format(backend))

        #set ProjectionOperator class from backend
        try:
            module = importlib.import_module('cil.plugins.'+backend)
        except ImportError:
            if backend == 'tigre':
                raise ImportError("Cannot import the {} plugin module. Please install TIGRE or select a different backend".format(self.backend))
            if backend == 'astra':
                raise ImportError("Cannot import the {} plugin module. Please install CIL-ASTRA or select a different backend".format(self.backend))

        self._PO_class = module.ProjectionOperator
        self._backend = backend


    def reset(self):
        """
        Resets all optional configuration parameters to their default values
        """
        raise NotImplementedError()


    def run(self, out=None, verbose=1):
        """
        Runs the configured recon and returns the reconstruction

        Parameters
        ----------
        out : ImageData, optional
           Fills the referenced ImageData with the reconstructed volume and suppresses the return
        
        verbose : int, default=1
           Contols the verbosity of the reconstructor. 0: No output is logged, 1: Full configuration is logged

        Returns
        -------
        ImageData
            The reconstructed volume. Suppressed if `out` is passed
        """

        raise NotImplementedError()


    def _str_data_size(self):

        repres = "\nInput Data:\n"
        for dim in  zip(self.acquisition_geometry.dimension_labels,self.acquisition_geometry.shape):
            repres += "\t" + str(dim[0]) + ': ' + str(dim[1])+'\n'

        repres += "\nReconstruction Volume:\n"
        for dim in zip(self.image_geometry.dimension_labels,self.image_geometry.shape):
            repres += "\t" + str(dim[0]) + ': ' + str(dim[1]) +'\n'

        return repres


class IterativeReconstructor(Reconstructor):
    
    """ Abstract class representing an iterative reconstructor 
    """

    @property
    def alpha(self):
        return self._alpha


    def __init__ (self, input, image_geometry=None, backend='tigre'):

        super().__init__(input, image_geometry, backend)

        self._device='gpu'
        self._algorithm = None
        self._alpha = 1.0
        self.set_attenuation_window()


    def set_input(self, input):
        super().set_input(input)


    def set_device(self,device='gpu'):
        '''
        Run on GPU or CPU
        '''
        if device == 'cpu':
            if self.backend == 'astra': #and parallel?
                self._device='cpu'
            else:
                print("cannot")
                self._device='gpu'

        self.configure_operators()


    def set_alpha(self,alpha=1):
        '''
        Sets the ratio between the data fidelity and the regularistion
        '''
        self._alpha=float(alpha)
        self._algorithm = None


    def set_initial(self, initial=None):
        '''
        can be image, or string 'FBP'
        '''
        self.initial = ImageGeometry.allocate(0)
        self._algorithm = None


    def set_image_geometry(self, image_geometry=None):

        self._algorithm = None
        super().set_image_geometry(image_geometry)
        if not hasattr(self,'_Op_proj') or self._Op_proj.domain_geometry != self.image_geometry:
            self.configure_operators()


    def configure_operators(self):

        if self.backend=='tigre':
            self._Op_proj = self._PO_class(self.image_geometry, self.acquisition_geometry)
        else:
            self._Op_proj = self._PO_class(self.image_geometry, self.acquisition_geometry, device=self._device)


    def reset_algorithm(self):
        """
        Resets the algorithm to it's initial state
        """
        self._algorithm = None
        self._algorithm = self._initialise_algorithm()
        print("resetting the algorithm to it's initial state")


    def reset(self):
        self.set_alpha()
        self.set_device()
        self.set_image_geometry()
        self.set_initial()


    def _initialise_algorithm(self):
        raise NotImplementedError()


    def run(self, iterations=1, verbose=1):
        """
        Runs the configured reconstructor and returns the reconstruction

        Parameters
        ----------
        verbose : int, default=1
           Contols the verbosity of the reconstructor. 0: No output is logged, 1: Full configuration is logged

        Returns
        -------
        ImageData
            The reconstructed volume. Suppressed if `out` is passed
        """

        if verbose:
            print(self)

        if self._algorithm is None:
            self._initialise_algorithm()

        self._algorithm.run(iterations)

        return self._algorithm.get_output()


    def _str_options(self):
           
        repres += "\nReconstruction Options:\n"
        repres += "\tBackend: {}\n".format(self._backend)             
        repres += "\tDevice: {}\n".format(self._device)             
        repres += "\tAlpha: {}\n".format(self._alpha)

        return repres   