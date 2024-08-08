#  Copyright 2021 United Kingdom Research and Innovation
#  Copyright 2021 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from cil.framework import Processor, AcquisitionData, DataOrder
from .Binner import Binner
from .Slicer import Slicer

import matplotlib.pyplot as plt
import scipy
import numpy as np
import logging
import math
import importlib

log = logging.getLogger(__name__)

class CofR_image_sharpness(Processor):

    """This creates a CentreOfRotationCorrector processor.

    The processor will find the centre offset by maximising the sharpness of a reconstructed slice.

    Can be used on single slice parallel-beam, and centre slice cone beam geometry. For use only with datasets that can be reconstructed with FBP/FDK.

    Parameters
    ----------

    slice_index : int, str, default='centre'
        An integer defining the vertical slice to run the algorithm on. The special case slice 'centre' is the default.

    backend : {'tigre', 'astra'}
        The backend to use for the reconstruction

    tolerance : float, default=0.005
        The tolerance of the fit in pixels, the default is 1/200 of a pixel. This is a stopping criteria, not a statement of accuracy of the algorithm.

    search_range : int
        The range in pixels to search either side of the panel centre. If `None` a quarter of the width of the panel is used.

    initial_binning : int
        The size of the bins for the initial search. If `None` will bin the image to a step corresponding to <128 pixels. The fine search will be on unbinned data.


    Example
    -------
    from cil.processors import CentreOfRotationCorrector

    processor = CentreOfRotationCorrector.image_sharpness('centre', 'tigre')
    processor.set_input(data)
    data_centred = processor.get_output()


    Example
    -------
    from cil.processors import CentreOfRotationCorrector

    processor = CentreOfRotationCorrector.image_sharpness(slice_index=120, 'astra')
    processor.set_input(data)
    processor.get_output(out=data)


    Note
    ----
    For best results data should be 360deg which leads to blurring with incorrect geometry.
    This method is unreliable on half-scan data with 'tuning-fork' style artifacts.

    """
    _supported_backends = ['astra', 'tigre']

    def __init__(self, slice_index='centre', backend='tigre', tolerance=0.005, search_range=None, initial_binning=None):

        FBP = self._configure_FBP(backend)


        kwargs = {
                    'slice_index': slice_index,
                    'FBP': FBP,
                    'backend' : backend,
                    'tolerance': tolerance,
                    'search_range': search_range,
                    'initial_binning': initial_binning,
                    '_validated_indices' : []
                 }

        super(CofR_image_sharpness, self).__init__(**kwargs)

    def check_input(self, data):
        if not isinstance(data, AcquisitionData):
            raise Exception('Processor supports only AcquisitionData')

        if data.geometry == None:
            raise Exception('Geometry is not defined.')

        if data.geometry.system_description not in ['simple','offset','advanced']:
            raise NotImplementedError("Geometry not supported")

        if data.geometry.channels > 1:
            raise ValueError("Only single channel data is supported with this algorithm")

        if self.slice_index == 'centre' or isinstance(self.slice_index, int):
            self.slice_index = [self.slice_index]

        if isinstance(self.slice_index, (list, tuple)) and len(self.slice_index) <= 2:
            
            # Validate each slice index
            for index in self.slice_index:
                if index == 'centre':
                    index = (data.get_dimension_size('vertical')-1) /2
                    self._validated_indices.append(index)
                else:
                    try:
                        index = int(index)
                    except:
                        raise ValueError(f"slice_index expected to be a positive integer or the string 'centre'. Got {index}")

                    if index >= 0 and index < data.get_dimension_size('vertical'):
                        self._validated_indices.append(index)
                    else:
                        raise ValueError(f"slice_index is out of range. Must be in range 0-{data.get_dimension_size('vertical')}. Got {index}")
        else:
            raise ValueError("self.slice_index must be a list or tuple of length 2, an integer, or the string 'centre'")

        if not DataOrder.check_order_for_engine(self.backend, data.geometry):
            raise ValueError("Input data must be reordered for use with selected backend. Use input.reorder{'{0}')".format(self.backend))

        return True


    def _configure_FBP(self, backend='tigre'):
        """
        Configures the processor for the right engine. Checks the data order.
        """
        if backend not in self._supported_backends:
            raise ValueError("Backend unsupported. Supported backends: {}".format(self._supported_backends))

        #set FBPOperator class from backend
        try:
            module = importlib.import_module(f'cil.plugins.{backend}')
        except ImportError as exc:
            msg = {'tigre': "TIGRE (e.g. `conda install conda-forge::tigre`)",
                   'astra': "ASTRA (e.g. `conda install astra-toolbox::astra-toolbox`)"}.get(backend, backend)
            raise ImportError(f"Please install {msg} or select a different backend") from exc

        return module.FBP


    def gss(self, data, ig, search_range, tolerance, binning):
        '''Golden section search'''
        # intervals c:cr:c where r = φ − 1=0.619... and c = 1 − r = 0.381..., φ
        log.debug("GSS between %f and %f", *search_range)
        phi = (1 + math.sqrt(5))*0.5
        r = phi - 1
        #1/(r+2)
        r2inv = 1/ (r+2)
        #c = 1 - r

        all_data = {}
        #set up
        sample_points = [np.nan]*4
        evaluation = [np.nan]*4

        sample_points[0] = search_range[0]
        sample_points[3] = search_range[1]

        interval = sample_points[-1] - sample_points[0]
        step_c = interval *r2inv
        sample_points[1] = search_range[0] + step_c
        sample_points[2] = search_range[1] - step_c

        for i in range(4):
            evaluation[i] = self.calculate(data, ig, sample_points[i])
            all_data[sample_points[i]] = evaluation[i]

        count = 0
        while(count < 30):
            ind = np.argmin(evaluation)
            if ind == 1:
                del sample_points[-1]
                del evaluation[-1]

                interval = sample_points[-1] - sample_points[0]
                step_c = interval *r2inv
                new_point = sample_points[0] + step_c

            elif ind == 2:
                del sample_points[0]
                del evaluation[0]

                interval = sample_points[-1] - sample_points[0]
                step_c = interval *r2inv
                new_point = sample_points[-1]- step_c

            else:
                raise ValueError("The centre of rotation could not be located to the requested tolerance. Try increasing the search tolerance.")

            if interval < tolerance:
                break

            sample_points.insert(ind, new_point)
            obj = self.calculate(data, ig, new_point)
            evaluation.insert(ind, obj)
            all_data[new_point] = obj

            count +=1

        log.info("evaluated %d points",len(all_data))
        if log.isEnabledFor(logging.DEBUG):
            keys, values = zip(*all_data.items())
            self.plot(keys, values, ig.voxel_size_x/binning)

        z = np.polyfit(sample_points, evaluation, 2)
        min_point = -z[1] / (2*z[0])

        if np.sign(z[0]) == 1 and min_point < sample_points[2] and min_point > sample_points[0]:
            return min_point
        else:
            ind = np.argmin(evaluation)
            return sample_points[ind]

    def calculate(self, data, ig, offset):
        ag_shift = data.geometry.copy()
        shift = [0]*data.ndim
        shift[0] = offset
        ag_shift.config.system.rotation_axis.position = shift
        reco = self.FBP(ig, ag_shift)(data)
        return (reco*reco).sum()

    def plot(self, offsets,values, vox_size):
        x=[x / vox_size for x in offsets]
        y=values

        plt.figure()
        plt.scatter(x,y)
        plt.show()

    def get_min(self, offsets, values, ind):
        #calculate quadratic from 3 points around ind  (-1,0,1)
        a = (values[ind+1] + values[ind-1] - 2*values[ind]) * 0.5
        b = a + values[ind] - values[ind-1]
        ind_centre = -b / (2*a)+ind

        ind0 = int(ind_centre)
        w1 = ind_centre - ind0
        return (1.0 - w1) * offsets[ind0] + w1 * offsets[ind0+1]


    def _bin_and_filter_data(self, data):

        binner = Binner(roi={'horizontal':(None, None, self.initial_binning)},accelerated=False)
        binner.set_input(data.geometry)
        geom_binned = binner.get_output()
        data_binned = geom_binned.allocate()
        
        if self.backend=='astra':
            geom_temp = data.geometry.get_slice(vertical='centre')
            #astra requires cubic voxels
            data_binned.geometry.pixel_size_v = geom_temp.pixel_size_v * self.initial_binning
        else:
            geom_temp = data.geometry.get_slice(angle=0)

        proj_single = geom_temp.allocate()
        filter_kernel = (data_binned.ndim -1)* [1]
        filter_kernel[-1] = self.initial_binning//2

        for i in range(data.shape[0]):
            proj_single.fill(scipy.ndimage.gaussian_filter(data.array[i], filter_kernel))
            binner.set_input(proj_single)
            proj_binned = binner.get_output()
            np.copyto(data_binned.array[i],proj_binned.array)

        data_binned.fill(scipy.ndimage.sobel(data_binned.as_array(), axis=-1, mode='reflect', cval=0.0))
        return data_binned


    def _coarse_search(self, ig, data):
        # coarse grid search
        vox_rad = np.ceil(self.search_range /self.initial_binning)
        steps = int(4*vox_rad + 1)
        offsets = np.linspace(-vox_rad, vox_rad, steps) * ig.voxel_size_x
        obj_vals = []

        for offset in offsets:
            obj_vals.append(self.calculate(data, ig, offset))

        if log.isEnabledFor(logging.DEBUG):
            self.plot(offsets,obj_vals,ig.voxel_size_x / self.initial_binning)

        ind = np.argmin(obj_vals)
        if ind == 0 or ind == len(obj_vals)-1:
            raise ValueError ("Unable to minimise function within set search_range")
       
        return self.get_min(offsets, obj_vals, ind)

    def process(self, out=None):

        data_in = self.get_input()
        data_in.geometry.config.system.align_reference_frame('cil')
        width = data_in.geometry.config.panel.num_pixels[0]

        #initial search setup
        if self.search_range is None:
            self.search_range = width //4

        if self.initial_binning is None:
            self.initial_binning = min(int(np.ceil(width / 128)),16)

        log.debug("Initial search:")
        log.debug("search range is %d", self.search_range)
        log.debug("initial binning is %d", self.initial_binning)

        num_slices = len(self._validated_indices)
        found_offsets = [None] * num_slices
        use_full_data = False
        try:
            # fails if it can't slice the geometry and will default to single slice reconstructions of the full data
            for slice_index in self._validated_indices:
                data_filtered_list.append(data_in.get_slice(vertical=slice_index))
        except:
            data = data_in.copy()
            # use the same data for all the slices
            data_filtered_list = [data]
            use_full_data = True

        # bin and filter data
        if self.initial_binning > 1:
            data_binned = [self._bin_and_filter_data(data) for data in data_filtered_list]

        # filter unbinned data in place
        for i, data in enumerate(data_filtered_list):
            data_filtered_list[i].fill(scipy.ndimage.sobel(data.array, axis=-1, mode='reflect', cval=0.0))

        # set reference to binned data if exists
        if self.initial_binning > 1:
            data_initial = data_binned
        else:
            data_initial = data_filtered_list
            
        # coarse search for each slice on
        for i, slice_index in enumerate(self._validated_indices):
            log.debug(f"Coarse search for offset in slice {i+1} of {num_slices} on data with binning {self.initial_binning}")

            if len(data_initial) > 1:
                data = data_initial[i]
            else:
                data = data_initial[0]

            ig = data.geometry.get_ImageGeometry()
            if use_full_data:
                ig = Slicer(roi={'vertical':(slice_index, slice_index+1,None)})(ig)

            found_offsets[i] = self._coarse_search(ig, data)
    
            if self.initial_binning > 8:
                # if had a lot of binning then do a second coarse search around the minimum
                log.debug("binned search starting at %f", found_offsets[i])
                a = found_offsets[i] - ig.voxel_size_x *2
                b = found_offsets[i] + ig.voxel_size_x *2
                found_offsets[i] = self.gss(data,ig, (a, b), self.tolerance *ig.voxel_size_x, self.initial_binning )
    

        # fine search on unbinned data
        for i, slice_index in enumerate(self._validated_indices):
            log.debug(f"Fine search for offset in slice {i+1} of {num_slices}")

            if len(data_initial) > 1:
                data = data_filtered_list[i]
            else:
                data = data_filtered_list[0]

            ig = data.geometry.get_ImageGeometry()
            if use_full_data:
                ig = Slicer(roi={'vertical':(slice_index, slice_index+1,None)})(ig)

            a = found_offsets[i] - ig.voxel_size_x *2
            b = found_offsets[i] + ig.voxel_size_x *2
            found_offsets[i] = self.gss(data, ig, (a, b), self.tolerance*ig.voxel_size_x, 1 )
            log.info(f"Offset in slice {slice_index} = {found_offsets[i]} units")
    
        log.info("Centre of rotation correction found using image_sharpness")
        log.info(f"backend FBP/FDK {self.backend}")

        if out is None:
            out = data_in.copy()

        #return found_offsets
        if len(found_offsets) > 1:
        
            slice_pos1 = (self._validated_indices[0]-(data.geometry.pixel_num_v-1)/2) * data.geometry.pixel_size_v / data.geometry.magnification
            slice_pos2 = (self._validated_indices[1]-(data.geometry.pixel_num_v-1)/2) * data.geometry.pixel_size_v / data.geometry.magnification

            x_diff = found_offsets[1] - found_offsets[0]
            y_diff = slice_pos2 - slice_pos1

            offset_x_y0 = found_offsets[0] - slice_pos1 * x_diff/y_diff

            out.geometry.config.system.rotation_axis.position = [offset_x_y0, 0, 0]
            out.geometry.config.system.rotation_axis.direction = [x_diff, 0, y_diff]

        else:
            out.geometry.config.system.rotation_axis.position = [found_offsets[0], 0, 0]

            if out.geometry.dimension == '3D':
                out.geometry.config.system.rotation_axis.direction = [0, 0, 1]

        return out

