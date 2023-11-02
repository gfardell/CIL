# -*- coding: utf-8 -*-
#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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

from cil.framework import AcquisitionData, AcquisitionGeometry, ImageGeometry, ImageData
from .utilities import Tiff_utilities
import os
import glob
import re
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TIFFStackReader(object):

    """
    A TIFF reader. 

    Parameters
    ----------
    file_name : str, list, abspath to folder/file
        The absolute file path to a TIFF file, or a list of absolute file paths to TIFF files. Or an absolute directory path to a folder containing TIFF files.


    deprecated_kwargs
    -----------------

    roi : dictionary, default `None`, pending deprecation
        Use methods `set_image_indices()` and `set_image_roi()` to configure your reader instead.
    
        Dictionary with ROI to load:
        ``{'axis_0': (start, end, step),
            'axis_1': (start, end, step),
            'axis_2': (start, end, step)}``

    mode : str, {'bin', 'slice'}, default 'bin', pending deprecation
        Use `set_image_roi()` to set the 'bin'/'slicing' behaviour.
        In 'bin' mode, N pixels are averaged together. In 'slice' mode 1 in N pixels are used. N is determined by the 'step' parameter in the ROI dictionary.

        
    transpose : tuple, default `None`, deprecated
        This has been deprecated. Please define your geometry accordingly. 

    dtype : numpy type, string, default np.float32
        Requested type of the read image. If set to None it defaults to the type of the saved file. Use read(dtype) instead.


    Notes
    -----
    ROI behaviour (deprecated):
        Files are stacked along ``axis_0``, in alphabetical order.

        ``axis_1`` and ``axis_2`` correspond to row and column dimensions, respectively.

        To skip files or to change the number of files to load, adjust ``axis_0``. For instance, ``'axis_0': (100, 300)``
        will skip the first 100 files and will load 200 files.

        ``'axis_0': -1`` is a shortcut to load all elements along axis 0.

        ``start`` and ``end`` can be specified as ``None`` which is equivalent to ``start = 0`` and ``end = load everything to the end``, respectively.

        Start and end also can be negative.

        ROI is specified for axes before transpose.


    Example
    -------

    Select spcific indices from a TIFF stack and read only them in: 

    >>> reader = TIFFStackReader(file_name = '/path/to/folder')
    >>> reader.get_image_list()
    ['/path/to/folder/0001.tif', '/path/to/folder/0002.tif', '/path/to/folder/0003.tif', ...]
    >>> reader.set_image_indices([0, 1, 2, 8, 9])
    >>> data = reader.read()

    Select a region of interest (ROI) from a TIFF and read it:

    >>> reader = TIFFStackReader(file_name = '/path/to/file.tiff')
    >>> reader.set_image_roi(height = (10, -10, 1), width = (0, 100, 1))
    >>> data = reader.read()

    Read in a TIFF stack with binning factor 2:

    >>> reader = TIFFStackReader(file_name = '/path/to/folder')
    >>> reader.set_image_roi(height = (None, None, 2), width = (None, None, 2), mode = 'bin')
    >>> data = reader.read()

    Read in 1 in 10 images in a TIFF stack:

    >>> reader = TIFFStackReader(file_name = '/path/to/folder')
    >>> reader.set_image_indices((None, None, 10))
    >>> data = reader.read()

    You can rescale the read data as `rescaled_data = (read_data - offset)/scale` with the following code:

    >>> reader = TIFFStackReader(file_name = '/path/to/folder')
    >>> rescaled_data = reader.read_rescaled(scale, offset)

    Alternatively, if TIFFWriter has been used to save data with lossy compression, then you can rescale the
    read data to approximately the original data with the following code:

    >>> writer = TIFFWriter(file_name = '/path/to/folder', compression='uint8')
    >>> writer.write(original_data)
    >>> reader = TIFFStackReader(file_name = '/path/to/folder')
    >>> about_original_data = reader.read_rescaled()
    """


    def _deprecated_kwargs(self, deprecated_kwargs):
        """
        Handle deprecated keyword arguments for backward compatibility.

        Parameters
        ----------
        deprecated_kwargs : dict
            Dictionary of deprecated keyword arguments.

        Notes
        -----
        This method is called by the __init__ method.

        `mode` and `roi` will be removed in the future. If passed they will be used to configure the reader.

        `transpose` has been removed. Please define your geometry accordingly.

        `dtype` has been removed. Please use the dtype argument in the read method instead.
        """

        mode = 'bin'
        if deprecated_kwargs.get('mode', False):
            logging.warning("Input argument `mode` has been deprecated. Please define binning/slicing with method 'set_image_roi()' instead")
            mode = deprecated_kwargs.pop('mode')

        if deprecated_kwargs.get('roi', False):
            logging.warning("Input argument `roi` has been deprecated. Please use methods 'set_image_roi()' and 'set_frame_indices()' instead")
            roi = deprecated_kwargs.pop('roi')
            self.set_image_indices(roi.get('axis_0'), mode)
            self.set_image_roi(roi.get('axis_1'), roi.get('axis_2'), mode)

        if deprecated_kwargs.pop('transpose', None) is not None:
            raise ValueError("Input argument `transpose` has been deprecated. Please define your geometry accordingly")

        if deprecated_kwargs.pop('dtype', None) is not None:
            raise ValueError("Input argument `dtype` has been deprecated. Please use the dtype argument in the read method instead.")

        if deprecated_kwargs:
            logging.warning("Additional keyword arguments passed but not used: {}".format(deprecated_kwargs))


    def __init__(self, file_name, **deprecated_kwargs):

        if (Tiff_utilities.pilAvailable == False):
            raise Exception("PIL (pillow) is not available, cannot load TIFF files.")

        self.set_up(file_name = file_name,
                     **deprecated_kwargs)


    def set_up(self,
               file_name,
               **deprecated_kwargs):
        """
        Sets up the TIFFReader object.

        Parameters
        ----------
        file_name : str, abspath to folder, list
            Path to folder with TIFF files, list of paths of TIFFs, or path to single TIFF file

        """

        self._deprecated_kwargs(deprecated_kwargs)

        # find all tiff files
        if isinstance(file_name, list):
            self._tiff_files = file_name
        elif os.path.isfile(file_name):
            self._tiff_files = [file_name]
        elif os.path.isdir(file_name):
            self._tiff_files = glob.glob(os.path.join(glob.escape(file_name),"*.tif"))

            if not self._tiff_files:
                self._tiff_files = glob.glob(os.path.join(glob.escape(file_name),"*.tiff"))

            if not self._tiff_files:
                raise Exception("No tiff files were found in the directory \n{}".format(file_name))
        else:
            raise Exception("file_name expects a tiff file, a list of tiffs, or a directory containing tiffs.\n{}".format(file_name))

        # check all matched file paths are valid
        for i, fn in enumerate(self._tiff_files):
            if '.tif' in fn:
                if not(os.path.exists(fn)):
                    raise Exception('File \n {}\n does not exist.'.format(fn))
                self._tiff_files[i] = os.path.abspath(fn)
            else:
                raise Exception("file_name expects a tiff file, a list of tiffs, or a directory containing tiffs.\n{}".format(file_name))

        # sort the files alphabetically
        self._tiff_files.sort(key=self.__natural_keys)

        # use the first image to determine the image parameters
        self._image_param_in = Tiff_utilities.get_dataset_metadata(self._tiff_files[0])
        num_images = len(self._tiff_files)
        self._image_param_in['num_images'] = num_images

        self._shape_in = [num_images, self._image_param_in['height'], self._image_param_in['width']]


        # set up the default ROI and indices
        self._roi_image = {'height':slice(None),'width':slice(None),'method_downsample':None}
        self._indices_stack = {'images':range(num_images),'method_downsample':'slice'}
        self._shape_out = self._shape_in.copy()


    @property
    def file_name(self):
        """Returns the name of the TIFF file being read."""

        return self._file_name


    def get_image_list(self):
        """
        Returns an ordered list of TIFF files that have been found by the reader.

        Returns:
            list: An ordered list of TIFF files.
        """
        return self._tiff_files.copy()


    def reset(self):
        """
        Resets the region of interest (ROI) to the full image, and the indices to all tiffs.
        """
        self.set_image_roi()
        self.set_image_indices()
        self._shape_out = self._shape_in.copy()


    def _parse_slice(self, roi, axis):
        """
        Given a roi object returns a slice object

        axis = 'width' or 'height'

        Parameters
        ----------
        roi : slice, tuple, int, None
            The roi object to be used.

        Returns
        -------
        slice
            The slice object to be used.
        """

        if roi == -1 or roi is None:
            return slice(None)
        elif isinstance(roi,tuple):
            return slice(*roi)
        elif isinstance(roi, int):
            return slice(int(roi),int(roi)+1,1)
        elif isinstance(roi ,slice):
            return roi

        raise ValueError("Cannot intepret roi as slice object for axis {}".format(axis))


    def set_image_roi(self, height=None, width=None,  mode=None):
        """
        Sets the region of interest (ROI) for the image.

        Parameters
        ----------

        height : slice, tuple, int optional
            The height of the ROI.
        width : slice, tuple, int, optional
            The width of the ROI.
        mode : str, optional
            The downsampling mode to use. Can be 'bin', 'slice', or None.
        """

        if mode not in [None, 'bin', 'slice']:
            raise ValueError("Wrong mode, None, 'bin' or 'slice' is expected, got {}.".format(mode))

        self._roi_image['height'] = self._parse_slice(height,'height')
        self._roi_image['width'] = self._parse_slice(width,'width')


        if self._roi_image['height'] != slice(None) or self._roi_image['width'] != slice(None):
            if (self._roi_image['height'].step == 1 or self._roi_image['height'].step is None) and (self._roi_image['width'].step == 1 or self._roi_image['width'].step is None):
                mode = 'crop'

        self._roi_image['method_downsample'] = mode
        self._shape_out[1] = self._get_axis_length(self._roi_image['height'],'height', self._roi_image['method_downsample'])
        self._shape_out[2] = self._get_axis_length(self._roi_image['width'],'width', self._roi_image['method_downsample'])


    def _get_axis_length(self, roi_slice, axis, mode):
        """
        Given a slice object returns the axis length

        axis = 'width' or 'height'

        Parameters
        ----------
        roi_slice : slice
            The slice object to be used.
        axis : str
            The axis for which the length is being calculated.
        mode : str
            The downsampling mode to use. Can be 'bin' or 'slice'.
            
        Returns
        -------
        int
            The length of the axis.

        """
        length = self._image_param_in[axis]
        axis_range = range(length)[roi_slice]

        if mode is None or mode == 'crop':
            axis_length = int(axis_range.stop - axis_range.start)
        elif mode == 'slice':
            axis_length = int(np.ceil((axis_range.stop - axis_range.start) / axis_range.step))
        elif mode == 'bin':
            axis_length = int(np.ceil((axis_range.stop - axis_range.start) / axis_range.step))
        else:
            raise ValueError("Nope")

        return axis_length


    def set_image_indices(self, indices=None, mode='slice'):
        """
        Method to configure the image indices to be read. This will be applied to the list retrieved with `get_image_list`.

        Parameters
        ----------

        indices: int, tuple, list, optional
            Takes an integer for a single image, a tuple of (start, stop, step), or a list of image indices. If None, all images will be read.
        mode : str, optional
            The downsampling mode to use. Can be 'bin' or 'slice'. If 'bin' indices bust be specified as (start, stop, step). 
        """

        if isinstance(indices, (list, np.ndarray)):
            try:
                indices = np.arange(self._image_param_in['num_images']).take(indices)
            except IndexError:
                raise ValueError("Index out of range")
            
            if mode == 'bin':
                raise ValueError("Cannot use binning with a list of indices. Please use a tuple instead")
            
            num_images_out = len(indices)
        else:
            indices = self._parse_slice(indices,'images')
            num_images_out = self._get_axis_length(indices,'num_images', mode)
            indices = range(self._image_param_in['num_images'])[indices]


        if num_images_out < 1:
            raise ValueError("No frames selected. Please select at least 1 frame")

        self._indices_stack['images'] = indices
        self._indices_stack['method_downsample'] = mode
        self._shape_out[0] = num_images_out


    def read(self, dtype=np.float32):
        """
        Reads the images and ROI determined by methods `set_image_indices()` an `set_image_roi()` and returns a numpy array.

        Parameters
        ----------

        dtype : numpy type, string, default np.float32
            Requested type of the read image. If set to None it defaults to the type of the saved file.
        """
        indices = self._indices_stack['images']
        image_height_slice = self._roi_image['height']
        image_width_slice = self._roi_image['width']


        if dtype is None:
            dtype =  self._image_param_in['dtype']

        if self._roi_image['method_downsample'] is None:
            crop_box = None
        else:
            crop_box = [0, 0 , self._image_param_in['width'], self._image_param_in['height']]
           
            if image_height_slice is not None:
                axis_range_h = range(self._image_param_in['height'])[image_height_slice]
                crop_box[1] = axis_range_h.start
                crop_box[3] = axis_range_h.start + self._shape_out[1] * axis_range_h.step

            if image_width_slice is not None:
                axis_range_w = range(self._image_param_in['width'])[image_width_slice]
                crop_box[0] = axis_range_w.start
                crop_box[2] = axis_range_w.start + self._shape_out[2] * axis_range_w.step

        # create empty data container for downsized array
        array_full = np.empty(self._shape_out, dtype=dtype)
        if self._indices_stack['method_downsample'] == 'slice':
            count = 0
            for i in indices:
                Tiff_utilities.read_to(self._tiff_files[i], array_full, np.s_[count:count+1,:,:], crop_box, self._roi_image['method_downsample'], self._shape_out[1::])
                count+=1
        else:

            array_single = np.empty((1,*self._shape_out[1::]), dtype=dtype)

            excess = len(indices) % self._indices_stack['images'].step

            ind_empty = np.s_[:,:,:]
            for i in range(self._shape_out[0]-1):

                ind = i*self._indices_stack['images'].step

                Tiff_utilities.read_to(self._tiff_files[ind], array_full, np.s_[i,:,:], crop_box, self._roi_image['method_downsample'], self._shape_out[1::])

                for j in range(1, self._indices_stack['images'].step):
                    # accumulates the images
                    Tiff_utilities.read_to(self._tiff_files[ind+j], array_single,ind_empty, crop_box, self._roi_image['method_downsample'], self._shape_out[1::])
                    array_full[i] +=array_single[0]

            np.divide(array_full, self._indices_stack['images'].step, out=array_full)

            i+=1
            ind = i*self._indices_stack['images'].step


            Tiff_utilities.read_to(self._tiff_files[ind], array_full, np.s_[i,:,:], crop_box, self._roi_image['method_downsample'], self._shape_out[1::])

            for j in range(1, excess):
                # accumulates the images
                Tiff_utilities.read_to(self._tiff_files[ind+j], array_single, ind_empty, crop_box, self._roi_image['method_downsample'], self._shape_out[1::])
                array_full[i] +=array_single[0]

            array_full[i] /=excess
            
        return np.squeeze(array_full)


    def __atoi(self, text):
        return int(text) if text.isdigit() else text


    def __natural_keys(self, text):
        '''
        https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [self.__atoi(c) for c in re.split(r'(\d+)', text) ]


    def _read_as(self, geometry):
        '''reads the data as an ImageData or AcquisitionData with the provided geometry'''

        data = self.read(geometry.dtype)


        try:
            data.shape = geometry.shape
        except AssertionError:
            raise ValueError('data {} and requested {} shapes are not compatible'\
                    .format(data.shape, geometry.shape))

        if data.shape != geometry.shape:
            raise ValueError('Requested {} shape is incompatible with data. Expected {}, got {}'\
                .format(geometry.__class__.__name__, data.shape, geometry.shape))

        if self.dimension_labels != geometry.dimension_labels:
            raise ValueError('Requested geometry is ordered differently to dataset. Expected {}, got {}'\
                .format(self.dimension_labels, geometry.shape))

        return self._return_appropriate_data(data, geometry.dimension_labels)


    def _return_appropriate_data(self, data, geometry):
        if isinstance (geometry, ImageGeometry):
            return ImageData(data, deep=False, geometry=geometry.copy(), suppress_warning=True)
        elif isinstance (geometry, AcquisitionGeometry):
            return AcquisitionData(data, deep=False, geometry=geometry.copy(), suppress_warning=True)
        else:
            raise TypeError("Unsupported Geometry type. Expected ImageGeometry or AcquisitionGeometry, got {}"\
                .format(type(geometry)))


    def read_as_ImageData(self, image_geometry):
        '''reads the TIFF stack as an ImageData with the provided geometry
        
        Notice that the data will be reshaped to what requested in the geometry but there is 
        no warranty that the data will be read in the right order! 
        In facts you can reshape a (2,3,4) array as (3,4,2), however we do not check if the reshape
        leads to sensible data.
        '''
        return self._read_as(image_geometry)


    def read_as_AcquisitionData(self, acquisition_geometry):
        '''reads the TIFF stack as an AcquisitionData with the provided geometry
        
        Notice that the data will be reshaped to what requested in the geometry but there is 
        no warranty that the data will be read in the right order! 
        In facts you can reshape a (2,3,4) array as (3,4,2), however we do not check if the reshape
        leads to sensible data.
        '''
        return self._read_as(acquisition_geometry)


    def read_scale_offset(self):
        '''Reads the scale and offset from a json file in the same folder as the tiff stack
        
        This is a courtesy method that will work only if the tiff stack is saved with the TIFFWriter

        Returns:
        --------

        tuple: (scale, offset)
        '''
        # load first image to find out dimensions and type
        path = os.path.dirname(self._tiff_files[0])
        with open(os.path.join(path, "scaleoffset.json"), 'r') as f:
            d = json.load(f)

        return (d['scale'], d['offset'])


    def read_rescaled(self, scale=None, offset=None):
        '''Reads the TIFF stack and rescales it with the provided scale and offset, or with the ones in the json file if not provided
        
        This is a courtesy method that will work only if the tiff stack is saved with the TIFFWriter

        Parameters:
        -----------

        scale: float, default None
            scale to apply to the data. If None, the scale will be read from the json file saved by TIFFWriter.
        offset: float, default None
            offset to apply to the data. If None, the offset will be read from the json file saved by TIFFWriter.

        Returns:
        --------

        numpy.ndarray in float32
        '''
        data = self.read(dtype=np.float32)
        if scale is None or offset is None:
            scale, offset = self.read_scale_offset()
        data -= offset
        data /= scale
        return data
