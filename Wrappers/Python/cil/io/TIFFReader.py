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

from cil.framework import AcquisitionData, AcquisitionGeometry, ImageGeometry, ImageData, DataContainer
from .utilities import Tiff_utilities
import os
import glob
import re
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TIFFReader(object):

    """
    A TIFF reader. 

    Parameters
    ----------
    file_name : str, list, abspath to folder/file
        The absolute file path to a TIFF file, or a list of absolute file paths to TIFF files. Or an absolute directory path to a folder containing TIFF files.

    dimension_labels : tuple of strings
        The labels of the dimensions of the TIFF file(s) being read.
        The data will be structured as 'images', 'image height', 'image width' so the dimension labels should be in this order.

    dtype : numpy.dtype, default np.float32
        The data type returned with 'read'


    deprecated_kwargs
    -----------------

    roi : dictionary, default `None`, deprecated
        This has been deprecated. Use proccessor `Slicer` or `Binner` instead (see examples below).

    mode : str, {'bin', 'slice'}, default 'bin', deprecated
        This has been deprecated. Use proccessor `Slicer` or `Binner` instead (see examples below).
        
    transpose : tuple, default `None`, deprecated
        This has been deprecated. Please define your geometry accordingly. 

        
    Example
    -------

    To read a TIFF stack, use the following code:

    >>> reader = TIFFStackReader(file_name = '/path/to/folder', dimension_labels=('vertical','horizontal_y','horizontal_x')
    >>> reader.get_image_list()
    ['/path/to/folder/0001.tif', '/path/to/folder/0002.tif', '/path/to/folder/0003.tif', ...]
    >>> data = reader.read()

    
    To read as subset of you data use this reader with the processor `Slicer` as follows:

    >>> reader = TIFFStackReader(file_name = '/path/to/folder', dimension_labels=('angles','vertical','horizontal')
    >>> roi = {'angles':(None,None,10), 'vertical':(100), 'horizontal':(50,-50)}
    >>> slicer = Slicer(roi)
    >>> slicer.set_input(reader)
    >>> data = slicer.get_output()

        
    To return an ImageData, use the following code:

    >>> reader = TIFFStackReader(file_name = '/path/to/folder')
    >>> reader.get_image_list()
    ['/path/to/folder/0001.tif', '/path/to/folder/0002.tif', '/path/to/folder/0003.tif', ...]
    >>> data = reader.read_as_ImageData(image_geometry)

    
    You can rescale the read data as `rescaled_data = (read_data - offset)/scale` with the following code:

    >>> reader = TIFFStackReader(file_name = '/path/to/folder')
    >>> rescaled_data = reader.read_rescaled(scale, offset)

    
    Alternatively, if TIFFWriter has been used to save data with lossy compression, then you can rescale the
    read data to approximately the original data with the following code:

    >>> writer = TIFFWriter(file_name = '/path/to/folder', compression='uint8')
    >>> writer.write(original_data)
    >>> reader = TIFFStackReader(file_name = '/path/to/folder')
    >>> about_original_data = reader.read_rescaled()


    Notes
    -----

    When reading a TIFF stack, the reader will sort the TIFF files found in the folder in to natural sort order.
    This will sort first alphabetically, then numerically leading to the sorting: 1.tif, 2.tif, 3.tif, 10.tif, 11.tif
    The users can see the order with the method `get_image_list()`. The user can also pass an ordered list of TIFF
    files to the reader, in which case the user's order will be used.

    """


    class data(object):
        
        """
        spoof array in reader
        """
        def __init__(self, shape, dtype, read_function):
            self._shape = shape
            self._dtype = dtype
            self._read_slice = read_function

        @property
        def ndim(self):
            return 3 if self.shape[0] > 1 else 2
        
        @property
        def shape(self):
            return self._shape
        
        @property
        def dtype(self):
            return self._dtype
        
        @property
        def size(self):
            return np.prod(self.shape)
        
        
        def __getitem__(self, key):

            if isinstance(key, tuple):
                len_key = len(key)

                list_slices = [item if isinstance(item, slice) else slice(item, item+1) for item in key]

                if len_key == 2 and self.ndim == 2:
                    return self._read_slice(images_slice=None, height_slice=list_slices[0], width_slice=list_slices[1])
                elif len_key == 3 and self.ndim == 3:
                    return self._read_slice(images_slice=list_slices[0], height_slice=list_slices[1], width_slice=list_slices[2])
                else:
                    raise ValueError("Expected a tuple of ndim slices or ints, expected length {} got length {}".format(self.ndim, len_key))
                
            else:
                raise ValueError("Expected a tuple of ndim slices or ints, got key as {}".format(type(key)))
            
            
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
        """

        if deprecated_kwargs.get('mode', False):
            raise ValueError("Input argument `mode` has been deprecated. Please use processors 'Binner' or 'Slicer' instead")

        if deprecated_kwargs.get('roi', False):
           raise ValueError("Input argument `roi` has been deprecated. Please use processors 'Binner' or 'Slicer' instead")

        if deprecated_kwargs.pop('transpose', None) is not None:
            raise ValueError("Input argument `transpose` has been deprecated. Please define your geometry accordingly")

        if deprecated_kwargs:
            logging.warning("Additional keyword arguments passed but not used: {}".format(deprecated_kwargs))


    def __init__(self, file_name, dimension_labels, dtype=np.float32, **deprecated_kwargs):

        if (Tiff_utilities.pilAvailable == False):
            raise Exception("PIL (pillow) is not available, cannot load TIFF files.")

        self.set_up(file_name = file_name, dimension_labels=dimension_labels, dtype=dtype,
                     **deprecated_kwargs)


    def set_up(self,
               file_name,
               dimension_labels,
               dtype=np.float32,
               **deprecated_kwargs):
        """
        Reconfigures the TIFFReader object.

        Parameters
        ----------
        file_name : str, list, abspath to folder/file
            The absolute file path to a TIFF file, or a list of absolute file paths to TIFF files. Or an absolute directory path to a folder containing TIFF files.

        dimension_labels : tuple of strings
            The labels of the dimensions of the TIFF file(s) being read.
            The data will be structured as 'images', 'image height', 'image width' so the dimension labels should be in this order.

        dtype : numpy.dtype, default np.float32
            The data type returned with 'read'

        """

        self._deprecated_kwargs(deprecated_kwargs)

        # find all tiff files
        sorted = False
        if isinstance(file_name, list):
            self._tiff_files = file_name
            sorted = True
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
        if not sorted:
            self._tiff_files.sort(key=self.__natural_keys)

        # use the first image to determine the image parameters
        image_param_in = Tiff_utilities.get_dataset_metadata(self._tiff_files[0])
        num_images = len(self._tiff_files)

        dtype = image_param_in['dtype'] if dtype is None else dtype
        shape = [num_images, image_param_in['height'], image_param_in['width']]

        self._dimension_labels = dimension_labels 
        self._array = self.data(shape, dtype, self._read_slice)


    @property
    def array(self):
        """Returns the array-like object of the TIFF file(s) being read. Elements can be accessed via numpy advanced indexing. Slicing order is defined by the dimension labels"""

        return self._array
        
    
    @property
    def ndim(self):
        """Returns the number of dimensions of the TIFF file(s) being read."""

        return self.array.ndim
    
    @property
    def shape(self):
        """Returns the shape of the TIFF file(s) being read."""

        return self.array.shape

    @property
    def dtype(self):
        """Returns the data type of the TIFF file(s) being read."""

        return self.array.dtype
    
    @property
    def dimension_labels(self):
        """Returns the dimension labels of the TIFF file(s) being read."""

        return self._dimension_labels
    

    def get_image_list(self):
        """
        Returns an ordered list of TIFF files that have been found by the reader.

        Returns:
            list: An ordered list of TIFF files.
        """
        return self._tiff_files.copy()


    def read(self):
        """
        Reads the images and returns a datacontainer. Dimension labels and dtype are configured by the reader.

        Returns
        -------
        DataContainer: The read data
        """

        # create empty data container for the array
        array_full = np.empty(self.shape, dtype=self.dtype)
        image_shape_PIL = (self.shape[2], self.shape[1])

        for i in range(self.shape[0]):
            Tiff_utilities.read_to(self._tiff_files[i], array_full, image_shape_PIL, np.s_[i,:,:])
    
        array_full = np.squeeze(array_full)
        return DataContainer(array_full, dimension_labels=self.dimension_labels)



    def _read_slice(self, images_slice=None, height_slice=None, width_slice=None):
        """
        Reads the ROI of an image and returns as a numpy array. This is used by `array.__getitem__` to allow numpy advanced indexing.
        The dtype is configured by the reader.

        Parameters
        ----------
        images_slice : slice, default None
            Slice defining images to be read. If None, all images are read. This will be applied to the list retrieved with `get_image_list`
        height_slice : slice, default None
            Slice defining ROI of image height. Applied to all images.
        width_slice : slice, default None
            Slice defining ROI of image width. Applied to all images.

        Returns
        -------
        numpy narray: The read data

        """

        shape_out = self.shape.copy()
        crop_box = [0, 0, self.shape[2], self.shape[1]]

        if images_slice is not None:
            axis_range_images = range(shape_out[0])[images_slice]
            shape_out[0] = len(axis_range_images)
        else:
            axis_range_images = range(shape_out[0])

        if height_slice is not None:
            axis_range = range(shape_out[1])[height_slice]
            shape_out[1] = len(axis_range)
            crop_box[1] = axis_range.start
            crop_box[3] = axis_range.start + axis_range.step * (shape_out[1]-1) +1

        if width_slice is not None:
            axis_range = range(shape_out[2])[width_slice]
            shape_out[2] = len(axis_range)
            crop_box[0] = axis_range.start
            crop_box[2] = axis_range.start + axis_range.step * (shape_out[2]-1) +1


        # PIL specific set up
        shape_PIL_image = (shape_out[2], shape_out[1])

        # create empty data container for the array
        array_full = np.empty(shape_out, dtype=self.dtype)

        ind_out = 0
        for i in axis_range_images:
            Tiff_utilities.read_to(self._tiff_files[i], array_full, shape_PIL_image, np.s_[ind_out,:,:], crop_box)
            ind_out +=1

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
