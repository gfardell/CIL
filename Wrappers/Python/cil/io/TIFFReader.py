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
from cil.processors import Binner
from cil.io.utilities import Tiff_utilities
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

    rescale : bool, tuple, default False
        If False, the data will be returned as is.
        If a tuple (scale, offset) is provided, the data will be rescaled as `rescaled_data = (read_data - offset)/scale`
        If True, the data will be rescaled by the offset and scale found in the accompanying json file. This is a curtosy method that will onyl work if the data was saved with TIFFWriter.

        
    Example
    -------

    To read a TIFF stack, use the following code:

    >>> reader = TIFFStackReader(file_name = '/path/to/folder', dimension_labels=('vertical','horizontal_y','horizontal_x')
    >>> reader.get_image_list()
    ['/path/to/folder/0001.tif', '/path/to/folder/0002.tif', '/path/to/folder/0003.tif', ...]
    >>> data = reader.read()

        
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


    def __init__(self, file_name, dimension_labels, dtype=np.float32, rescale=False, **deprecated_kwargs):

        if (Tiff_utilities.pilAvailable == False):
            raise Exception("PIL (pillow) is not available, cannot load TIFF files.")

        self.set_up(file_name = file_name, dimension_labels=dimension_labels, dtype=dtype, rescale=rescale,
                     **deprecated_kwargs)


    def set_up(self,
               file_name,
               dimension_labels,
               rescale,
               dtype=np.float32):
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

        rescale : bool, tuple, default False
            If False, the data will be returned as is.
            If a tuple (scale, offset) is provided, the data will be rescaled as `rescaled_data = (read_data - offset)/scale`
            If True, the data will be rescaled by the offset and scale found in the accompanying json file. This is a curtosy method that will onyl work if the data was saved with TIFFWriter.

        """

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

        self._dtype = image_param_in['dtype'] if dtype is None else dtype
        self._shape = [num_images, image_param_in['height'], image_param_in['width']]

        if self._shape[0] >1:
            if len(dimension_labels) == 3:
                self._dimension_labels_full = dimension_labels
            else:
                raise ValueError("dimension_labels must be a tuple of length 3 as reading in multiple Tiffs. Got {}".format(len(dimension_labels)))
        else:
            if len(dimension_labels) == 2:
                self._dimension_labels_full = ['None',*dimension_labels]
            elif len(dimension_labels) == 3:
                self._dimension_labels_full = dimension_labels
                logging.WARNING("dimension_labels is a tuple of length 3 but only 1 Tiff file was found. The first dimension will be set to 'None'")
            else:
                raise ValueError("dimension_labels must be a tuple of length 2. Got {}".format(len(dimension_labels)))
    
        # set up scaling
        if rescale is False:
            self._rescale = False
        else:
            self._rescale = True

            if rescale is True:
                scale, offset = self.read_scale_offset()
            else:
                scale, offset = rescale

            self._rescale_values = (1.0/scale, -offset/scale)

    @property
    def shape(self):
        """Returns the shape of the TIFF file(s) being read."""
        return self._shape
        
    @property
    def dimension_labels(self):
        """Returns the dimension labels of the TIFF file(s) being read."""
        if self._shape[0] > 1:
            return self._dimension_labels_full
        else:
            return self._dimension_labels_full[1::]
    
    @property
    def dtype(self):
        """Returns the data type of the TIFF file(s) being read."""
        return self._dtype
  
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

            if self._rescale:
                array_full[np.s_[i,:,:]] *= self._rescale_values[0]
                array_full[np.s_[i,:,:]] += self._rescale_values[1]

        array_full = array_full.squeeze()
        return DataContainer(array_full, dimension_labels=self.dimension_labels)
    


    def read_binned(self, images_roi=None, height_roi=None, width_roi=None):
        """
    
        Reads the ROI of an image and returns as a `DataContainer`. The dtype is configured by the reader.

        step defines number of pixels to average together.

        Parameters
        ----------
        images_roi : tuple, default None
            Tuple (start, stop, step) defining images to be read. If None, all images are read. This will be applied to the list retrieved with `get_image_list`
        height_roi : tuple, default None
            Tuple (start, stop, step) defining ROI of image height. Applied to all images.
        width_roi : tuple, default None
            Tuple (start, stop, step) defining ROI of image width. Applied to all images.

        Returns
        -------
        DataContainer: The read data

        """

        shape_out = self.shape.copy()
        shape_PIL_image = [*shape_out[1::]]

        crop_box = [0, 0, self.shape[2], self.shape[1]]
        dimension_labels_keep = [1]*len(self._dimension_labels_full)
        
        roi = {}
        if images_roi is not None:
            if self.shape[0] > 1:
                axis_range = range(shape_out[0])[images_roi]
                N_images = axis_range.step

                length = (axis_range.stop - axis_range.start) // axis_range.step
                start = axis_range.start

                roi[self.dimension_labels[0]] = (0, axis_range.step, axis_range.step)
                shape_out[0] = length

                images_range = range(axis_range.start, start + length * axis_range.step, axis_range.step)

            else:
                images_range = range(1)
                N_images = 1
        else:
            images_range = range(self.shape[0])
            N_images = 1

        if len(images_range) <= 1:
            dimension_labels_keep[-3] = 0


        if height_roi is not None:
            axis_range = range(shape_out[1])[height_roi]
            length = (axis_range.stop - axis_range.start) // axis_range.step

            roi[self.dimension_labels[-2]] = (0, length * axis_range.step, axis_range.step)
            shape_out[-2] = length
            crop_box[1] = axis_range.start
            crop_box[3] = axis_range.start + length * axis_range.step
            shape_PIL_image[1] = length * axis_range.step

            if length < 2:
                dimension_labels_keep[-2] = 0


        if width_roi is not None:        
            axis_range = range(shape_out[2])[width_roi]
            length = (axis_range.stop - axis_range.start) // axis_range.step

            roi[self.dimension_labels[-1]] = (0, length * axis_range.step, axis_range.step)
            shape_out[-1] = length
            crop_box[0] = axis_range.start
            crop_box[2] = axis_range.start + length * axis_range.step
            shape_PIL_image[0] = length * axis_range.step

            if length < 2:
                dimension_labels_keep[-1] = 0


        # create empty data container for the array
        array_full = np.empty(shape_out, dtype=self.dtype)

        binner = Binner(roi, accelerated=False)
        arr_unbinned = np.empty((N_images, shape_PIL_image[1], shape_PIL_image[0]), dtype=self.dtype)

        image_unbinned = DataContainer(arr_unbinned, False, dimension_labels=self._dimension_labels_full)

        binner.set_input(image_unbinned)


        count = 0
        for i in images_range:
            for j in range(images_range.step):
                Tiff_utilities.read_to(self._tiff_files[i+j], arr_unbinned, shape_PIL_image, np.s_[j,:,:], crop_box)

            array_full[count] = binner.get_output().array
            count+=1

        array_full = array_full.squeeze()


        dimension_labels_out = [self._dimension_labels_full[i] for i, x in enumerate(dimension_labels_keep) if x == 1]
        data =  DataContainer(array_full, False, dimension_labels=dimension_labels_out)
        
        if self._rescale:
            return self._rescale_data(data)
    
        return data


    def read_sliced(self, images_slice=None, height_slice=None, width_slice=None):
        """
        Reads the ROI of an image and returns as a `DataContainer`. The dtype is configured by the reader.

        Parameters
        ----------
        images_slice : slice, list, default None
            Slice or list defining images to be read. If None, all images are read. This will be applied to the list retrieved with `get_image_list`
        height_slice : slice, default None
            Slice defining ROI of image height. Applied to all images.
        width_slice : slice, default None
            Slice defining ROI of image width. Applied to all images.

        Returns
        -------
        DataContainer: The read data

        """

        shape_out = self.shape.copy()
        crop_box = [0, 0, self.shape[2], self.shape[1]]
        dimension_labels_keep = [1]*len(self.dimension_labels)


        if images_slice is not None and  self.shape[0] > 1:
            if isinstance(images_slice,list):
                axis_range_images = images_slice
            elif isinstance(images_slice,slice):
                axis_range_images = range(shape_out[0])[images_slice]
            else:
                raise TypeError("Unsupported type for images_slice. Expected list or slice, got {}"\
                    .format(type(images_slice)))
            
            shape_out[0] = len(axis_range_images)

            if len(axis_range_images) < 2:
                dimension_labels_keep[-3] = 0
            
        else:
            axis_range_images = range(shape_out[0])

        if height_slice is not None:
            axis_range = range(shape_out[1])[height_slice]
            shape_out[1] = len(axis_range)
            crop_box[1] = axis_range.start
            crop_box[3] = axis_range.start + axis_range.step * (shape_out[1]-1) +1
            if len(axis_range) < 2:
                dimension_labels_keep[-2] = 0

        if width_slice is not None:
            axis_range = range(shape_out[2])[width_slice]
            shape_out[2] = len(axis_range)
            crop_box[0] = axis_range.start
            crop_box[2] = axis_range.start + axis_range.step * (shape_out[2]-1) +1
            if len(axis_range) < 2:
                dimension_labels_keep[-1] = 0

        # PIL specific set up
        shape_PIL_image = (shape_out[2], shape_out[1])

        # create empty data container for the array
        array_full = np.empty(shape_out, dtype=self.dtype)

        ind_out = 0
        for i in axis_range_images:
            Tiff_utilities.read_to(self._tiff_files[i], array_full, shape_PIL_image, np.s_[ind_out,:,:], crop_box)
            ind_out +=1

        array_full = array_full.squeeze()
        dimension_labels_out = [self.dimension_labels[i] for i, x in enumerate(dimension_labels_keep) if x == 1]
        data =  DataContainer(array_full, False, dimension_labels=dimension_labels_out)
        
        if self._rescale:
            return self._rescale_data(data)
    
        return data

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


    def _rescale_data(self, data):
        """
        Rescales the data by the offset and scale requested by the user as `rescaled_data = (read_data - offset)/scale`
        
        Parameters:
        -----------
        data: DataContainer
            The data to be rescaled  
        """

        data.multiply(self._rescale_values[0], out=data)
        data.add(self._rescale_values[1], out=data)

        return data
