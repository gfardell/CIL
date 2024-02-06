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


import os
import re
import json
import logging
import dask_image.imread as dask_imread
import dask.array as da
import numpy as np
import glob 
import pims
from PIL import Image

from cil.framework import (
    AcquisitionData,
    AcquisitionGeometry,
    ImageGeometry,
    ImageData,
    DataContainer,
)
from cil.processors import Binner, Slicer
from cil.io.utilities import Tiff_utilities
from cil.io import utilities

class TIFFReader(object):

    """
    Configures the TIFFReader. The TIFFReader can read a single TIFF file, a list of TIFF files, or all TIFF files in a directory.
    The TIFFReader can read a subset of the TIFF files found in a directory. The TIFFReader can read a subset of the image found in a TIFF file.
    The TIFFReader can bin the data read in. The TIFFReader can rescale the data read in.

    
    Parameters
    ----------
    file_name : str, list, abspath to folder/file
        The absolute file path to a TIFF file, or a list of absolute file paths to TIFF files. Or an absolute directory path to a folder containing TIFF files.

    rescale : bool, default True
        If True, and if accompanying JSON is found then the data will be rescaled by the offset and scale found in the file. This is a courtesy method that will only work if the data was saved with TIFFWriter.
        If False, or if the json file can't be found, the data will be returned as is.

    dimension_labels : tuple of strings, optional
        The labels of the dimensions of the TIFF file(s) being read If None, default labels will be used.

    dtype : numpy.dtype, default np.float32
        The data type returned with 'read'. If None, the dtype of the TIFF file(s) will be used.

    Notes
    -----

    Advanced configuration of the TIFFReader can be done with the following methods:
        - set_input_geometry
        - set_rescale
        - set_custom_pixelwise_operation
        - read_binned
        - read_sliced
              
    Example
    -------
    
    To read a single TIFF image (i.e. a sinogram):

    >>> reader = TIFFReader(file_name = '/path/to/file.tiff', dimension_labels=('angles','horizontal'))
    >>> data = reader.read()

    To read all TIFF files in a directory:

    >>> reader = TIFFReader(file_name = '/path/to/folder', dimension_labels=('angles','vertical','horizontal'))
    >>> data = reader.read()

    To read all TIFF files in a directory whose names start with a common prefix:

    >>> reader = TIFFReader(file_name = '/path/to/folder/projection', dimension_labels=('angles','vertical','horizontal'))

    To read a subset of TIFF files in a directory:

    >>> reader = TIFFReader(file_name = '/path/to/folder', dimension_labels=('z','y','x'))
    >>> reader.get_image_list()
    ['/path/to/folder/0001.tif', '/path/to/folder/0002.tif', '/path/to/folder/0003.tif', ...]
    >>> data = reader.read_sliced(images_slice=slice(0,10,2))


    Notes
    -----

    When reading a TIFF stack, the reader will sort the TIFF files found in the folder in to natural sort order.
    This will sort first alphabetically, then numerically leading to the sorting: 1.tif, 2.tif, 3.tif, 10.tif, 11.tif
    The users can see the order with the method `get_image_list()`. The user can also pass an ordered list of TIFF
    files to the reader, in which case the user's order will be used.

    """



    def __init__(self, file_name, rescale=True,  dimension_labels=None, dtype=np.float32):

        if (Tiff_utilities.pilAvailable == False):
            raise ImportError("PIL (pillow) is not available, cannot load TIFF files. Please install PIL (pillow) to use TIFFReader.")

        self._file_name = file_name
        self._shape = None
        self._dtype = None
        self._dimension_labels_full = None
        self._geometry_full = None

        self._rescale_pwop = None
        self._custom_pwop = None

        self._tiff_files = Tiff_utilities.get_tiff_paths(self._file_name)
        self.set_rescale(rescale)
        
        # use the first image to determine the image parameters
        self._image_param_in = Tiff_utilities.get_dataset_metadata(self._tiff_files[0])
        num_images = len(self._tiff_files)

        self._shape = [num_images, self._image_param_in['height'], self._image_param_in['width']]
        self._dtype = self._image_param_in['dtype'] if dtype is None else dtype

        # set up scaling

        if dimension_labels is None:
            dimension_labels = ['images', 'image_height', 'image_width']

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
                logging.info("dimension_labels length should match the input dimensions. Using {}.".format(dimension_labels[-2::]))
            else:
                raise ValueError("dimension_labels must be a tuple of length 2. Got {}".format(len(dimension_labels)))


    def set_input_geometry(self, geometry=None):
        """
        Sets the geometry of the data to be read. This determines the return type of read methods.
        The geometry shape must match the full TIFF file(s) being read.
        
        Parameters
        ----------
        geometry : ImageGeometry or AcquisitionGeometry, optional
            The geometry of the data to be read, if `None` the data will be returned as a DataContainer.

        Example
        -------
                
        >>> reader = TIFFReader(file_name = '/path/to/folder')
        >>> geometry = ImageGeometry(voxel_num_x=100, voxel_num_y=100)
        >>> reader.set_input_geometry(geometry)
        >>> data = reader.read()
        >>> print(data)
        ImageData
        Dimension labels: ['horizontal_y', 'horizontal_x']
        Data dimensions: (100, 100)
        voxel sizes: (1.0, 1.0)

        
        If you wish to bin the data, the binned geometry will be created from the input geometry:

        >>> reader = TIFFReader(file_name = '/path/to/folder')
        >>> geometry = ImageGeometry(voxel_num_x=100, voxel_num_y=100)
        >>> reader.set_input_geometry(geometry)
        >>> data = reader.read_binned(width_roi=slice(0,100,2), height_roi=slice(0,100,2))
        >>> print(data)
        ImageData
        Dimension labels: ['horizontal_y', 'horizontal_x']
        Data dimensions: (50, 50)
        voxel sizes: (2.0, 2.0)

        """
        self._geometry_full = None

        if geometry is not None:
            #check input data matches geometry
            if not isinstance(geometry, (ImageGeometry, AcquisitionGeometry)):
                raise TypeError("Unsupported Geometry type. Expected ImageGeometry or AcquisitionGeometry, got {}"\
                    .format(type(geometry)))
            
            shape = tuple([x for x in self._shape if x != 1])

            if shape != geometry.shape:
                raise ValueError('Requested {} shape is incompatible with data. Expected {}, got {}'\
                    .format(geometry.__class__.__name__, self._shape, geometry.shape))
               
            self._geometry_full = geometry.copy()


    def set_rescale(self, rescale=True):
        """
        Sets the rescale parameters if rescale files are found in the directory. This is a courtesy method that will only work if the data was saved with TIFFWriter.
         
        Parameters
        ----------
        rescale : bool, default True
            If True, and if accompanying JSON is found then the data will be rescaled by the offset and scale found in the file. This is a courtesy method that will only work if the data was saved with TIFFWriter.
            If False, or if the json file can't be found, the data will be returned as is.

        """

        if rescale is True:
            try:
                scale, offset = utilities.read_scale_offset(os.path.dirname(self._tiff_files[0]))
            except FileNotFoundError:
                logging.info("No json file with scale and offset found in directory. Returning data as is.")
                self._rescale_pwop = None
            else:
                self._rescale_pwop = lambda x: (x - offset)*(1.0 / scale)
        else:
            self._rescale_pwop = None
                
            
    def set_custom_pixelwise_operation(self, function = None ):
        """
        Sets a custom pixelwise operation to be applied to the data read in.

        Parameters:
        -----------
        function: callable[x] -> y, optional
            A function handle or lambda function that defines the custom pixelwise operation.
            The function must take a single value and return a single value.

        Example:
        --------
        To rescale the data manually, you can set a custom pixelwise operation that divides the input by a constant:

        >>> reader = TIFFReader(file_name='/path/to/folder')
        >>> reader.set_custom_pixelwise_operation(function=lambda x: x/60000)
        >>> data = reader.read()

        
        To set a custom pixelwise operation that takes the negative logarithm (base 10) of the input:

        >>> reader = TIFFReader(file_name='/path/to/folder')
        >>> reader.set_custom_pixelwise_operation(function=lambda x: -np.log10(x)) 
        >>> data = reader.read()
        """

        self._custom_pwop = function



    def _get_combined_pwop(self, dtype=np.float32):
        """
        Returns a combined pixelwise operation that applies the rescale and custom pixelwise operation to the data read in.

        Parameters:
        -----------
        dtype: numpy.dtype, default np.float32
            The data type of the output of the combined pixelwise operations.

        Returns:
        --------
        combined_ufunc: callable[x] -> y
            A function handle that defines the combined pixelwise operation with a return type of otype.

        """
        
        func1 = self._rescale_pwop
        func2 = self._custom_pwop

        if func1 is not None and func2 is not None:
            func = lambda x: self._custom_pwop(self._rescale_pwop(x))
            combined_ufunc = np.frompyfunc(func,1,1)

        elif func1 is not None:
            combined_ufunc = np.frompyfunc(func1,1,1)
        elif func2 is not None:
            combined_ufunc = np.frompyfunc(func2,1,1)
        else:
            combined_ufunc = None

        return combined_ufunc


    @property
    def shape(self):
        """Returns the shape of the TIFF file(s) being read."""
        return self._shape
        
    @property
    def dimension_labels(self):
        """Returns the dimension labels of the TIFF file(s) being read."""

        if self._geometry_full is None:
            if self._shape[0] > 1:
                return self._dimension_labels_full
            else:
                return self._dimension_labels_full[1::]
        else:
            return self._geometry_full.dimension_labels


    @property
    def dtype(self):
        """Returns the data type of the TIFF file(s) being read. If `set_geometry` has been called the datatype returned will match."""
        if self._geometry_full is None:
            return self._dtype
        else:
            return self._geometry_full.dtype

    def get_image_paths(self):
        """
        Returns a list of TIFF files that have been found by the reader. This list will be sorted in natural order. Modifications to 
        this list will not affect the reader.

        Returns
        -------
        list
            An ordered list of TIFF files.
        """
        return self._tiff_files.copy()


    def read(self):
        """
        Reads the TIFF file(s) and returns as a `DataContainer`, `ImageData` of `AcquisitionData`.

        Returns
        -------
        DataContainer, ImageData or AcquisitionData :
            The full read data

        Notes
        -----
        If `set_geometry` has been called, the data will be returned as an ImageData or AcquisitionData.

        If `set_geometry` has been called, the `dtype` and `dimension_labels` will be set by the geometry. 
        Otherwise, the `dtype` and `dimension_labels` will be set by the reader, if None was passed to the constructor, 
        the `dtype` will be set by the TIFF file(s) and the `dimension_labels` will be set to ['images', 'image_height', 'image_width']
        if multiple TIFF files are read, or ['image_height', 'image_width'] if a single TIFF file is read.

        """
        if self._geometry_full is None:
            dtype = self.dtype
        else:
            dtype = self._geometry_full.dtype

        pwop = self._get_combined_pwop(dtype)

        # create empty data container for the array
        array_full = np.empty(self.shape, dtype=dtype)

        for i in range(self.shape[0]):
            Tiff_utilities.read_to(self._tiff_files[i], array_full, np.s_[i,:,:])
        
        if pwop is not None:
            np.clip(array_full, 1, 65535, out=array_full)
            np.multiply(array_full, 1.0/65535.0, out=array_full)
            np.log(array_full, out=array_full)
            np.negative(array_full, out=array_full)

        array_full = array_full.squeeze()

        if self._geometry_full is None:
            return DataContainer(array_full, dimension_labels=self.dimension_labels)

        if isinstance (self._geometry_full, AcquisitionGeometry):
            return AcquisitionData(array_full, deep=False, geometry=self._geometry_full.copy(), suppress_warning=True)

        if isinstance (self._geometry_full, ImageGeometry):
            return ImageData(array_full, deep=False, geometry=self._geometry_full.copy(), suppress_warning=True)


    def read_dask(self):
        """
        Reads the TIFF file(s) and returns as a Dask array.

        Returns
        -------
        dask.array.Array :
            The full read data
        """
        if self._geometry_full is None:
            dtype = self.dtype
        else:
            dtype = self._geometry_full.dtype

        pwop = self._get_combined_pwop(dtype)

        images = [dask_imread.imread(file, ) for file in self._tiff_files]
        images =  da.stack(images)
        images = images.squeeze()
        images.astype(np.float32)

        if pwop is not None:
            np.clip(images, 1, 65535, out=images)
            np.multiply(images, 1.0/65535.0, out=images)
            np.log(images, out=images)
            np.negative(images, out=images)

        array_full = images.compute()

        if self._geometry_full is None:
            return DataContainer(array_full, dimension_labels=self.dimension_labels)

        if isinstance (self._geometry_full, AcquisitionGeometry):
            return AcquisitionData(array_full, deep=False, geometry=self._geometry_full.copy(), suppress_warning=True)

        if isinstance (self._geometry_full, ImageGeometry):
            return ImageData(array_full, deep=False, geometry=self._geometry_full.copy(), suppress_warning=True)


    def read_async(self):
        """
        Reads the TIFF file(s) and returns as a `DataContainer`, `ImageData` of `AcquisitionData`.

        Returns
        -------
        DataContainer, ImageData or AcquisitionData :
            The full read data

        Notes
        -----
        If `set_geometry` has been called, the data will be returned as an ImageData or AcquisitionData.

        If `set_geometry` has been called, the `dtype` and `dimension_labels` will be set by the geometry. 
        Otherwise, the `dtype` and `dimension_labels` will be set by the reader, if None was passed to the constructor, 
        the `dtype` will be set by the TIFF file(s) and the `dimension_labels` will be set to ['images', 'image_height', 'image_width']
        if multiple TIFF files are read, or ['image_height', 'image_width'] if a single TIFF file is read.

        """
        if self._geometry_full is None:
            dtype = self.dtype
        else:
            dtype = self._geometry_full.dtype

        pwop = self._get_combined_pwop(dtype)

        # create empty data container for the array
        array_full = np.empty(self.shape, dtype=dtype)

        # read to the array and apply the pixelwise operation
        Tiff_utilities.read_to_all_async(self._tiff_files, array_full)

        if pwop is not None:
            np.clip(array_full, 1, 65535, out=array_full)
            np.multiply(array_full, 1.0/65535.0, out=array_full)
            np.log(array_full, out=array_full)
            np.negative(array_full, out=array_full)

        array = array_full.squeeze()

        if self._geometry_full is None:
            return DataContainer(array, dimension_labels=self.dimension_labels)

        if isinstance (self._geometry_full, AcquisitionGeometry):
            return AcquisitionData(array, deep=False, geometry=self._geometry_full.copy(), suppress_warning=True)

        if isinstance (self._geometry_full, ImageGeometry):
            return ImageData(array, deep=False, geometry=self._geometry_full.copy(), suppress_warning=True)
        

    def read_binned(self, images=None, height_roi=None, width_roi=None):
        return self._read_downsampled(images=images, height_roi=height_roi, width_roi=width_roi, downsample_mode='bin')

    def read_sliced(self, images=None, height_roi=None, width_roi=None):
        return self._read_downsampled(images=images, height_roi=height_roi, width_roi=width_roi, downsample_mode='slice')        

    def _read_downsampled(self, images=None, height_roi=None, width_roi=None, downsample_mode='slice'):     
        """
        Reads the ROI of an image and returns as a `DataContainer`, `ImageData` of `AcquisitionData`. 

        Parameters
        ----------

        images : slice, list, int, optional
            Slice or list or int defining images to be read. If None, all images are read. This will be applied to the list retrieved with `get_image_list`
        height_roi : slice, optional
            Slice defining ROI of image height. Applied to all images.
        width_roi : slice, optional
            Slice defining ROI of image width. Applied to all images.

        Returns
        -------
        DataContainer, ImageData or AcquisitionData :
            The read data binned according to the ROI requested.

        Notes
        -----
        If `set_geometry` has been called, the data will be returned as an ImageData or AcquisitionData.

        If `set_geometry` has been called, the `dtype` and `dimension_labels` will be set by the geometry.
        Otherwise, the `dtype` and `dimension_labels` will be set by the reader, if None was passed to the constructor,
        the `dtype` will be set by the TIFF file(s) and the `dimension_labels` will be set to ['images', 'image_height', 'image_width']
        if multiple TIFF files are read, or ['image_height', 'image_width'] if a single TIFF file is read.
        """

        if downsample_mode == 'bin':
            DownSampler = Binner
            extra_args = (False,)
        elif downsample_mode == 'slice':
            DownSampler = Slicer
            extra_args = ()
        else:
            raise ValueError("downsample_mode must be either 'bin' or 'slice'. Got {}".format(downsample_mode))
            

        if self._geometry_full is None:
            dtype = self.dtype
            # create a dummy geometry to pass to the downsampler
            geometry = ImageGeometry(voxel_num_x=self.shape[-1], voxel_num_y=self.shape[-2], voxel_num_z=self.shape[-3])

        else:
            dtype = self._geometry_full.dtype
            geometry = self._geometry_full

        pwop = self._get_combined_pwop(dtype)

        # parse the images to read, this can include an int of slice which slicer cannot parse
        pass_to_downsampler = False
        num_images_bin = 1
        if images is None:
            image_iter = range(self.shape[0])
        elif isinstance(images, slice):
            image_iter = range(self.shape[0])[images]
            pass_to_downsampler = True
            if image_iter.step > 1:
                num_images_bin = images.step
        elif isinstance(images, int):
            image_iter = slice(images, images+1)
            pass_to_downsampler = True
        elif isinstance(images, list):
            raise NotImplementedError("Binning of a list of images is not implemented. Please use a slice or int.")
            # TODO will need to update geometry to kept list of images
            image_iter = images
        else:
            raise TypeError
        
        # get new geometry, and use this to set the image crop box to read in
        roi_geom = {}
        roi_geom[geometry.dimension_labels[-1]]=width_roi
        roi_geom[geometry.dimension_labels[-2]]=height_roi
        if self._shape[0] > 1 and pass_to_downsampler:
            roi_geom[geometry.dimension_labels[-3]]=images

        downsampler = DownSampler(roi_geom,*extra_args)
        downsampler.set_input(geometry)
        geometry_new = downsampler.get_output()

        pixel_indices_width = downsampler._pixel_indices[-1]
        pixel_indices_height = downsampler._pixel_indices[-2]

        # read in roi
        image_size = [ pixel_indices_height[1]-pixel_indices_height[0]+1,
                      pixel_indices_width[1]-pixel_indices_width[0]+1,
                      ]
        
        crop_box = [pixel_indices_width[0], pixel_indices_height[0], pixel_indices_width[1]+1, pixel_indices_height[1]+1]

        # binner for each chunk of images to be downsampled
        roi_data = {}
        roi_data['image_width'] = (None, None, downsampler._roi_ordered[-1].step)  
        roi_data['image_height'] = (None, None, downsampler._roi_ordered[-2].step)  
        if self._shape[0] > 1 and num_images_bin > 1:
            roi_data['images']= (None, None, num_images_bin)

        # create empty data container for the full output array
        shape_out = [len(image_iter), downsampler._shape_out[-2], downsampler._shape_out[-1]]
        array_out = np.empty(shape_out, dtype=dtype)

        # create empty data container for the array passed to the down sampler
        arr_unbinned = np.empty((num_images_bin, *image_size), dtype=dtype)
        image_unbinned = DataContainer(arr_unbinned, deep_copy=False, dimension_labels=['images', 'image_height', 'image_width'])

        # create the downsampler

        downsampler = DownSampler(roi_data, *extra_args)
        downsampler.set_input(image_unbinned)

        # read in the first N images and pass to the downsampler
        count = 0
        for i in image_iter:
            for j in range(num_images_bin):
                Tiff_utilities.read_to(self._tiff_files[i+j], arr_unbinned, np.s_[j,:,:], crop_box)

            array_out[count] = downsampler.get_output().array

            if pwop is not None:
                array_out[np.s_[count,:,:]] = pwop(array_out[np.s_[count,:,:]] )

            count+=1

        array_out = array_out.squeeze()

        if self._geometry_full is None:
            dimension_labels_out = []
            if shape_out[0] > 1:
                dimension_labels_out.append(self.dimension_labels[-3])
            if shape_out[1] > 1:
                dimension_labels_out.append(self.dimension_labels[-2])
            if shape_out[2] > 1:
                dimension_labels_out.append(self.dimension_labels[-1])
            
            data = DataContainer(array_out, deep_copy=False, dimension_labels=dimension_labels_out)

        elif isinstance (self._geometry_full, AcquisitionGeometry):
            data = AcquisitionData(array_out, deep=False, geometry=geometry_new, suppress_warning=True)
        else:
            data = ImageData(array_out, deep=False, geometry=geometry_new, suppress_warning=True)
        
        return data
