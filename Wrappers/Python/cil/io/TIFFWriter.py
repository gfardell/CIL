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
import os, re
from cil.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry

pilAvailable = True
try:    
    from PIL import Image
except:
    pilAvailable = False
import functools
import glob
import re
import numpy as np
from cil.io import utilities
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

def save_scale_offset(fname, scale, offset):
    '''Save scale and offset to file
    
    Parameters
    ----------
    fname : string
    scale : float
    offset : float
    '''
    dirname = os.path.dirname(fname)
    txt = os.path.join(dirname, 'scaleoffset.json')
    d = {'scale': scale, 'offset': offset}
    utilities.save_dict_to_file(txt, d)

class TIFFWriter(object):
    '''Write a DataSet to disk as a TIFF file or stack of TIFF files


        Parameters
        ----------
        data : DataContainer, AcquisitionData or ImageData
            This represents the data to save to TIFF file(s)
        file_name : string
            This defines the file name prefix, i.e. the file name without the extension.
        counter_offset : int, default 0.
            counter_offset indicates at which number the ordinal index should start.
            For instance, if you have to save 10 files the index would by default go from 0 to 9.
            By counter_offset you can offset the index: from `counter_offset` to `9+counter_offset`
        compression : str, default None. Accepted values None, 'uint8', 'uint16'
            The lossy compression to apply. The default None will not compress data. 
            'uint8' or 'unit16' will compress to unsigned int 8 and 16 bit respectively.


        Note
        ----

          If compression ``uint8`` or ``unit16`` are used, the scale and offset used to compress the data are saved 
          in a file called ``scaleoffset.json`` in the same directory as the TIFF file(s).

          The original data can be obtained by: ``original_data = (compressed_data - offset) / scale``
        
        Note
        ----
        
          In the case of 3D or 4D data this writer will save the data as a stack of multiple TIFF files,
          not as a single multi-page TIFF file.
        '''

    
    def __init__(self, data=None, file_name=None, counter_offset=0, compression=None):
        
        self.data_container = data
        self.file_name = file_name
        self.counter_offset = counter_offset
        if ((data is not None) and (file_name is not None)):
            self.set_up(data = data, file_name = file_name, 
                        counter_offset=counter_offset,
                        compression=compression)
        
    def set_up(self,
               data = None,
               file_name = None,
               counter_offset = 0,
               compression=0):
        
        self.data_container = data
        file_name = os.path.abspath(file_name)
        self.file_name = os.path.splitext(
            os.path.basename(
                file_name
                )
            )[0]
        
        self.dir_name = os.path.dirname(file_name)
        logger.info ("dir_name {}".format(self.dir_name))
        logger.info ("file_name {}".format(self.file_name))
        self.counter_offset = counter_offset
        
        if not ((isinstance(self.data_container, ImageData)) or 
                (isinstance(self.data_container, AcquisitionData))):
            raise Exception('Writer supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')

        # Deal with compression
        self.compress           = utilities.get_compress(compression)
        self.dtype              = utilities.get_compressed_dtype(data, compression)
        self.scale, self.offset = utilities.get_compression_scale_offset(data, compression)
        self.compression        = compression

    
    def write(self):
        '''Write data to disk'''
        if not os.path.isdir(self.dir_name):
            os.mkdir(self.dir_name)

        ndim = len(self.data_container.shape)
        if ndim == 2:
            # save single slice
            
            if self.counter_offset >= 0:
                fname = "{}_idx_{:04d}.tiff".format(os.path.join(self.dir_name, self.file_name), self.counter_offset)
            else:
                fname = "{}.tiff".format(os.path.join(self.dir_name, self.file_name))
            with open(fname, 'wb') as f:
                Image.fromarray(
                    utilities.compress_data(self.data_container.as_array() , self.scale, self.offset, self.dtype)
                    ).save(f, 'tiff')
        elif ndim == 3:
            for sliceno in range(self.data_container.shape[0]):
                # save single slice
                # pattern = self.file_name.split('.')
                dimension = self.data_container.dimension_labels[0]
                fname = "{}_idx_{:04d}.tiff".format(
                    os.path.join(self.dir_name, self.file_name),
                    sliceno + self.counter_offset)
                with open(fname, 'wb') as f:
                    Image.fromarray(
                            utilities.compress_data(self.data_container.as_array()[sliceno] , self.scale, self.offset, self.dtype)
                        ).save(f, 'tiff')
        elif ndim == 4:
            # find how many decimal places self.data_container.shape[0] and shape[1] have
            zero_padding = self._zero_padding(self.data_container.shape[0])
            zero_padding += '_' + self._zero_padding(self.data_container.shape[1])
            format_string = "{}_{}x{}x{}x{}_"+"{}.tiff".format(zero_padding)

            for sliceno1 in range(self.data_container.shape[0]):
                # save single slice
                # pattern = self.file_name.split('.')
                dimension = [ self.data_container.dimension_labels[0] ]
                for sliceno2 in range(self.data_container.shape[1]):
                    fname = format_string.format(os.path.join(self.dir_name, self.file_name), 
                        self.data_container.shape[0], self.data_container.shape[1], self.data_container.shape[2],
                        self.data_container.shape[3] , sliceno1, sliceno2)
                    with open(fname, 'wb') as f:
                        Image.fromarray(
                            utilities.compress_data(self.data_container.as_array()[sliceno1][sliceno2] , self.scale, self.offset, self.dtype)
                        ).save(f, 'tiff')
        else:
            raise ValueError('Cannot handle more than 4 dimensions')
        if self.compress:
            save_scale_offset(fname, self.scale, self.offset)
    
    def _zero_padding(self, number):
        i = 0
        while 10**i < number:
            i+=1
        i+=1 
        zero_padding_string = '{:0'+str(i)+'d}'
        return zero_padding_string

