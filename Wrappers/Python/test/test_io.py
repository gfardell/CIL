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

import unittest
from unittest.mock import patch, MagicMock
from utils import initialise_tests
from cil.framework import AcquisitionGeometry, ImageGeometry, AcquisitionData, ImageData, DataContainer
import numpy as np
import os
from cil.io import TXRMDataReader, NEXUSDataReader, NikonDataReader, ZEISSDataReader
from cil.io import TIFFReader 
from cil.io import TIFFWriter, TIFFStackReader
from cil.io.utilities import HDF5_utilities, Tiff_utilities
from cil.processors import Slicer, Binner
from utils import has_astra, has_nvidia
from cil.utilities.dataexample import data_dir
from cil.utilities.quality_measures import mse
from cil.utilities import dataexample
import shutil
import logging
import glob
import json
from cil.io import utilities
from cil.io import RAWFileWriter
import configparser
import tempfile
        

initialise_tests()

has_dxchange = True
try:
    import dxchange
except ImportError as ie:
    has_dxchange = False
has_olefile = True
try:
    import olefile
except ImportError as ie:
    has_olefile = False
has_wget = True
try:
    import wget
except ImportError as ie:
    has_wget = False
if has_astra:
    from cil.plugins.astra import FBP
has_pil = True
try:    
    from PIL import Image
except:
    has_pil = False


# change basedir to point to the location of the walnut dataset which can
# be downloaded from https://zenodo.org/record/4822516
# basedir = os.path.abspath('/home/edo/scratch/Data/Walnut/valnut_2014-03-21_643_28/tomo-A/')
basedir = data_dir
filename = os.path.join(basedir, "valnut_tomo-A.txrm")
has_file = os.path.isfile(filename)


has_prerequisites = has_olefile and has_dxchange and has_astra and has_nvidia and has_file \
    and has_wget

# Change the level of the logger to WARNING (or whichever you want) to see more information
logging.basicConfig(level=logging.WARNING)

logging.info ("has_astra {}".format(has_astra))
logging.info ("has_wget {}".format(has_wget))
logging.info ("has_olefile {}".format(has_olefile))
logging.info ("has_dxchange {}".format(has_dxchange))
logging.info ("has_file {}".format(has_file))

if not has_file:
    logging.info("This unittest requires the walnut Zeiss dataset saved in {}".format(data_dir))


class TestTXRMDataReader(unittest.TestCase):
    

    def setUp(self):
        logging.info ("has_astra {}".format(has_astra))
        logging.info ("has_wget {}".format(has_wget))
        logging.info ("has_olefile {}".format(has_olefile))
        logging.info ("has_dxchange {}".format(has_dxchange))
        logging.info ("has_file {}".format(has_file))
        if has_file:
            self.reader = TXRMDataReader()
            angle_unit = AcquisitionGeometry.RADIAN
            
            self.reader.set_up(file_name=filename, 
                               angle_unit=angle_unit)
            data = self.reader.read()
            if data.geometry is None:
                raise AssertionError("WTF")
            # Choose the number of voxels to reconstruct onto as number of detector pixels
            N = data.geometry.pixel_num_h
            
            # Geometric magnification
            mag = (np.abs(data.geometry.dist_center_detector) + \
                np.abs(data.geometry.dist_source_center)) / \
                np.abs(data.geometry.dist_source_center)
                
            # Voxel size is detector pixel size divided by mag
            voxel_size_h = data.geometry.pixel_size_h / mag
            voxel_size_v = data.geometry.pixel_size_v / mag

            self.mag = mag
            self.N = N
            self.voxel_size_h = voxel_size_h
            self.voxel_size_v = voxel_size_v

            self.data = data


    def tearDown(self):
        pass


    def test_run_test(self):
        print("run test Zeiss Reader")
        self.assertTrue(True)
    

    @unittest.skipIf(not has_prerequisites, "Prerequisites not met")
    def test_read_and_reconstruct_2D(self):
        
        # get central slice
        data2d = self.data.subset(vertical='centre')
        # d512 = self.data.subset(vertical=512)
        # data2d.fill(d512.as_array())
        # neg log
        data2d.log(out=data2d)
        data2d *= -1

        ig2d = data2d.geometry.get_ImageGeometry()
        # Construct the appropriate ImageGeometry
        ig2d = ImageGeometry(voxel_num_x=self.N,
                            voxel_num_y=self.N,
                            voxel_size_x=self.voxel_size_h, 
                            voxel_size_y=self.voxel_size_h)
        if data2d.geometry is None:
            raise AssertionError('What? None?')
        fbpalg = FBP(ig2d,data2d.geometry)
        fbpalg.set_input(data2d)
        
        recfbp = fbpalg.get_output()
        
        wget.download('https://www.ccpi.ac.uk/sites/www.ccpi.ac.uk/files/walnut_slice512.nxs',
                      out=data_dir)
        fname = os.path.join(data_dir, 'walnut_slice512.nxs')
        reader = NEXUSDataReader()
        reader.set_up(file_name=fname)
        gt = reader.read()

        qm = mse(gt, recfbp)
        logging.info ("MSE {}".format(qm) )

        np.testing.assert_almost_equal(qm, 0, decimal=3)
        fname = os.path.join(data_dir, 'walnut_slice512.nxs')
        os.remove(fname)


class TestTIFF(unittest.TestCase):
    def setUp(self) -> None:
        self.TMP = tempfile.TemporaryDirectory()
        self.cwd = os.path.join(self.TMP.name, 'rawtest')

    def tearDown(self) -> None:
        self.TMP.cleanup()
        
    def get_slice_imagedata(self, data):
        '''Returns only 2 slices of data'''
        # data = dataexample.SIMULATED_SPHERE_VOLUME.get()
        data.dimension_labels[0]
        roi = {data.dimension_labels[0]: (0,2,1), 
               data.dimension_labels[1]: (None, None, None), 
               data.dimension_labels[2]: (None, None, None)}
        return Slicer(roi=roi)(data)


    def test_tiff_stack_ImageData(self):
        # data = dataexample.SIMULATED_SPHERE_VOLUME.get()
        data = self.get_slice_imagedata(
            dataexample.SIMULATED_SPHERE_VOLUME.get()
        )
        
        fname = os.path.join(self.cwd, "unittest")

        writer = TIFFWriter(data=data, file_name=fname)
        writer.write()

        reader = TIFFStackReader(file_name=self.cwd)
        read_array = reader.read()

        np.testing.assert_allclose(data.as_array(), read_array)

        read = reader.read_as_ImageData(data.geometry)
        np.testing.assert_allclose(data.as_array(), read.as_array())


    def test_tiff_stack_AcquisitionData(self):
        # data = dataexample.SIMULATED_CONE_BEAM_DATA.get()
        data = self.get_slice_imagedata(
            dataexample.SIMULATED_CONE_BEAM_DATA.get()
        )
        
        
        fname = os.path.join(self.cwd, "unittest")

        writer = TIFFWriter(data=data, file_name=fname)
        writer.write()

        reader = TIFFStackReader(file_name=self.cwd)
        read_array = reader.read()

        np.testing.assert_allclose(data.as_array(), read_array)

        read = reader.read_as_AcquisitionData(data.geometry)
        np.testing.assert_allclose(data.as_array(), read.as_array())


    def test_tiff_stack_ImageDataSlice(self):
        data = dataexample.SIMULATED_SPHERE_VOLUME.get()
        
        fname = os.path.join(self.cwd, "unittest")

        writer = TIFFWriter(data=data, file_name=fname)
        writer.write()

        roi = {'axis_0': -1, 'axis_1': -1, 'axis_2': (None, None, 2)}

        reader = TIFFStackReader(file_name=self.cwd, roi=roi, mode='slice')
        read_array = reader.read()

        shape = [el for el in data.shape]
        shape[2] /= 2

        np.testing.assert_allclose(shape, read_array.shape )
        
        roi = {'axis_0': (0, 2, None), 'axis_1': -1, 'axis_2': -1}

        reader = TIFFStackReader(file_name=self.cwd, roi=roi, mode='slice')
        read_array = reader.read()

        np.testing.assert_allclose(data.as_array()[:2], read_array)


    def test_tiff_stack_ImageData_wrong_file(self):
        # data = dataexample.SIMULATED_SPHERE_VOLUME.get()
        data = self.get_slice_imagedata(
            dataexample.SIMULATED_SPHERE_VOLUME.get()
        )
        
        fname = os.path.join(self.cwd, "unittest")

        writer = TIFFWriter(data=data, file_name=fname)
        writer.write()

        for el in glob.glob(os.path.join(self.cwd , "unittest*.tiff")):
            # print (f"modifying {el}")
            with open(el, 'w') as f:
                f.write('BOOM')
            break
                
        reader = TIFFStackReader(file_name=self.cwd)
        try:
            read_array = reader.read()
            assert False
        except:
            assert True

    def test_TIFF_compression3D_0(self):
        self.TIFF_compression_test(None)
    
    def test_TIFF_compression3D_1(self):
        self.TIFF_compression_test('uint8')

    def test_TIFF_compression3D_2(self):
        self.TIFF_compression_test('uint16')

    def test_TIFF_compression3D_3(self):
        with self.assertRaises(ValueError) as context:
            self.TIFF_compression_test('whatever_compression')
            
    def test_TIFF_compression4D_0(self):
        self.TIFF_compression_test(None,2)
        
    def test_TIFF_compression4D_1(self):
        self.TIFF_compression_test('uint8',2)

    def test_TIFF_compression4D_2(self):
        self.TIFF_compression_test('uint16',2)
    
    def test_TIFF_compression4D_3(self):
        with self.assertRaises(ValueError) as context:
            self.TIFF_compression_test('whatever_compression',2)

    def TIFF_compression_test(self, compression, channels=1):
        X=4
        Y=5
        Z=6
        C=channels
        if C == 1:
            ig = ImageGeometry(voxel_num_x=4, voxel_num_y=5, voxel_num_z=6)
        else:
            ig = ImageGeometry(voxel_num_x=4, voxel_num_y=5, voxel_num_z=6, channels=C)
        data = ig.allocate(0)
        data.fill(np.arange(X*Y*Z*C).reshape(ig.shape))

        compress = utilities.get_compress(compression)
        dtype = utilities.get_compressed_dtype(data.array, compression)
        scale, offset = utilities.get_compression_scale_offset(data.array, compression)
        if C > 1:
            assert data.ndim == 4
        fname = os.path.join(self.cwd, "unittest")
        writer = TIFFWriter(data=data, file_name=fname, compression=compression)
        writer.write()
        # force the reader to use the native TIFF dtype by setting dtype=None
        reader = TIFFStackReader(file_name=self.cwd, dtype=None)
        read_array = reader.read()
        if C > 1:
            read_array = reader.read_as_ImageData(ig).array

        
        if compress:
            tmp = data.array * scale + offset
            tmp = np.asarray(tmp, dtype=dtype)
            # test if the scale and offset are written to the json file
            with open(os.path.join(self.cwd, "scaleoffset.json"), 'r') as f:
                d = json.load(f)
            assert d['scale'] == scale
            assert d['offset'] == offset
            # test if the scale and offset are read from the json file
            sc, of = utilities.read_scale_offset(self.cwd)
            assert sc == scale
            assert of == offset
            
            recovered_data = (read_array - of)/sc
            np.testing.assert_allclose(recovered_data, data.array, rtol=1e-1, atol=1e-2)

            # test read_rescaled
            approx = reader.read_rescaled()
            np.testing.assert_allclose(approx.ravel(), data.array.ravel(), rtol=1e-1, atol=1e-2)

            approx = reader.read_rescaled(sc, of)
            np.testing.assert_allclose(approx.ravel(), data.array.ravel(), rtol=1e-1, atol=1e-2)
        else:
            tmp = data.array
            # if the compression is None, the scale and offset should not be written to the json file
            with self.assertRaises(OSError) as context:
                sc, of = utilities.read_scale_offset(self.cwd)
        
        assert tmp.dtype == read_array.dtype
        
        np.testing.assert_array_equal(tmp, read_array)

class TestRAW(unittest.TestCase):
    def setUp(self) -> None:
        self.TMP = tempfile.TemporaryDirectory()
        self.cwd = os.path.join(self.TMP.name, 'rawtest')

    def tearDown(self) -> None:
        self.TMP.cleanup()
        # pass

    def test_raw_nocompression_0(self):
        self.RAW_compression_test(None,1)
    
    def test_raw_compression_0(self):
        self.RAW_compression_test('uint8',1)

    def test_raw_compression_1(self):
        self.RAW_compression_test('uint16',1)

    def test_raw_nocompression_1(self):
        self.RAW_compression_test(None,1)
    
    def test_raw_compression_2(self):
        with self.assertRaises(ValueError) as context:
            self.RAW_compression_test(12,1)

    def RAW_compression_test(self, compression, channels=1):
        X=4
        Y=5
        Z=6
        C=channels
        if C == 1:
            ig = ImageGeometry(voxel_num_x=4, voxel_num_y=5, voxel_num_z=6)
        else:
            ig = ImageGeometry(voxel_num_x=4, voxel_num_y=5, voxel_num_z=6, channels=C)
        data = ig.allocate(0)
        data.fill(np.arange(X*Y*Z*C).reshape(ig.shape))

        compress = utilities.get_compress(compression)
        dtype = utilities.get_compressed_dtype(data.array, compression)
        scale, offset = utilities.get_compression_scale_offset(data.array, compression)
        if C > 1:
            assert data.ndim == 4
        raw = "unittest.raw"
        fname = os.path.join(self.cwd, raw)
        
        writer = RAWFileWriter(data=data, file_name=fname, compression=compression)
        writer.write()

        # read the data from the ini file
        ini = "unittest.ini"
        config = configparser.ConfigParser()
        inifname = os.path.join(self.cwd, ini)
        config.read(inifname)
        

        assert raw == config['MINIMAL INFO']['file_name']

        # read how to read the data from the ini file
        read_dtype = config['MINIMAL INFO']['data_type']
        read_array = np.fromfile(fname, dtype=read_dtype)
        read_shape = eval(config['MINIMAL INFO']['shape'])
        
        # reshape read in array
        read_array = read_array.reshape(read_shape)

        if compress:
            # rescale the dataset to the original data
            sc = float(config['COMPRESSION']['scale'])
            of = float(config['COMPRESSION']['offset'])
            assert sc == scale
            assert of == offset

            recovered_data = (read_array - of)/sc
            np.testing.assert_allclose(recovered_data, data.array, rtol=1e-1, atol=1e-2)

            # rescale the original data with scale and offset and compare with what saved
            tmp = data.array * scale + offset
            tmp = np.asarray(tmp, dtype=dtype)
        else:
            tmp = data.array
            
        assert tmp.dtype == read_array.dtype
        
        np.testing.assert_array_equal(tmp, read_array)

class Test_HDF5_utilities(unittest.TestCase):
    def setUp(self) -> None:
        self.path = os.path.join(os.path.abspath(data_dir), '24737_fd_normalised.nxs')

        
        self.dset_path ='/entry1/tomo_entry/data/data'


    def test_print_metadata(self):
        devnull = open(os.devnull, 'w') #suppress stdout
        with patch('sys.stdout', devnull):
            HDF5_utilities.print_metadata(self.path)    


    def test_get_dataset_metadata(self):
        dset_dict = HDF5_utilities.get_dataset_metadata(self.path, self.dset_path)

        dict_by_hand  ={'ndim': 3, 'shape': (91, 135, 160), 'size': 1965600, 'dtype': np.float32, 'compression': None, 'chunks': None, 'is_virtual': False}
        self.assertDictContainsSubset(dict_by_hand,dset_dict)


    def test_read(self):

        data_full = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()

        # full dataset
        data_read_full = HDF5_utilities.read(self.path, self.dset_path)
        np.testing.assert_allclose(data_full.array,data_read_full)

        # subset of input
        subset = np.s_[44:45,70:90:2,80]
        data_read_subset = HDF5_utilities.read(self.path, self.dset_path, subset)
        self.assertTrue(data_read_subset.dtype == np.float32)
        np.testing.assert_allclose(data_full.array[subset],data_read_subset)

        # read as dtype
        subset = np.s_[44:45,70:90:2,80]
        data_read_dtype = HDF5_utilities.read(self.path, self.dset_path, subset, dtype=np.float64)
        self.assertTrue(data_read_dtype.dtype == np.float64)
        np.testing.assert_allclose(data_full.array[subset],data_read_dtype)


    def test_read_to(self):
        data_full = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()

        # full dataset
        data_full_out = np.empty_like(data_full.array, dtype=np.float32)
        HDF5_utilities.read_to(self.path, self.dset_path, data_full_out)
        np.testing.assert_allclose(data_full.array,data_full_out)

        # subset of input, continuous output
        subset = np.s_[44:45,70:90:2,80]
        data_subset_out = np.empty((1,10), dtype=np.float32)
        HDF5_utilities.read_to(self.path, self.dset_path, data_subset_out, source_sel=subset)
        np.testing.assert_allclose(data_full.array[subset],data_subset_out)

        # subset of input, continuous output, change of dtype
        subset = np.s_[44:45,70:90:2,80]
        data_subset_out = np.empty((1,10), dtype=np.float64)
        HDF5_utilities.read_to(self.path, self.dset_path, data_subset_out, source_sel=subset)
        np.testing.assert_allclose(data_full.array[subset],data_subset_out)

        # subset of input written to subset of  output
        data_partial_by_hand = np.zeros_like(data_full.array, dtype=np.float32)
        data_partial = np.zeros_like(data_full.array, dtype=np.float32)

        data_partial_by_hand[subset] = data_full.array[subset]

        HDF5_utilities.read_to(self.path, self.dset_path, data_partial, source_sel=subset, dest_sel=subset)
        np.testing.assert_allclose(data_partial_by_hand,data_partial)


class TestNikonReader(unittest.TestCase):

    def test_setup(self):

        reader = NikonDataReader()
        self.assertEqual(reader.file_name, None)
        self.assertEqual(reader.roi, None)
        self.assertTrue(reader.normalise)
        self.assertEqual(reader.mode, 'bin')
        self.assertFalse(reader.fliplr)

        roi = {'vertical':(1,-1),'horizontal':(1,-1),'angle':(1,-1)}
        reader = NikonDataReader(file_name=None, roi=roi, normalise=False, mode='slice', fliplr=True)
        self.assertEqual(reader.file_name, None)
        self.assertEqual(reader.roi, roi)
        self.assertFalse(reader.normalise)
        self.assertEqual(reader.mode, 'slice')
        self.assertTrue(reader.fliplr)

        with self.assertRaises(FileNotFoundError):
            reader = NikonDataReader(file_name='no-file')
        

class TestZeissReader(unittest.TestCase):

    def test_setup(self):

        reader = ZEISSDataReader()
        self.assertEqual(reader.file_name, None)

        with self.assertRaises(FileNotFoundError):
            reader = ZEISSDataReader(file_name='no-file')


class TestTiffUtilities(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.join(data_dir, 'resolution_chart.tiff')

        self.x_ind = (128,192)
        self.y_ind = (64,128)
        self.test_full_shape_out = (256,256)
        
        self.test_crop_roi = [self.x_ind[0], self.y_ind[0], self.x_ind[1], self.y_ind[1]]
        self.test_crop_shape_out = (self.y_ind[1]-self.y_ind[0] , self.x_ind[1]- self.x_ind[0])

        self.cropped_arr_mean= 205.647705078125     
        self.full_arr_mean= 225.91497802734375

        self.test_dtype = np.uint8

    @patch('PIL.Image.open')
    @patch.object(Tiff_utilities, '_get_file_type', return_value='uint8')
    def test_get_dataset_metadata_mock(self, mock_get_file_type, mock_open):
        mock_img = MagicMock()
        mock_img.size = (10, 20)
        mock_open.return_value.__enter__.return_value = mock_img

        test_filename = '/path/to/test/file'
        metadata = Tiff_utilities.get_dataset_metadata(test_filename)
        mock_open.assert_called_once_with(test_filename)
        mock_get_file_type.assert_called_once_with(mock_img)
        self.assertEqual(metadata, {'dtype': 'uint8', 'height': 20, 'width': 10})

    def test_get_file_type_mock(self):
        # Create a mock image for each mode
        modes = ['1', 'L', 'F', 'I', 'I;16']
        dtypes = [np.bool_, np.uint8, np.float32, np.int32, np.uint16]

        for mode, dtype in zip(modes, dtypes):
            mock_img = MagicMock()
            mock_img.mode = mode

            # Call the _get_file_type method
            result_dtype = Tiff_utilities._get_file_type(mock_img)

            # Check that the returned dtype is correct
            self.assertEqual(result_dtype, dtype)

        # Test that a ValueError is raised for an unsupported mode
        mock_img = MagicMock()
        mock_img.mode = 'unsupported'
        with self.assertRaises(ValueError):
            Tiff_utilities._get_file_type(mock_img)

    def test_get_dataset_metadata(self):
        metadata = Tiff_utilities.get_dataset_metadata(self.test_path)
        self.assertEqual(metadata, {'dtype': np.uint8, 'height': 256, 'width': 256})

    def test_get_file_type(self):
        img = Image.open(self.test_path)
        dtype = Tiff_utilities._get_file_type(img)
        self.assertEqual(dtype, np.uint8)

    def test_read(self):
        # Call the read method with the crop roi
        cropped_arr = Tiff_utilities.read(self.test_path, self.test_crop_roi, self.test_dtype)

        # Call the read method with the full image
        full_arr = Tiff_utilities.read(self.test_path, None, self.test_dtype)

        # Check the shape of the returned arrays
        self.assertEqual(cropped_arr.shape, self.test_crop_shape_out)
        self.assertEqual(full_arr.shape, self.test_full_shape_out)

        # Check the dtype of the returned arrays
        self.assertEqual(cropped_arr.dtype, self.test_dtype)
        self.assertEqual(full_arr.dtype, self.test_dtype)

        # Check that the mean of the returned arrays is correct (i.e. both images aren't empty)
        self.assertAlmostEqual(cropped_arr.mean(), self.cropped_arr_mean, places=3)        
        self.assertAlmostEqual(full_arr.mean(), self.full_arr_mean, places=3)        

        # Compare the cropped array with the corresponding part of the full array
        np.testing.assert_array_equal(cropped_arr, full_arr[self.y_ind[0]:self.y_ind[1], self.x_ind[0]:self.x_ind[1]])

        # Check read with forced dtype
        forced_dtype = np.float32
        forced_arr = Tiff_utilities.read(self.test_path, None, forced_dtype)
        self.assertEqual(forced_arr.dtype, forced_dtype)
        self.assertAlmostEqual(forced_arr.mean(), self.full_arr_mean, places=3)

    def test_read_to(self):
        # Call the read and read_to methods with the crop roi
        gold_full = Tiff_utilities.read(self.test_path, dtype=self.test_dtype)
        gold_crop = gold_full[self.y_ind[0]:self.y_ind[1], self.x_ind[0]:self.x_ind[1]]

        # Test when filling a full array
        out = np.empty(self.test_full_shape_out, dtype=self.test_dtype)
        Tiff_utilities.read_to(self.test_path, out)

        # Check that the read_to method returns the same array as the read method
        np.testing.assert_array_equal(out, gold_full)

        # Check the dtype of the returned array
        self.assertEqual(out.dtype, self.test_dtype)

        # Check read_to with forced dtype
        forced_dtype = np.float32
        forced_out = np.empty(self.test_full_shape_out, dtype=forced_dtype)
        Tiff_utilities.read_to(self.test_path, forced_out, None, None)
        self.assertEqual(forced_out.dtype, forced_dtype)
        np.testing.assert_array_equal(out, np.asarray(gold_full,dtype=np.float32))

        # Check read_to with filling a subset of an array
        out = np.zeros((3, *self.test_full_shape_out), dtype=self.test_dtype)
        Tiff_utilities.read_to(self.test_path, out, (slice(1,2),slice(None),slice(None)))
        np.testing.assert_array_equal(out[1], gold_full)
        np.testing.assert_array_equal(out[0], 0)
        np.testing.assert_array_equal(out[2], 0)

        # Check read_to with cropped ROI
        out = np.zeros(self.test_crop_shape_out, dtype=self.test_dtype)
        Tiff_utilities.read_to(self.test_path, out, crop_roi=self.test_crop_roi)

        # Check that the read_to method returns the same array as the read method
        np.testing.assert_array_equal(out, gold_crop)

        # Check read_to with cropped ROI and dest_roi
        out = np.zeros(self.test_full_shape_out, dtype=self.test_dtype)
        Tiff_utilities.read_to(self.test_path, out,np.s_[self.y_ind[0]:self.y_ind[1],self.x_ind[0]:self.x_ind[1]] ,crop_roi=self.test_crop_roi)

        # Check that the read_to method returns the same array as the read method
        np.testing.assert_array_equal(out[self.y_ind[0]:self.y_ind[1],self.x_ind[0]:self.x_ind[1]], gold_crop)



class TestTiffReader(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory
        cls.test_dir = tempfile.mkdtemp()

        # Create a numpy array
        np.random.seed(0)
        #cls.test_data = np.random.randint(0, 65535, (3,8,16), dtype=np.uint16)
        cls.test_data = np.arange(3*8*16).reshape((3,8,16)).astype(np.uint16)

        # Create a JSON file with the scale and offset
        utilities.save_scale_offset(cls.test_dir, 50000, -10)
        cls.test_data_scaled = (cls.test_data + 10)/50000

        # Write the numpy array as TIFFs to the temporary directory
        for i in range(cls.test_data.shape[0]):
            img = Image.fromarray(cls.test_data[i])
            img.save(os.path.join(cls.test_dir, f'proj_{i}.tiff'))

        cls.flatfield = np.random.randint(0, 65535, (8,16), dtype=np.uint16)
        img = Image.fromarray(cls.flatfield)
        img.save(os.path.join(cls.test_dir, f'flatfield.tiff'))

        cls.darkfield = np.zeros((8,16), dtype=np.uint8)
        img = Image.fromarray(cls.darkfield)
        img.save(os.path.join(cls.test_dir, f'darkfield.tiff'))

        # Add some extra files to the temporary directory
        with open(os.path.join(cls.test_dir, f'some_parameters.txt'), 'w') as f:
            f.write('Metadata')


        cls.paths = []
        cls.paths.append(os.path.join(cls.test_dir, f'darkfield.tiff'))
        cls.paths.append(os.path.join(cls.test_dir, f'flatfield.tiff'))
        cls.paths += [os.path.join(cls.test_dir, f'proj_{i}.tiff') for i in range(3)]

    @classmethod
    def tearDownClass(cls):
        # Delete the temporary directory
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_read_scale_offset(self):
        """
        Test that the scale and offset are correctly read from a JSON file
        This function is called by the __init__ method, but is in io.utilities
        The test is here as it makes use of the temporary directory created in setUpClass
        """
        scale, offset = utilities.read_scale_offset(self.test_dir)
        self.assertEqual(scale, 50000)
        self.assertEqual(offset, -10)

    def test_set_file_paths(self):
        """
        Test that _set_file_paths correctly sets the _tiff_files attribute
        This method is called by the __init__ method, but is in io.TIFF_utilities
        The test is here as it makes use of the temporary directory created in setUpClass
        """
        
        # Test with one tiff file
        found_tiffs = Tiff_utilities.get_tiff_paths(self.paths[3])
        self.assertEqual(found_tiffs, [self.paths[3]])

        # Test with a custom order list of tiff files
        found_tiffs = Tiff_utilities.get_tiff_paths([self.paths[2],self.paths[0], self.paths[4]])
        self.assertEqual(found_tiffs, [self.paths[2],self.paths[0],self.paths[4]])

        # Test reading all tiffs from a directory in sorted order
        found_tiffs = Tiff_utilities.get_tiff_paths(self.test_dir)
        self.assertEqual(found_tiffs, self.paths)

        # Test reading partial matches in a directory in sorted order
        found_tiffs = Tiff_utilities.get_tiff_paths(os.path.join(self.test_dir, 'proj'))
        self.assertEqual(found_tiffs, self.paths[2::])

        # Test that _check_tiff_files raises a FileNotFoundError if no TIFF files are found
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(FileNotFoundError):
                found_tiffs = Tiff_utilities.get_tiff_paths(empty_dir)

        # Test that _check_tiff_files raises a ValueError if _file_name is not a list, a file, or a directory
        with self.assertRaises(ValueError):
            found_tiffs = Tiff_utilities.get_tiff_paths(12345)  # _file_name is an integer


    def test_init(self):
        # Test that a TiffReader can be created with a file name
        reader = TIFFReader(self.test_dir)
        self.assertIsInstance(reader, TIFFReader)

        # Test that an ImportError is raised if PIL is not available
        with patch('cil.io.utilities.Tiff_utilities.pilAvailable', False):
            with self.assertRaises(ImportError):
                reader = TIFFReader(self.test_dir)

        # Test that a ValueError is raised if the length of dimension_labels is not correct
        with self.assertRaises(ValueError):
            reader = TIFFReader(self.test_dir, dimension_labels=['images'])

        # Test that a ValueError is raised if the length of dimension_labels is not correct
        with self.assertRaises(ValueError):
            reader = TIFFReader(self.test_dir, dimension_labels=['images', 'x', 'y', 'z'])

        # 3 labels are allowed for a single image as the 1st label is ignored but it uses logging.info
        with patch('logging.info') as mock_info:
            reader = TIFFReader(self.paths[2], dimension_labels=('x', 'y', 'z'))

        logged_messages = [call[0][0] for call in mock_info.call_args_list]
        self.assertTrue(any("dimension_labels length should match the input dimensions. Using ('y', 'z')." in message for message in logged_messages))


    def test_set_input_geometry(self):
        # Test with ImageGeometry
        reader = TIFFReader(os.path.join(self.test_dir, 'proj'))
        geometry = ImageGeometry(voxel_num_x=16, voxel_num_y=8, voxel_num_z=3)
        reader.set_input_geometry(geometry)
        self.assertEqual(reader._geometry_full, geometry)

        # Test with AcquisitionGeometry
        geometry = AcquisitionGeometry.create_Parallel3D().set_panel([16,8]).set_angles([0,1,2])
        reader.set_input_geometry(geometry)
        self.assertEqual(reader._geometry_full, geometry)

        reader.set_input_geometry(None)
        self.assertIsNone(reader._geometry_full)

        # Test with incompatible geometry
        geometry = ImageGeometry(voxel_num_x=50, voxel_num_y=50)
        with self.assertRaises(ValueError):
            reader.set_input_geometry(geometry)

        # Test with unsupported geometry type
        with self.assertRaises(TypeError):
            reader.set_input_geometry('unsupported type')


    def test_set_rescale(self):   

        reader = TIFFReader(self.paths, rescale=False)
        self.assertIsNone(reader._rescale_pwop)

        # From file scale= 50000 offset= -10
        reader = TIFFReader(self.paths, rescale=True)
        self.assertEquals(reader._rescale_pwop(50000), (50000+10)/50000)

        # Test with rescale=False
        reader.set_rescale(False)
        self.assertIsNone(reader._rescale_pwop)

        # Mock the read_scale_offset function to return (2, 1)
        with patch('cil.io.utilities.read_scale_offset', return_value=(2, 1)):
            reader.set_rescale(True)
        self.assertIsNotNone(reader._rescale_pwop)
        self.assertEquals(reader._rescale_pwop(2), 0.5)

        # Test with missing JSON file``
        # Mock the read_scale_offset function to raise a FileNotFoundError
        with patch('cil.io.utilities.read_scale_offset', side_effect=FileNotFoundError):
            reader.set_rescale(True)
            self.assertIsNone(reader._rescale_pwop)


    def test_set_custom_pixelwise_operation(self):

        reader = TIFFReader(self.paths, rescale=False)

        # Test with Non
        reader.set_custom_pixelwise_operation(None)
        self.assertIsNone(reader._custom_pwop)

        # Test with a custom function
        custom_function = lambda x: x * 0.5 

        reader.set_custom_pixelwise_operation(custom_function)
        self.assertIsNotNone(reader._custom_pwop)

        # Test the custom function
        self.assertEquals(reader._custom_pwop(1), 0.5)


    def test_set_combined_pwop(self):

        arr_in = np.array([65534, 50000, 49546, 525,52054], dtype=np.uint16)

        # From file scale= 50000 offset= -10
        scale = 50000
        offset = -10
        reader = TIFFReader(self.paths, rescale=True)

        # # Test with only rescale operation
        reader_op = reader._get_combined_pwop()
        expected_result = (arr_in.astype(np.float32) - offset) / scale

        arr_calc = reader_op(arr_in)
        np.testing.assert_array_almost_equal(arr_calc, expected_result)
        self.assertEqual(arr_calc.dtype, np.float32)            

        # Test with custom operations -log10(x) and rescale
        custom_function = lambda x: -np.log10(x)
        reader._custom_pwop = custom_function
        reader_op = reader._get_combined_pwop()

        expected_result = -np.log10(expected_result)

        arr_calc = reader_op(arr_in)
        np.testing.assert_array_almost_equal(arr_calc, expected_result)
        self.assertEqual(arr_calc.dtype, np.float32)

        # Test with only custom operation
        reader._rescale_pwop = None
        reader_op = reader._get_combined_pwop()

        expected_result = -np.log10(arr_in.astype(np.float32))
        arr_calc = reader_op(arr_in)
        np.testing.assert_array_almost_equal(arr_calc, expected_result)
        self.assertEqual(arr_calc.dtype, np.float32)

        # Test with no operations
        reader._custom_pwop = None
        reader_op = reader._get_combined_pwop()
        self.assertIsNone(reader_op)

        # Test with different output dtype
        reader._custom_pwop = custom_function
        reader_op = reader._get_combined_pwop(dtype=np.float64)
        arr_calc = reader_op(arr_in)
        np.testing.assert_array_almost_equal(arr_calc, expected_result)
        self.assertEqual(arr_calc.dtype, np.float64)


    def test_get_image_paths(self):

        reader = TIFFReader(self.test_dir)
        image_list = reader.get_image_paths()
        self.assertEqual(image_list, reader._tiff_files)

        image_list.remove(os.path.join(self.test_dir, 'proj_0.tiff'))
        self.assertNotEqual(image_list, reader._tiff_files)

    def get_result(self, reader, method, *args):
        """
        Used to test functionality of read, read_binned and read_sliced
        """
        
        if method == 'read':
            return reader.read()
        elif method == 'read_binned':
            return reader.read_binned(*args)
        elif method == 'read_sliced':
            return reader.read_sliced(*args)
        
    def read_as_data_container(self, read_method):

        # Test with default parameters for a single image
        reader = TIFFReader(self.paths[2])
        result = self.get_result(reader, read_method)
        self.assertIsInstance(result, DataContainer)
        self.assertTupleEqual(result.dimension_labels, ('image_height', 'image_width'))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_almost_equal(result.as_array(), self.test_data_scaled[0])

        # Test toggling rescale
        reader.set_rescale(False)
        result = self.get_result(reader, read_method)
        np.testing.assert_array_almost_equal(result.as_array(), self.test_data[0])
        reader.set_rescale(True)
        result = self.get_result(reader, read_method)
        np.testing.assert_array_almost_equal(result.as_array(), self.test_data_scaled[0])

        # Test with default parameters for multiple images
        reader = TIFFReader(self.paths[2::])
        result = self.get_result(reader, read_method)
        self.assertIsInstance(result, DataContainer)
        self.assertTupleEqual(result.dimension_labels, ('images', 'image_height', 'image_width'))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_almost_equal(result.as_array(), self.test_data_scaled)

        # Test with custom dimension labels, dtype and rescale=False
        # Test for a single image
        reader = TIFFReader(self.paths[2], False, dimension_labels=('y', 'x'), dtype=np.complex64)
        result = self.get_result(reader, read_method)
        self.assertIsInstance(result, DataContainer)
        self.assertTupleEqual(result.dimension_labels, ('y', 'x'))
        self.assertEqual(result.dtype, np.complex64)
        np.testing.assert_array_almost_equal(np.real(result.as_array()), self.test_data[0])

        # Test multiple images
        reader = TIFFReader(self.paths[2::], False, dimension_labels=('z', 'y', 'x'), dtype=np.complex64)
        result = self.get_result(reader, read_method)
        self.assertIsInstance(result, DataContainer)
        self.assertTupleEqual(result.dimension_labels, ('z', 'y', 'x'))
        self.assertEqual(result.dtype, np.complex64)
        np.testing.assert_array_almost_equal(np.real(result.as_array()), self.test_data)


    def read_with_geometry(self, read_method, geometry=None):

        if geometry is None:
            return_class = DataContainer
            geometry_2D = None
            geometry_3D = None
        elif geometry == 'ag':
            return_class = AcquisitionData
            geometry_3D = AcquisitionGeometry.create_Parallel3D().set_panel([16,8]).set_angles([0,1,2])
            geometry_2D = AcquisitionGeometry.create_Parallel2D().set_panel(16).set_angles(np.linspace(0, np.pi, 8, endpoint=False))
        elif geometry == 'ig':
            return_class = ImageData
            geometry_3D = ImageGeometry(voxel_num_x=16, voxel_num_y=8, voxel_num_z=3)
            geometry_2D = ImageGeometry(voxel_num_x=16, voxel_num_y=8)

        # Read with default parameters for a single image
        reader = TIFFReader(self.paths[2::], True, dimension_labels=('z', 'y', 'x'), dtype=np.float64)

        # Read with geometry (ignores defaults)
        reader.set_input_geometry(geometry_3D)
        result = self.get_result(reader, read_method)
        self.assertIsInstance(result, return_class)
        self.assertTupleEqual(result.dimension_labels, geometry_3D.dimension_labels)
        self.assertEqual(result.dtype, geometry_3D.dtype)
        np.testing.assert_array_almost_equal(result.as_array(), self.test_data_scaled)

        # Change the dtype of the geometry
        geometry_3D.dtype = np.complex64
        reader.set_input_geometry(geometry_3D)
        result = self.get_result(reader, read_method)
        self.assertEqual(result.dtype, np.complex64)

        # Set the geometry to None
        reader.set_input_geometry(None)
        result = self.get_result(reader, read_method)
        self.assertIsInstance(result, DataContainer)
        self.assertTupleEqual(result.dimension_labels, ('z', 'y', 'x'))
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_almost_equal(result.as_array(), self.test_data_scaled)

        # read 2D
        reader = TIFFReader(self.paths[2], True)
        reader.set_input_geometry(geometry_2D)
        result = self.get_result(reader, read_method)
        self.assertIsInstance(result, return_class)
        self.assertTupleEqual(result.dimension_labels, geometry_2D.dimension_labels)
        self.assertEqual(result.dtype, geometry_2D.dtype)
        np.testing.assert_array_almost_equal(result.as_array(), self.test_data_scaled[0])

        # Test toggling rescale
        reader.set_rescale(False)
        result = self.get_result(reader, read_method)
        np.testing.assert_array_almost_equal(result.as_array(), self.test_data[0])
        reader.set_rescale(True)
        result = self.get_result(reader, read_method)
        np.testing.assert_array_almost_equal(result.as_array(), self.test_data_scaled[0])

        # add custom pwop
        custom_function = lambda x: -np.log10(x)
        reader.set_custom_pixelwise_operation(custom_function)
        result = self.get_result(reader, read_method)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_almost_equal(result.as_array(), -np.log10(self.test_data_scaled[0]))
                    

    def downsampling(self, read_method, geometry=None):
        
        if read_method == 'read_binned':
            DownSampler = Binner
        else:
            DownSampler = Slicer

        if geometry is None:
            return_class = DataContainer
            geometry_2D = None
            geometry_3D = None
        elif geometry == 'ag':
            return_class = AcquisitionData
            geometry_3D = AcquisitionGeometry.create_Parallel3D().set_panel([16,8]).set_angles([0,1,2])
            geometry_2D = AcquisitionGeometry.create_Parallel2D().set_panel(16).set_angles(np.linspace(0, np.pi, 8, endpoint=False))
        elif geometry == 'ig':
            return_class = ImageData
            geometry_3D = ImageGeometry(voxel_num_x=16, voxel_num_y=8, voxel_num_z=3)
            geometry_2D = ImageGeometry(voxel_num_x=16, voxel_num_y=8)

        # Test with default parameters for a single image
        reader = TIFFReader(self.paths[2], False)
        reader.set_input_geometry(geometry_2D)

        data_full = reader.read()

        downsample_y_ls = [(2,-2, 2),(0,-1, 1),(None,None, 2),(None,None,3),(2,-2, 2),(0,-1,3)]
        downsample_x_ls = [(2,-2, 2),(0,-1, 1),(None,None, 2),(None,None,3),(2,-2, 2),(0,-1,3)]


        for downsample_x, downsample_y in zip(downsample_x_ls, downsample_y_ls):

            try:
                data_out = self.get_result(reader, read_method, None, slice(*downsample_y), slice(*downsample_x))
            except Exception as e:
                self.fail(f"An exception of type {type(e).__name__} occurred.\nMessage: {e.args}.\n\
                        In unit test with read_method={read_method}, geometry={geometry}, downsample_x={downsample_x}, downsample_y={downsample_y}")

            roi = {reader.dimension_labels[0]:downsample_y,reader.dimension_labels[1]:downsample_x}
            data_gold = DownSampler(roi)(data_full)

            self.assertIsInstance(data_out, return_class, msg=f"downsample_x={downsample_x}, downsample_y={downsample_y}")
            self.assertTupleEqual(data_out.dimension_labels, data_gold.dimension_labels, msg=f"downsample_x={downsample_x}, downsample_y={downsample_y}")
            self.assertEqual(data_out.dtype, np.float32, msg=f"downsample_x={downsample_x}, downsample_y={downsample_y}")
            np.testing.assert_array_almost_equal(data_out.as_array(), data_gold.as_array(), err_msg=f"downsample_x={downsample_x}, downsample_y={downsample_y}")
            self.assertEqual(data_out.geometry, data_gold.geometry, msg=f"downsample_x={downsample_x}, downsample_y={downsample_y}")


        # Test with default parameters for multiple images
        reader = TIFFReader(self.paths[2::])
        reader.set_input_geometry(geometry_3D)

        data_full = reader.read()
        data_out = self.get_result(reader, read_method, slice(1,3,2), slice(2,-2, 2), slice(2,-2, 2))

        roi = {reader.dimension_labels[0]:(1,3,2),reader.dimension_labels[1]:(2,-2, 2),reader.dimension_labels[2]:(2,-2, 2)}
        data_gold = DownSampler(roi)(data_full)

        self.assertIsInstance(data_out, return_class)
        self.assertTupleEqual(data_out.dimension_labels, data_gold.dimension_labels)
        self.assertEqual(data_out.dtype, np.float32)
        np.testing.assert_array_almost_equal(data_out.as_array(), data_gold.as_array())
        self.assertEqual(data_out.geometry, data_gold.geometry)

        
        #TODO if custom pwop then this is applied after binning. These tests all apply it after



    def test_read(self):

        # Check all basic read functionality
        self.read_as_data_container('read')
        self.read_with_geometry('read', 'ag')
        self.read_with_geometry('read', 'ig')
        
    def test_read_sliced(self):

        # Check all basic read functionality
        self.read_as_data_container('read_sliced')
        self.read_with_geometry('read_sliced', 'ag')
        self.read_with_geometry('read_sliced', 'ig')

        # # Check with downsampling
        self.downsampling('read_sliced',geometry=None)
        self.downsampling('read_sliced',geometry='ag')
        self.downsampling('read_sliced',geometry='ig')


    def test_read_binned(self):

        # # Check all basic read functionality
        self.read_as_data_container('read_binned')
        self.read_with_geometry('read_binned', 'ag')
        self.read_with_geometry('read_binned', 'ig')

        # # Check with downsampling
        self.downsampling('read_binned',geometry=None)
        self.downsampling('read_binned',geometry='ag')
        self.downsampling('read_binned',geometry='ig')