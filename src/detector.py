import cv2
import osr
import gdal
import random
import zipfile
import numpy as np
import numpy.ma as ma

from skimage.measure import label
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation

import matplotlib.pyplot as plt

class Detector:

    def __init__( self ):

        """
        constructor
        """

        # constants - as defined here: https://github.com/hfisser/Truck_Detection_Sentinel2_COVID19/blob/master/Detect_trucks_sentinel2_COVID19.ipynb
        self._min_rgb = 0.04
        self._max_red = 0.15
        self._max_green = 0.15
        self._max_blue  = 0.4

        # swir band
        self._min_b11 = 0.05
        self._max_b11 = 0.55

        # ratios
        self._max_ndvi = 0.7 
        self._max_ndwi = 0.001
        self._max_ndsi = 0.0001

        self._min_green_ratio = 0.05
        self._min_red_ratio = 0.1

        # compile scl categories into indexed dict
        names = [   'NODATA',
                    'SATURATED_DEFECTIVE',
                    'DARK_FEATURE_SHADOW',
                    'CLOUD_SHADOW',
                    'VEGETATION',
                    'BARE_SOIL_DESERT',
                    'WATER',
                    'UNCLASSIFIED',
                    'CLOUD_MEDIUM_PROBA',
                    'CLOUD_HIGH_PROBA',
                    'THIN_CIRRUS',
                    'SNOW_ICE' ]

        self._qa_lut = dict(zip( names, range( len(names) )) )
        return


    def process( self, scene, mask_pathname=None ):

        """
        main path of execution
        compile datasets, detect candidate locations of intradetector parallax effect and apply road mask 
        """

        # retrieve info from dataset file
        data, geo, qa = self.getDatasets( scene )

        # get pixels demonstrating possible parallax rainbow effect 
        candidates = self.getCandidateMap( data )
        candidates = np.bitwise_and( candidates, ~qa['MASK'] )

        if mask_pathname is None:

            #  derive road mask from multispectral signatures - prone to false detections
            roads = self.getRoadMask ( data )

        else:

            #  derive road mask from multispectral signatures - prone to false detections
            roads = self.loadRoadMask( mask_pathname, geo )

        #  mask candidates with road mask 
        candidates = candidates * roads

        # assign object ids to vehicle detections
        blobs = label( candidates )
        pts = np.nonzero( blobs )

        x = []; y = []
        for blob_id in range( 1, np.max( blobs ) + 1 ):

            idx = np.where( blobs[pts]==blob_id )

            # record mean x, y of detected object
            x.append( np.mean( pts[1][idx] ) )
            y.append( np.mean( pts[0][idx] ) )






        # create and show 24bit rgb image
        rgb = self.getRgbImage( data, [ 'B2', 'B3', 'B4' ] )
        plt.imshow( rgb )

        # plot locations of 'vehicles'
        plt.plot( x, y, 'or', markersize=20, fillstyle='none' )
        plt.show()

        # create plot canvas
        nrows = 4; ncols = 4; size = 64
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(80, 80))

        # iterate axes
        for row in range( nrows ):
            for col in range( ncols ):

                while True:
                
                    # randomly pick candidate location
                    idx = random.randrange( len( x ) )
                    x1 = int ( x[ idx ] - size ); y1 = int ( y[ idx ] - size )

                    # check valid image is plotable
                    if y1 > 0 and x1 > 0 and y1 < ( data[ 'B4' ].shape[0] - size ) and x1 < ( data[ 'B4' ].shape[1] - size ):
                        break

                # plot rgb subimage
                axes[ row ][ col ].imshow( rgb [ y1: (y1+size*2), x1 : (x1+size*2), : ] )

                # clear ticks
                axes[ row ][ col ].get_xaxis().set_ticks([])
                axes[ row ][ col ].get_yaxis().set_ticks([])

        # tighten up
        fig.tight_layout()        
        plt.show()

        return


    def getDatasets( self, scene ):

        """
        load channel reflectances and scene classification layer for qa
        upscale band 11 and scene classification layer to 10m resolution
        """

        # open sentinel-2 dataset
        ds = gdal.Open( scene, gdal.GA_ReadOnly )
        if ds is not None:

            # read required subdatasets into numpy array dictionary
            meta = ds.GetMetadata('SUBDATASETS')
            data = self.getSubDatasets( meta[ 'SUBDATASET_1_NAME' ],  [ 'B2', 'B3', 'B4', 'B8' ] )

            # add B11 and upscale to 10m
            data.update ( self.getSubDatasets( meta[ 'SUBDATASET_2_NAME' ],  [ 'B11' ] ) )
            data[ 'B11' ] = cv2.resize( data[ 'B11'], dsize=data[ 'B2' ].shape, interpolation=cv2.INTER_CUBIC )
            
            # convert 16bit channel data to masked reflectance images
            for key, value in data.items():
                    data[ key ] = np.asarray( value, dtype=np.float ) / 10000.0

            # retrieve geolocation info
            geo = self.getGeoParameters( meta[ 'SUBDATASET_1_NAME' ] )

            # load SCL image and upscale to 10m
            qa = self.getSclDataset( scene )
            qa[ 'SCL' ] = cv2.resize( qa[ 'SCL'], dsize=data[ 'B2' ].shape, interpolation=cv2.INTER_NEAREST )
            
            # generate cloud mask from SCL
            qa[ 'CLOUDS' ] = np.asarray(    ( qa['SCL'] == self._qa_lut['CLOUD_SHADOW'] ) |
                                            ( qa['SCL'] == self._qa_lut['CLOUD_MEDIUM_PROBA'] ) |
                                            ( qa['SCL'] == self._qa_lut['CLOUD_HIGH_PROBA'] ) )

            # remove isolated pixels and dilate cloud mask
            qa[ 'CLOUDS' ] = binary_erosion( qa[ 'CLOUDS' ] ).astype( qa[ 'CLOUDS' ].dtype )
            qa[ 'CLOUDS' ] = binary_dilation( qa[ 'CLOUDS' ], iterations=30 ).astype( qa[ 'CLOUDS' ].dtype )

            # generate qa mask from SCL
            qa[ 'MASK' ] = np.asarray(  ( qa['SCL'] == self._qa_lut['NODATA'] ) |
                                        ( qa['SCL'] == self._qa_lut['SATURATED_DEFECTIVE'] ) |
                                        ( qa['SCL'] == self._qa_lut['WATER'] ) )

            # combine qa mask with cloud mask
            qa[ 'MASK' ] = np.bitwise_or( qa[ 'MASK' ], qa[ 'CLOUDS' ] )


        return data, geo, qa


    def getSubDatasets( self, name, channels ):

        """
        load subdataset reflectance arrays listed in argument 
        """

        # initialise properties
        data = None

        # open subdataset
        sds = gdal.Open( name, gdal.GA_ReadOnly )
        if sds is not None:

            # read bands into dict
            data = dict( zip( channels, [None]*len(channels) ) )
            for idx in range ( sds.RasterCount ):

                # get band metadata                
                band = sds.GetRasterBand( idx + 1 )
                meta = band.GetMetadata()

                # read data if name in channel list
                if meta[ 'BANDNAME' ] in channels:                
                    data[ meta[ 'BANDNAME' ] ] = band.ReadAsArray()
        
        return data


    def getGeoParameters( self, name ):

        """
        get subdataset geolocation parameters
        """

        # open subdataset
        geo = None

        sds = gdal.Open( name, gdal.GA_ReadOnly )
        if sds is not None:

            # retrieve geolocation parameters
            geo = { 'TRANSFORM' : sds.GetGeoTransform(),
                    'SRS' : osr.SpatialReference( wkt=sds.GetProjection() ),
                    'SHAPE' : ( sds.RasterYSize, sds.RasterXSize ) }

        return geo


    def getSclDataset( self, pathname ):

        """
        read scene classification layer into numpy array 
        """

        # open product zip
        data = None

        zz = zipfile.ZipFile( pathname )
        for f in zz.filelist:

            # locate scl dataset at 20m resolution
            if f.filename.find('SCL_20m') >= 0:

                # open and read subdataset 
                sds = gdal.Open('/vsizip/{}/{}'.format( pathname, f.filename ) )
                data = { 'SCL' : sds.GetRasterBand( 1 ).ReadAsArray() }

        return data


    def getCandidateMap( self, data ):

        """
        identify locations of likely intradetector parallax effect
        """

        # compute rgb ratios - look for evidence of intradetector parallax effects
        bg_ratio = ( data[ 'B2' ] - data[ 'B3' ] ) / ( data[ 'B2' ] + data[ 'B3' ] ) 
        br_ratio = ( data[ 'B2' ] - data[ 'B4' ] ) / ( data[ 'B2' ] + data[ 'B4' ] )

        # threshold points with significant shift in red, green and blue values 
        bg_low = bg_ratio > self._min_green_ratio 
        br_low = br_ratio > self._min_red_ratio

        return bg_low * br_low


    def getRoadMask( self, data ):

        """
        generate road mask using multispectral reflectance
        prone to false detections
        """

        # compute thresholded band ratio products
        ndvi_mask = (( data[ 'B8' ] - data[ 'B4' ] ) / ( data[ 'B8' ] + data[ 'B4' ] ) ) < self._max_ndvi
        ndwi_mask = (( data[ 'B2' ] - data[ 'B11' ] ) / ( data[ 'B2' ] + data[ 'B11' ] ) ) < self._max_ndwi
        ndsi_mask = (( data[ 'B3' ] - data[ 'B11' ] ) / ( data[ 'B3' ] + data[ 'B11' ] ) ) < self._max_ndsi

        # compute thresholded rgb minmax products
        low_rgb_mask = ( data[ 'B2' ] > self._min_rgb ) * ( data[ 'B3' ] > self._min_rgb ) * ( data[ 'B4' ] > self._min_rgb )
        high_rgb_mask = ( data[ 'B2' ] < self._max_blue ) * ( data[ 'B3' ] < self._max_green ) * ( data[ 'B4' ] < self._max_red )

        # compute thresholded swir product
        b11_mask = (( data[ 'B11' ] - data[ 'B3' ] ) / ( data[ 'B11' ] + data[ 'B3' ] ) ) < self._max_b11 
        b11_mask_abs = ( data[ 'B11' ] > self._min_b11 ) * ( data[ 'B11' ] < self._max_b11 )

        # generate road mask
        return ndvi_mask * ndwi_mask * ndsi_mask * low_rgb_mask * high_rgb_mask * b11_mask * b11_mask_abs


    def loadRoadMask ( self, mask_pathname, geo ):

        """
        initialise road mask from file
        """

        # load road mask from file
        roads = None

        ds = gdal.Open( mask_pathname )
        if ds is not None:

            # check projection equivalence
            prj = osr.SpatialReference( wkt=ds.GetProjection() )
            if prj.GetAttrValue('AUTHORITY', 1 ) == geo[ 'SRS' ].GetAttrValue('AUTHORITY', 1 ):

                # check transform equivalence
                tfm = ds.GetGeoTransform()
                if tfm[ 1 ] == geo[ 'TRANSFORM' ][ 1 ] and tfm[ 5 ] == geo[ 'TRANSFORM' ][ 5 ]:

                    # get scene top left origin row / col location relative to mask
                    c1 = int ( ( geo[ 'TRANSFORM' ][ 0 ] - tfm[ 0 ] ) / tfm[ 1 ] )
                    r1 = int ( ( geo[ 'TRANSFORM' ][ 3 ] - tfm[ 3 ] ) / tfm[ 5 ] )

                    # get bottom right coordinates
                    c2 = c1 + geo[ 'SHAPE'][ 1 ]
                    r2 = c1 + geo[ 'SHAPE'][ 0 ]

                    # correct out-of-bounds pixel coordinates
                    c1 = max( c1, 0 ); r1 = max ( r1, 0 )
                    c2 = min( c2, ds.RasterXSize ); r2 = min( r2, ds.RasterYSize )

                    # read mask by offset
                    data = ds.GetRasterBand(1).ReadAsArray( c1, r1, c2 - c1, r2 - r1 )

                    # create roads mask                    
                    roads = np.zeros( geo[ 'SHAPE'], dtype=np.uint8 )
                    roads[ 0 : data.shape[ 0 ], 0: data.shape[ 1 ] ] = data

        return roads


    def getRgbImage( self, data, channels ):

        """
        create 24bit image from numpy floating point arrays
        """

        # create 24bit rgb image for display
        rgb = []
        for channel in channels:
            
            # rescale to 8bit
            limits = np.percentile( data[ channel ], [ 5, 95 ] )
            image = np.asarray(  ( ( data[ channel ] - limits[ 0 ] ) / ( limits[ 1 ] - limits[ 0 ] ) ) * 255 )
            
            # clip and append
            rgb.append( np.asarray( np.clip( image, 0, 255 ), np.uint8 ) )

        # stack list into 3d array
        return np.dstack( rgb )

