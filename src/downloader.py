import os
import re
import json
import yaml

from sentinelsat.sentinel import SentinelAPI
from sentinelsat.sentinel import read_geojson, geojson_to_wkt


class Downloader:

    # norad identifiers
    norad = {   'Sentinel-1A': 39634,
                'Sentinel-1B': 41456,
                'Sentinel-2A': 40697,
                'Sentinel-2B': 42063,
                'Sentinel-3A': 41335,
                'Sentinel-3B': 0 }

    def __init__( self, args ):

        """
        constructor
        """

        try:

            # load config parameters from file
            with open(  args.downloader_config, 'r' ) as f:
                self._config = yaml.safe_load( f )
            
            # create api object
            self._api = SentinelAPI(    self._config[ 'user' ], 
                                        self._config[ 'password' ], 
                                        'https://scihub.copernicus.eu/dhus' )

            # initialise keywords
            self._keywords = ''
            if 'keywords' in self._config:

                # split keywords string on whitespace
                matches = re.findall(r'\w+=".+?"', self._config[ 'keywords' ] ) + re.findall(r'\w+=[\d.]+', self._config[ 'keywords' ] )
                matches = [ m.split('=', 1) for m in matches ]

                # use results to make a dict
                self._keywords = dict(matches)
                for key, value in self._keywords.copy().items():

                    m = re.search( '\(\d+(,\d+)*\)', value )
                    if m:
                        # create tuple from 2 numbers inside brackets separated by comma
                        a = re.findall(r'\d+', value)
                        if len( a ) == 2:
                            self._keywords [ key ] = [ a[ 0 ], a[ 1 ] ]


        # error opening config file
        except EnvironmentError: 
            print ( 'Unable to open config file: {}'.format( args.config_file ) )

        return


    def getProductList( self, start_dt='NOW-14DAYS', end_dt='NOW' ):

        """
        get product details satisfying spatiotemporal filter constraints
        """

        # query api by aoi footprint and start / datetimes
        footprint = geojson_to_wkt( read_geojson( self._config[ 'aoi' ] ) )
        products = self._api.query(     footprint,
                                        date=( start_dt, end_dt ),
                                        **self._keywords )

        return products


    def exists( self, product ):

        """
        check json metadata file exists signifying successful download
        """

        # get path and metadata
        pathname = None

        odata = self._api.get_product_odata( product, full=True )
        path = self.getPath( odata )

        # check download meta file exists
        download_file = path + '/download.json'
        if os.path.exists( download_file ):

            # extract pathname
            with open( download_file ) as jf:  
                data = json.load(jf)
        
                if os.path.exists( data[ 'path' ] ):
                    pathname = data[ 'path' ]

        return pathname


    def getProduct( self, product ):

        """
        download product to local directory
        """

        # get path and metadata
        odata = self._api.get_product_odata( product, full=True )
        path = self.getPath( odata )

        # create path if not exists
        if not os.path.exists( path ):
            os.makedirs( path, 0o755 )

        # dump meta data to file
        with open ( path + '/meta.json', 'w' ) as fp:
            fp.write( json.dumps( odata, indent=4, sort_keys=True, default=str ) )

        # download product file 
        print( 'Downloading {} -> {}'.format( odata[ 'Filename' ], path ) )
        out = self._api.download( product, directory_path=path )

        # signify successful download by creating download.json
        with open ( path + '/download.json', 'w' ) as fp:
            fp.write( json.dumps( out, indent=4, sort_keys=True, default=str ) )

        return out[ 'path' ]


    def getPath( self, odata ):

        """
        construct path for dataset
        """

        # replace tokens with values taken from meta data
        path = self._config[ 'root_path' ]

        path = path.replace( '!norad!', str( self.getNoradId( odata[ 'Satellite name' ] + odata[ 'Satellite number' ] ) ) )
        path = path.replace( '!datetime!', odata[ 'Sensing start' ].strftime( '%Y%m%d_%H%M%S' ) )

        # tile id optional include
        tid = self.getTileId( odata[ 'Filename' ] )
        if tid is not None:
            path = path.replace( '!tile!', tid )

        return path


    def getNoradId( self, platform  ):

        """
        convert platform name to norad id
        """

        nid = None

        # retrieve from dictionary
        if platform in self.norad:
            nid = self.norad[ platform ]

        if nid is None:
            raise ValueError( 'Unable to determine norad id for sentinel platform: {} '.format( platform ) )

        return nid


    def getTileId( self, filename ):

        """
        extract tile id from filename
        """

        # parse for date time sub directory
        tid = None

        m = re.findall( '^S2.*(T\d{2}[A-Z]{3}).*', filename )
        if len( m ) == 1 :
            tid = str( m[ 0 ] )

        return tid
