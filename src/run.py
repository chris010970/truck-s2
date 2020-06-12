import os
import time
import glob
import argparse

from detector import Detector
from downloader import Downloader


def parseArguments(args=None):

    """
    parse command line arguments
    """

    # parse configuration
    parser = argparse.ArgumentParser(description='truck-s2 product generator')
    parser.add_argument('-downloader_config', action='store', default=None)
    parser.add_argument('-scene', action='store', default=None)
    parser.add_argument('-scene_path', action='store', default=None)
    parser.add_argument('-mask_pathname', action='store', default=None)

    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()

    # get scene list
    scenes = []
    if args.downloader_config is not None:

        # create objects
        downloader = Downloader( args )

        # get product list
        products = downloader.getProductList( start_dt='20190401', end_dt='20200531' )
        print( 'Scenes targeted for download: {}'.format( len( products ) ) )
        
        for product in products:

            # check scene exists before download
            pathname = downloader.exists( product )
            if pathname is None:
                scenes.append( downloader.getProduct( product ) )

    # add single scene
    if args.scene is not None:
        scenes.append( args.scene )

    # add scenes in a directory
    if args.scene_path is not None:
        scenes.extend( glob.glob( os.path.join( args.scene_path, '**/S2*.zip', recursive=True ) ) )

    # execute truck detector for collated scenes
    detector = Detector()
    for scene in scenes:    
        detector.process( scene, args.mask_pathname )
        
    return


# execute main
if __name__ == '__main__':
    main()
