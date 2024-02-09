#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EODAL SENTINEL-2 QUERY EXAMPLE FOR PIXEL ANALYSIS
This script allows you to:
    _ query the sentinel-2 image collection using EODAL given
    _ downloading long timeseries of images spanning multiple years for
        a given bounding box geometry
    _ stocking the data as form of numpy array, with all images on the same grid,
        conveninet for vector operations and pixel analysis.

To adapt the script to your application, only the section marked with EDIT need
to be edited.

Created on Thu Apr 13 11:36:47 2023

@author: Fabio Oriani, Agroscope, fabio.oriani@agroscope.admin.ch
"""


import numpy as np
import geopandas as gpd
import rasterio
#import shapefile
#from osgeo import osr
from datetime import datetime, timedelta
from eodal.config import get_settings
from eodal.core.scene import SceneCollection
from eodal.core.sensors.sentinel2 import Sentinel2
from eodal.mapper.feature import Feature
from eodal.mapper.filter import Filter
from eodal.mapper.mapper import Mapper, MapperConfigs
from pathlib import Path
from typing import List
from pandas import Series
import geopandas
from scipy.interpolate import interp2d
import json
import re
import requests
from rasterio.warp import reproject, calculate_default_transform
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.enums import Resampling
import time


# STAC PROTOCOL SETTINGS: set to False to use a local data archive
Settings = get_settings()
Settings.USE_STAC = True

#%% USER FUNCTIONS

def preprocess_sentinel2_scenes(
    ds: Sentinel2, # this is a scene, i.e. a RasterCollection object
    target_resolution: int, 
    
    # ADD HERE MORE ARGUMENTS (E.G.packages for preprocessing the images) 
    # and add these also in the dictionary below  'scene_modifier_kwargs'
    ) -> Sentinel2:
    """
    Resample Sentinel-2 scenes and mask clouds, shadows, and snow
    # resample scene
    .resample(inplace=True, target_resolution=target_resolution)
    
    ask clouds, shadows, and snowbased on the Scene Classification Layer (SCL).
    
    	NOTE:
    		Depending on your needs, the pre-processing function can be
    		fully customized using the full power of EOdal and its
    		interfacing libraries!
    
    	:param target_resolution:
    		spatial target resolution to resample all bands to.
    	:returns:
    		resampled, cloud-masked Sentinel-2 scene.
	"""
    
    # resample scene
    ds.resample(inplace=True, target_resolution=target_resolution) 
    
    # mask clouds, shadows, but leave snow (class 11), see page 304 https://sentinel.esa.int/documents/247904/685211/sentinel-2-products-specification-document
    # Label Classification
    # 0 NO_DATA
    # 1 SATURATED_OR_DEFECTIVE
    # 2 DARK_AREA_PIXELS
    # 3 CLOUD_SHADOWS
    # 4 VEGETATION
    # 5 BARE_SOILS
    # 6 WATER
    # 7 UNCLASSIFIED 
    # 8 CLOUD_MEDIUM_PROBABILITY
    # 9 CLOUD_HIGH_PROBABILITY 
    # 10 THIN_CIRRUS
    # 11 SNOW /ICE
    ds.mask_clouds_and_shadows(inplace=True, cloud_classes=[1, 2, 3, 7, 8, 9, 10])   # MASKED BY EODAL
    
    return ds

def scene_to_array(sc,tx,ty):
    """
    Generate an numpy array (image stack) from a given Eodal SceneCollection.
    The scenes are resampled on a costant coordinate grid allowing pixel analysis.
    Missing data location are marked as nans.
    ft.write_geotiff(NDVI,
    Parameters
    ----------
    sc : Eodal SceneCollection
        The given Scene Collection generated from Eodal
    tx : Float Vector
        x coordinate vector for the resample grid.
    ty : Float Vector
        x coordinate vector for the resample grid.

    Returns
    -------
    im : float 4D numpy array.
        Array containing the stack of all scenes.
        4 dimensions: [x, y, bands, scenes]

    """
    
    ts = sc.timestamps # time stamps for each image
    bands = sc[sc.timestamps[0]].band_names # bands
    im_size = [len(ty),len(tx)] # image size
    im = np.empty(np.hstack([im_size,len(bands),len(ts)])) # preallocate matrix

    for i, scene_iterator in enumerate(sc):
        
        # REGRID SCENE TO BBOX AND TARGET RESOLUTION
        scene = scene_iterator[1]        
        for idx, band_iterator in enumerate(scene):
            
            # extract data with masked ones = 0
            band = band_iterator[1]
            Gv = np.copy(band.values.data)             
            Gv[band.values.mask==1]=0
            
            #original grid coordinates
            ny,nx = np.shape(Gv)
            vx = band.coordinates['x']
            vy = band.coordinates['y']
           
            # create interpolator
            
            Gv_no_nans = Gv.copy()
            Gv_no_nans[np.isnan(Gv)] = 0
            f = interp2d(vx,vy,Gv_no_nans,kind='linear',fill_value=0)
            
            # interpolate band on the target grid
            Tv = np.flipud(f(tx,ty))
            
             # assign interpolated band [i = scene , b = band]
            im[:,:,idx,i] = Tv.copy()
            del Tv
    
    return im

def imrisc(im,qmin=1,qmax=99): 
    """
    Percentile-based 0-1 rescale for multiband images

    Parameters
    ----------
    im : Float Array
        The image to rescale, can be multiband on the 3rd dimension
    qmin : Float Scalar
        Percentile to set the bottom of the value range e.g. 0.01
    qmax : Float Scalar
        Percentile to set the top of the value range e.g. 0.99

    Returns
    -------Quantile
    im_out : Float Array
        Rescaled image

    """

    if len(np.shape(im))==2:
        band=im.copy()
        band2=band[~np.isnan(band)]
        vmin=np.percentile(band2,qmin)
        vmax=np.percentile(band2,qmax)
        band[band<vmin]=vmin
        band[band>vmax]=vmax
        band=(band-vmin)/(vmax-vmin)
        im_out=band
    else:
        im_out=im.copy()
        for i in range(np.shape(im)[2]):
            band=im[:,:,i].copy()
            band2=band[~np.isnan(band)]
            vmin=np.percentile(band2,qmin)
            vmax=np.percentile(band2,qmax)
            band[band<vmin]=vmin
            band[band>vmax]=vmax
            band=(band-vmin)/(vmax-vmin)
            im_out[:,:,i]=band
            
    return im_out

def reproject_raster(src,out_crs):
    """
    REPROJECT RASTER reproject a given rasterio object into a wanted CRS.

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        rasterio dataset to reproject.
        For a geoTiff, it can be obtained from:    
        src = rasterio.open(file.tif,'r')
            
    out_crs : int
        epgs code of the wanted output CRS

    Returns
    -------
    dst : rasterio.io.DatasetReader
        output rasterio dataset written in-memory (rasterio MemoryFile)
        can be written to file with:
        
        out_meta = src.meta.copy()
        with rasterio.open('out_file.tif','w', **out_meta) as out_file: 
            out_file.write(dst.read().copy())
            
        out_file.close()

    """
    
    src_crs = src.crs
    transform, width, height = calculate_default_transform(src_crs, out_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    
    memfile = MemoryFile()
    
    kwargs.update({
        #'driver':'Gtiff',
        'crs': out_crs,
        'transform': transform,
        'width': width,
        'height': height,
        "BigTIFF" : "yes"})
    
    dst = memfile.open(**kwargs)

          
    for i in range(1, src.count + 1):
        reproject(
            source=rasterio.band(src, i),
            destination=rasterio.band(dst, i),
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=out_crs,
            resampling=Resampling.nearest)
    
    return dst


def download_SA3D_STAC(
        bbox_path: str, 
        out_crs: str, 
        out_res: float,
        out_path: str,
        server_url: str =  'https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d',
        product_res_label: str = '2'
    ):
    """
    DOWNLOAD_SA3D_STAC downloads the SwissAlti3D product for the bounding box 
    of a given shapefile and output resolution. The projection grid will start 
    from the lower and left box bounds. 
    
    Parameters
    ----------
    bbox_path : str
        path to the input shapefile, it can be a .gpkg or .shp with georef files
        in any crs
    out_crs : int
        epgs code of the output CRS (e.g. 4326)
    out_res : float
        output resolution
    out_path : str
        output .tif file path to create
    server_url : str, optional
       Swisstopo STAC server url for the product.
       The default is 'https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d'.
    product_res_label : str, optional
        the original product comes in 0.5 or 2-m resolution. 
        The default is '2'.

    Returns
    -------
    None.
    The projected DEM is written in out_path

    """
    
   
    
    # RETRIEVE TILE LINKS FOR DOWNLOAD
    
    shp = gpd.read_file(bbox_path).to_crs('epsg:4326') # WG84 necessary to the query
    lef = np.min(shp.bounds.minx)
    rig = np.max(shp.bounds.maxx)
    bot = np.min(shp.bounds.miny)
    top = np.max(shp.bounds.maxy)
    
    # If bbox is big divide the bbox in a series of chunks to override server limtiations
    cell_size = 0.05 #temporary bounding size in degree to define the query chunks 
    xbb = np.arange(lef,rig,cell_size)
    if xbb[-1] < rig:
        xbb = np.append(xbb,rig)
    ybb = np.arange(bot,top,cell_size)
    if ybb[-1] < top:
        ybb = np.append(ybb,top)
    
    files = []
    print("Querying tile links...")
    for i in range(len(xbb)-1):
        for j in range(len(ybb)-1):
            bbox_tmp = [xbb[i],ybb[j],xbb[i+1],ybb[j+1]]
            # construct bbox specification required by STAC
            bbox_expr = f'{bbox_tmp[0]},{bbox_tmp[1]},{bbox_tmp[2]},{bbox_tmp[3]}'
            # construct API GET call
            url = server_url + '/items?bbox=' + bbox_expr
            # send the request and check response code
            res_get = requests.get(url)
            res_get.raise_for_status()         
            # get content and extract the tile URLs
            content = json.loads(res_get.content)
            features = content['features']
            for feature in features:
                assets = feature['assets']
                tif_pattern = re.compile(r"^.*\.(tif)$")
                tif_files = [tif_pattern.match(key) for key in assets.keys()]
                tif_files =[x for x in tif_files if x is not None]
                tif_file = [x.string if x.string.find(f'_{product_res_label}_') > 0 \
                            else None for x in tif_files]
                tif_file = [x for x in tif_file if x is not None][0]
                # get download link
                link = assets[tif_file]['href']
                files.append(link)
    
    # query_parameters = {"downloadformat": "tif"}
    
    # reprojects tiles in out crs
    file_handler = []
    iter_max = len(files)
    t_start = time.time() # start clock
    print("Downloading tiles...")
    n = 0
    for row in files:
        n = n+1
        src = rasterio.open(row,'r')
        dst = reproject_raster(src,out_crs)
        file_handler.append(dst)
        t_end = time.time()
        print("\r %.2f %% completed, ET %.2f minutes, ETA %.2f minutes" % (n/iter_max*100,(t_end-t_start)/60,((t_end-t_start)/60)/(n/iter_max)*(1-n/iter_max)),end="\r")
    
    shp = gpd.read_file(bbox_fname).to_crs(out_crs) # WG84 necessary to the query
    lef = np.min(shp.bounds.minx)
    rig = np.max(shp.bounds.maxx)
    bot = np.min(shp.bounds.miny)
    top = np.max(shp.bounds.maxy)
    
    print('Merging tiles...')
    merge(datasets=file_handler, # list of dataset objects opened in 'r' mode
    bounds=(lef, bot, rig, top), # tuple
    nodata=0, # float
    dtype='uint16', # dtype
    res=out_res,
    resampling=Resampling.nearest,
    method='first', # strategy to combine overlapping rasters
    dst_path=out_path # str or PathLike to save raster
    )
    print('Output saved in ' + out_path)
    



#%% (EDIT) READ BBOX SHAPEFILE TO USE AS GEOMETRY IN THE QUERY
bbox_fname = "data/bbox_site2_ok.shp"

#%% (EDIT) EODAL QUERY PARAMETERS
   
# user-inputs
# -------------------------- Collection -------------------------------
collection: str = 'sentinel2-msi'

# ------------------------- Time Range ---------------------------------
time_start: datetime = datetime(2016,1,1)  		# year, month, day (incl.)
time_end: datetime = datetime(2022,12,31)     	# year, month, day (incl.)

# ---------------------- Spatial Feature  ------------------------------
geom: Path = Path(bbox_fname) # BBOX as geometry for the query
	
# ------------------------- Metadata Filters ---------------------------
metadata_filters: List[Filter] = [
  	Filter('cloudy_pixel_percentage','<', 30),
  	Filter('processing_level', '==', 'Level-2A')
    ]

#  ---------------------- query params for STAC  ------------------------------
# See available bands for sentinel-2 L2A here:
# https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/
# SCL (Scene Classification Layer) is added as extra band
scene_kwargs = {
    'scene_constructor': Sentinel2.from_safe,
    'scene_constructor_kwargs': {'band_selection': ['B01',
'B02',
'B03',
'B04',
'B05',
'B06',
'B07',
'B08',
'B8A',
'B09',
'B11',
'B12']},
    'scene_modifier': preprocess_sentinel2_scenes,
    'scene_modifier_kwargs': {'target_resolution': 10}
}

feature = Feature.from_geoseries(gpd.read_file(geom).geometry)

#%% DOWNLOAD THE IMAGES

# split the wanted date range in 1-yr chuncks to override download limit
date_vec = [time_start]
date_new = time_start + timedelta(days = 30)
n = 1
while date_new < time_end and n < 100:
    date_vec.append(date_new)
    date_new = date_new + timedelta(days = 30)        
    n = n+1
date_vec.append(time_end)

#%% define target grid based on original bbox (local crs) and target resolution

shp = gpd.read_file(bbox_fname)
lef = np.min(shp.bounds.minx)
rig = np.max(shp.bounds.maxx)
bot = np.min(shp.bounds.miny)
top = np.max(shp.bounds.maxy)

res = scene_kwargs['scene_modifier_kwargs']['target_resolution']
tx = np.arange(lef, rig, res)
ty = np.arange(top, bot, -res)

im_date = Series([])
im_cloud_perc = Series([])


OR_switch = True # overwrite previous data stored

########## CHUNK COUNTER RESTART (comment to resume interrupted download)
n = 0 # to start the downloading from 0
np.save('data/counter.npy',n)
n_chunks = np.save('data/nchunks.npy',len(date_vec)-1)
##########

n = np.load('data/counter.npy') # counter to resume from last completed chunk
n_chunks = np.load('data/nchunks.npy')

for i in range(n,n_chunks):

    fpath_pickle = Path('data/s2_data_' + str(i)  + '_of_' + str(n_chunks-1) + '.pkl')
    fpath_metadata = Path('data/s2_data_' + str(i)  + '_of_' + str(n_chunks-1) + '.gpkg')
    
    print("DOWNLOADING DATA CHUNK " + str(i) + ' of ' + str(n_chunks-1))
    mapper_configs = MapperConfigs(
         collection = collection,
         time_start = date_vec[i],
         time_end = date_vec[i+1],
         feature = feature,
         metadata_filters = metadata_filters
     )

    if not fpath_pickle.exists() or OR_switch == True:

        # now, a new Mapper instance is created
        mapper = Mapper(mapper_configs)
       
        try:
            mapper.query_scenes()
        
        except Exception as e: 
            
            # if no images available are found skip to the next data chunk
            if e.args[0] == "Querying STAC catalog failed: 'sensing_time'":
                print('No images found, continuing to the next data chunk')
                continue # skip this data chunk download
            
            else:
                print(e) 
                break
    
        # download the images
        mapper.load_scenes(scene_kwargs=scene_kwargs)
    
        # display image headers
        mapper.data

        for _, scene in mapper.data:
            # reproject scene    
            scene.reproject(inplace=True, target_crs=2056)
            
        with open(fpath_pickle, 'wb+') as dst:
            dst.write(mapper.data.to_pickle())
        mapper.metadata.sensing_time = mapper.metadata.sensing_time.astype(str)
        mapper.metadata = mapper.metadata[['product_uri', 'scene_id', 'sensing_time', 'cloudy_pixel_percentage', 'geom']].copy()
        mapper.metadata.to_file(fpath_metadata)

    else:
        data = SceneCollection.from_pickle(fpath_pickle)
        metadata = gpd.read_file(fpath_metadata)
        mapper = Mapper(mapper_configs)
        mapper.data = data
        mapper.metadata = metadata
    
    sc = mapper.data
    bands = sc[sc.timestamps[0]].band_names # bands
    
    # extract image dates, cloud percentage, and images
    if not mapper.data.empty:
        
        im = scene_to_array(mapper.data,tx,ty)
        im_date = mapper.metadata['sensing_time']
        im_cloud_perc = im_cloud_perc
        
        # (EDIT) SAVE BANDS as a .npz file
        np.savez('data/s2_data_' + str(i)  + '_of_' + str(n_chunks-1) + '_snow.npz',
                 im_date = im_date, # dates vector
                 im_cloud_perc = im_cloud_perc, # cloud pecentage vector
                 bands = bands, # band names
                 im = im, # images array [x,y,band,scene]
                 tx = tx, # x coord vector
                 ty = ty,  # y coord vector
                 shp = shp # roi shapefile
                 )
        
        # missing data as nan
        im[im==0] = np.nan
        
        # compute NDVI
        RED = np.squeeze(im[:,:,np.array(bands)==['B04'],:])
        NIR = np.squeeze(im[:,:,np.array(bands)==['B08'],:])
        NDVI = (NIR-RED)/(NIR+RED)
        
        # (EDIT) SAVE NDVI as a .npz file        
        np.savez('data/s2_ndvi_' + str(i)  + '_of_' + str(n_chunks-2) + '_snow.npz',        im_date = im_date, # dates vector
                 im_cloud_perc = im_cloud_perc, # cloud pecentage vector
                 im = NDVI, # images array [x,y,band,scene]
                 tx = tx, # x coord vector
                 ty = ty,  # y coord vector
                 shp = shp # roi shapefile
                 )
        
        del im, mapper, NDVI, RED, NIR

    # update counter
    n = n+1 
    np.save('data/counter.npy',n)
                 
#%% DOWNLOAD DEM FOR THE ROI

download_SA3D_STAC(
        bbox_path = bbox_fname, # bbox in any crs
        out_crs = 2056, # wanted output crs EPGS code
        out_res = 10, # wanted output resolution (compatible with out crs)
        out_path = 'data/dem_merged_10m.tif',
        server_url = 'https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d',
        product_res_label = '2' # can only be 0.5 or 2, original product resolution 
    )