#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:32:13 2022

@author: orianif
"""

import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'DejaVu Sans',
        'size'   : 14}
from matplotlib import rc
rc('font', **font)
import fabio_tools as ft
from shapefile import Reader
from scipy.interpolate import interp2d
from osgeo import osr
from osgeo import ogr
import python_dem_shadows as so
from joblib import Parallel, delayed
import multiprocessing
import richdem as rd
from datetime import datetime


#%% IMPORT VECTOR VEGETATION MAP

fname='data/all_20110318_2056.shp'

# create shapefile object
sf=Reader(fname,encoding='UTF-8') # specify encoding

# retrieve shapes
sh=np.array(sf.shapes()) # shape

# retrieve fields
fi=np.array(sf.fields) # [field name, type, data length, decimal places]

# retrieve records (associated to the shapes, contain all fields for every shape)
re=np.array(sf.records())

# retrieve shape and associated records together
sr=np.array(sf.shapeRecords())
#sr[0].shape.points # shape points for shaperecord 0
#sr[0].record # associated record

# keep only records with Veg 2 missing and Veg 1 informed
ind=np.logical_and(re[:,2]=='',re[:,1]!='')
re=re[ind,:]
sh=sh[ind]
sr=sr[ind]

#%% Vegetation class LUT
vt_list=np.genfromtxt('data/veg1_vegclass_LUT.txt',usecols=0,skip_header=1)
veg1_list=np.genfromtxt('data/veg1_vegclass_LUT.txt',usecols=1,skip_header=1)

#% extract vegetation type field and shape points

vt=[] # vegetation class
shx=[] # xcoords for every shape
shy=[] # y coords for every shape

for i in range(len(re)):
    # consider only records with non-zero shape AND with unique veg1 label (exclude uncertain labels like 12x38)
    if len(sh[i].points)>0 and len(re[i,1])==2 and np.any(veg1_list==np.array(re[i,1],dtype='float64')): 
        vt.append(vt_list[veg1_list==np.array(re[i,1],dtype='float64')][0]) # it's the 38th field
        shx.append(np.array(sh[i].points)[:,0])
        shy.append(np.array(sh[i].points)[:,1])

vt=np.array(vt)
shx=np.array(shx)
shy=np.array(shy)

#% plot shapes with Veg type        
plt.figure()
for i in range(len(vt)):
    plt.plot(shx[i],shy[i])
    plt.text(np.mean(shx[i]),np.mean(shy[i]),str(vt[i]))
plt.axis('equal')

#% reduce vt_list as a list of unique values (AFTER ALL SHAPEFILE ARE RASTERIZED)
vt_list=np.unique(vt_list)

#%% IMPORT GRID FROM FIRST SAT IMAGE CHUNK
variables = np.load('data/s2_ndvi_0_of_85_snow.npz')
variables.allow_pickle=True
locals().update(variables)
del variables
xx, yy = np.meshgrid(tx,ty,indexing='xy')
res = tx[1]-tx[0]
im_extent = [tx[0]-res/2,tx[-1]+res/2,ty[0]+res/2,ty[-1]-res/2]
plt.figure(), plt.imshow(im[:,:,0], extent=im_extent)

#%% RASTERIZE THE SHAPEFILE TO SAT IMAGE GRID

##%% uncomment to generate vt_rast (now present in data)
# vt_rast=np.zeros_like(xx)
# points = np.array((xx.flatten(), yy.flatten())).T
# for i in range(len(vt)):
#     mpath = mplp.Path(np.hstack((shx[i][:,None],shy[i][:,None])))
#     mask = mpath.contains_points(points).reshape(xx.shape)
#     vt_rast[mask]=vt[i]
#     print(str(i) + '/' + str(len(vt)))

# # group units 7.* in 7.1 (wetlands)
# vt_rast[np.logical_or(vt_rast==7.2,vt_rast==7.3)] = 7.1
# np.save('data/vt_rast.npy',vt_rast)

vt_rast = np.load('data/vt_rast.npy')

# List of selected units
u_list = [2,8,9,10,13,14,15,16,22] # unit list
#vt_list = vt_list[u_list]

#%% COMPUTE UNIQUE ID FOR ALL PARCELS 
## using 8-connected components + unit divisions
## precomputed, uncomment to compute from scratch

# vt_bin = vt_rast != 0
# cc_rast = label(vt_bin,connectivity=2) # coccented components
# cc_list = np.unique(cc_rast[cc_rast!=0])

# pid_rast = np.empty_like(vt_rast)*0
# pid = 0 # initialize parcel id 
# for i in vt_list:
#     for j in cc_list:
#         print('i=' + str(i) + 'j=' + str(j))
#         pid = pid+1
#         pid_rast[np.logical_and(vt_rast==i,cc_rast==j)] = pid
#         pid_mask = pid_rast==pid
        
#         if not(np.sum(pid_mask)==0):
#             # split parcels having same unit and original patch 
#             cc_pid = label(pid_mask,connectivity=2)
#             cc_pid_list = np.unique(cc_pid)
#             pid2 = 0
#             for k in range(1,len(cc_pid_list)):
#                 pid2 = pid2 + 0.001
#                 pid_rast[cc_pid==cc_pid_list[k]] = pid + pid2

        
# plt.imshow(pid_rast)
# np.save('data/pid_rast.npy',pid_rast)
pid_rast = np.load('data/pid_rast.npy')

#%% IMPORT DEM

dem,R_dem = ft.read_geotiff('data/dem_merged_10m.tif')
dem = np.flipud(dem)

#original grid coordinates
lefc = R_dem.xllc
vx = np.arange(R_dem.xllc,R_dem.xllc+R_dem.xres*(R_dem.ncols),step=R_dem.xres)
vy = np.arange(R_dem.yllc,R_dem.yllc+R_dem.yres*(R_dem.nrows),step=R_dem.yres)

# create interpolator
f = interp2d(vx,vy,dem,kind='linear',fill_value=-9999)

# interpolate band on the target grid
dem = np.flipud(f(tx,ty))
ft.write_geotiff(filename="data/dem_merged_interpolated_10m.tif",
                 im = dem,
                 xmin = tx[0]-5,
                 ymax = ty[0]+5,
                 xres = 10,
                 yres = 10,
                 epsg = 2056,
                 nodata_val = -9999)
dem[dem==-9999]=np.nan

# compute dem derivatives
dem_tmp = rd.LoadGDAL('data/dem_merged_interpolated_10m.tif')
#dem_tmp = np.flipud(dem_tmp)
dem_tmp[dem_tmp==-9999]=np.nan
aspect = rd.TerrainAttribute(dem_tmp, attrib='aspect')
slope = rd.TerrainAttribute(dem_tmp, attrib='slope_degrees')
curv = rd.TerrainAttribute(dem_tmp, attrib='curvature')
#dem_der = np.stack((dem,aspect,slope,curv),axis=2)


plt.figure()
plt.subplot(2,2,1)
plt.imshow(dem,extent = im_extent,alpha = 1)
plt.title('dem')
plt.subplot(2,2,2)
plt.imshow(aspect,extent = im_extent,alpha = 1)
plt.title('aspect')
plt.subplot(2,2,3)
plt.imshow(slope,extent = im_extent,alpha = 1)
plt.title('slope')
plt.subplot(2,2,4)
plt.imshow(ft.imrisc(curv,10,90),extent = im_extent,alpha = 1)
plt.title('curvature')

#%% DEM PARAMS FOR SHADOW COMPUTATION 

# compute image center grid coordinates in WGS84
InSR = osr.SpatialReference()
InSR.ImportFromEPSG(2056)       # original CRS
OutSR = osr.SpatialReference()
OutSR.ImportFromEPSG(4326)     # WGS84
cx=xx[0,int(np.shape(xx)[1]/2)]
cy=yy[int(np.shape(yy)[0]/2),0]
Point = ogr.Geometry(ogr.wkbPoint)
Point.AddPoint(cx,cy) # use your coordinates here
Point.AssignSpatialReference(InSR)    # tell the point what coordinates it's in
Point.TransformTo(OutSR)
lon=Point.GetY()
lat=Point.GetX()
tzone=0 # if shot time is Zulu then put 0 here (UTM 0)
dx=xx[0,1]-xx[0,0]

#%% MONTHLY NDVI SHADOW/SUN DATA site 1

def compute_shadow(dem,lon,lat,date,tzone,dx):
    date = datetime.strptime(date[:19],'%Y-%m-%d %H:%M:%S')
    jd=so.to_juliandate(date) # transform to julian date
    sun_vector= so.sun_vector(jd,lat,lon,tzone)
    shad=so.project_shadows(dem, sun_vector, dx, dy=dx)
    shad[shad==1]=np.nan
    shad[shad==0]=1
    shad[np.isnan(shad)]=0
    return shad

#%% INITIALIZE DATABASE AS DICTIONARY
# # BASIC DICTIONARY STRUCTURE: EXAMPLE

# mydict = {
#     "ndvi": [0.1,0.2,0.5,0.8],
#     "unit":[2,3,5,6,9],
#     "date":[datetime(2016,2,1),datetime(2016,6,1),datetime(2017,2,1),datetime(2019,2,1)],
#     "shadow":[0,1,1,0]
#     }
#
# variables can then be extracted as numpy arrays,ex:
# ndvi = np.concatenate(mydict['ndvi'])
#

m_list = np.arange(1,13) # months list now including winter


mykeys = ['ndvi', # NDVI pixel value
          'date', # original image date
          'data_chunk_n', # data chunk the image belongs to
          'year', # image year
          'month', # image month
          'week', # image week
          'doy', # image day of the year
          'unit', # pixel vegetation unit
          'pid', # parcel id the pixel belongs to
          'shadow', # shadow indicator
          'elevation', # elevation
          'aspect', # aspect
          'slope', # slope
          'curvature', # curvature
          'x', # x coord
          'y'] # y coord

#inizialize dictionarîes for shadow and sun data, dictionary structure is nested [unit][month]
ndvi_dict = dict.fromkeys(mykeys)

for i in ndvi_dict.keys():
      ndvi_dict[i] = list([])

# Initialize saved data
#np.savez('data/ndvi_dict.npz',
np.savez('data/ndvi_dict_snow.npz',
          ndvi_dict = ndvi_dict,
          q_tmp = 0, # current data chunk to process, 0 at the beginning 
          )

#%% EXTRACT NDVI FROM DATA CHUNKS, 

plt.close('all')
variables = np.load('data/ndvi_dict_snow.npz')
variables.allow_pickle=True
locals().update(variables)
del variables

ndvi_dict = ndvi_dict.all()
vt_ind = np.in1d(vt_rast,vt_list).reshape(np.shape(vt_rast))
n_chunks = np.load('data/nchunks.npy')

for q in range(q_tmp,n_chunks):
    print('processing chunk ' + str(q) + '_of_' + str(n_chunks))
    # import images chunk    
    variables = np.load('data/s2_ndvi_' + str(q) + '_of_' + str(n_chunks-1) + '_snow.npz')
    variables.allow_pickle=True
    locals().update(variables)
    del variables
    
    if np.ndim(im)==2:
        im = im[:,:,None]
    nim = np.shape(im)[2]
    
    # skip whole data chunk if there is no image belonging to the month list
    month = []
    for i in range(nim):
        month.append(datetime.strptime(im_date[i][:19],'%Y-%m-%d %H:%M:%S').month)    
    
    if not(np.any(np.in1d(month,m_list))): # exclude data outside the wanted month range
            continue   
    
    
    # Shadow computation (parallelized)
    num_cores = multiprocessing.cpu_count()-2
    shad = Parallel(n_jobs=num_cores,backend="multiprocessing")(delayed(compute_shadow)(dem,lon,lat,im_date[i],tzone,dx)for i in range(len(im_date))) # parallelized loop
    
    # populate data dictionary
    for i in range(nim):
        date_tmp = datetime.strptime(im_date[i][:19],'%Y-%m-%d %H:%M:%S')    
        
        # skip image if there is no image belonging to the month list
        if not(np.in1d(date_tmp.month,m_list)):
            continue
        
        # skip image if there if it contains too few data
        data_ind = np.logical_and(vt_ind,~np.isnan(im[:,:,i]))
        df = np.sum(data_ind)/np.sum(vt_ind) # data fraction in the image
        dth = 0.1 # 10% of miniumum data required
        if df < dth:
            continue
        
        # skip image if 
        
        ndvi_dict['date'].append([date_tmp]*np.sum(data_ind))
        ndvi_dict['data_chunk_n'].append([q]*np.sum(data_ind))
        ndvi_dict['year'].append([date_tmp.year]*np.sum(data_ind))
        ndvi_dict['month'].append([date_tmp.month]*np.sum(data_ind))
        ndvi_dict['week'].append([date_tmp.isocalendar()[1]]*np.sum(data_ind))
        ndvi_dict['doy'].append([date_tmp.timetuple().tm_yday]*np.sum(data_ind))
        ndvi_dict['ndvi'].append(im[data_ind,i])
        ndvi_dict['unit'].append(vt_rast[data_ind])
        ndvi_dict['pid'].append(pid_rast[data_ind])
        ndvi_dict['shadow'].append(shad[i][data_ind])
        ndvi_dict['elevation'].append(dem[data_ind])
        ndvi_dict['aspect'].append(aspect[data_ind])
        ndvi_dict['slope'].append(slope[data_ind])
        ndvi_dict['curvature'].append(curv[data_ind])
        ndvi_dict['x'].append(xx[data_ind])
        ndvi_dict['y'].append(yy[data_ind])
        
        
    # save data
    np.savez('data/ndvi_dict_snow.npz',
             ndvi_dict = ndvi_dict,
              q_tmp = q+1, # current data chunk to process, 0 at the beginning 
              )
    
