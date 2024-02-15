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
#from matplotlib import colors
from scipy.stats import pearsonr
from datetime import datetime as dt 
import richdem as rd
from scipy.optimize import curve_fit
import ndvi_tools as ndt
import matplotlib

#% LOAD DATA BACK FROM PREPROCESSING

# map of selected units
vt_rast = np.load('data/vt_rast.npy')
vt_list= [4.2,4.3,4.4,2.1,6.1,6.2,7.1,6.3,9.1]
vt_rast[np.logical_not(np.in1d(vt_rast,vt_list).reshape(np.shape(vt_rast)))]=np.nan

# ndvi dictionary
#variables = np.load('data/ndvi_dict.npz')
variables = np.load('data/ndvi_dict_snow.npz')
variables.allow_pickle=True
locals().update(variables)
del variables
ndvi_dict = ndvi_dict.all()

unit = np.concatenate(ndvi_dict['unit'])
unit_ind = np.in1d(unit,vt_list)
unit = unit[unit_ind]

date = np.concatenate(ndvi_dict['date'])[unit_ind]
doy = np.concatenate(ndvi_dict['doy'])[unit_ind]
month = np.concatenate(ndvi_dict['month'])[unit_ind]
ndvi = np.concatenate(ndvi_dict['ndvi'])[unit_ind]
shadow = np.concatenate(ndvi_dict['shadow'])[unit_ind]
week = np.concatenate(ndvi_dict['week'])[unit_ind]
year = np.concatenate(ndvi_dict['year'])[unit_ind]
elevation = np.concatenate(ndvi_dict['elevation'])[unit_ind]
aspect = np.concatenate(ndvi_dict['aspect'])[unit_ind]
pid = np.concatenate(ndvi_dict['pid'])[unit_ind]
x = np.concatenate(ndvi_dict['x'])[unit_ind]

y = np.concatenate(ndvi_dict['y'])[unit_ind]
dcn = np.concatenate(ndvi_dict['data_chunk_n'])[unit_ind]


#% COLOR MAP AND LABELS
verb_labels=[
'a) Mesic nutrient-rich pastures',
'b) Wet nutrient-rich pastures',
'c) Resting areas',
'd) Dry nutrient-poor pastures',
'e) Acidic nutrient-poor pastures',
'f) Mesic nutrient-poor pastures',
'g) Wetland',
'h) Dwarf shrubs',
'i) Tall shrubs'
]

verb_labels_short=[
'a) Mesic nutrient-rich p.',
'b) Wet nutrient-rich p.',
'c) Resting areas',
'd) Dry nutrient-poor p.',
'e) Acidic nutrient-poor p.',
'f) Mesic nutrient-poor p.',
'g) Wetland',
'h) Dwarf shrubs',
'i) Tall shrubs'
]


prop_cycle = plt.rcParams['axes.prop_cycle']
cols = prop_cycle.by_key()['color']

cols = ["#158c21", # mesi fertile
        "#abe26b", # wet fertile
        "#987db7", # resting areas
        "#ffe601", # dry pastures
        "#e77148", # acidic infertile
        "#ba1600", # mesi infertile
        "#0084d0", # wetland
        "#b9af8c", # dwarf shr
        "#8e5c00"] # tall shr

# import snow and rad time series
st = np.genfromtxt('data/order_114180_data.txt',usecols=0,dtype='str',delimiter=';',skip_header=1)
t_tmp = np.genfromtxt('data/order_114180_data.txt',usecols=1,dtype='str',delimiter=';',skip_header=1)

t=[]
d=[]
for i in range(len(t_tmp)):
    t.append(dt(int(t_tmp[i][:4]),int(t_tmp[i][4:6]),int(t_tmp[i][6:])))
    d.append(t[i].timetuple().tm_yday)
t = np.array(t)
d = np.array(d)

sno = np.genfromtxt('data/order_114180_data.txt',usecols=3,dtype='float',delimiter=';',skip_header=1)
rad = np.genfromtxt('data/order_114180_data_2.txt',usecols=2,dtype='float',delimiter=';',skip_header=1)
sno_SCU = sno[st=='SCU']
sno_BUF = sno[st=='BUF']
rad_SCU = rad[st=='SCU']
rad_BUF = rad[st=='BUF']
t_SCU = t[st=='SCU']
t_BUF = t[st=='BUF']
d_SCU = d[st=='SCU']
d_BUF = d[st=='BUF']

### INPUT PARAMS ########################################################

# years to generate the ndvi annual curves (one curve per year)
years = np.arange(2016,2023) # chosen dry/wet years 
time_res = 'doy' # time resolution of the data display, it can be 'doy' 'week' or 'month'
ndvi_th = 0.05 # greening / browning threshold
pth = 5 # greening / browning threshold period
os = 0.2 # graph bars offset

# aspect filter
alb = 90 # lower aspect boundary to select the ROI pixels
aub = 270 # upper aspect boundary to select the ROI pixels, = 365 for no filter

# elevation filter
elb = [2000,2200,2400,2600] # elevation boundaries to select the ROI pixels, = 0 for no filter
eub = [2200,2400,2600,2800] # elevation boundaries to select the ROI pixels, = 9999 for no filter

# dates to exclude from the images (ex. snowfall days), leave empty for no dates excluded

excl_dates = ["2016-01-04 10:24:32",
              "2016-04-19 10:10:32",
              "2017-10-24 10:21:11",
              "2017-09-21 10:10:21",
              "2017-08-20 10:20:19",
              "2017-10-09 10:20:09",
              "2017-07-31 10:20:19",
              "2019-09-09 10:20:29",
              "2019-10-16 10:10:29",
              "2020-09-30 10:07:29",
              "2020-10-18 10:20:41",
              "2020-10-08 10:20:31",
              "2020-10-10 10:10:29",
              "2020-10-28 10:21:41",
              "2020-10-30 10:11:39",
              "2021-10-08 10:18:29",
              "2022-09-18 10:17:01"]

for i in range(len(excl_dates)):
    excl_dates[i] = dt.strptime(excl_dates[i],'%Y-%m-%d %H:%M:%S')
excl_ind = ~np.in1d(date,excl_dates) 

###############################################################################

#%% TAB1 UNIT DESCRIPTION TABLE

# compute dem derivatives
dem_tmp = rd.LoadGDAL('data/dem_merged_interpolated_10m.tif')
dem_tmp[dem_tmp==-9999]=np.nan
aspect_rast = rd.TerrainAttribute(dem_tmp, attrib='aspect')
asp_ind = np.logical_and(aspect_rast > alb, aspect_rast < aub)

f = open('export/units_table.txt','w')
for u in range(len(vt_list)):

    # data selection for unit, no shadow, and aspect
    data_ind = np.logical_and(vt_rast==vt_list[u],asp_ind)
    
    # unit surface
    f.write(verb_labels[u][3:] + ' & ') 
    y_data = vt_rast[data_ind]
    y_A = "%2.2f" % (len(y_data)*10*10/1000/1000) # surface of the data in sqkm
    f.write(y_A + ' & ')
    
    #elevation distribution
    vt_elev = dem_tmp[data_ind]
    q_elev = np.quantile(vt_elev,[0.25,0.5,0.75])
    f.write("%d & %d & %d " % (q_elev[0],q_elev[1],q_elev[2]))
    f.write('\\\ \n')
f.close()


#%% FIG2 NDVI CURVE SKETCH

# data selection indicators
tmp_year = 2020
year_ind = np.in1d(year,tmp_year) # year indicator
sun_ind = shadow==0 # sun indicator
asp_ind = np.logical_and(aspect > alb, aspect < aub)
excl_ind = ~np.in1d(date,excl_dates) 

f=plt.figure(figsize=[11.1 ,  7.17]) # f.get_size_inches()
n = 0
t_tmp = np.concatenate(ndvi_dict[time_res])[unit_ind]
u = 0  

# data selection for unit, no shadow, aspect, and elevation
data_ind = np.logical_and(unit==vt_list[u],sun_ind)
data_ind = np.logical_and(data_ind,asp_ind)
data_ind = np.logical_and(data_ind,excl_ind)

# dry/wet years selection
data_ind = np.logical_and(data_ind,year_ind)
ndvi_data = ndvi[data_ind]
ndvi_time = t_tmp[data_ind]

# ndvi plot
dates = ndvi_time
data = ndvi_data
lb = 0
rb = 0
dcol = 'tab:green'
dlabel = 'NDVI median interpolation'
clabel = 'NDVI 0.25-0.75 quantile interpolation'
plabel = 'NDVI median data'
envelope = True
f_range = [-0.1,1]
lw = 2 # line width

d_list = np.unique(dates)
plt.grid(axis='y',linestyle='--')

qm = np.array([],dtype=float)
for d in d_list:
    qm = np.hstack((qm,np.nanquantile(data[dates==d],0.5)))

# filter out data with median outside given range
fil = np.logical_and(qm > f_range[0],qm < f_range[1])
d_list = d_list[fil]
qm = qm[fil]

# envelop interpolation
if envelope==True:
    q1 = np.array([],dtype=float)
    q2 = np.array([],dtype=float)
    for d in d_list:
        q1 = np.hstack((q1,np.nanquantile(data[dates==d],0.75)))
        q2 = np.hstack((q2,np.nanquantile(data[dates==d],0.25)))
    d_list1,q1i,*tmp = ndt.annual_interp(d_list,q1,time_res=time_res,lb=lb,rb=rb)
    d_list2,q2i,*tmp = ndt.annual_interp(d_list,q2,time_res=time_res,lb=lb,rb=rb)
    q2i_f = np.flip(q2i)
    qi = np.hstack((q1i,q2i_f))
    d = np.hstack((d_list1,np.flip(d_list1)))
    d = d[~np.isnan(qi)]
    qi = qi[~np.isnan(qi)]
    plt.fill(d,qi,alpha=0.5,c=dcol,label=clabel)

# median interpolation
d_listm,qmi,*tmp = ndt.annual_interp(d_list,qm,time_res=time_res,lb=lb,rb=rb)
plt.plot(d_listm,qmi,linestyle = '--',c=dcol,markersize=15,label=dlabel,linewidth=lw)

# median data
plt.scatter(d_list,qm,c=dcol,label=plabel)

# snow plot
stime,sdata = ndt.snow_plot(t_SCU,sno_SCU,tmp_year,'tab:blue',stat='mean',lb='data',rb='data',slabel='RSD interpolation')

# sog
gt = ndt.sog(ndvi_time,ndvi_data,time_res='doy',ndvi_th = 0.05,pth=10,envelope=False)
plt.plot([gt,gt],[-0.08,0.2],'--',color='k',linewidth=lw)
plt.text(gt-6,-0.13,s='SOG')#,c='tab:green',rotation = 'vertical')
#plt.annotate("SOG", xy=(gt,0), xytext=(gt,-0.1),
#            arrowprops=dict(arrowstyle="->"))

# EOS
se = ndt.eos(ndvi_time,ndvi_data,time_res='doy',ndvi_th = 0.05,pth=10,envelope=False)
plt.plot([se,se],[-0.08,0.2],'--',color='k',linewidth=lw)
plt.text(se-6,-0.13,s='EOS')#,c='tab:green',rotation = 'vertical')
#plt.annotate("EOS", xy=(se,0), xytext=(se-40,-0.1),
#            arrowprops=dict(arrowstyle="->"))

# AUC
f_ind = np.logical_and(d_listm > gt,d_listm < se)
xv = np.hstack([gt,d_listm[f_ind],se])
yv = np.hstack([0,qmi[f_ind],0])
plt.fill(xv,yv,alpha=0.5,fill=False,hatch='/')
plt.plot([213,213],[-0.08,0.8],'--',color='k',linewidth=lw)
plt.text(213-5,-0.12,s='1st of August')
t = plt.text(170,0.4,s='AUC1')
t.set_bbox(dict(facecolor='white', alpha=1,linewidth=0))
t = plt.text(240,0.4,s='AUC2')
t.set_bbox(dict(facecolor='white', alpha=1,linewidth=0))

# Gompertz model
fp,C = curve_fit(ndt.gomp,
                  d_listm[:300],
                  qmi[:300])
gom_time = np.arange(0,213)
gom_data = ndt.gomp(gom_time,*fp)
plt.plot(gom_time,gom_data,'--',c='tab:orange',label="Gompertz model",linewidth=lw)

# growth slope
xi = [gom_time[120],gom_time[130]]
yi = [gom_data[120],gom_data[130]]
b =(yi[1]-yi[0])/(xi[1]-xi[0])
a = yi[0]-b*xi[0]
xl = np.arange(110,145)
yl = a + b*xl
plt.plot(xl,yl,'--k',linewidth=lw)
plt.text(xl[-1]+2,yl[-1],s='growth slope',color='k')

# graph cosmetics
plt.ylim([-0.15,1.1])
plt.xlabel('Day of the year (DOY)')
plt.ylabel('NDVI [-1,1]')
plt.legend(loc='upper left',prop={'size': 12})
plt.grid(axis='y',linestyle='--',alpha=0.5)
plt.grid(axis='x',linestyle='--',alpha=0.5)
plt.tight_layout()
plt.savefig('export/curve_sketch.pdf')

#%% FIG3 ANNUAL NDVI CURVE FOR DIFFERENT DRY-WET YEARS

### params
dry_year = [2016] # chosen dry/wet years 
wet_year = [2017] # to group years: np.array([[2016,2017],[2022,2023]])
time_res = 'doy' # time resolution of the data display, it can be 'doy' 'week' or 'month'
###

# data selection indicators
dry_ind = np.in1d(year,dry_year) # dry year indicator
wet_ind = np.in1d(year,wet_year) # wet year indicator
sun_ind = shadow==0 # sun indicator
asp_ind = np.logical_and(aspect > alb, aspect < aub)
excl_ind = ~np.in1d(date,excl_dates) 

f=plt.figure(figsize=[15, 10.87]) # f.get_size_inches()
n = 0

t_tmp = np.concatenate(ndvi_dict[time_res])[unit_ind]
for u in range(len(vt_list)):
    
    # data selection for unit, no shadow, aspect, and elevation
    data_ind = np.logical_and(unit==vt_list[u],sun_ind)
    data_ind = np.logical_and(data_ind,asp_ind)
    data_ind = np.logical_and(data_ind,excl_ind)
    
    # dry/wet years selection
    dry_data_ind = np.logical_and(data_ind,dry_ind)
    dry_data = ndvi[dry_data_ind]
    dry_time = t_tmp[dry_data_ind]
    
    wet_data_ind = np.logical_and(data_ind,wet_ind)
    wet_data = ndvi[wet_data_ind]
    wet_time = t_tmp[wet_data_ind]
      
    # plot
    n = n+1
    ax = plt.subplot(3,3,n)
    ndt.annual_plot(wet_time,wet_data,dcol='tab:blue',dlabel='NDVI ' + str(wet_year[0]),time_res=time_res)
    ndt.annual_plot(dry_time,dry_data,dcol='tab:orange',dlabel='NDVI ' + str(dry_year[0]),time_res=time_res)
    
    # snow data
    ndt.snow_plot(t_BUF,sno_BUF,wet_year,'tab:blue',stat='mean',lb='data',rb=0,slabel='RSD ' + str(wet_year[0]))
    ndt.snow_plot(t_BUF,sno_BUF,dry_year,'tab:orange',stat='mean',lb='data',rb=0,slabel='RSD ' + str(dry_year[0]))
    
    plt.title(verb_labels[u])
    plt.ylim([-0.15,1.1])
    plt.xlabel('Day of the year (DOY)')
    plt.ylabel('NDVI [-1,1]')
    if np.in1d(n,[1,2,3,4,5,6]):
        plt.xlabel(None)
        ax.set_xticklabels([])
        
    if np.in1d(n,[2,3,5,6,8,9]):
        plt.ylabel(None)
        ax.set_yticklabels([])
        
    if n==1:
        plt.legend(loc='upper right',prop={'size': 11})
    
    plt.grid(axis='y',linestyle='--',alpha=0.5)
    plt.grid(axis='x',linestyle='--',alpha=0.5)
 
plt.tight_layout()
plt.savefig('export/ndvi_drywet_' + str(dry_year[0]) + '_' + str(wet_year[0]) +'.pdf')

# %% INTERACTIVE NDVI CURVE TO SPOT WRONG IMAGES

### params
time_res= 'doy' #  year, month, week, doy
year_tmp = [2019] # to group years: np.array([[2016,2017],[2022,2023]])
vt_tmp = 4.3
###

sun_ind = shadow==0 # sun indicator
asp_ind = np.logical_and(aspect > alb, aspect < aub)

f=plt.figure(figsize=[11.08,  9.68]) # f.get_size_inches()
data_ind = np.logical_and(unit==vt_tmp,np.in1d(year,year_tmp)) # unit and year indicators
data_ind = np.logical_and(data_ind,shadow==0) # sun indicator
data_ind = np.logical_and(data_ind,asp_ind)
data_ind = np.logical_and(data_ind,excl_ind)
ndvi_sun = ndvi[data_ind]
time_sun = doy[data_ind]
dcn_sun = dcn[data_ind]
date_sun = date[data_ind]

# plot
h = plt.subplot(1,3,1)
ndt.annual_plot(time_sun,ndvi_sun,'r','sun',time_res='doy')
plt.title(verb_labels[vt_list==vt_tmp])
if time_res == 'month':
    plt.xticks(ticks=np.arange(3,11),labels=['Mar','Apr','May','Jun','Jul','Aug','Sep','Oct'])
plt.ylim([-0.15,1.1])
plt.xlabel(time_res + ' number')#plt.xlabel('week number')
plt.ylabel('NDVI [-1,1]')
f.text(0.05,0.97,str(i),fontsize=16,weight='bold')        
plt.tight_layout()
global h2
h2 = plt.subplot(1,3,2)
pcm = h2.imshow(np.empty([10,10])*np.nan,vmin=-0.1,vmax=0.8)
h2.set_xticks([],[])
h2.set_yticks([],[])
#plt.colorbar(pcm,ax=h2,orientation='horizontal')
global h3
h3 = plt.subplot(1,3,3)
pcm = h3.imshow(np.empty([10,10])*np.nan,vmin=-0.1,vmax=0.8)
h3.set_xticks([],[])
h3.set_yticks([],[])
    
# DEFINE CALLBACK FUNCTIONS
def onclick(event):
    global ix, iy, ni 
    ix, iy = event.xdata, event.ydata
    dx=(ix-time_sun)/np.std(time_sun) 
    dy=(iy-ndvi_sun)/np.std(ndvi_sun)
    D=np.sqrt(dx**2+dy**2)
    ind=np.argmin(D)
    print((time_sun[ind],ndvi_sun[ind]))
    
    # detect the point in the plot and map
    h.scatter(time_sun[ind],ndvi_sun[ind],marker='+',s=100,c='k') 
    plt.draw()
    
    # load related data chunk

    variables = np.load('data/s2_ndvi_' + str(dcn_sun[ind].astype('int')) + '_of_85_snow.npz')
    variables.allow_pickle=True
    globals().update(variables)
    del variables
    
    for j in range(len(im_date)):
        print(j)
        if date_sun[ind] == dt.strptime(im_date[j][:-4],'%Y-%m-%d %H:%M:%S'):
            print('image found!')
            print('point date ' + str(date_sun[ind]))
            print('found date ' + im_date[j][:-4])
            break
    
    print('j = ' + str(j) + ' data_chunk = ' + str(dcn_sun[ind]))
    #h2 = plt.subplot(1,2,2)
    h2.clear()
    plt.draw()
    if im.ndim==2:
        h2.imshow(np.squeeze(im),vmin=-0.1,vmax=0.8,interpolation='None')
        data_ind = np.logical_and(~np.isnan(im),~np.isnan(vt_rast))
        df = np.sum(data_ind)/np.sum(~np.isnan(vt_rast)) # data fraction in the image
    else:
        h2.imshow(np.squeeze(im[:,:,j]),vmin=-0.1,vmax=0.8,interpolation='None')
        data_ind = np.logical_and(~np.isnan(im[:,:,j]),~np.isnan(vt_rast))
        df = np.sum(data_ind)/np.sum(~np.isnan(vt_rast)) # data fraction in the image
    h2.set_title('ndvi ' + str(im_date[j])) # + '\n data fraction = ' + str(df)))
    plt.tight_layout()
    plt.draw()
    
    variables = np.load('data/s2_data_' + str(dcn_sun[ind].astype('int')) + '_of_85_snow.npz')
    variables.allow_pickle=True
    globals().update(variables)
    del variables
    
    h3.clear()
    plt.draw()
    if im.ndim==2:
        h3.imshow(ft.imrisc(np.squeeze(im[:,:,[3,2,1],:]),2,98),interpolation='None')
    else:
        h3.imshow(ft.imrisc(np.squeeze(im[:,:,[3,2,1],j]),2,98),interpolation='None')
    h3.set_title('RGB')
    plt.tight_layout()
    plt.draw()

def onclose(event): # on figure close do..
    # disconnect listening functions
    f.canvas.mpl_disconnect(cid)
    f.canvas.mpl_disconnect(cid2)
    print('figure closed')

# CONNECT LISTENING FUNCTIONS TO FIGURE
cid = f.canvas.mpl_connect('button_press_event', onclick)
cid2 = f.canvas.mpl_connect('close_event', onclose)

#%% FIG4 AUC BOXPLOT AND BAR - 1st season half

tos = 0.5 # tick offset
sun_ind = shadow==0 # sun indicator
asp_ind = np.logical_and(aspect > alb, aspect < aub) # aspect indicator
t_tmp = np.concatenate(ndvi_dict[time_res])[unit_ind] # time vector

# initialize figure
myticks = []
mytlabels = []
fig,ax = plt.subplots(3,figsize=[11.08,  11.08]) #fig.get_size_inches()

plt.subplot(3,1,1)

n = 0

for i in range(len(elb)):
        
    # elevation indicator
    elev_ind = np.logical_and(elevation > elb[i], aspect <= eub[i])
        
    for u in range(len(vt_list)):
            
        u_ind = unit==vt_list[u] # unit indicator
        
        for y in range(len(years)):
            
            n = n+1
            
            #year indicator
            year_ind = year==years[y]
            
            # COMBINE INDICATORS
            data_ind = np.logical_and(sun_ind,asp_ind)
            data_ind = np.logical_and(data_ind,u_ind)
            data_ind = np.logical_and(data_ind,elev_ind)
            data_ind = np.logical_and(data_ind,year_ind)
            data_ind = np.logical_and(data_ind,excl_ind)
            
            # data to plot
            ndvi_data = ndvi[data_ind]
            ndvi_time = t_tmp[data_ind]
            
            if len(ndvi_data)==0:
                dq = [0,0,0]
            else:
                gt = ndt.sog(ndvi_time,ndvi_data,time_res=time_res,envelope=False,ndvi_th =ndvi_th,pth=pth)
                [*dq] = ndt.auc(ndvi_time,ndvi_data,time_res=time_res,sttt=gt,entt=213)
            
            if not(np.all(dq==0)):
                if y==0 and i==0:
                    ax[0].plot([n*os+i*os*3,n*os+i*os*3],[dq[0],dq[2]],'-',c=cols[u],label=verb_labels_short[u])
                    
                else:
                    ax[0].plot([n*os+i*os*3,n*os+i*os*3],[dq[0],dq[2]],'-',c=cols[u])
                    
                ax[0].plot(n*os+i*os*3,dq[1],'o',c=cols[u])

            if u==4 and y==3:
                mytlabels.append(str(elb[i]) + '-' + str(eub[i]))
                myticks.append(n*os+i*os*3)

myticks = np.array(myticks)
tos = (myticks[1]-myticks[0])/2
myticks_st = myticks[0]-tos
myticks_new = myticks + tos
myticks_new = np.append(myticks_st,myticks_new)
ax[0].set_xticks(myticks_new)
plt.grid()
mytlabels_new = np.append(mytlabels,'') 
ax[0].set_xticklabels(mytlabels_new)
offset = matplotlib.transforms.ScaledTranslation(tos/5, 0, fig.dpi_scale_trans)
for label in ax[0].xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

#plt.xlabel('elevation [m]')
plt.ylabel('NDVI AUC')
ax[0].legend(loc='upper right',prop={'size': 10},ncol=3)

ax[0].set_title('a) 1-st half season NDVI AUC - annual median and Q.25-.75 interval')
ax[0].set_xlim([myticks_new[0],myticks_new[-1]])
plt.tight_layout()

# AUC Inter-quartile range bar - 1st season half
plt.subplot(3,1,2)

sun_ind = shadow==0 # sun indicator
asp_ind = np.logical_and(aspect > alb, aspect < aub) # aspect indicator
t_tmp = np.concatenate(ndvi_dict[time_res])[unit_ind] # time vector

# initialize figure
myticks = []
mytlabels = []

n = 0
cv = []    
for i in range(len(elb)):
    
    # elevation indicator
    elev_ind = np.logical_and(elevation > elb[i], aspect <= eub[i])
    
    for u in range(len(vt_list)):
        
        u_ind = unit==vt_list[u] # unit indicator        
        iqr=[]
        n = n+1
        for y in range(len(years)):
            #year indicator
            year_ind = year==years[y]
            
            # COMBINE INDICATORS
            data_ind = np.logical_and(sun_ind,asp_ind)
            data_ind = np.logical_and(data_ind,u_ind)
            data_ind = np.logical_and(data_ind,elev_ind)
            data_ind = np.logical_and(data_ind,year_ind)
            data_ind = np.logical_and(data_ind,excl_ind)            
            
            # data to plot
            ndvi_data = ndvi[data_ind]
            ndvi_time = t_tmp[data_ind]
            
            if len(ndvi_data)==0:
                iqr.append([])
            else:
                # area under the curve
                gt = ndt.sog(ndvi_time,ndvi_data,time_res=time_res,envelope=False,ndvi_th =ndvi_th,pth=pth)
                AUC = ndt.auc(ndvi_time,ndvi_data,time_res=time_res,envelope=True,sttt=gt,entt=213)
                iqr = AUC[-1]-AUC[0] # interquartile range of AUC
           
        if u==4:
            mytlabels.append(str(elb[i]) + '-' + str(eub[i]))
            myticks.append(n)       

    
        miqr = np.nanmean(iqr)
        if i==0:
            ax[1].bar(n,miqr,color=cols[u],label=verb_labels_short[u])
        else:
            ax[1].bar(n,miqr,color=cols[u])

myticks = np.array(myticks)
tos = (myticks[1]-myticks[0])/2
myticks_st = myticks[0]-tos
myticks_new = myticks + tos
myticks_new = np.append(myticks_st,myticks_new)
ax[1].set_xticks(myticks_new)
plt.grid()
mytlabels_new = np.append(mytlabels,'') 
ax[1].set_xticklabels(mytlabels_new)
offset = matplotlib.transforms.ScaledTranslation(tos/3.2, 0, fig.dpi_scale_trans)
for label in ax[1].xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

#plt.xticks(rotation=90)
#plt.xlabel('elevation [m]')
ax[1].set_ylabel('mean interquartile range')
#ax[1].legend(loc='lower left',prop={'size': 11},ncol=3)
ax[1].set_title('b) 1-st half season NDVI AUC - mean interquartile range')
ax[1].set_xlim([myticks_new[0],myticks_new[-1]])
plt.tight_layout()

# CV bar - 2nd season half - version b
plt.subplot(3,1,3)

sun_ind = shadow==0 # sun indicator
asp_ind = np.logical_and(aspect > alb, aspect < aub) # aspect indicator
t_tmp = np.concatenate(ndvi_dict[time_res])[unit_ind] # time vector

# initialize figure
myticks = []
mytlabels = []
#fig,ax = plt.subplots(figsize=[15.59,  8.65])

n = 0
cv = []    
for i in range(len(elb)):
    
    # elevation indicator
    elev_ind = np.logical_and(elevation > elb[i], aspect <= eub[i])
    
    for u in range(len(vt_list)):
        
        u_ind = unit==vt_list[u] # unit indicator        
        mq=[]
        n = n+1
        for y in range(len(years)):
            #year indicator
            year_ind = year==years[y]
            
            # COMBINE INDICATORS
            data_ind = np.logical_and(sun_ind,asp_ind)
            data_ind = np.logical_and(data_ind,u_ind)
            data_ind = np.logical_and(data_ind,elev_ind)
            data_ind = np.logical_and(data_ind,year_ind)
            data_ind = np.logical_and(data_ind,excl_ind)            
            
            # data to plot
            ndvi_data = ndvi[data_ind]
            ndvi_time = t_tmp[data_ind]
            
            if len(ndvi_data)==0:
                mq.append(0)
            else:
                # area under the curve
                gt = ndt.sog(ndvi_time,ndvi_data,time_res=time_res,envelope=False,ndvi_th =ndvi_th,pth=pth)
                mq.append(ndt.auc(ndvi_time,ndvi_data,time_res=time_res,envelope=False,sttt=gt,entt=213))
           
        if u==4:
            mytlabels.append(str(elb[i]) + '-' + str(eub[i]))
            myticks.append(n)       

    
        cv = np.std(mq)/np.mean(mq)
        if i==0:
            ax[2].bar(n,cv,color=cols[u],label=verb_labels_short[u])
        else:
            ax[2].bar(n,cv,color=cols[u])

myticks = np.array(myticks)
tos = (myticks[1]-myticks[0])/2
myticks_st = myticks[0]-tos
myticks_new = myticks + tos
myticks_new = np.append(myticks_st,myticks_new)
ax[2].set_xticks(myticks_new)
plt.grid()
mytlabels_new = np.append(mytlabels,'') 
ax[2].set_xticklabels(mytlabels_new)
offset = matplotlib.transforms.ScaledTranslation(tos/3.2, 0, fig.dpi_scale_trans)
for label in ax[2].xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

#plt.xticks(rotation=90)
plt.xlabel('elevation [m]')
ax[2].set_ylabel('coefficient of variation')
ax[2].legend(loc='lower left',prop={'size': 11},ncol=3)
ax[2].set_title('c) 1-st half season NDVI AUC - coefficient of variation of the annual medians')
ax[2].set_xlim([myticks_new[0],myticks_new[-1]])
plt.tight_layout()

# save figure
plt.savefig('export/AUC_1st.pdf')
plt.close('all')

# FIG5 AUC BOXPLOT AND BAR - 2nd season half

tos = 0.5 # tick offset
sun_ind = shadow==0 # sun indicator
asp_ind = np.logical_and(aspect > alb, aspect < aub) # aspect indicator
t_tmp = np.concatenate(ndvi_dict[time_res])[unit_ind] # time vector

# initialize figure
myticks = []
mytlabels = []
fig,ax = plt.subplots(3,figsize=[11.08,  11.08]) #fig.get_size_inches()

plt.subplot(3,1,1)

n = 0

for i in range(len(elb)):
        
    # elevation indicator
    elev_ind = np.logical_and(elevation > elb[i], aspect <= eub[i])
        
    for u in range(len(vt_list)):
            
        u_ind = unit==vt_list[u] # unit indicator
        
        for y in range(len(years)):
            
            n = n+1
            
            #year indicator
            year_ind = year==years[y]
            
            # COMBINE INDICATORS
            data_ind = np.logical_and(sun_ind,asp_ind)
            data_ind = np.logical_and(data_ind,u_ind)
            data_ind = np.logical_and(data_ind,elev_ind)
            data_ind = np.logical_and(data_ind,year_ind)
            data_ind = np.logical_and(data_ind,excl_ind)
            
            # data to plot
            ndvi_data = ndvi[data_ind]
            ndvi_time = t_tmp[data_ind]
            
            if len(ndvi_data)==0:
                dq = [0,0,0]
            else:
                se = ndt.eos(ndvi_time,ndvi_data,time_res=time_res,envelope=False,ndvi_th =ndvi_th,pth=pth)
                [*dq] = ndt.auc(ndvi_time,ndvi_data,time_res=time_res,sttt=214,entt=se)
                
                if not(np.all(dq==0)):
                    if y==0 and i==0:
                        ax[0].plot([n*os+i*os*3,n*os+i*os*3],[dq[0],dq[2]],'-',c=cols[u],label=verb_labels_short[u])
                        
                    else:
                        ax[0].plot([n*os+i*os*3,n*os+i*os*3],[dq[0],dq[2]],'-',c=cols[u])
                        
                    ax[0].plot(n*os+i*os*3,dq[1],'o',c=cols[u])

            if u==4 and y==3:
                mytlabels.append(str(elb[i]) + '-' + str(eub[i]))
                myticks.append(n*os+i*os*3)

myticks = np.array(myticks)
tos = (myticks[1]-myticks[0])/2
myticks_st = myticks[0]-tos
myticks_new = myticks + tos
myticks_new = np.append(myticks_st,myticks_new)
ax[0].set_xticks(myticks_new)
plt.grid()
mytlabels_new = np.append(mytlabels,'') 
ax[0].set_xticklabels(mytlabels_new)
offset = matplotlib.transforms.ScaledTranslation(tos/5, 0, fig.dpi_scale_trans)
for label in ax[0].xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

#plt.xlabel('elevation [m]')
plt.ylabel('NDVI AUC')
ax[0].legend(loc='upper right',prop={'size': 10},ncol=3)

ax[0].set_title('a) 2-nd half season NDVI AUC - annual median and Q.25-.75 interval')
ax[0].set_xlim([myticks_new[0],myticks_new[-1]])
ax[0].set_ylim([7,107])
plt.tight_layout()

# AUC Inter-quartile range bar - 2nd season half
plt.subplot(3,1,2)

sun_ind = shadow==0 # sun indicator
asp_ind = np.logical_and(aspect > alb, aspect < aub) # aspect indicator
t_tmp = np.concatenate(ndvi_dict[time_res])[unit_ind] # time vector

# initialize figure
myticks = []
mytlabels = []
#fig,ax = plt.subplots(figsize=[15.59,  8.65])

n = 0
cv = []    
for i in range(len(elb)):
    
    # elevation indicator
    elev_ind = np.logical_and(elevation > elb[i], aspect <= eub[i])
    
    for u in range(len(vt_list)):
        
        u_ind = unit==vt_list[u] # unit indicator        
        iqr=[]
        n = n+1
        for y in range(len(years)):
            #year indicator
            year_ind = year==years[y]
            
            # COMBINE INDICATORS
            data_ind = np.logical_and(sun_ind,asp_ind)
            data_ind = np.logical_and(data_ind,u_ind)
            data_ind = np.logical_and(data_ind,elev_ind)
            data_ind = np.logical_and(data_ind,year_ind)
            data_ind = np.logical_and(data_ind,excl_ind)            
            
            # data to plot
            ndvi_data = ndvi[data_ind]
            ndvi_time = t_tmp[data_ind]
            
            if len(ndvi_data)==0:
                iqr.append([])
            else:
                # area under the curve
                EOS = ndt.eos(ndvi_time,ndvi_data,time_res=time_res,envelope=False,ndvi_th =ndvi_th,pth=pth)
                AUC = ndt.auc(ndvi_time,ndvi_data,time_res=time_res,envelope=True,sttt=214,entt=EOS)
                iqr = AUC[-1]-AUC[0] # interquartile range of AUC
           
        if u==4:
            mytlabels.append(str(elb[i]) + '-' + str(eub[i]))
            myticks.append(n)       

    
        miqr = np.nanmean(iqr)
        if i==0:
            ax[1].bar(n,miqr,color=cols[u],label=verb_labels_short[u])
        else:
            ax[1].bar(n,miqr,color=cols[u])

myticks = np.array(myticks)
tos = (myticks[1]-myticks[0])/2
myticks_st = myticks[0]-tos
myticks_new = myticks + tos
myticks_new = np.append(myticks_st,myticks_new)
ax[1].set_xticks(myticks_new)
plt.grid()
mytlabels_new = np.append(mytlabels,'') 
ax[1].set_xticklabels(mytlabels_new)
offset = matplotlib.transforms.ScaledTranslation(tos/3.2, 0, fig.dpi_scale_trans)
for label in ax[1].xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

#plt.xticks(rotation=90)
#plt.xlabel('elevation [m]')
ax[1].set_ylabel('mean interquartile range')
#ax[1].legend(loc='lower left',prop={'size': 11},ncol=3)
ax[1].set_title('b) 2-nd half season NDVI AUC - mean interquartile range')
ax[1].set_xlim([myticks_new[0],myticks_new[-1]])
plt.tight_layout()


# CV bar - 2nd season half - version b
plt.subplot(3,1,3)

sun_ind = shadow==0 # sun indicator
asp_ind = np.logical_and(aspect > alb, aspect < aub) # aspect indicator
t_tmp = np.concatenate(ndvi_dict[time_res])[unit_ind] # time vector

# initialize figure
myticks = []
mytlabels = []
#fig,ax = plt.subplots(figsize=[15.59,  8.65])

n = 0
cv = []    
for i in range(len(elb)):
    
    # elevation indicator
    elev_ind = np.logical_and(elevation > elb[i], aspect <= eub[i])
    
    for u in range(len(vt_list)):
        
        u_ind = unit==vt_list[u] # unit indicator        
        mq=[]
        n = n+1
        for y in range(len(years)):
            #year indicator
            year_ind = year==years[y]
            
            # COMBINE INDICATORS
            data_ind = np.logical_and(sun_ind,asp_ind)
            data_ind = np.logical_and(data_ind,u_ind)
            data_ind = np.logical_and(data_ind,elev_ind)
            data_ind = np.logical_and(data_ind,year_ind)
            data_ind = np.logical_and(data_ind,excl_ind)            
            
            # data to plot
            ndvi_data = ndvi[data_ind]
            ndvi_time = t_tmp[data_ind]
            
            if len(ndvi_data)==0:
                mq.append(0)
            else:
                # area under the curve
                EOS = ndt.eos(ndvi_time,ndvi_data,time_res=time_res,envelope=False,ndvi_th =ndvi_th,pth=pth)
                mq.append(ndt.auc(ndvi_time,ndvi_data,time_res=time_res,envelope=False,sttt=214,entt=EOS))
           
        if u==4:
            mytlabels.append(str(elb[i]) + '-' + str(eub[i]))
            myticks.append(n)       

    
        cv = np.std(mq)/np.mean(mq)
        if i==0:
            ax[2].bar(n,cv,color=cols[u],label=verb_labels_short[u])
        else:
            ax[2].bar(n,cv,color=cols[u])

myticks = np.array(myticks)
tos = (myticks[1]-myticks[0])/2
myticks_st = myticks[0]-tos
myticks_new = myticks + tos
myticks_new = np.append(myticks_st,myticks_new)
ax[2].set_xticks(myticks_new)
plt.grid()
mytlabels_new = np.append(mytlabels,'') 
ax[2].set_xticklabels(mytlabels_new)
offset = matplotlib.transforms.ScaledTranslation(tos/3.2, 0, fig.dpi_scale_trans)
for label in ax[2].xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

#plt.xticks(rotation=90)
plt.xlabel('elevation [m]')
ax[2].set_ylabel('coefficient of variation')
ax[2].legend(loc='lower left',prop={'size': 11},ncol=3)
ax[2].set_title('c) 2-nd half season NDVI AUC - coefficient of variation of the annual medians')
ax[2].set_xlim([myticks_new[0],myticks_new[-1]])
ax[2].set_ylim([0.04,0.20])
plt.tight_layout()

# save figure
plt.savefig('export/AUC_2nd.pdf')

#%% AUC vs elevation whole season table

sun_ind = shadow==0 # sun indicator
asp_ind = np.logical_and(aspect > alb, aspect < aub) # aspect indicator
t_tmp = np.concatenate(ndvi_dict[time_res])[unit_ind] # time vector

f = open('export/AUCvsALT_table.txt','w')
for u in range(len(vt_list)):
    
    u_ind = unit==vt_list[u] # unit indicator        
    f.write(verb_labels[u][3:] + ' & ') 
    
    R = [] # annual corr vector
    
    for y in range(len(years)):
        #year indicator
        year_ind = year==years[y]
        
        # COMBINE INDICATORS
        data_ind = np.logical_and(sun_ind,asp_ind)
        data_ind = np.logical_and(data_ind,u_ind)
        data_ind = np.logical_and(data_ind,year_ind)
        data_ind = np.logical_and(data_ind,excl_ind)
        
        # select data
        ndvi_data = ndvi[data_ind]
        ndvi_time = t_tmp[data_ind]
        ndvi_elev = elevation[data_ind]
        pid_list = np.unique(pid[data_ind]) # list of parcels
        
        # auc
        auc = []
        elev = []
        for i in range(len(pid_list)):
            ind_tmp = np.logical_and(data_ind,pid==pid_list[i])
            if np.sum(ind_tmp)>=5: # at least 5 point in the annual curve per parcel
                gt = ndt.sog(t_tmp[ind_tmp],ndvi[ind_tmp],time_res=time_res,envelope=False,ndvi_th =ndvi_th,pth=pth)
                se = ndt.eos(t_tmp[ind_tmp],ndvi[ind_tmp],time_res=time_res,envelope=False,ndvi_th =ndvi_th,pth=pth)
                if gt<se: # if a correct curve is detected
                    q = ndt.auc(t_tmp[ind_tmp],ndvi[ind_tmp],time_res=time_res,envelope=False,sttt=gt,entt=se)
                    auc.append(q)
                    elev.append(np.median(elevation[ind_tmp]))
        
        # linear regression
        R.append(np.corrcoef(elev,auc)[0,1])
        
        if R[y] >= 0.6 or R[y] <= -0.6:
            f.write('\\bf %.2f &' % (R[y]))
        else:
            f.write(' %.2f &' % (R[y]))
        
    Rm = np.mean(R) # mean corr
    f.write(str(Rm)[:5] + ' & ')
    
    Rs = np.std(R) # std corr
    f.write(str(Rs)[:5])
    
    f.write('\\\ \n')
f.close()

#%% PEARSON CORRELATION WITH P-VALUE TABLE

sun_ind = shadow==0 # sun indicator
asp_ind = np.logical_and(aspect > alb, aspect < aub) # aspect indicator
t_tmp = np.concatenate(ndvi_dict[time_res])[unit_ind] # time vector
#gtime_ind = t_tmp < 213 # only growing season
stv_name = ['g', # sog
            's', # growth slope
            'a1', # AUC of the 1st season half 
            'm', # greening maximum 
            'a2', # AUC of the 2nd season half 
            'e' # EOS
            ]

f = open('export/rp_table.txt','w')

f.write('unit & ')

# header
N = len(stv_name)
for i in range(N):
    for j in range(i+1,N):
        f.write('$r_{%s,%s}$ ' % (stv_name[i],stv_name[j]))
        if i != N-2 or j != N-1:
            f.write('& ')
f.write('\\\ \n')
              
for u in range(len(vt_list)):
    
    u_ind = unit==vt_list[u] # unit indicator        
    f.write(verb_labels_short[u][3:] + ' & ') 
    
    sl = []
    gt = []
    a1 = []
    a2 = []
    se = []
    m = []
    for y in range(len(years)):
        #year indicator
        year_ind = year==years[y]
    
        # COMBINE INDICATORS
        data_ind = np.logical_and(sun_ind,asp_ind)
        data_ind = np.logical_and(data_ind,u_ind)
        data_ind = np.logical_and(data_ind,year_ind)
        data_ind = np.logical_and(data_ind,excl_ind)
        
        # select data
        ndvi_data = ndvi[data_ind]
        ndvi_time = t_tmp[data_ind]
        
        # slope and sog
        
        if len(ndvi_data)==0:
            continue

        gt_tmp = ndt.sog(ndvi_time,ndvi_data,time_res='doy',ndvi_th=ndvi_th,pth=pth,envelope=False)
        se_tmp = ndt.eos(ndvi_time,ndvi_data,time_res='doy',ndvi_th=ndvi_th,pth=pth,envelope=False)           
        
        if gt_tmp < se_tmp: # if a correct curve is detected
            a1_tmp = ndt.auc(ndvi_time,ndvi_data,time_res='doy',envelope=False,sttt=gt_tmp,entt=213)
            a2_tmp = ndt.auc(ndvi_time,ndvi_data,time_res='doy',envelope=False,sttt=214,entt=se_tmp)
   
            try:
                sl_tmp = ndt.greening_slope(ndvi_time,ndvi_data,envelope=False,plot=False) # plot = True to inspect the curve fitting
                m_tmp = ndt.greening_max(ndvi_time,ndvi_data,envelope=False,plot=False)
                
            except:
                print('error in gompertz')
                continue
        
        else:
            print('error in SOG EOS')
            continue
        
        sl.append(sl_tmp)
        gt.append(gt_tmp)
        a1.append(a1_tmp)
        a2.append(a2_tmp)
        se.append(se_tmp)
        m.append(m_tmp)
    
    stv = np.vstack([gt,sl,a1,a2,m,se]).T
    
    for i in range(N):
        for j in range(i+1,N): 
            nonan_ind = np.logical_and(~np.isnan(stv[:,i]),~np.isnan(stv[:,j]))
            Rtmp = pearsonr(stv[nonan_ind,i],stv[nonan_ind,j])
            
            if (Rtmp.statistic >= 0.6 or Rtmp.statistic <= -0.6) and Rtmp.pvalue < 0.05:
                f.write('\\bf %.2f ' % (Rtmp.statistic))
                
            else:
                f.write('%.2f ' % (Rtmp.statistic))
            
            if i != N-2 or j != N-1:
                f.write('& ')
        
    f.write('\\\ \n')
f.close()