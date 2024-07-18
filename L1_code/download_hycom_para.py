#download_hycom_para
#attempt to parallize the downloading of hycom data
#not clean code
# import
import xarray as xr
import matplotlib.pyplot as plt
import os, sys, shutil
from pathlib import Path 
from datetime import datetime, timedelta
import netCDF4 as nc
import numpy as np
import time
import pickle
from scipy.spatial import cKDTree
import seawater
import subprocess
import requests
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

#####modify these variables

grid_file = '/home/rmsanche/research/LV1_2017/GRID_SDTJRE_LV1_rx020_hmask.nc' #LV1 grid
datestring_start = '2017.10.01.21'
datestring_end = '2017.12.05.12'
#out_dir = 'Data/test_hycom' # processes in a different file
#hout_dir = '/home/rmsanche/research/LV1_2017/Hycom/Data/' #where the history files are dumped
#^^^ edit this in the code below full_fn_out is the variable

###### Important, before running make sure the URL is over the correct time stamp


#####functions


def para_loop(dtff,aa,ds_fmt):
    north = aa[0]
    south = aa[1]
    west = aa[2]
    east = aa[3]



    # template for url from Michael MacDonald March 2020
    """
    https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_93.0/FMRC/GLBy0.08_930_FMRC_best.ncd
    ?var=surf_el,water_temp,salinity,water_u,water_v
    &north=53&south=39&west=229&east=239
    &time=2020-03-11-T00:00:00Z
    &addLatLon=true&accept=netcdf4
    """
    #/thredds/ncss/GLBy0.08/expt_93.0 different url because example is out of range
    # THE URL BELOW IS FOR AFTER 2018
    """
    https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_93.0
    ?var=surf_el&var=salinity&var=water_temp&var=water_u&var=water_v
    &north=36.39&west=236.28&east=244.22&south=28.52&disableProjSubset=on&horizStride=1&time=2018-12-15T12%3A00%3A00Z&vertCoord=&accept=netcdf4
    """
    # THE URL BELOW IS FOR May 2016 through Jan 31 2017
    """
    https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_57.2
    ?var=surf_el&var=salinity&var=water_temp&var=water_u&var=water_v
    &north=36.39&west=236.28&east=244.22&south=28.52&disableProjSubset=on&horizStride=1&time=2018-12-15T12%3A00%3A00Z&vertCoord=&accept=netcdf4
    """
    # THE URL BELOW IS FOR Feb 1st 2017 through May 31st
    """
    https://tds.hycom.org/thredds/ncss/GLBv0.08/expt_92.8
    ?var=surf_el&var=salinity&var=water_temp&var=water_u&var=water_v
    &north=36.39&west=236.28&east=244.22&south=28.52&disableProjSubset=on&horizStride=1&time=2018-12-15T12%3A00%3A00Z&vertCoord=&accept=netcdf4
    """
    #the URL below is for June 1st (noon) through
    
    #https://tds.hycom.org/thredds/ncss/GLBv0.08/expt_57.7
    ## MODIFY THIS URL, PAY ATTENTION TO lowercase letter in GLBv or GLBy etc
        

    data_out_fn =  h_out_dir /  ('h' + dtff.strftime(ds_fmt)+ '.nc')   
    if verbose:
        print(data_out_fn)
    sys.stdout.flush()


    # time limits
    dtff_cop = dtff
    dtff_adv = dtff+timedelta(hours=2)
    dstr0 = dtff.strftime('%Y-%m-%dT%H:%M')
    dstr1 = dtff_adv.strftime('%Y-%m-%dT%H:%M')
    # use subprocess.call() to execute the ncks command
    vstr = 'surf_el,water_temp,salinity,water_u,water_v'
    #where to save the data
    full_fn_out=  '/home/rmsanche/research/LV1_2017/Hycom/Data/'+'h' + dtff_cop.strftime(ds_fmt)+ '.nc'  
    cmd_list = ['ncks',
        '-d', 'time,'+dstr0+','+dstr1,
        '-d', 'lon,'+str(west-.05)+','+str(east)+'',
        '-d', 'lat,'+str(south-.08)+','+str(north)+'',
        '-v', vstr,
        'https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.9',
        '-4', '-O', full_fn_out]

    print(cmd_list)

    # run ncks
    ret1 = subprocess.call(cmd_list)
    return dtff








### code that gets aa
grid = xr.open_dataset(grid_file)
north = max(grid.lat_v.max(),grid.lat_rho.max(),grid.lat_u.max()).values #north boundary
dlat = 0.04
dlon = 0.04
south = min(grid.lat_v.min(),grid.lat_rho.min(),grid.lat_u.min()).values #south boundary
east = max(grid.lon_v.max(),grid.lon_rho.max(),grid.lon_u.max()).values +360 #east boundary
west = min(grid.lon_v.min(),grid.lon_rho.min(),grid.lon_u.min()).values +360#west boundary
aa = [north+dlat, south-dlat, west-dlon, east+dlon]
aa = [np.round(i,2) for i in aa]

#establish timestamp

ds_fmt = '%Y.%m.%d.%H'
ds_fmt2 = '%Y-%m-%d-T%H:%M:%SZ'
this_dt = datetime.strptime(datestring_start, ds_fmt)



end_dt = datetime.strptime(datestring_end, ds_fmt)
dstr = this_dt.strftime(ds_fmt)
# time string in HYCOM format
dstr_hy = this_dt.strftime('%Y-%m-%d-T%H:%M:%SZ')


#make list

#out_dir = pathlib.Path(out_dir)

h_out_dir = pathlib.Path(hout_dir)

# form list of days to get, datetimes
dt0 = this_dt
dt1 = end_dt 
dt_list_full = []
dtff = dt0

#timestamps
while dtff <= dt1:
    dt_list_full.append(dtff)
    dtff = dtff + timedelta(hours=3)
    
verbose = True
#this is where we para
tt0 = time.time()



# create parallel executor
with ThreadPoolExecutor() as executor:
    threads = []
    for dtff in dt_list_full:
        fn = para_loop #define function
        args = [dtff,aa,ds_fmt] #define args
        kwargs = {} #
        # start thread by submitting it to the executor
        threads.append(executor.submit(fn, *args, **kwargs))
    for future in as_completed(threads):
            # retrieve the result
            result = future.result()
            # report the result
 


        
print('Time to get full file using get url = %0.2f sec' % (time.time()-tt0))


