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

#####modify these variables

grid_file = '/home/rmsanche/research/LV1_2017/GRID_SDTJRE_LV1_rx020_hmask.nc' #LV1 grid
datestring_start = '2017.01.17.03'
datestring_end = '2017.01.31.21'
#out_dir = 'Data/test_hycom' # processes in a different file
hout_dir = '/home/rmsanche/research/LV1_2017/Hycom/Data/' #where the history files are dumped

#####functions
#lightly modified from Parkers original
def get_data(this_dt, out_fn, aa):
    """"
    From LO package hycom data using the FMRC_best file.
    It gets only a single time, per the new guidance from Michael McDonald
    at HYCOM, 2020.03.16.
    
    Note that this hard-codes HYCOM experiment info like GLBy0.08/expt_93.0
    and so could fail when this is superseded.

    Also it is no longer daily but every three hours

    L1 grid is hardcoded
    """
    ds_fmt = '%Y.%m.%d.%H'
    dstr = this_dt.strftime(ds_fmt)
    # time string in HYCOM format
    dstr_hy = this_dt.strftime('%Y-%m-%d-T%H:%M:%SZ')
    
    
    print(' - getting hycom fields for ' + dstr)
    
    # specify spatial limits

    north = aa[0]
    south = aa[1]
    west = aa[2]
    east = aa[3]
    testing = False #turn off for run
    if testing == True:
        var_list = 'surf_el'
    else:
        #var_list = 'surf_el,water_temp,salinity,water_u,water_v'
        var_list = 'surf_el&var=salinity&var=water_temp&var=water_u&var=water_v'

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
    https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_57.2
    ?var=surf_el&var=salinity&var=water_temp&var=water_u&var=water_v
    &north=36.39&west=236.28&east=244.22&south=28.52&disableProjSubset=on&horizStride=1&time=2018-12-15T12%3A00%3A00Z&vertCoord=&accept=netcdf4
    """
    url = ('https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_57.2'+
        '?var='+var_list +
        '&north='+str(north)+'&south='+str(south)+'&west='+str(west)+'&east='+str(east) +
        '&disableProjSubset=on&&horizStride=1' +
        '&time='+dstr_hy +
        '&vertCoord=&accept=netcdf4') 
        #'&addLatLon=true&accept=netcdf4')    
    if verbose:
        print(url)
    # new version 2020.04.22 using requests
    counter = 1
    got_fmrc = False
    while (counter <= 10) and (got_fmrc == False):
        print(' - Attempting to get data, counter = ' + str(counter))
        time.sleep(10) # pause before each request
        tt0 = time.time()
        try:
            r = requests.get(url, timeout=200)
            if r.ok:
                with open(out_fn,'wb') as f:
                    f.write(r.content)
                got_fmrc = True
                r.close()
            elif not r.ok:
                print(' - Failed with status code:')
                print(r.status_code)
        except Exception as e:
            print(' - Exception from requests:')
            print(e)
        counter += 1
        print(' - took %0.1f seconds' % (time.time() - tt0))
        print(datetime.now())
        print('')
        sys.stdout.flush()
    return got_fmrc


### code that gets run
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
#will be converted to a list of days

ds_fmt = '%Y.%m.%d.%H'
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

while dtff <= dt1:
    dt_list_full.append(dtff)
    dtff = dtff + timedelta(hours=3)
    
verbose = True

tt0 = time.time()
for dtff in dt_list_full:
    got_fmrc = False
    data_out_fn =  h_out_dir /  ('h' + dtff.strftime(ds_fmt)+ '.nc')   
    if verbose:
        print(data_out_fn)
    sys.stdout.flush()
    # get hycom forecast data from the web, and save it in the file "data_out_fn".
    # it tries 10 times before ending
    got_fmrc = get_data(dtff, data_out_fn,aa)
    if got_fmrc == False:
        # this should break out of the dtff loop at the first failure
        # and send the code to Plan C
        print('- error getting forecast files using fmrc')
        
print('Time to get full file using get url = %0.2f sec' % (time.time()-tt0))

