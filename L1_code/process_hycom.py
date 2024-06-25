import sys, os
from datetime import datetime, timedelta
from pathlib import Path
from time import time, sleep
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.spatial import cKDTree
from scipy import signal
import netCDF4 as nc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os, sys, shutil
from datetime import datetime, timedelta
import pickle
import seawater

#####modify these variables

grid_file = '/home/rmsanche/research/LV1_2017/GRID_SDTJRE_LV1_rx020_hmask.nc' #LV1 grid

datestring_start = '2016.11.15.00' #initial file time, note it will process all the times in a folder after this time

hout_dir = '/home/rmsanche/research/LV1_2017/Hycom/Data/' #where the history files are located
out_dir = '/home/rmsanche/research/LV1_2017/Hycom/' #where the processed files are to be dumped
S_info_dict = {'N' : 40, 'VSTRETCHING': 4, 'VTRANSFORM': 2, 'TCLINE': 50.0, 'THETA_S': 8.0, 'THETA_B': 3.0}

#the output file names ocn_clm.nc, ocn_bry.nc and ocn_ini.nc are standardized but can updated at the bottom of the code

#####functions
#lightly modified from Parkers original
def get_hnc_short_list(h_out_dir):
    '''generates a shortlist that has just hycom nc files'''
    hnc_short_list = sorted([item.name for item in h_out_dir.iterdir()
        if (item.name[0]=='h' and item.name[-3:]=='.nc')])
    hnc_short_list = [h_out_dir / i for i in hnc_short_list]
    return hnc_short_list


def convert_extraction(fn):
    """
    This converts a hycom NetCDF extraction file into a
    pickled dict of fields, doing some renaming and packing things
    bottom to top.  It could probably be skipped over in future
    versions, and is here because subsequent workflow depends on the
    pickled dict existing.
    """
    
    testing = False
    
    # initialize an output dict
    out_dict = dict()
    
    # load the file
    ds = nc.Dataset(fn)
    # fn can be either a Path object or a string, in either case
    # corresponding to a NetCDF file.
    
    if testing:
        print('opening ' + fn)
    
    # get time info
    t = ds['time'][0]
    if isinstance(t, np.ma.MaskedArray):
        th = t.data
    else:
        th = t
    tu = ds['time'].units
    # e.g. 'hours since 2018-11-20 12:00:00.000 UTC'
    # Warning: Brittle code below!
    ymd = tu.split()[2]
    hmss = tu.split()[3]
    hms = hmss.split('.')[0]
    hycom_dt0 = datetime.strptime(ymd + ' ' + hms, '%Y-%m-%d %H:%M:%S')
    this_dt = hycom_dt0 + timedelta(days=(th/24))
    out_dict['dt'] = this_dt # datetime time of this snapshot
    # print('- for dt = ' + str(this_dt))
    
    if testing == False:
        # create z from the depth
        depth = ds.variables['depth'][:]
        z = -depth[::-1] # you reverse an axis with a -1 step!
        out_dict['z'] = z
        N = len(z)
        
    # get full coordinates (vectors for the plaid grid)
    lon_offset = 0
    if np.min(ds.variables['lon'][:]) > 0:
        lon_offset = 360
    lon = ds.variables['lon'][:] - lon_offset  # convert from 0:360 to -360:0 format
    lat = ds.variables['lat'][:]
    # and save them   
    out_dict['lon'] = lon
    out_dict['lat'] = lat
    
    if testing == True:
        var_list = ['surf_el']
    else:
        var_list = ['surf_el','water_temp','salinity','water_u','water_v']
        
    # get the dynamical variables
    for var_name in var_list:
        # Get the variables, renaming to be consistent with what we want,
        # and pack bottom to top
        #np.warnings.filterwarnings('ignore') # suppress warning related to nans in fields
        if var_name == 'surf_el':
            ssh = ds['surf_el'][0, :, :]
            out_dict['ssh'] = ssh
        elif var_name == 'water_temp':
            t3d = ds['water_temp'][0, :, :, :]
            t3d = t3d[::-1, :, :] # pack bottom to top
            out_dict['t3d'] = t3d
        elif var_name == 'salinity':
            s3d = ds['salinity'][0, :, :, :]
            s3d = s3d[::-1, :, :]
            out_dict['s3d'] = s3d
        elif var_name == 'water_u':
            u3d = ds['water_u'][0, :, :, :]
            u3d = u3d[::-1, :, :]
            out_dict['u3d'] = u3d
        elif var_name == 'water_v':
            v3d = ds['water_v'][0, :, :, :]
            v3d = v3d[::-1, :, :] # pack bottom to top
            out_dict['v3d'] = v3d
    ds.close()
    
    return out_dict # the keys of this dictionary are separate variables

def time_filter(in_dir, h_list, out_dir,datestring_start):
    """
    Filter the files in time to get rid of inertial oscillations that
    are aliased in the sampling. The points are then subsampled down. Currently does 5 days filter.
    """
    print('-Filtering in time')
    vl = ['ssh', 'u3d', 'v3d', 't3d', 's3d']
    #use initial date
    #dts0 = Ldir['date_string']
    ds_fmt_H = '%Y.%m.%d.%H'
    ds_fmt = '%Y.%m.%d'
    dts0 =datestring_start #set intial time
    dt0 = datetime.strptime(dts0, ds_fmt_H)
    nh = len(h_list)    
    # test for gaps in h_list
    no_gaps = True
    dtsg = h_list[0].strip('h').strip('.p')
    dtg0 = datetime.strptime(dtsg, ds_fmt_H)
    for hh in h_list[1:]:
        dtsg = hh.strip('h').strip('.p')
        dtg1 = datetime.strptime(dtsg, ds_fmt_H)
        if (dtg1-dtg0).seconds > 3600*3: #3 hours
            no_gaps = False
            print('** HAS GAPS **')
            break
        else:
            dtg0 = dtg1    
    # inverse weighting factors for a Hanning window of length 5
    #fac_list_H = [12, 4, 3, 4, 12]
    #inverse weighting for Hanning length 40 (8*5 days)
    fac_list_H = 1/(signal.windows.hann(42)/(np.sum(signal.windows.hann(42))))[1:-1]
   
    #no filtering
    nfilt = len(fac_list_H)
    #nd_f = Ldir['forecast_days']
    nd_f = 7*8 #*8 to account for days, 
    nhmin_f = nfilt + nd_f #min number of points to complete a 5 day hanning window
    if (nh>=nhmin_f) and no_gaps:
        print('--Using Hanning window')
        fac_list = fac_list_H
        for nt in range(0,nh - 39,8): #39 is 4 if using days, 8 is 1 if using days. This is what subsamples it down
            #loop through timestamps
            n_center = nt + 16 #is 2 if days
            aa = dict()
            for n in range(nfilt):
                #loop through files in hanning
                nn = n + nt #
                fn = in_dir / h_list[nn]
                a = pickle.load(open(fn, 'rb'))
                for v in vl:
                    if n == 0:
                        aa[v] = a[v]/fac_list[n]
                    else:
                        aa[v] = aa[v] + a[v]/fac_list[n]
            out_name = 'f' + h_list[n_center]
            dts = out_name.strip('fh').strip('.p') 
            dt = datetime.strptime(dts, ds_fmt_H)
            aa['dt'] = dt.date()
            out_name= 'fh'+datetime.strftime(dt.date(), ds_fmt)+'.p'
            print('   ' + out_name)
            pickle.dump(aa, open(out_dir / out_name, 'wb'))
    else:
        print('--Using block average (NOT TESTED FOR PFM')
        # make a simple average and use it for everything
        fac_list = list(nh * np.ones(nh))
        print(fac_list)
        aa = dict()
        for n in range(nh):
            fn = in_dir / h_list[n]
            a = pickle.load(open(fn, 'rb'))
            for v in vl:
                if n == 0:
                    aa[v] = a[v]/fac_list[n]
                else:
                    aa[v] = aa[v] + a[v]/fac_list[n]        
        if rtp == 'backfill':
            nd = 1
        else:
            nd = 3
        # saving the first file
        out_name0 = 'fh' + dts0 + '.p' 
        aa['dt'] = dt0
        # print('   ' + out_name0)
        pickle.dump(aa, open(out_dir / out_name0, 'wb'))
        # saving the last file
        dt1 = dt0 + timedelta(days=nd)
        dts1 = datetime.strftime(dt1, ds_fmt2)
        out_name1 = 'fh' + dts1 + '.p'            
        aa['dt'] = dt1           
        # print('   ' + out_name1)
        pickle.dump(aa, open(out_dir / out_name1, 'wb'))


def earth_rad(lat_deg):
    """
    Calculate the Earth radius (m) at a latitude
    (from http://en.wikipedia.org/wiki/Earth_radius) for oblate spheroid

    INPUT: latitude in degrees

    OUTPUT: Earth radius (m) at that latitute
    """
    a = 6378.137 * 1000; # equatorial radius (m)
    b = 6356.7523 * 1000; # polar radius (m)
    cl = np.cos(np.pi*lat_deg/180)
    sl = np.sin(np.pi*lat_deg/180)
    RE = np.sqrt(((a*a*cl)**2 + (b*b*sl)**2) / ((a*cl)**2 + (b*sl)**2))
    return RE

def ll2xy(lon, lat, lon0, lat0):
    """
    This converts lon, lat into meters relative to lon0, lat0.
    It should work for lon, lat scalars or arrays.
    NOTE: lat and lon are in degrees!!
    """
    R = earth_rad(lat0)
    clat = np.cos(np.pi*lat0/180)
    x = R * clat * np.pi * (lon - lon0) / 180
    y = R * np.pi * (lat - lat0) / 180
    return x, y

def get_coords(in_dir):
    """
    get coordinate fields and sizes
    """
    coord_dict = pickle.load(open(in_dir / 'coord_dict.p', 'rb'))
    lon = coord_dict['lon']
    lat = coord_dict['lat']
    z = coord_dict['z']
    L = len(lon)
    M = len(lat)
    N = len(z)
    # Create arrays of distance from the center (m) so that the
    # nearest neighbor extrapolation is based on physical distance
    Lon, Lat = np.meshgrid(lon,lat)
    X, Y = ll2xy(Lon, Lat, lon.mean(), lat.mean())
    return (lon, lat, z, L, M, N, X, Y)


#extrap functions
def checknan(fld):
    """
    A utility function that issues a working if there are nans in fld.
    """
    if np.isnan(fld).sum() > 0:
        print('WARNING: nans in data field')  
        
def extrap_nearest_to_masked(X, Y, fld, fld0=0):
    """
    INPUT: fld is a 2D array (np.ndarray or np.ma.MaskedArray) on spatial grid X, Y
    OUTPUT: a numpy array of the same size with no mask
    and no missing values.        
    If input is a masked array:        
        * If it is ALL masked then return an array filled with fld0.         
        * If it is PARTLY masked use nearest neighbor interpolation to
        fill missing values, and then return data.        
        * If it is all unmasked then return the data.    
    If input is not a masked array:        
        * Return the array.    
    """
    # first make sure nans are masked
    if np.ma.is_masked(fld) == False:
        fld = np.ma.masked_where(np.isnan(fld), fld)
        
    if fld.all() is np.ma.masked:
        #print('  filling with ' + str(fld0))
        fldf = fld0 * np.ones(fld.data.shape)
        fldd = fldf.data
        checknan(fldd)
        return fldd
    else:
        # do the extrapolation using nearest neighbor
        fldf = fld.copy() # initialize the "filled" field
        xyorig = np.array((X[~fld.mask],Y[~fld.mask])).T
        xynew = np.array((X[fld.mask],Y[fld.mask])).T
        a = cKDTree(xyorig).query(xynew)
        aa = a[1]
        fldf[fld.mask] = fld[~fld.mask][aa]
        fldd = fldf.data
        checknan(fldd)
        return fldd


def get_extrapolated(in_fn, L, M, N, X, Y, lon, lat, z, add_CTD=False):
    """
    Make use of extrap_nearest_to_masked() to fill fields completely
    before interpolating to the ROMS grid.  It also adds CTD data if asked to,
    creates ubar and vbar, and converts the temperature to potential temperature. Modified with Matt code to interpolate
    """
    b = pickle.load(open(in_fn, 'rb'))
    vn_list = list(b.keys())    
    # check that things are the expected shape
    def check_coords(shape_tuple, arr_shape):
        if arr_shape != shape_tuple:
            print('WARNING: array shape mismatch')
    for vn in vn_list:
        if vn == 'dt':
            pass
        elif vn == 'ssh':
            check_coords((M, L), b[vn].shape)
        else:
            check_coords((N, M, L), b[vn].shape)    
    # creat output array and add dt to it.
    vn_list.remove('dt')
    V = dict()
    for vn in vn_list:
        V[vn] = np.nan + np.ones(b[vn].shape)
    V['dt'] = b['dt']    
    # extrapolate ssh
    vn = 'ssh'
    v = b[vn]
    #instead use nearest
    vv = extrap_nearest_to_masked(X, Y, v)
    V[vn] = vv
    vn_list.remove('ssh')    
    # extrapolate 3D fields
    for vn in vn_list:
        v = b[vn]
        if vn == 't3d':
            v0 = np.nanmin(v)
        elif vn == 's3d':
            v0 = np.nanmax(v)
        if vn in ['t3d', 's3d']:
            # print(' -- extrapolating ' + vn)
            if add_CTD==False:
                for k in range(N):
                    fld = v[k, :, :]
                    fldf = extrap_nearest_to_masked(X, Y, fld, fld0=v0)
                    V[vn][k, :, :] = fldf
            elif add_CTD==True:
                continue
                '''
                print(vn + ' Adding CTD data before extrapolating')
                Cast_dict, sta_df = Ofun_CTD.get_casts(Ldir)
                for k in range(N):
                    fld = v[k, :, :]
                    zz = z[k]
                    xyorig, fldorig = Ofun_CTD.get_orig(Cast_dict, sta_df,
                        X, Y, fld, lon, lat, zz, vn)
                    fldf = Ofun_CTD.extrap_nearest_to_masked_CTD(X,Y,fld,
                        xyorig=xyorig,fldorig=fldorig,fld0=v0)
                    V[vn][k, :, :] = fldf
                '''
        elif vn in ['u3d', 'v3d']:
            # print(' -- extrapolating ' + vn)
            vv = v.copy()
            vv = np.ma.masked_where(np.isnan(vv), vv)
            vv[vv.mask] = 0
            V[vn] = vv.data
    # Create ubar and vbar.
    # Note: this is slightly imperfect because the z levels are at the same
    # position as the velocity levels.
    dz = np.nan * np.ones((N, 1, 1))
    dz[1:, 0, 0]= np.diff(z)
    dz[0, 0, 0] = dz[1, 0, 0]
    
    # account for the fact that the new hycom fields do not show up masked
    u3d = np.ma.masked_where(np.isnan(b['u3d']),b['u3d'])
    v3d = np.ma.masked_where(np.isnan(b['v3d']),b['v3d'])
    dz3 = dz * np.ones_like(u3d) # make dz a masked array
    b['ubar'] = np.sum(u3d*dz3, axis=0) / np.sum(dz3, axis=0)
    b['vbar'] = np.sum(v3d*dz3, axis=0) / np.sum(dz3, axis=0)
    
    for vn in ['ubar', 'vbar']:
        v = b[vn]
        vv = v.copy()
        vv = np.ma.masked_where(np.isnan(vv), vv)
        vv[vv.mask] = 0
        V[vn] = vv.data  
    # calculate potential temperature
    press_db = -z.reshape((N,1,1))
    V['theta'] = seawater.ptmp(V['s3d'], V['t3d'], press_db)    
    return V

def get_basic_info(fn, only_G=False, only_S=False, only_T=False):
    """
    Gets grid, vertical coordinate, and time info from a ROMS NetCDF
    history file with full name 'fn'
    Input: the filename (with path if needed)
    Output: dicts G, S, and T
    Example calls:
    G, S, T = zfun.get_basic_info(fn)
    T = zfun.get_basic_info(fn, only_T=True)
    """
    ds = xr.open_dataset(fn)
    def make_G(ds):
        # get grid and bathymetry info
        g_varlist = ['h', 'lon_rho', 'lat_rho', 'lon_u', 'lat_u', 'lon_v', 'lat_v',
        'lon_psi', 'lat_psi', 'mask_rho', 'mask_u', 'mask_v', 'pm', 'pn','angle']
        G = dict()
        for vv in g_varlist:
            G[vv] = ds[vv].values
        G['DX'] = 1/G['pm']
        G['DY'] = 1/G['pn']
        G['M'], G['L'] = np.shape(G['lon_rho']) # M = rows, L = columns
        return G
    def make_S(ds):
        # get vertical sigma-coordinate info (vectors are bottom to top)
        s_varlist = ['s_rho', 's_w', 'hc', 'Cs_r', 'Cs_w', 'Vtransform']
        S = dict()
        for vv in s_varlist:
            S[vv] = ds[vv].values
        S['N'] = len(S['s_rho']) # number of vertical levels
        return S
    def make_T(ds):
        # returns two single values, one a datatime, and one a float
        ot = ds.ocean_time.values # an array with dtype='datetime64[ns]'
        dti = pd.to_datetime(ot) # a pandas DatetimeIndex with dtype='datetime64[ns]'
        dt = dti.to_pydatetime() # an array of datetimes
        T = dict()
        T['dt'] = dt[0] # a datetime object
        T['ocean_time'] = Lfun.datetime_to_modtime(dt[0]) # a float, "seconds since..."
        return T
    # return results
    if only_G:
        return make_G(ds)
    elif only_S:
        return make_S(ds)
    elif only_T:
        return make_T(ds)
    else:
        return make_G(ds), make_S(ds), make_T(ds)
    ds.close()

def get_S(S_info_dict):
    """
    Code to calculate S-coordinate vectors from the parameters
    in S_COORDINATE_INFO.csv.
    Need to check this carefully against the matlab version.
    # recoded for python on 7/7/2016 from:
    # Z_scoord.m  5/21/2007  Parker MacCready
    # this creates the structure S, which would be used for example by
    # Z_s2z.m, given basic grid parameters
    # edited by DAS to include more things in S stucture
    # edited by SNG March 2011 to include all of the current available ROMS
    # stretching functions, 1-4 see:
    # https://www.myroms.org/wiki/index.php/Vertical_S-coordinate#Vertical_Stretching_Functions
    
    NOTES 2019.09.11
    (1) I checked that Cs_r and _w made by this program are identical to those which are
    given in the ROMS history files.  They are.
    (2) I also made some inquiries on the ROMS forum to make sure that the parameter 'hc' is
    being done correctly.  The short answer is that yes it is.  With Vtransform = 2 (my
    new default) it is given by Tcline from the .in file.  In older runs with Vtransform = 1
    is it min(hmin, Tcline) and this REQUIRES that Tcline be less than hmin.  Since all those
    older runs used Tcline = 0 then hc = 0.
    
    """
    S = dict()
    for item in S_info_dict.keys():
        if item in ['N', 'VSTRETCHING', 'VTRANSFORM']:
            S[item.title()] = int(S_info_dict[item])
        elif item in ['TCLINE', 'THETA_S', 'THETA_B']:
            S[item.lower()] = float(S_info_dict[item])
        else:
            pass
    N = S['N']
    Vstretching = S['Vstretching']
    Vtransform = S['Vtransform']
    tcline = S['tcline']
    theta_s = S['theta_s']
    theta_b = S['theta_b']
    hmin = 3 # a placeholder, used only for Vtransform = 1.
    if Vtransform == 1:
        hc = min(hmin,tcline)
    elif Vtransform == 2:
        hc = tcline
    S['hc'] = hc
    s_rho = (np.linspace(-(N-1), 0, N) - 0.5)/N
    s_w = np.linspace(-N, 0, N+1)/N
    S['s_rho'] = s_rho
    S['s_w'] = s_w
    if Vstretching == 1:
        if theta_s != 0:
            cff1 = 1/np.sinh(theta_s)
            cff2 = 0.5/np.tanh(0.5*theta_s)
            Cs_r = ( (1-theta_b)*cff1*np.sinh(theta_s*s_rho)
                    + theta_b*( cff2*np.tanh(theta_s*(s_rho + 0.5)) - 0.5 ) )
            Cs_w = ( (1-theta_b)*cff1*np.sinh(theta_s*s_w)
                    + theta_b*( cff2*np.tanh(theta_s*(s_w + 0.5)) - 0.5 ) )
        else:
            Cs_r = s_rho
            Cs_w = s_w
    elif Vstretching == 2:
        alpha = 1
        beta = 1
        if theta_s!=0 and theta_b!=0:
            Csur = (1-np.cosh(theta_s*s_rho))/(np.cosh(theta_s)-1)
            Cbot = ((np.sinh(theta_b*(s_rho+1)))/(np.sinh(theta_b)))-1
            u = ((s_rho+1)**alpha)*(1+(alpha/beta)*(1-((s_rho+1)**beta)))
            Cs_r = u*Csur+(1-u)*Cbot
            Csur_w = (1-np.cosh(theta_s*s_w))/(np.cosh(theta_s)-1)
            Cbot_w = ((np.sinh(theta_b*(s_w+1)))/(np.sinh(theta_b)))-1
            u_w = ((s_w+1)**alpha)*(1+(alpha/beta)*(1-((s_w+1)**beta)))
            Cs_w = u_w*Csur_w+(1-u_w)*Cbot_w
        else:
            Cs_r = s_rho
            Cs_w = s_w
    elif Vstretching == 3:
        # Geyer function for high bbl resolution in shallow applications
        gamma = 3
        Csur = -(np.log(np.cosh(gamma*abs(s_rho)**theta_s)))/np.log(np.cosh(gamma))
        Cbot = ((np.log(np.cosh(gamma*(s_rho+1)**theta_b)))/np.log(np.cosh(gamma)))-1
        mu = 0.5*(1-np.tanh(gamma*(s_rho+0.5)))
        Cs_r = mu*Cbot+(1-mu)*Csur
        Csur_w = -(np.log(np.cosh(gamma*abs(s_w)**theta_s)))/np.log(np.cosh(gamma))
        Cbot_w = ((np.log(np.cosh(gamma*(s_w+1)**theta_b)))/np.log(np.cosh(gamma)))-1
        mu_w = 0.5*(1-np.tanh(gamma*(s_w+0.5)))
        Cs_w = mu_w*Cbot_w+(1-mu_w)*Csur_w
    elif Vstretching == 4:
        # newest ROMS default as of March 2011 (theta_s between 0 and 10,
        # theta_b between 0 and 4)
        if theta_s>0:
            Cs_r = (1-np.cosh(theta_s*s_rho))/(np.cosh(theta_s)-1)
            Cs_w = (1-np.cosh(theta_s*s_w))/(np.cosh(theta_s)-1)
        elif theta_s<=0:
            Cs_r = -(s_rho**2)
            Cs_w = -(s_w**2)
        if theta_b > 0:
            Cs_r = (np.exp(theta_b*Cs_r)-1)/(1-np.exp(-theta_b))
            Cs_w = (np.exp(theta_b*Cs_w)-1)/(1-np.exp(-theta_b))
    S['Cs_r'] = Cs_r
    S['Cs_w'] = Cs_w
    return S
    
def get_z(h, zeta, S, only_rho=False, only_w=False):
    """
    Used to calculate the z position of fields in a ROMS history file

    Input: arrays h (bathymetry depth) and zeta (sea surface height)
    which must be the same size, and dict S created by get_basic_info()

    Output: 3-D arrays of z_rho and z_w

    NOTE: one foible is that if you input arrays of h and zeta that are
    vectors of length VL, the output array (e.g. z_rho) will have size (N, VL)
    (i.e. it will never return an array with size (N, VL, 1), even if (VL, 1) was
    the input shape).  This is a result of the initial and final squeeze calls.
    """
    # input error checking
    if ( (not isinstance(h, np.ndarray))
        or (not isinstance(zeta, (np.ndarray, np.ma.core.MaskedArray))) ):
        print('WARNING from get_z(): Inputs must be numpy arrays')
    if not isinstance(S, dict):
        print('WARNING from get_z(): S must be a dict')
    # number of vertical levels
    N = S['N']
    # remove singleton dimensions
    h = h.squeeze()
    zeta = zeta.squeeze()
    # ensure that we have enough dimensions
    h = np.atleast_2d(h)
    zeta = np.atleast_2d(zeta)
    # check that the dimensions are the same
    if h.shape != zeta.shape:
        print('WARNING from get_z(): h and zeta must be the same shape')
    M, L = h.shape
    def make_z_rho(h, zeta, S, N, M, L):
        # rho
        # create some useful arrays
        csr = S['Cs_r']
        csrr = csr.reshape(N, 1, 1).copy()
        Cs_r = np.tile(csrr, [1, M, L])
        H_r = np.tile(h.reshape(1, M, L).copy(), [N, 1, 1])
        Zeta_r = np.tile(zeta.reshape(1, M, L).copy(), [N, 1, 1])
        if S['hc'] == 0: # if hc = 0 the transform is simpler (and faster)
            z_rho = H_r*Cs_r + Zeta_r + Zeta_r*Cs_r
        elif S['hc'] != 0: # need to calculate a few more useful arrays
            sr = S['s_rho'] # PM edit 2019.01.24
            srr = sr.reshape(N, 1, 1).copy()
            S_rho = np.tile(srr, [1, M, L])
            Hc_r = np.tile(S['hc'], [N, M, L])
            if S['Vtransform'] == 1:
                zr0 = (S_rho - Cs_r) * Hc_r + Cs_r*H_r
                z_rho = zr0 + Zeta_r * (1 + zr0/H_r)
            elif S['Vtransform'] == 2:
                zr0 = (S_rho*Hc_r + Cs_r*H_r) / (Hc_r + H_r)
                z_rho = Zeta_r + (Zeta_r + H_r)*zr0
        z_rho = z_rho.squeeze()
        return z_rho
    def make_z_w(h, zeta, S, N, M, L):
        # w
        # create some useful arrays
        csw = S['Cs_w']
        csww = csw.reshape(N+1, 1, 1).copy()
        Cs_w = np.tile(csww, [1, M, L])
        H_w = np.tile(h.reshape(1, M, L).copy(), [N+1, 1, 1])
        Zeta_w = np.tile(zeta.reshape(1, M, L).copy(), [N+1, 1, 1])
        if S['hc'] == 0: # if hc = 0 the transform is simpler (and faster)
            z_w = H_w*Cs_w + Zeta_w + Zeta_w*Cs_w
        elif S['hc'] != 0: # need to calculate a few more useful arrays
            #sw = S['s_w']
            sw = S['s_w'] # PM edit 2019.01.24
            sww = sw.reshape(N+1, 1, 1).copy()
            S_w = np.tile(sww, [1, M, L])    #
            Hc_w = np.tile(S['hc'], [N+1, M, L])
            if S['Vtransform'] == 1:
                zw0 = (S_w - Cs_w) * Hc_w + Cs_w*H_w
                z_w = zw0 + Zeta_w * (1 + zw0/H_w)
            elif S['Vtransform'] == 2:
                zw0 = (S_w*Hc_w  + Cs_w*H_w) / (Hc_w + H_w)
                z_w = Zeta_w + (Zeta_w + H_w)*zw0
        z_w = z_w.squeeze()
        return z_w
    # return results
    if only_rho:
        return make_z_rho(h, zeta, S, N, M, L)
    elif only_w:
        return make_z_w(h, zeta, S, N, M, L)
    else :
        return make_z_rho(h, zeta, S, N, M, L), make_z_w(h, zeta, S, N, M, L)

def get_zinds(h, S, z):
    """
    Precalculate the array of indices to go from HYCOM z to ROMS z.
    This just finds the vertical index in HYCOM z for each ROMS z_rho
    value in the whole 3D array, with the index being the UPPER one of
    the two HYCOM z indices that any ROMS z falls between.
    """
    
    tt0 = time()
    zr = get_z(h, 0*h, S, only_rho=True)
    zrf = zr.flatten()
    zinds = np.nan * np.ones_like(zrf)
    if isinstance(z, np.ma.MaskedArray):
        z = z.data
    for ii in range(len(z)-1):
        zlo = z[ii]; zhi = z[ii+1]
        mask = (zrf>zlo) & (zrf<=zhi)
        zinds[mask] = ii+1 # this is where the UPPER index is enforced
    #lower limit (added by BS)
    nandex = np.squeeze(np.argwhere(np.isnan(zinds)))
    zinds[nandex] = 0
    zinds = zinds.astype(int)
    if isinstance(zinds, np.ma.MaskedArray):
        zinds = zinds.data
    if verbose:
        print(' --create zinds array took %0.1f seconds' % (time() - tt0))
    return zinds

def get_basic_info(fn, only_G=False, only_S=False, only_T=False):
    """
    Gets grid, vertical coordinate, and time info from a ROMS NetCDF
    history file with full name 'fn'
    Input: the filename (with path if needed)
    Output: dicts G, S, and T
    Example calls:
    G, S, T = zfun.get_basic_info(fn)
    T = zfun.get_basic_info(fn, only_T=True)
    """
    ds = xr.open_dataset(fn)
    def make_G(ds):
        # get grid and bathymetry info
        g_varlist = ['h', 'lon_rho', 'lat_rho', 'lon_u', 'lat_u', 'lon_v', 'lat_v',
        'lon_psi', 'lat_psi', 'mask_rho', 'mask_u', 'mask_v', 'pm', 'pn',]
        G = dict()
        for vv in g_varlist:
            G[vv] = ds[vv].values
        G['DX'] = 1/G['pm']
        G['DY'] = 1/G['pn']
        G['M'], G['L'] = np.shape(G['lon_rho']) # M = rows, L = columns
        return G
    def make_S(ds):
        # get vertical sigma-coordinate info (vectors are bottom to top)
        s_varlist = ['s_rho', 's_w', 'hc', 'Cs_r', 'Cs_w', 'Vtransform']
        S = dict()
        for vv in s_varlist:
            S[vv] = ds[vv].values
        S['N'] = len(S['s_rho']) # number of vertical levels
        return S
    def make_T(ds):
        # returns two single values, one a datatime, and one a float
        ot = ds.ocean_time.values # an array with dtype='datetime64[ns]'
        dti = pd.to_datetime(ot) # a pandas DatetimeIndex with dtype='datetime64[ns]'
        dt = dti.to_pydatetime() # an array of datetimes
        T = dict()
        T['dt'] = dt[0] # a datetime object
        T['ocean_time'] = Lfun.datetime_to_modtime(dt[0]) # a float, "seconds since..."
        return T
    # return results
    if only_G:
        return make_G(ds)
    elif only_S:
        return make_S(ds)
    elif only_T:
        return make_T(ds)
    else:
        return make_G(ds), make_S(ds), make_T(ds)
    ds.close()

def get_interpolated_z(G, S, b, lon, lat, z, N, zr):
    """
    This does the horizontal and vertical interpolation to get from
    extrapolated, filtered HYCOM fields to ROMS fields.

    We use fast nearest neighbor interpolation as much as possible.
    Also we interpolate everything to the ROMS rho grid, and then crudely
    interpolate to the u and v grids at the last moment.  Much simpler.

    Modified by Bob S. to use linear interp
    """
    
    # start input dict
    c = {}
        
    #msk = gr.variables['mask_rho'][:,:]
    
    # precalculate useful arrays that are used for horizontal interpolation
    if isinstance(lon, np.ma.MaskedArray):
        lon = lon.data
    if isinstance(lat, np.ma.MaskedArray):
        lat = lat.data
    mf = 6 #fineness factor how much finer is the grid
    
    loni = np.linspace(lon[0],lon[-1],mf*len(lon))
    lati = np.linspace(lat[0],lat[-1],mf*len(lat))
    
    Lnhi, Lthi = np.meshgrid(loni,lati, indexing='xy') #now we have a fine grid
    #old code
    #Lon, Lat = np.meshgrid(lon,lat)
    #XYin = np.array((Lon.flatten(), Lat.flatten())).T
    #XYr = np.array((G['lon_rho'].flatten(), G['lat_rho'].flatten())).T
    
    XYin = np.array((Lnhi.flatten(), Lthi.flatten())).T
    XYr = np.array((G['lon_rho'].flatten(), G['lat_rho'].flatten())).T
    # nearest neighbor interpolation from XYin to XYr is done below...
    h = G['h']
    angle = G['angle']
    #IMr = cKDTree(XYin).query(XYr)[1]  #old nearest neighbor
    interp_func = RegularGridInterpolator((lat, lon), b['ssh'])
    
    # 2D fields
    for vn in ['ssh', 'ubar', 'vbar']:
        # interp hycom to finer grid
        setattr(interp_func,'values',b[vn])
        vn_fi=interp_func((G['lat_rho'],G['lon_rho']))
        #vn_fi=interp_func((Lthi,Lnhi))
        
        # interp to roms from the fine hycom ssh
        vv = vn_fi#.flatten()[IMr].reshape(h.shape)    
    
        vvc = vv.copy()
        # always a good idea to make sure dict entries are not just pointers
        # to arrays that might be changed later, hence the .copy()
        c[vn] = vvc
        checknan(vvc)
        
     #rotate velocites
    vu = c['ubar']
    vv = c['vbar']

    u = np.cos(angle) * vu + np.sin(angle) * vv
    v = np.cos(angle) * vv - np.sin(angle) * vu
    u = (u[:,:-1] + u[:,1:])/2
    v = (v[:-1,:] + v[1:,:])/2
    vvc = u.copy()
    c['ubar'] = vvc
    vvc = v.copy()
    c['vbar'] = vvc

     

        
        
    # 3D fields
    # create intermediate arrays which are on the ROMS lon_rho, lat_rho grid
    # but have the HYCOM vertical grid (N layers)
    F = np.nan * np.ones(((N,) + h.shape))
    vi_dict = {}
    for vn in ['theta', 's3d', 'u3d', 'v3d']:
        FF = F.copy()
        for nn in range(N):
            vin = b[vn][nn,:,:]
            # interp hycom to finer grid
            setattr(interp_func,'values',vin)
            vn_fi=interp_func((G['lat_rho'],G['lon_rho']))
            #interp_func = RegularGridInterpolator((lat, lon), vin)
            #vn_fi=interp_func((Lthi,Lnhi))
            
            # interp to roms from the fine hycom ssh
            FF[nn,:,:] = vn_fi#.flatten()[IMr].reshape(h.shape)    
            #FF[nn,:,:] = vin[IMr].reshape(h.shape)
        checknan(FF)
        vi_dict[vn] = FF

    interp_z = interp1d(z, vi_dict['theta'][:,0,0],  kind='linear', fill_value="extrapolate")   
    # do the vertical interpolation from HYCOM to ROMS z positions
    for vn in [ 's3d', 'theta','u3d', 'v3d']:
        vi = vi_dict[vn]
        for k in range(G['M']):
            for j in range(G['L']):
                setattr(interp_z,'values',vi[:,k,j])

                #hinds = np.indices((S['N'], G['M'], G['L']))
                #vvf = vi[zinds, hinds[1].flatten(), hinds[2].flatten()]
                #vv = vvf.reshape((S['N'], G['M'], G['L']))
                vv[:,k,j] = interp_z(zr[:,k,j])
        vvc = vv.copy()

        checknan(vvc)
        c[vn] = vvc

        #hinds = np.indices((S['N'], G['M'], G['L']))
        #vvf = vi[zinds, hinds[1].flatten(), hinds[2].flatten()]
        #vv = vvf.reshape((S['N'], G['M'], G['L']))


     #rotate velocites
    vu = c['u3d']
    vv = c['v3d']
    u = np.nan * np.ones(((N,) + h.shape))
    v = np.nan * np.ones(((N,) + h.shape))
    for nn in range(S['N']):
        u[nn,:,:] = np.cos(angle) * vu[nn,:,:] + np.sin(angle) * vv[nn,:,:]
        v[nn,:,:] = np.cos(angle) * vv[nn,:,:] - np.sin(angle) * vu[nn,:,:]
        
    u = (u[:,:,:-1] + u[:,:,1:])/2
    v = (v[:,:-1,:] + v[:,1:,:])/2
    vvc = u.copy()
    c['u3d'] = vvc
    vvc = v.copy()
    c['v3d'] = vvc

    return c


def datetime_to_modtime(dt):
    """
    This is where we define how time will be treated
    in all the model forcing files.

    INPUT: dt is a single datetime value
    OUTPUT: dt as seconds since modtime0 (float)
    """
    dt = datetime.combine(dt, datetime.min.time())
    modtime0 = datetime(1970,1,1,0,0)
    t = (dt - modtime0).total_seconds()
    return t


def get_varinfo(vn, vartype='state'):
    """
    This looks through the pre-parsed varinfo.yaml and returns a dict for a given variable.
    """
    data_path = Path('/home/rmsanche/models/LO_data') #this points to the folder LO_data which has my varinfo
    in_fn = data_path / 'varinfo' / 'varinfo_list.p'
    if in_fn.is_file():
        short_list = pickle.load(open(in_fn, 'rb'))
    else:
        make_varinfo_list()
        short_list = pickle.load(open(in_fn, 'rb'))
        
    # get the dict for the requested variable
    short_list = [item for item in short_list if item['variable']==vn]
        
    # Associate grid_type with a tuple of spatial dimensions
    grid_type_dict = {
            'r2dvar': ('eta_rho', 'xi_rho'),
            'u2dvar': ('eta_u', 'xi_u'),
            'v2dvar': ('eta_v', 'xi_v'),
            'r3dvar': ('s_rho', 'eta_rho', 'xi_rho'),
            'u3dvar': ('s_rho', 'eta_u', 'xi_u'),
            'v3dvar': ('s_rho', 'eta_v', 'xi_v'),
            'nulvar': ('BLANK',)
            }
    
    if vartype=='state':
        short_list = [item for item in short_list if 'climatology' not in item['long_name']]
    elif vartype=='climatology':
        if vn in ['zeta', 'u', 'v', 'ubar', 'vbar']:
            # only these have explicity climatology versions
            short_list = [item for item in short_list if 'climatology' in item['long_name']]
        else:
            # the rest are the same as state variables
            short_list = [item for item in short_list if 'climatology' not in item['long_name']]
    else:
        print('Error in zrfun.get_varinfo(), unknown vartype: ' + vartype)
        sys.exit()
    
    if len(short_list)==1:
        vinfo = short_list[0]
        vinfo['space_dims_tup'] = grid_type_dict[vinfo['type']]
        vinfo['time_name'] = vinfo['time'] # for compatibility with forcing code
        
        # rename the time variable to work with climatology
        # NOTE: this does not work for the bry files - for these you just use vinfo['time'].
        if vartype=='climatology':
            if vn in ['zeta', 'u', 'v', 'ubar', 'vbar']:
                pass
            else:
                vinfo['time_name'] = vn + '_time'
        
        if vn in ['ubar', 'vbar']:
            vinfo['long_name'] = vinfo['long_name'].replace('integrated', 'averaged')
        return vinfo
    else:
        print('Error in zrfun.get_varinfo: vn not unique for ' + vn)
        vinfo = short_list
        return vinfo
    
"""
This is a dict to use for compression when saving an xarray Dataset, e.g. with lines like:
    Enc_dict = {vn:zrfun.enc_dict for vn in ds.data_vars}
    ds.to_netcdf(out_fn, encoding=Enc_dict)
Using compression (zlib=True, complevel=1) results in files that are just 2% of the
uncompressed files (for hc0, which has a lot of nan's).
Using complevel=9 makes the files half as big as complevel=1, but takes about 10x longer.
"""
enc_dict = {'zlib':True, 'complevel':1, '_FillValue':1e20}

def make_ini_file(in_fn, out_fn):
    """
    Create the ini file from the first time of the clm file.
    """
    roms_time_units = 'seconds since 1970-01-01 00:00:00'
    ds0 = xr.open_dataset(in_fn, decode_times=False)
    ot_vec = ds0.zeta_time.values
    ds = xr.Dataset()
    ds['ocean_time'] = (('ocean_time',), [ot_vec[0]])
    ds['ocean_time'].attrs['units'] = roms_time_units
    for vn in ds0.data_vars:
        vinfo = get_varinfo(vn)
        ndims = len(ds0[vn].dims)
        if ndims == 3:
            ds[vn] = (('ocean_time',) + ds0[vn].dims[1:], ds0[vn].values[[0],:,:])
            # Note: we use [0] instead of 0 to retain the singleton dimension
        elif ndims == 4:
            ds[vn] = (('ocean_time',) + ds0[vn].dims[1:], ds0[vn].values[[0],:,:,:])
        ds[vn].attrs['units'] = vinfo['units']
        ds[vn].attrs['long_name'] = vinfo['long_name']
    ds0.close()
    enc_dict = {'zlib':True, 'complevel':1, '_FillValue':1e20}
    Enc_dict = {vn:enc_dict for vn in ds.data_vars}
    # and save to NetCDF
    ds.to_netcdf(out_fn, encoding=Enc_dict)
    ds.close()

def make_bry_file(in_fn, out_fn):
    """
    Create the bry file from the edges of the clm file.
    """
    roms_time_units = 'seconds since 1970-01-01 00:00:00'
    ds0 = xr.open_dataset(in_fn, decode_times=False)
    ot_vec = ds0.salt_time.values
    ds = xr.Dataset()
    for vn in ds0.data_vars:
        dm = ds0[vn].dims
        ndims = len(dm)
        for D in ['north', 'south', 'east', 'west']:
            # rename variable
            # We have to deal with some inconsistency in the naming of bio boundary variables.
            bvn_dict = {'phytoplankton': 'phyt', 'zooplankton': 'zoop', 'chlorophyll': 'chlo',
                    'LdetritusN': 'LDeN', 'SdetritusN': 'SDeN',
                    'LdetritusC': 'LdeC', 'SdetritusC': 'SdeC'}
            if vn in bvn_dict.keys():
                vn_new = bvn_dict[vn]
            else:
                vn_new = vn
            Vn = vn_new + '_' + D
            try:
                vinfo = get_varinfo(Vn, vartype='climatology')
            except Exception as e:
                print(Vn)
                print(e)
            tname = vinfo['time']
            
            # create time coordinate
            ds[tname] = ((tname,), ot_vec)
            ds[tname].attrs['units'] = roms_time_units
            
            # trim dimensions
            if D in ['east','west']:
                Dm = tuple(item for item in dm if 'xi_' not in item)
            elif D in ['north','south']:
                Dm = tuple(item for item in dm if (('eta_' not in item) or ('zeta' in item)))
            
            # replace time dimension
            Dm = tuple(tname if '_time' in item else item for item in Dm) 

            # write boundary arrays
            if ndims == 3:
                if D == 'north':
                    ds[Vn] = (Dm, ds0[vn].values[:,-1,:])
                elif D == 'south':
                    ds[Vn] = (Dm, ds0[vn].values[:,0,:])
                elif D == 'east':
                    ds[Vn] = (Dm, ds0[vn].values[:,:,-1])
                elif D == 'west':
                    ds[Vn] = (Dm, ds0[vn].values[:,:,0])
            elif ndims == 4:
                if D == 'north':
                    ds[Vn] = (Dm, ds0[vn].values[:,:,-1,:])
                elif D == 'south':
                    ds[Vn] = (Dm, ds0[vn].values[:,:,0,:])
                elif D == 'east':
                    ds[Vn] = (Dm, ds0[vn].values[:,:,:,-1])
                elif D == 'west':
                    ds[Vn] = (Dm, ds0[vn].values[:,:,:,0])
                    
            # add attributes
            ds[Vn].attrs['units'] = vinfo['units']
            ds[Vn].attrs['long_name'] = vinfo['long_name']
    
    ds0.close()
    enc_dict = {'zlib':True, 'complevel':1, '_FillValue':1e20}
    Enc_dict = {vn:enc_dict for vn in ds.data_vars}
    # and save to NetCDF
    ds.to_netcdf(out_fn, encoding=Enc_dict)
    ds.close()


#### Actual code below


ds_fmt_hr = '%Y.%m.%d.%H'

h_out_dir = Path(hout_dir)
out_dir = Path(out_dir)

hnc_short_list = get_hnc_short_list(h_out_dir)

#generate a pickle file for each hycom nc
for fn in hnc_short_list:
    a = convert_extraction(fn)
    dts = datetime.strftime(a['dt'], ds_fmt_hr)
    out_fn = h_out_dir / ('h' + dts + '.p')
    pickle.dump(a, open(out_fn, 'wb'))

#make another list of just those pickle files
hp_list = sorted([item.name for item in h_out_dir.iterdir()
        if (item.name[0]=='h' and item.name[-2:]=='.p')])


# copy in the coordinates (assume those from first file work)
this_h_dict = pickle.load(open(h_out_dir / hp_list[0], 'rb'))
coord_dict = dict()
for vn in ['lon', 'lat', 'z']:
    coord_dict[vn] = this_h_dict[vn]
pickle.dump(coord_dict, open(h_out_dir / 'coord_dict.p', 'wb'))

#now we filter in time a Hanning window to remove inertial oscillations and then backfill land

time_filter(h_out_dir, hp_list, h_out_dir,datestring_start)


# now prep list for extrapolation
lon, lat, z, L, M, N, X, Y = get_coords(h_out_dir)
fh_list = sorted([item.name for item in h_out_dir.iterdir()
        if item.name[:2]=='fh'])


#now extrapolate to fill land with nearest neighbor, also makes ubar and vbar
add_CTD=False #dont gen ctd profiles
for fn in fh_list:
    print('-Extrapolating ' + fn)
    in_fn = h_out_dir / fn
    V = get_extrapolated(in_fn, L, M, N, X, Y, lon, lat, z,
        add_CTD=add_CTD)
    pickle.dump(V, open(h_out_dir / ('x' + fn), 'wb'))

#now code to interpolate to roms
# get grid and S info
verbose = True
G = get_basic_info(grid_file, only_G=True)
#I put this based on my reading of the roms code

S = get_S(S_info_dict)
# make list of files to process
xfh_list = sorted([item.name for item in h_out_dir.iterdir()
        if item.name[:3]=='xfh'])

# load a dict of hycom fields
dt_list = []
count = 0
c_dict = dict()
#zinds = get_zinds(G['h'], S, z) #enforce -5000 m
zr = get_z(G['h'], 0*G['h'], S, only_rho=True) #gets zr

#now interpolate to ROMS grid
for fn in xfh_list:
    print('-Interpolating ' + fn + ' to ROMS grid')
    b = pickle.load(open(h_out_dir / fn, 'rb'))
    dt_list.append(b['dt'])
    c = get_interpolated_z(G, S, b, lon, lat, z, N, zr) #modified to use linear interp instead
    c_dict[count] = c
    count += 1

# Write to ROMS forcing files
# The fields we want to write are in c_dict, whose keys are indices of time.
# Then c_dict[i] is another dict of arrays at that time, with keys:
# ['ssh', 'ubar', 'vbar', 'theta', 's3d', 'u3d', 'v3d']
# and the associated time vector is dt_list (datetimes).

# Here I will write these to the dict V as expected below
hycom_names = ['ssh', 'ubar', 'vbar', 'theta', 's3d', 'u3d', 'v3d']
roms_names = ['zeta', 'ubar', 'vbar', 'temp', 'salt', 'u', 'v']
names_dict = dict(zip(hycom_names, roms_names))

# get sizes
NZ = S['N']; NR = G['M']; NC = G['L']

# Make the time vector.
ot_vec = np.array([datetime_to_modtime(item) for item in dt_list])
NT = len(ot_vec)

# Create a dict of fields for the state variables.
V = dict()
V['zeta'] = np.zeros((NT, NR, NC))
V['ubar'] = np.zeros((NT, NR, NC-1))
V['vbar'] = np.zeros((NT, NR-1, NC))
V['salt'] = np.zeros((NT, NZ, NR, NC))
V['temp'] = np.zeros((NT, NZ, NR, NC))
V['u'] = np.zeros((NT, NZ, NR, NC-1))
V['v'] = np.zeros((NT, NZ, NR-1, NC))

# Fill the V dict
for ii in range(NT):
    C = c_dict[ii]
    for vnh in hycom_names:
        vnr = names_dict[vnh]
        # note that the : here represents all axes after 0
        # and that it retains the correct shape
        V[vnr][ii, :] = C[vnh]
        
# Create masks
mr2 = np.ones((NT, NR, NC)) * G['mask_rho'].reshape((1, NR, NC))
mr3 = np.ones((NT, NZ, NR, NC)) * G['mask_rho'].reshape((1, 1, NR, NC))
mu2 = np.ones((NT, NR, NC-1)) * G['mask_u'].reshape((1, NR, NC-1))
mu3 = np.ones((NT, NZ, NR, NC-1)) * G['mask_u'].reshape((1, 1, NR, NC-1))
mv2 = np.ones((NT, NR-1, NC)) * G['mask_v'].reshape((1, NR-1, NC))
mv3 = np.ones((NT, NZ, NR-1, NC)) * G['mask_v'].reshape((1, 1, NR-1, NC))

# Apply masks
V['zeta'][mr2==0] = np.nan
V['ubar'][mu2==0] = np.nan
V['vbar'][mv2==0] = np.nan
V['salt'][mr3==0] = np.nan
V['temp'][mr3==0] = np.nan
V['u'][mu3==0] = np.nan
V['v'][mv3==0] = np.nan


roms_time_units = 'seconds since 1970-01-01 00:00:00'
out_fn = out_dir / 'ocean_clm.nc'
ds = xr.Dataset()    
for vn in V.keys():
    # tt00 = time()
    vinfo =get_varinfo(vn, vartype='climatology')
    # print(' -- time to get varinfo: %0.2f sec' % (time()-tt00))
    tname = vinfo['time_name']
    dims = (vinfo['time_name'],) + vinfo['space_dims_tup']
    ds[vn] = (dims, V[vn])
    ds[vn].attrs['units'] = vinfo['units']
    ds[vn].attrs['long_name'] = vinfo['long_name']
    # time coordinate
    ds[tname] = ((tname,), ot_vec)
    ds[tname].attrs['units'] = roms_time_units
# and save to NetCDF
enc_dict = {'zlib':True, 'complevel':1, '_FillValue':1e20}
Enc_dict = {vn:enc_dict for vn in ds.data_vars}
ds.to_netcdf(out_fn, encoding=Enc_dict)
ds.close()
sys.stdout.flush()

# Write initial condition file if needed
tt0 = time()
in_fn = out_dir / 'ocean_clm.nc'
out_fn = out_dir / 'ocean_ini.nc'
out_fn.unlink(missing_ok=True)
make_ini_file(in_fn, out_fn)
print('- Write ini file: %0.2f sec' % (time()-tt0))
sys.stdout.flush()

# Write boundary file
tt0 = time()
in_fn = out_dir / 'ocean_clm.nc'
out_fn = out_dir / 'ocean_bry.nc'
out_fn.unlink(missing_ok=True)
make_bry_file(in_fn, out_fn)
print('- Write bry file: %0.2f sec' % (time()-tt0))
sys.stdout.flush()





