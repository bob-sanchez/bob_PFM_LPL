#python script to merge netcdfs to create less boundary files
# %%
import xarray as xr



# %%
low = [0,12,24,36,48,60] 
high= [12,24,36,48,60,73]
for k in range(1,6):
    paths = []
    for i in range(low[k],high[k]):
        path_i = '/scratch/bob/Hycom/ocean_bry_LV1_'+str(i)+'.nc'
        paths.append(path_i)

    ds = xr.open_mfdataset(paths,combine = 'by_coords',data_vars='minimal')




    enc_dict = {'zlib':True, 'complevel':1, '_FillValue':1e20}
    Enc_dict = {vn:enc_dict for vn in ds.data_vars}
    out_fn = '/scratch/bob/Hycom/ocean_bry_LV1_L'+str(k+1)+'.nc'
    print(out_fn)
    ds.to_netcdf(out_fn,encoding=Enc_dict)
    ds.close()

