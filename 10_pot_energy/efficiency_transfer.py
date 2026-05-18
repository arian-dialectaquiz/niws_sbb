
import numpy as np
from scipy.signal import welch, csd
import matplotlib.pyplot as plt
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask


import pandas as pd
import gc
from scipy.signal import butter, filtfilt,windows
from cartopy.feature import NaturalEarthFeature, COLORS
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from scipy import signal
from matplotlib.ticker import ScalarFormatter, MaxNLocator, LogLocator, NullFormatter, FixedLocator


path_wp = '/data1/roms_dd_waves/analysis_outs/NIW/pressure_b/'
path = '/data1/roms_dd_waves/analysis_outs/NIW/pot_energy/'
dzs = '/data1/roms_dd_waves/analysis_outs/dz/'

z = xr.open_mfdataset(dzs + f'dz_*.nc').isel(s_rho = 0).z_rho #bottom
lon_rho = z.lon_rho.compute()
lat_rho = z.lat_rho.compute()



######################################################################
####################-----> shear of potential energy


EP1 = xr.open_mfdataset(path_wp + 'w_p*_1.nc').w_p#*1025
EP2 = xr.open_mfdataset(path_wp + 'w_p*_2.nc').w_p#*1025
EP3 = xr.open_mfdataset(path_wp + 'w_p*_3.nc').w_p#*1025
EP4 = xr.open_mfdataset(path_wp + 'w_p*_4.nc').w_p#*1025


EP = xr.concat([EP1, EP2, EP3, EP4], dim='ocean_time')
EP = EP.sortby('ocean_time')
EP = EP.sel(ocean_time=EP.ocean_time.drop_duplicates(dim='ocean_time')).isel(ocean_time=slice(40,None))

#############----> cold fronts
date_ranges = [
	("2001-12-10", "2001-12-16"),
	("2001-12-23", "2001-12-27"),
	("2001-12-30", "2002-01-02"),
	("2002-01-06", "2002-01-10"),
	("2002-01-15", "2002-01-20"),
	("2002-01-24", "2002-01-25"),
	("2002-01-28", "2002-01-30"),
	("2002-02-14", "2002-02-22"),
	("2002-03-19", "2002-03-22"),
]

dates = EP.ocean_time.to_index()


# Get indices and datetime values that fall in each range
selected_idx = []
selected_dates = []

for start, end in date_ranges:
	mask = (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))
	idx = np.where(mask)[0]        # positions
	vals = dates[mask]             # datetime values	
	selected_idx.append(idx)
	selected_dates.append(vals)


cold_idx = np.concatenate(selected_idx)
EP_cold_fronts = EP.isel(ocean_time=cold_idx)
dates_cf = EP.ocean_time.isel(ocean_time=cold_idx)

###############----> normal wind
all_times_idx = np.arange(len(EP.ocean_time))
normal_idx = np.setdiff1d(all_times_idx, cold_idx)

dates_normal = EP.ocean_time.isel(ocean_time=normal_idx)
EP_normal = EP.isel(ocean_time=normal_idx)

EP_cf_mean = EP_cold_fronts.mean(dim='ocean_time').compute()
EP_normal_mean = EP_normal.mean(dim='ocean_time').compute()

######################################################################
####################-----> MLD

ml1 = xr.open_mfdataset(path + 'mld_slice*_1.nc').mld
ml2 = xr.open_mfdataset(path + 'mld_slice*_2.nc').mld
ml3 = xr.open_mfdataset(path + 'mld_slice*_3.nc').mld
ml4 = xr.open_mfdataset(path + 'mld_slice*_4.nc').mld

ml = xr.concat([ml1, ml2, ml3,ml4], dim='ocean_time')
ml = ml.sortby('ocean_time')
ml = ml.sel(ocean_time=ml.ocean_time.drop_duplicates(dim='ocean_time')).compute().isel(ocean_time=slice(40,None))

assert ml.sizes['ocean_time'] == EP.sizes['ocean_time']  # sanity check
ml_on_EP = ml.assign_coords(ocean_time=EP['ocean_time'].values)
ml_on_EP['ocean_time'].attrs = EP['ocean_time'].attrs



ml_cold_fronts = ml_on_EP.isel(ocean_time=cold_idx)
ml_normal = ml_on_EP.isel(ocean_time=normal_idx)


ml_cf_mean = ml_cold_fronts.mean(dim='ocean_time')#.compute()
ml_normal_mean = ml_normal.mean(dim='ocean_time')#.compute()


dz = xr.open_mfdataset(dzs + f'dz_*.nc')
depth = (-dz['z_rho']).rename('depth').compute()


ml_cf_mean_3d = ml_cf_mean.broadcast_like(depth)       # (s_rho, eta_rho, xi_rho)
ml_normal_mean_3d = ml_normal_mean.broadcast_like(depth)       # (s_rho, eta_rho, xi_rho)


diff_mean = xr.apply_ufunc(np.abs, depth - ml_normal_mean_3d)
diff_mean = diff_mean.where(np.isfinite(diff_mean), np.inf)
k_near_mean = diff_mean.argmin('s_rho')  


diff_cf = xr.apply_ufunc(np.abs, depth - ml_cf_mean_3d)
diff_cf = diff_cf.where(np.isfinite(diff_cf), np.inf)
k_near_cf = diff_cf.argmin('s_rho')  


EP_at_MLD_cf = EP_cf_mean.isel(s_rho=k_near_cf)
EP_at_MLD_normal = EP_normal_mean.isel(s_rho=k_near_mean)



##################-----> Wind input map <------###################################

winds = '/data1/roms_dd_waves/analysis_outs/NIW/wind_input/' #already filtered data
speeds = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'
T_fixed = 12.42  # [hours] fixed boundary between "super" and "mid-high"
fs = 1/6


## tu_prime
tu1 = xr.open_mfdataset(winds + 'tu_prime*_1.nc').tu_prime
tu2 = xr.open_mfdataset(winds + 'tu_prime*_2.nc').tu_prime
tu3 = xr.open_mfdataset(winds + 'tu_prime*_3.nc').tu_prime
tu4 = xr.open_mfdataset(winds + 'tu_prime*_4.nc').tu_prime


tu = xr.concat([tu1, tu2, tu3, tu4], dim='ocean_time')
tu = tu.sortby('ocean_time')
tu = tu.sel(ocean_time=tu.ocean_time.drop_duplicates(dim='ocean_time'))

## tv_prime
tv1 = xr.open_mfdataset(winds + 'tv_prime*_1.nc').tv_prime
tv2 = xr.open_mfdataset(winds + 'tv_prime*_2.nc').tv_prime
tv3 = xr.open_mfdataset(winds + 'tv_prime*_3.nc').tv_prime
tv4 = xr.open_mfdataset(winds + 'tv_prime*_4.nc').tv_prime


tv = xr.concat([tv1, tv2, tv3, tv4], dim='ocean_time')
tv = tv.sortby('ocean_time')
tv = tv.sel(ocean_time=tv.ocean_time.drop_duplicates(dim='ocean_time'))

## u prime
u1 = xr.open_mfdataset(speeds + 'u_prime_*_1.nc').u_prime.isel(s_rho=-1)
u2 = xr.open_mfdataset(speeds + 'u_prime_*_2.nc').u_prime.isel(s_rho=-1)
u3 = xr.open_mfdataset(speeds + 'u_prime_*_3.nc').u_prime.isel(s_rho=-1)
u4 = xr.open_mfdataset(speeds + 'u_prime_*_4.nc').u_prime.isel(s_rho=-1)


u = xr.concat([u1, u2, u3, u4], dim='ocean_time')
u = u.sortby('ocean_time')
u = u.sel(ocean_time=u.ocean_time.drop_duplicates(dim='ocean_time'))


## v prime
v1 = xr.open_mfdataset(speeds + 'v_prime_*_1.nc').v_prime.isel(s_rho=-1)
v2 = xr.open_mfdataset(speeds + 'v_prime_*_2.nc').v_prime.isel(s_rho=-1)
v3 = xr.open_mfdataset(speeds + 'v_prime_*_3.nc').v_prime.isel(s_rho=-1)
v4 = xr.open_mfdataset(speeds + 'v_prime_*_4.nc').v_prime.isel(s_rho=-1)


v = xr.concat([v1, v2, v3, v4], dim='ocean_time')
v = v.sortby('ocean_time')
v = v.sel(ocean_time=v.ocean_time.drop_duplicates(dim='ocean_time'))


#### --> Wind input values <---####
WI = (tu*u) + (tv*v)


WI_cold_fronts = WI.isel(ocean_time=cold_idx)
WI_normal = WI.isel(ocean_time=normal_idx)


WI_cf_mean = WI_cold_fronts.mean(dim='ocean_time').compute()
WI_normal_mean = WI_normal.mean(dim='ocean_time').compute()

WI_normal_meanfile = WI_normal_mean.to_dataset(name='WI_normal_mean')		
WI_normal_meanfile.to_netcdf('WI_normal_mean.nc')

WI_cf_meanfile = WI_cf_mean.to_dataset(name='WI_cf_mean')		
WI_cf_meanfile.to_netcdf('WI_cf_mean.nc')


#######################----> wind input efficiency

leak_normal = EP_at_MLD_normal/WI_normal_mean
leak_cf = EP_at_MLD_cf/WI_cf_mean

leak_normalfile = leak_normal.to_dataset(name='leak_norm')		
leak_normalfile.to_netcdf('leak_norm_time.nc')

leak_cffile = leak_cf.to_dataset(name='leak_cf')		
leak_cffile.to_netcdf('leak_cf_time.nc')