
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

EP1 = xr.open_mfdataset(path_wp + 'w_p*_1.nc').w_p
EP2 = xr.open_mfdataset(path_wp + 'w_p*_2.nc').w_p
EP3 = xr.open_mfdataset(path_wp + 'w_p*_3.nc').w_p
EP4 = xr.open_mfdataset(path_wp + 'w_p*_4.nc').w_p


EP = xr.concat([EP1, EP2, EP3, EP4], dim='ocean_time')
EP = EP.sortby('ocean_time')
EP = EP.sel(ocean_time=EP.ocean_time.drop_duplicates(dim='ocean_time'))
EP = EP[40:]


#EP_time = (EP*20).mean(dim='ocean_time').compute()
#EP_tfile = EP_time.to_dataset(name='pot_ex_time')		
#EP_tfile.to_netcdf('pot_ex_time.nc')


#EP_time_file = xr.open_dataset('pot_ex_time.nc').pot_ex_time
#######################################################################################################################
################### ----- MAPS of EP through MLD -----#################################################################

ml1 = xr.open_mfdataset(path + 'mld_slice*_1.nc').mld
ml2 = xr.open_mfdataset(path + 'mld_slice*_2.nc').mld
ml3 = xr.open_mfdataset(path + 'mld_slice*_3.nc').mld
ml4 = xr.open_mfdataset(path + 'mld_slice*_4.nc').mld

ml = xr.concat([ml1, ml2, ml3,ml4], dim='ocean_time')
ml = ml.sortby('ocean_time')
ml = ml.sel(ocean_time=ml.ocean_time.drop_duplicates(dim='ocean_time')).compute()
ml = ml[40:]

assert ml.sizes['ocean_time'] == EP.sizes['ocean_time']  # sanity check
ml_on_EP = ml.assign_coords(ocean_time=EP['ocean_time'].values)
ml_on_EP['ocean_time'].attrs = EP['ocean_time'].attrs


ml_time = ml_on_EP.mean(dim='ocean_time')
dz = xr.open_mfdataset(dzs + f'dz_*.nc')

depth = (-dz['z_rho']).rename('depth').compute()
mld3 = ml_time.broadcast_like(depth)       # (s_rho, eta_rho, xi_rho)

diff = xr.apply_ufunc(np.abs, depth - mld3)

diff = diff.where(np.isfinite(diff), np.inf)
k_near = diff.argmin('s_rho')  

EP_at_MLD = EP_time.isel(s_rho=k_near)*20
valid_col = np.isfinite(depth).any('s_rho')


EP_at_MLD = EP_at_MLD.where(np.isfinite(ml_time) & valid_col)




#leak_time =  EP_at_MLD*ml_time

#######################################################################################################################
################### ----- Sections and point -----#################################################################
ml = ml_on_EP

################### ----- A -----####################################################################################

## - point per time
EP_A_p = EP.isel(eta_rho=905,xi_rho=380)*20
ml_A_p = ml.isel(eta_rho=905,xi_rho=380) #to be contoured

time = ml_A_p.ocean_time

ii, jj = 905, 380
z_p      = dz['z_rho'].isel(eta_rho=ii, xi_rho=jj)        # (ocean_time, s_rho) or (s_rho) depending on your z
depth_A_p = depth.isel(eta_rho=ii,xi_rho=jj)

ml_A_p_like = ml_A_p.broadcast_like(depth_A_p)

diff = xr.apply_ufunc(np.abs, depth_A_p - ml_A_p_like)

# Robust to land/NaNs: make NaNs "inf" so argmin works
diff = diff.where(np.isfinite(diff), np.inf)

# For each time, nearest vertical index
k_near_t = diff.argmin('s_rho')                           # (ocean_time), integer indices

# Select EP at that level, time by time (vectorized advanced indexing)
EP_ml_A = EP_A_p.isel(s_rho=k_near_t).compute()                 # (ocean_time)

# Optional masking: invalid where column has no finite depths or MLD NaN
valid_col_t = np.isfinite(depth_A_p).any('s_rho')
EP_ml_A_t = EP_ml_A.where(np.isfinite(ml_A_p) & valid_col_t)

zcross_A = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=905,xi_rho=slice(0,-1)).z_rho.values

### - cross section
#EP_A = EP.isel(eta_rho=905,xi_rho=slice(0,None),s_rho=slice(0,-1)).compute().mean(dim='ocean_time')
#zcross_A = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=905,xi_rho=slice(0,-1)).z_rho.values



################### ----- B -----####################################################################################
ii, jj = 500, 238


## - point per time
EP_B_p = EP.isel(eta_rho=ii,xi_rho=jj).compute()*20
ml_B_p = ml.isel(eta_rho=ii,xi_rho=jj) #to be contoured

z_p      = dz['z_rho'].isel(eta_rho=ii, xi_rho=jj)        # (ocean_time, s_rho) or (s_rho) depending on your z
depth_B_p = depth.isel(eta_rho=ii,xi_rho=jj)

ml_B_p_like = ml_B_p.broadcast_like(depth_B_p)

diff = xr.apply_ufunc(np.abs, depth_B_p - ml_B_p_like)

# Robust to land/NaNs: make NaNs "inf" so argmin works
diff = diff.where(np.isfinite(diff), np.inf)

# For each time, nearest vertical index
k_near_t = diff.argmin('s_rho')                           # (ocean_time), integer indices

# Select EP at that level, time by time (vectorized advanced indexing)
EP_ml_B = EP_B_p.isel(s_rho=k_near_t).compute()                 # (ocean_time)

# Optional masking: invalid where column has no finite depths or MLD NaN
valid_col_t = np.isfinite(depth_B_p).any('s_rho')
EP_ml_B_t = EP_ml_B.where(np.isfinite(ml_B_p) & valid_col_t)

zcross_B = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=500,xi_rho=slice(0,-1)).z_rho.values


## - cross section
#EP_B = EP.isel(eta_rho=500,xi_rho=slice(0,None),s_rho=slice(0,-1)).compute().mean(dim='ocean_time')
#zcross_B = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=500,xi_rho=slice(0,-1)).z_rho.values


################### ----- C -----####################################################################################

ii, jj = 280, 221
## - point per time
EP_C_p = EP.isel(eta_rho=ii,xi_rho=jj).compute()*20
ml_C_p = ml.isel(eta_rho=ii,xi_rho=jj) #to be contoured

ml_C_p = ml.isel(eta_rho=ii,xi_rho=jj) #to be contoured

                

z_p      = dz['z_rho'].isel(eta_rho=ii, xi_rho=jj)        # (ocean_time, s_rho) or (s_rho) depending on your z
depth_C_p = depth.isel(eta_rho=ii,xi_rho=jj)

ml_C_p_like = ml_C_p.broadcast_like(depth_C_p)

diff = xr.apply_ufunc(np.abs, depth_C_p - ml_C_p_like)

# Robust to land/NaNs: make NaNs "inf" so argmin works
diff = diff.where(np.isfinite(diff), np.inf)

# For each time, nearest vertical index
k_near_t = diff.argmin('s_rho')                           # (ocean_time), integer indices

# Select EP at that level, time by time (vectorized advanced indexing)
EP_ml_C = EP_C_p.isel(s_rho=k_near_t).compute()                 # (ocean_time)

# Optional masking: invalid where column has no finite depths or MLD NaN
valid_col_t = np.isfinite(depth_C_p).any('s_rho')
EP_ml_C_t = EP_ml_C.where(np.isfinite(ml_C_p) & valid_col_t)

zcross_C = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=280,xi_rho=slice(0,-1)).z_rho.values

## - cross section
#EP_C = EP.isel(eta_rho=280,xi_rho=slice(0,None),s_rho=slice(0,-1)).compute().mean(dim='ocean_time')
#zcross_C = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=280,xi_rho=slice(0,-1)).z_rho.values



################### ----- D -----####################################################################################


## - point per time
ii, jj = 120, 275
## - point per time
EP_D_p = EP.isel(eta_rho=ii,xi_rho=jj).compute()*20
ml_D_p = ml.isel(eta_rho=ii,xi_rho=jj) #to be contoured

ml_D_p = ml.isel(eta_rho=ii,xi_rho=jj) #to be contoured


z_p      = dz['z_rho'].isel(eta_rho=ii, xi_rho=jj)        # (ocean_time, s_rho) or (s_rho) depending on your z
depth_D_p = depth.isel(eta_rho=ii,xi_rho=jj)

ml_D_p_like = ml_D_p.broadcast_like(depth_D_p)

diff = xr.apply_ufunc(np.abs, depth_D_p - ml_D_p_like)

# Robust to land/NaNs: make NaNs "inf" so argmin works
diff = diff.where(np.isfinite(diff), np.inf)

# For each time, nearest vertical index
k_near_t = diff.argmin('s_rho')                           # (ocean_time), integer indices

# Select EP at that level, time by time (vectorized advanced indexing)
EP_ml_D = EP_D_p.isel(s_rho=k_near_t).compute()                 # (ocean_time)

# Optional masking: invalid where column has no finite depths or MLD NaN
valid_Dol_t = np.isfinite(depth_D_p).any('s_rho')
EP_ml_D_t = EP_ml_D.where(np.isfinite(ml_D_p) & valid_Dol_t)

zcross_D = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=120,xi_rho=slice(0,-1)).z_rho.values

## - cross section
#EP_D = EP.isel(eta_rho=120,xi_rho=slice(0,None),s_rho=slice(0,-1)).compute().mean(dim='ocean_time')
#zcross_D = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=120,xi_rho=slice(0,-1)).z_rho.values



#######################################################################################################################
################### ----- Wind Input at each point -----###############################################################

winds = '/data1/roms_dd_waves/analysis_outs/NIW/wind_input/' #already filtered data
speeds = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'

#
#################### ----- A -----####################################################################################
#
#tu1 = xr.open_mfdataset(winds + 'tu_slice*_1.nc').isel(eta_rho=905,xi_rho=380).tu
#tu2 = xr.open_mfdataset(winds + 'tu_slice*_2.nc').isel(eta_rho=905,xi_rho=380).tu
#tu3 = xr.open_mfdataset(winds + 'tu_slice*_3.nc').isel(eta_rho=905,xi_rho=380).tu
#tu4 = xr.open_mfdataset(winds + 'tu_slice*_4.nc').isel(eta_rho=905,xi_rho=380).tu
#
#
#tu_A = xr.concat([tu1, tu2, tu3, tu4], dim='ocean_time')
#tu_A = tu_A.sortby('ocean_time')
#tu_A = tu_A.sel(ocean_time=tu_A.ocean_time.drop_duplicates(dim='ocean_time'))
#
### tv_prime
#tv1 = xr.open_mfdataset(winds + 'tv_slice*_1.nc').isel(eta_rho=905,xi_rho=380).tv
#tv2 = xr.open_mfdataset(winds + 'tv_slice*_2.nc').isel(eta_rho=905,xi_rho=380).tv
#tv3 = xr.open_mfdataset(winds + 'tv_slice*_3.nc').isel(eta_rho=905,xi_rho=380).tv
#tv4 = xr.open_mfdataset(winds + 'tv_slice*_4.nc').isel(eta_rho=905,xi_rho=380).tv
#
#
#tv_A = xr.concat([tv1, tv2, tv3, tv4], dim='ocean_time')
#tv_A = tv_A.sortby('ocean_time')
#tv_A = tv_A.sel(ocean_time=tv_A.ocean_time.drop_duplicates(dim='ocean_time'))
#
### u prime
#u1 = xr.open_mfdataset(speeds + 'u_slice_*_1.nc').isel(eta_rho=905,xi_rho=380).u.isel(s_rho=-1)
#u2 = xr.open_mfdataset(speeds + 'u_slice_*_2.nc').isel(eta_rho=905,xi_rho=380).u.isel(s_rho=-1)
#u3 = xr.open_mfdataset(speeds + 'u_slice_*_3.nc').isel(eta_rho=905,xi_rho=380).u.isel(s_rho=-1)
#u4 = xr.open_mfdataset(speeds + 'u_slice_*_4.nc').isel(eta_rho=905,xi_rho=380).u.isel(s_rho=-1)
#
#
#u_A = xr.concat([u1, u2, u3, u4], dim='ocean_time')
#u_A = u_A.sortby('ocean_time')
#u_A = u_A.sel(ocean_time=u_A.ocean_time.drop_duplicates(dim='ocean_time'))
#
#
### v prime
#v1 = xr.open_mfdataset(speeds + 'v_slice_*_1.nc').isel(eta_rho=905,xi_rho=380).v.isel(s_rho=-1)
#v2 = xr.open_mfdataset(speeds + 'v_slice_*_2.nc').isel(eta_rho=905,xi_rho=380).v.isel(s_rho=-1)
#v3 = xr.open_mfdataset(speeds + 'v_slice_*_3.nc').isel(eta_rho=905,xi_rho=380).v.isel(s_rho=-1)
#v4 = xr.open_mfdataset(speeds + 'v_slice_*_4.nc').isel(eta_rho=905,xi_rho=380).v.isel(s_rho=-1)
#
#
#v_A = xr.concat([v1, v2, v3, v4], dim='ocean_time')
#v_A = v_A.sortby('ocean_time')
#v_A = v_A.sel(ocean_time=v_A.ocean_time.drop_duplicates(dim='ocean_time'))
#
#################### ----- B -----####################################################################################
#
#tu1 = xr.open_mfdataset(winds + 'tu_slice*_1.nc').isel(eta_rho=500,xi_rho=238).tu
#tu2 = xr.open_mfdataset(winds + 'tu_slice*_2.nc').isel(eta_rho=500,xi_rho=238).tu
#tu3 = xr.open_mfdataset(winds + 'tu_slice*_3.nc').isel(eta_rho=500,xi_rho=238).tu
#tu4 = xr.open_mfdataset(winds + 'tu_slice*_4.nc').isel(eta_rho=500,xi_rho=238).tu
#
#
#tu_B = xr.concat([tu1, tu2, tu3, tu4], dim='ocean_time')
#tu_B = tu_B.sortby('ocean_time')
#tu_B = tu_B.sel(ocean_time=tu_B.ocean_time.drop_duplicates(dim='ocean_time'))
#
### tv_prime
#tv1 = xr.open_mfdataset(winds + 'tv_slice*_1.nc').isel(eta_rho=500,xi_rho=238).tv
#tv2 = xr.open_mfdataset(winds + 'tv_slice*_2.nc').isel(eta_rho=500,xi_rho=238).tv
#tv3 = xr.open_mfdataset(winds + 'tv_slice*_3.nc').isel(eta_rho=500,xi_rho=238).tv
#tv4 = xr.open_mfdataset(winds + 'tv_slice*_4.nc').isel(eta_rho=500,xi_rho=238).tv
#
#
#tv_B = xr.concat([tv1, tv2, tv3, tv4], dim='ocean_time')
#tv_B = tv_B.sortby('ocean_time')
#tv_B = tv_B.sel(ocean_time=tv_B.ocean_time.drop_duplicates(dim='ocean_time'))
#
### u prime
#u1 = xr.open_mfdataset(speeds + 'u_slice_*_1.nc').isel(eta_rho=500,xi_rho=238).u.isel(s_rho=-1)
#u2 = xr.open_mfdataset(speeds + 'u_slice_*_2.nc').isel(eta_rho=500,xi_rho=238).u.isel(s_rho=-1)
#u3 = xr.open_mfdataset(speeds + 'u_slice_*_3.nc').isel(eta_rho=500,xi_rho=238).u.isel(s_rho=-1)
#u4 = xr.open_mfdataset(speeds + 'u_slice_*_4.nc').isel(eta_rho=500,xi_rho=238).u.isel(s_rho=-1)
#
#
#u_B = xr.concat([u1, u2, u3, u4], dim='ocean_time')
#u_B = u_B.sortby('ocean_time')
#u_B = u_B.sel(ocean_time=u_B.ocean_time.drop_duplicates(dim='ocean_time'))
#
#
### v prime
#v1 = xr.open_mfdataset(speeds + 'v_slice_*_1.nc').isel(eta_rho=500,xi_rho=238).v.isel(s_rho=-1)
#v2 = xr.open_mfdataset(speeds + 'v_slice_*_2.nc').isel(eta_rho=500,xi_rho=238).v.isel(s_rho=-1)
#v3 = xr.open_mfdataset(speeds + 'v_slice_*_3.nc').isel(eta_rho=500,xi_rho=238).v.isel(s_rho=-1)
#v4 = xr.open_mfdataset(speeds + 'v_slice_*_4.nc').isel(eta_rho=500,xi_rho=238).v.isel(s_rho=-1)
#
#
#v_B = xr.concat([v1, v2, v3, v4], dim='ocean_time')
#v_B = v_B.sortby('ocean_time')
#v_B = v_B.sel(ocean_time=v_B.ocean_time.drop_duplicates(dim='ocean_time'))
#
#################### ----- C -----####################################################################################
#
#
### tu_prime
#tu1 = xr.open_mfdataset(winds + 'tu_slice*_1.nc').isel(eta_rho=280,xi_rho=221).tu
#tu2 = xr.open_mfdataset(winds + 'tu_slice*_2.nc').isel(eta_rho=280,xi_rho=221).tu
#tu3 = xr.open_mfdataset(winds + 'tu_slice*_3.nc').isel(eta_rho=280,xi_rho=221).tu
#tu4 = xr.open_mfdataset(winds + 'tu_slice*_4.nc').isel(eta_rho=280,xi_rho=221).tu
#
#
#tu_C = xr.concat([tu1, tu2, tu3, tu4], dim='ocean_time')
#tu_C = tu_C.sortby('ocean_time')
#tu_C = tu_C.sel(ocean_time=tu_C.ocean_time.drop_duplicates(dim='ocean_time'))
#
### tv_prime
#tv1 = xr.open_mfdataset(winds + 'tv_slice*_1.nc').isel(eta_rho=280,xi_rho=221).tv
#tv2 = xr.open_mfdataset(winds + 'tv_slice*_2.nc').isel(eta_rho=280,xi_rho=221).tv
#tv3 = xr.open_mfdataset(winds + 'tv_slice*_3.nc').isel(eta_rho=280,xi_rho=221).tv
#tv4 = xr.open_mfdataset(winds + 'tv_slice*_4.nc').isel(eta_rho=280,xi_rho=221).tv
#
#
#tv_C = xr.concat([tv1, tv2, tv3, tv4], dim='ocean_time')
#tv_C = tv_C.sortby('ocean_time')
#tv_C = tv_C.sel(ocean_time=tv_C.ocean_time.drop_duplicates(dim='ocean_time'))
#
### u prime
#u1 = xr.open_mfdataset(speeds + 'u_slice_*_1.nc').isel(eta_rho=280,xi_rho=221).u.isel(s_rho=-1)
#u2 = xr.open_mfdataset(speeds + 'u_slice_*_2.nc').isel(eta_rho=280,xi_rho=221).u.isel(s_rho=-1)
#u3 = xr.open_mfdataset(speeds + 'u_slice_*_3.nc').isel(eta_rho=280,xi_rho=221).u.isel(s_rho=-1)
#u4 = xr.open_mfdataset(speeds + 'u_slice_*_4.nc').isel(eta_rho=280,xi_rho=221).u.isel(s_rho=-1)
#
#
#u_C = xr.concat([u1, u2, u3, u4], dim='ocean_time')
#u_C = u_C.sortby('ocean_time')
#u_C = u_C.sel(ocean_time=u_C.ocean_time.drop_duplicates(dim='ocean_time'))
#
#
### v prime
#v1 = xr.open_mfdataset(speeds + 'v_slice_*_1.nc').isel(eta_rho=280,xi_rho=221).v.isel(s_rho=-1)
#v2 = xr.open_mfdataset(speeds + 'v_slice_*_2.nc').isel(eta_rho=280,xi_rho=221).v.isel(s_rho=-1)
#v3 = xr.open_mfdataset(speeds + 'v_slice_*_3.nc').isel(eta_rho=280,xi_rho=221).v.isel(s_rho=-1)
#v4 = xr.open_mfdataset(speeds + 'v_slice_*_4.nc').isel(eta_rho=280,xi_rho=221).v.isel(s_rho=-1)
#
#
#v_C = xr.concat([v1, v2, v3, v4], dim='ocean_time')
#v_C = v_C.sortby('ocean_time')
#v_C = v_C.sel(ocean_time=v_C.ocean_time.drop_duplicates(dim='ocean_time'))
#
#
#
#################### ----- D -----####################################################################################
#
#
### tu_prime
#tu1 = xr.open_mfdataset(winds + 'tu_slice*_1.nc').isel(eta_rho=120,xi_rho=275).tu
#tu2 = xr.open_mfdataset(winds + 'tu_slice*_2.nc').isel(eta_rho=120,xi_rho=275).tu
#tu3 = xr.open_mfdataset(winds + 'tu_slice*_3.nc').isel(eta_rho=120,xi_rho=275).tu
#tu4 = xr.open_mfdataset(winds + 'tu_slice*_4.nc').isel(eta_rho=120,xi_rho=275).tu
#
#
#tu_D = xr.concat([tu1, tu2, tu3, tu4], dim='ocean_time')
#tu_D = tu_D.sortby('ocean_time')
#tu_D = tu_D.sel(ocean_time=tu_D.ocean_time.drop_duplicates(dim='ocean_time'))
#
### tv_prime
#tv1 = xr.open_mfdataset(winds + 'tv_slice*_1.nc').isel(eta_rho=120,xi_rho=275).tv
#tv2 = xr.open_mfdataset(winds + 'tv_slice*_2.nc').isel(eta_rho=120,xi_rho=275).tv
#tv3 = xr.open_mfdataset(winds + 'tv_slice*_3.nc').isel(eta_rho=120,xi_rho=275).tv
#tv4 = xr.open_mfdataset(winds + 'tv_slice*_4.nc').isel(eta_rho=120,xi_rho=275).tv
#
#
#tv_D = xr.concat([tv1, tv2, tv3, tv4], dim='ocean_time')
#tv_D = tv_D.sortby('ocean_time')
#tv_D = tv_D.sel(ocean_time=tv_D.ocean_time.drop_duplicates(dim='ocean_time'))
#
### u prime
#u1 = xr.open_mfdataset(speeds + 'u_slice_*_1.nc').isel(eta_rho=120,xi_rho=275).u.isel(s_rho=-1)
#u2 = xr.open_mfdataset(speeds + 'u_slice_*_2.nc').isel(eta_rho=120,xi_rho=275).u.isel(s_rho=-1)
#u3 = xr.open_mfdataset(speeds + 'u_slice_*_3.nc').isel(eta_rho=120,xi_rho=275).u.isel(s_rho=-1)
#u4 = xr.open_mfdataset(speeds + 'u_slice_*_4.nc').isel(eta_rho=120,xi_rho=275).u.isel(s_rho=-1)
#
#
#u_D = xr.concat([u1, u2, u3, u4], dim='ocean_time')
#u_D = u_D.sortby('ocean_time')
#u_D = u_D.sel(ocean_time=u_D.ocean_time.drop_duplicates(dim='ocean_time'))
#
#
### v prime
#v1 = xr.open_mfdataset(speeds + 'v_slice_*_1.nc').isel(eta_rho=120,xi_rho=275).v.isel(s_rho=-1)
#v2 = xr.open_mfdataset(speeds + 'v_slice_*_2.nc').isel(eta_rho=120,xi_rho=275).v.isel(s_rho=-1)
#v3 = xr.open_mfdataset(speeds + 'v_slice_*_3.nc').isel(eta_rho=120,xi_rho=275).v.isel(s_rho=-1)
#v4 = xr.open_mfdataset(speeds + 'v_slice_*_4.nc').isel(eta_rho=120,xi_rho=275).v.isel(s_rho=-1)
#
#
#v_D = xr.concat([v1, v2, v3, v4], dim='ocean_time')
#v_D = v_D.sortby('ocean_time')
#v_D = v_D.sel(ocean_time=v_D.ocean_time.drop_duplicates(dim='ocean_time'))
#
###################-----> Wind input per time <------###################################
#WI_A = ((tu_A*u_A) + (tv_A*v_A)).compute()[40:]
#WI_B = ((tu_B*u_B) + (tv_B*v_B)).compute()[40:]
#WI_C = ((tu_C*u_C) + (tv_C*v_C)).compute()[40:]
#WI_D = ((tu_D*u_D) + (tv_D*v_D)).compute()[40:]

##################-----> Wind input map <------###################################



WI_map = xr.open_dataset('/home/arian/dd_waves/wind_driven_iw/analysis/1_wind_input/WI_mean_time.nc').WI_time.compute()



#######################################################################################################################
################### ----- Leakage through mld at each point and ratio with wind input -----############################

leak_mean = EP_at_MLD/WI_map


leak_map = leak_mean/ml_time


