
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt

out_path = '/data1/roms_dd_waves/analysis_outs/NIW/wind_input/'

wind_input = '/data1/roms_dd_waves/analysis_outs/NIW/wind_input/'
speeds = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'

# ---- Sampling frequency (hourly sampling) ----
dt_hours = 1.0
fs = 1 / dt_hours # sampling frequency in Hz

filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc'	
## Open the dataset with Dask, chunked by 'ocean_time' for efficiency
## Open datasets
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})	
ds, xgrid = xroms.roms_dataset(ds1)
f = ds.f.compute()
f_mean = float(np.abs(f).where(np.abs(f) > 0).mean().values)
T_mean = (2*np.pi/f_mean)/3600 #inertial period in hours
T1 = 0.8 * T_mean  # Hours
T2 = 1.5 * T_mean  # Hours

# Define eta and xi slices
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)



for i in range(0,len(etas)):
	print("Starting calculationf for etas:",{i})
	place =  True
	print('fetching Tu')
	tu = xr.open_dataset(wind_input + f'tu_slice_{i}.nc').tu
	print('filtering Tu')
	tu_prime = filter_da_bandpass(tu, T1, T2, fs).compute().transpose('ocean_time', 'eta_rho', 'xi_rho')#.isel(ocean_time=slice(0, None, 2))
	#tu_prime_file = tu_prime.to_dataset(name='tu_prime')		
	#tu_prime_file.to_netcdf(out_path+f'tu_prime_{i}.nc')
	print('u input')
	u = xr.open_dataset(speeds + f'u_prime_{i}.nc', chunks={'ocean_time': 'auto'}).u_prime.isel(s_rho=-1)#
	WU = tu_prime * u#
	WU.name = 'wind_input'
	WU.attrs['long_name'] = 'U Wind energy input to NIW'
	WU.attrs['units'] = 'W/m^2'
	WU_file = WU.to_dataset(name='WU')		
	WU_file.to_netcdf(wind_input+f'WU_Input_{i}.nc')
	del tu, tu_prime
	

	print('fetching Tv')
	tv = xr.open_dataset(wind_input + f'tv_slice_{i}.nc').tv
	print('filtering Tv')
	tv_prime = filter_da_bandpass(tv, T1, T2, fs).compute().transpose('ocean_time','eta_rho', 'xi_rho')#.isel(ocean_time=slice(0, None, 2))
	#tv_prime_file = tv_prime.to_dataset(name='tv_prime')		
	#tv_prime_file.to_netcdf(out_path+f'tv_prime_{i}.nc')
	print('v input')
	v = xr.open_dataset(speeds + f'v_prime_{i}.nc', chunks={'ocean_time': 'auto'}).v_prime.isel(s_rho=-1)#		
	WV = tv_prime * v#
	WV.name = 'wind_input'
	WV.attrs['long_name'] = 'V Wind energy input to NIW'
	WV.attrs['units'] = 'W/m^2'
	WV_file = WV.to_dataset(name='WV')		
	WV_file.to_netcdf(wind_input+f'WV_Input_{i}.nc')

	del tv, tv_prime		
	print("Jumping to new slice\n\tEnding saving process")
	print("Deleting variables")	
	target_variable = 'place'
	local_vars = locals()
	# Find the index of the target variable
	index_of_target = list(local_vars.keys()).index(target_variable)
	# Delete variables defined later than the target variable
	variables_to_delete = list(local_vars.keys())[index_of_target:]
	for var in variables_to_delete:
		del locals()[var]


del ds1,ds,eta,etas,xis,filename, f, f_mean,T_mean, T1, T2
gc.collect()
print("All slices processed and saved.")



