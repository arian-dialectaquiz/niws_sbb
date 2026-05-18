
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt

out_path = '/data1/roms_dd_waves/analysis_outs/NIW/wind_input/'

#wind_input = '/data1/roms_dd_waves/analysis_outs/NIW/wind_input/'
speeds = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'

# ---- Sampling frequency (hourly sampling) ----
dt_hours = 1.0
fs = 1 / dt_hours # sampling frequency in Hz


filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc'	
	
# Open the dataset with Dask, chunked by 'ocean_time' for efficiency
# Open datasets
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})	
ds, xgrid = xroms.roms_dataset(ds1)
f = ds.f.compute()
f_mean = float(np.abs(f).where(np.abs(f) > 0).mean().values)
T_mean = (2*np.pi/f_mean)/3600 #inertial period in hours
T1 = 0.8 * T_mean  # Hours
T2 = 1.5 * T_mean  # Hours

eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)


for i in range(0,len(etas)):
	print("Starting calculationf for etas:",{i})
	place =  True
	print('fetching Tu')
	tu = xroms.to_rho(ds.sustr, xgrid).isel(eta_rho=slice(etas[i][0],etas[i][-1]),xi_rho=xis).compute().transpose('ocean_time', 'eta_rho', 'xi_rho')#.isel(ocean_time=slice(0, None, 2))
	tu_file = tu.to_dataset(name='tu')		
	tu_file.to_netcdf(out_path+f'tu_slice_{i}.nc')

	print('fetching Tv')
	tv = xroms.to_rho(ds.svstr, xgrid).isel(eta_rho=slice(etas[i][0],etas[i][-1]),xi_rho=xis).compute().transpose('ocean_time', 'eta_rho', 'xi_rho')#.isel(ocean_time=slice(0, None, 2))
	tv_file = tv.to_dataset(name='tv')		
	tv_file.to_netcdf(out_path+f'tv_slice_{i}.nc')		
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
