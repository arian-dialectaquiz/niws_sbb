
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt

out_path = '/data1/roms_dd_waves/analysis_outs/NIW/pressure_b/'
speeds = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'
dzs = '/data1/roms_dd_waves/analysis_outs/dz/'

pressure = out_path


rho = 1025
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)	

for i in range(0,len(etas)):
	
	place =  True	
	u_prime = xr.open_dataset(speeds + f'u_prime_{i}.nc', chunks={'ocean_time': 'auto'}).u_prime.isel(s_rho=slice(0,39))
	v_prime = xr.open_dataset(speeds + f'v_prime_{i}.nc', chunks={'ocean_time': 'auto'}).v_prime.isel(s_rho=slice(0,39))
	P_prime = xr.open_dataset(pressure + f'pprime_slice_{i}.nc', chunks={'ocean_time': 'auto'}).pressure_prime
	u_prime, v_prime, P_prime = xr.align(u_prime, v_prime, P_prime, join="override")
	dzii = xr.open_dataset(dzs + f'dz_{i}.nc').dz
	
	Fx = (u_prime * P_prime)* dzii
	F_xsum = Fx.cumsum(dim='s_rho').compute()
	F_xsum.to_dataset(name='Fx').to_netcdf(out_path + f'X_energy_flux_slice_{i}.nc')

	Fy = (v_prime * P_prime)* dzii	
	F_ysum = Fy.cumsum(dim='s_rho').compute()
	F_ysum.to_dataset(name='Fy').to_netcdf(out_path + f'Y_energy_flux_slice_{i}.nc')	

	target_variable = 'place'
	local_vars = locals()
	# Find the index of the target variable
	index_of_target = list(local_vars.keys()).index(target_variable)
	# Delete variables defined later than the target variable
	variables_to_delete = list(local_vars.keys())[index_of_target:]
	for var in variables_to_delete:
		del locals()[var]			
	gc.collect()
	print("All eta slices processed and saved.")



