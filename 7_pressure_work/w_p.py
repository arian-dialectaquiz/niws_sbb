
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
pressure = out_path


rho = 1025

rho = 1025
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)	


dl = 1000

for i in range(len(etas)):
	place =  True
	w_prime = xr.open_dataset(speeds + f'w_prime_{i}.nc', chunks={'ocean_time': 'auto'}).w_prime.isel(s_rho=slice(0,39))
	P_prime = xr.open_dataset(pressure + f'pprime_slice_{i}.nc', chunks={'ocean_time': 'auto'}).pressure_prime
	w_prime, P_prime = xr.align(w_prime, P_prime, join="override")
	#they have different times, fix it
	pw = ((w_prime * P_prime)/rho).compute()
	#numx = num.differentiate('xi_rho')/dl
	#numy = num.differentiate('eta_rho')/dl
	#pw = numx + numy
	#pw_zsum = pw.compute()#.sum(dim='s_rho').compute()
	print(f"Saving file for eta slice {i}")		
	pw.to_dataset(name='w_p').to_netcdf(out_path + f'w_p_slice_{i}.nc')	
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
