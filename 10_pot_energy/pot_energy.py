
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt

out_path = '/data1/roms_dd_waves/analysis_outs/NIW/pot_energy/'
speeds = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'
pressure = '/data1/roms_dd_waves/analysis_outs/NIW/pressure_b/'


rho = 1025
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)	

dl = 1000

for i in range(9,len(etas)):
	place =  True
	w_prime = xr.open_dataset(speeds + f'w_prime_{i}.nc', chunks={'ocean_time': 'auto'}).w_prime
	b = xr.open_dataset(pressure + f'b_slice_{i}.nc', chunks={'ocean_time': 'auto'}).buoyancy		
	term = (rho *  w_prime * b).compute()
	term_zsum = term.compute()#.sum(dim='s_rho').compute()
	print(f"Saving file for eta slice {i}")		
	term_zsum.to_dataset(name='pot_ex').to_netcdf(out_path + f'pot_exchange_slice_{i}.nc')			
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
