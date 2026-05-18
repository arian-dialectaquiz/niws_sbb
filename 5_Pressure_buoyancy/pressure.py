###

p1_dir = '/home/arian/dd_waves/m2_internal_tide_gen_prop/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt


out_path = '/data1/roms_dd_waves/analysis_outs/NIW/pressure_b/'
displacement = '/data1/roms_dd_waves/analysis_outs/NIW/displacement/'
strat = '/data1/roms_dd_waves/analysis_outs/NIW/N2/'
dzs = '/data1/roms_dd_waves/analysis_outs/dz/'

rho0 = 1025
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)


	
for i in range(2,len(etas)):	

	place =  True
	eta = xr.open_dataset(displacement + f'eta_full_slice_{i}.nc').eta.compute()
	eta = eta.where((eta >= -200) & (eta <= 200), other=np.nan)
	dz = xr.open_dataset(dzs + f'dz_{i}.nc').dz.compute()
	N = xr.open_dataset(strat + f'N2_slice_{i}.nc').N2.isel(s_rho=slice(0, -1)).compute()	
	arg = N.values * eta.values * dz.values	
	arg_reversed = arg[:, ::-1, :]		
	arg_xr = xr.DataArray(arg_reversed, coords=[N.ocean_time, N.s_rho[::-1], N.eta_rho, N.xi_rho],
					 dims=["ocean_time", "s_rho", "eta_rho", "xi_rho"])
	P_bar = rho0 * (arg_xr.sum(dim='s_rho'))		

	P_prime = (rho0 * arg_xr.cumsum(dim='s_rho')) - P_bar	
	Pp = P_prime[:, ::-1, :].compute() #giving the original shape of roms	
	Pp.to_dataset(name='pressure_prime').to_netcdf(out_path + f'pprime_slice_{i}.nc')	
	del eta,dz,N,P_bar, arg, P_prime, Pp	
	target_variable = 'place'
	local_vars = locals()
	# Find the index of the target variable
	index_of_target = list(local_vars.keys()).index(target_variable)
	# Delete variables defined later than the target variable
	variables_to_delete = list(local_vars.keys())[index_of_target:]
	for var in variables_to_delete:
		del locals()[var]	
	print(f"Completed eta slice {i}.")

