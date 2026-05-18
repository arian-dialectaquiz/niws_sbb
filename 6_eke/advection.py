#####

p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt

out_path = '/data1/roms_dd_waves/analysis_outs/NIW/eke/'
speeds = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'

# ---- Sampling frequency (hourly sampling) ----

rho = 1025

eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)	

for i in range(len(etas)):

	place =  True	
	u_prime = xr.open_dataset(speeds + f'u_prime_{i}.nc', chunks={'ocean_time': 'auto'}).u_prime
	v_prime = xr.open_dataset(speeds + f'v_prime_{i}.nc', chunks={'ocean_time': 'auto'}).v_prime
	w_prime = xr.open_dataset(speeds + f'w_prime_{i}.nc', chunks={'ocean_time': 'auto'}).w_prime	
	u = xr.open_dataset(speeds + f'u_slice_{i}.nc', chunks={'ocean_time': 'auto'}).u
	v = xr.open_dataset(speeds + f'v_slice_{i}.nc', chunks={'ocean_time': 'auto'}).v	
	dl = 1000 #m given or taken
	#u_i'*u_i'
	d_uu_p, d_uu_pv = (u_prime * u_prime).differentiate('xi_rho')/dl , (u_prime * u_prime).differentiate('eta_rho')/dl
	d_uv_p, d_uv_pv = (u_prime * v_prime).differentiate('xi_rho')/dl , (u_prime * v_prime).differentiate('eta_rho')/dl
	d_uw_p,d_uw_pv = (u_prime * w_prime).differentiate('xi_rho')/dl , (u_prime * w_prime).differentiate('eta_rho')/dl
	d_vv_p, d_vv_pv= (v_prime * v_prime).differentiate('xi_rho')/dl , (v_prime * v_prime).differentiate('eta_rho')/dl
	d_vw_p,d_vw_pv = (v_prime * w_prime).differentiate('xi_rho')/dl , (v_prime * w_prime).differentiate('eta_rho')/dl
	d_ww_p,d_ww_pv = (w_prime * w_prime).differentiate('xi_rho')/dl , (w_prime * w_prime).differentiate('eta_rho')/dl						
	num_u = (d_uu_p + d_uv_p + d_uw_p + d_vv_p + d_vw_p + d_ww_p).compute()		
	num_v = (d_uu_pv + d_uv_pv + d_uw_pv + d_vv_pv + d_vw_pv + d_ww_pv).compute()

	#advection		
	adv = -rho* ((u * num_u) + (v*num_v))		
	print("computing...")		
	#adv_zsum = adv.sum('s_rho').compute()
	print(f"Saving file for eta slice {i}")		
	adv.to_dataset(name='eke_adv').to_netcdf(out_path + f'eke_adv_slice_{i}.nc')	
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