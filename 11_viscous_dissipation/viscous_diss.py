

p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt

speed = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'
dzs = '/data1/roms_dd_waves/analysis_outs/dz/'
bs = '/data1/roms_dd_waves/analysis_outs/NIW/pressure_b/'
ns = '/data1/roms_dd_waves/analysis_outs/NIW/N2/'
out_path = '/data1/roms_dd_waves/analysis_outs/NIW/viscous/'
dzs = '/data1/roms_dd_waves/analysis_outs/dz/'

	


# Define slices for eta and xi
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)	
		
rho_0 = 1025  # Reference density
dl = 1000     # Mean lateral distance	]
filename = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc'	
# Open datasets
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})

filename_dia = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/dia_paper_2_1km.nc'
# Open datasets
ds2 = xr.open_dataset(filename_dia, chunks={'ocean_time': 'auto'})

ds_Av, xgrid = xroms.roms_dataset(ds1)
ds_Ah, xgrid_h = xroms.roms_dataset(ds2)

del ds1,ds2

for i in range(1,len(etas)):
	place = True
	dz = xr.open_dataset(dzs + f'dz_{i}.nc').dz
	A_v = xroms.to_s_rho(ds_Av.AKv, xgrid).isel(eta_rho=slice(etas[i][0], etas[i][-1]), xi_rho=xis).resample(ocean_time='3H').mean()
	A_h = xroms.to_rho(ds_Ah.u_hvisc, xgrid_h).isel(eta_rho=slice(etas[i][0], etas[i][-1]), xi_rho=xis).assign_coords(ocean_time=A_v.ocean_time)
	u_prime = xr.open_dataset(speed + f'u_prime_{i}.nc').u_prime.resample(ocean_time='3H').mean()		
	v_prime = xr.open_dataset(speed + f'v_prime_{i}.nc').v_prime.resample(ocean_time='3H').mean()		
	w_prime = xr.open_dataset(speed + f'w_prime_{i}.nc').w_prime.resample(ocean_time='3H').mean()
	u_prime, v_prime, w_prime, A_v,A_h = xr.align(u_prime, v_prime, w_prime, A_v,A_h, join="inner")
	du_dx = u_prime.differentiate('xi_rho')/dl
	dv_dy = v_prime.differentiate('eta_rho')/dl
	dw_dz = w_prime.differentiate('s_rho')/dz

	dissipation = rho_0 * (A_h * (du_dx**2 + dv_dy**2) + A_v * dw_dz**2)
	vis = dissipation.compute()
	vis.to_dataset(name='viscous').drop_vars('z_rho').to_netcdf(out_path + f'visc_loss_{i}.nc')		
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