
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt


out_path = '/data1/roms_dd_waves/analysis_outs/NIW/vorticity/'

# Open dataset




# Define slices for eta and xi

eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)	
		
rho_0 = 1025  # Reference density
dl = 1000     # Mean lateral distance	
filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc'	
# Open datasets
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})
ds, xgrid = xroms.roms_dataset(ds1)	
# Get vorticity field
zeta_s = xroms.relative_vorticity(ds.u,ds.v,xgrid)
zeta_rho = xroms.to_rho(zeta_s, xgrid)
zeta = xroms.to_s_rho(zeta_rho, xgrid)
f = ds.f  # Coriolis parameter

for i in range(0,len(etas)):
	place =  True
	print(f"Slicing array for i = {i}")
	z = zeta.isel(eta_rho=slice(etas[i][0],etas[i][-1]),xi_rho=xis).compute().transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
	f = f.isel(eta_rho=slice(etas[i][0],etas[i][-1]),xi_rho=xis).compute().transpose('eta_rho', 'xi_rho')
	norm_vort = f / (f + z/2)
	norm_vort_file = norm_vort.to_dataset(name='f_eff')		
	norm_vort_file.to_netcdf(out_path+f'f_eff_{i}.nc')
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




