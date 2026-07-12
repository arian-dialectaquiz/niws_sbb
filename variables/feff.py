
import os
import sys
import gc
import numpy as np
import xarray as xr
import xroms
from scipy import signal

filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_1km_NIWS_V2.nc'	
out_path = '/data1/roms_dd_waves/analysis_outs/NIW/vorticity/'

# Open dataset




# Define slices for eta and xi

eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)	
		
# Open datasets
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})
ds, xgrid = xroms.roms_dataset(ds1)	
# Get vorticity field

f = ds.f  # Coriolis parameter

for i in range(0,len(etas)):
	place =  True
	print(f"Slicing array for i = {i}")

	vort = xr.open_dataset(out_path + f'relative_vorticity_slice_{i}.nc', chunks={'ocean_time': 'auto'}).zeta
	f = f.isel(eta_rho=slice(etas[i][0],etas[i][-1]),xi_rho=xis).compute()
	
	feff = (f + vort/2).compute()

	norm_vort_file = feff.to_dataset(name='f_eff')		
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




