

p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *


filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc'	
out_path = '/data1/roms_dd_waves/analysis_outs/dz/'

file = xr.open_dataset(filename,chunks={'ocean_time':'auto'})
ds, xgrid = xroms.roms_dataset(file)

eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)	

for i in range(len(etas)):
	
	place = True
	print(f"Slicing array for i = {i}")
	
	z_rho_ini = ds.z_rho.isel(eta_rho=slice(etas[i][0],etas[i][-1]),xi_rho=xis)
	
	dz = z_rho_ini.diff(dim='s_rho')
	dz_comp_da = dz[500].compute()
	# Convert to a dataset
	dz_comp_da = dz_comp_da.to_dataset(name='dz')

	dz_comp_da.to_netcdf(out_path + f'dz_{i}.nc')
		
	print("Jumping to new slice\n\tEnding saving proces")
	print("Deleting variables")     
	target_variable = 'place'       
	local_vars = locals()       
	# Find the index of the target variable 
	index_of_target = list(local_vars.keys()).index(target_variable)        
	# Delete variables defined later than the target variable   
	variables_to_delete = list(local_vars.keys())[index_of_target:]     
	for var in variables_to_delete:		
		del locals()[var]


