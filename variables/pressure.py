import os
import xarray as xr
import xgcm
import dask
import xroms
from dask.diagnostics import ProgressBar
import numpy as np
import gc


filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_1km_NIWS_V2.nc'	
out_path = '/data1/roms_dd_waves/analysis_outs/NIWS/pressure/'
os.makedirs(out_path, exist_ok=True)


def compute_p_bc(ds_slice,g):
	"""
	Computes baroclinic pressure lazily.
	"""
	# dz is calculated per time step based on the free surface
	dz = ds_slice.dz
	
	# Hydrostatic integration: Surface to Bottom
	# We reindex to start from the surface (s_rho top index), cumsum, then reindex back.
	p_total = (g * ds_slice.rho * dz).reindex(s_rho=ds_slice.s_rho[::-1]).cumsum(dim='s_rho').reindex(s_rho=ds_slice.s_rho)
	
	# Remove depth mean to get baroclinic component
	p_bc = p_total - p_total.mean(dim='s_rho')
	p_bc.name = 'p_bc'
	p_bc.attrs['units'] = 'Pa'
	p_bc.attrs['long_name'] = 'baroclinic hydrostatic pressure'
	
	return p_bc



eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)

chunks = {'ocean_time': 1}
ds1 = xr.open_dataset(filename, chunks=chunks)
ds,xgrid = xroms.roms_dataset(ds1)
gra = 9.8
for i in range(0,len(etas)):

	ds_sub = ds.isel(eta_rho=slice(etas[i][0],etas[i][-1]),xi_rho=xis)
	p_bc_slice = compute_p_bc(ds_sub,gra)
	# Drop z_rho if it exists in the dataset/dataarray to avoid saving it
	
	p_bc_slice = p_bc_slice.drop_vars('z_rho', errors='ignore')
	
	# Save each slice
	save_file = os.path.join(out_path, f'pprime_slice_{i}.nc')
	
	# This triggers the computation and streams it to disk
	with ProgressBar():
		p_bc_slice.to_netcdf(save_file)
	
	# Cleanup
	del p_bc_slice
	gc.collect()
ds1.close()

