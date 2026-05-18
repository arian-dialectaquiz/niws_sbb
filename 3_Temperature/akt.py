
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt
from pathlib import Path

dzs = '/data1/roms_dd_waves/analysis_outs/dz/'
fs = 1 / 1  


# Configuration
out_path = Path('/data1/roms_dd_waves/analysis_outs/NIW/theta/')
out_path.mkdir(parents=True, exist_ok=True)

#filename = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/dia_paper_2_1km.nc'	

filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc'
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})
# Define eta and xi slices

# Define eta and xi slices
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)	
ds, xgrid = xroms.roms_dataset(ds1)

eta_selected = np.concatenate([etas[i][:-1] for i in range(len(etas))])
eta_targets = [225]



def process_file(ds, xis, eta_selected, eta_targets, out_path):
	"""Process a single file and save outputs with custom naming"""
	
	ds_work = ds	

	for i, g in enumerate(eta_targets):
		# Compute and save
		print('Entering eta: ', g)
		ak = xroms.to_s_rho(ds_work.AKt, xgrid).isel(xi_rho=xis, eta_rho=eta_selected).isel(eta_rho=g, xi_rho=slice(None)).compute()
		output_file = out_path / f'AKt_eta_{i}.nc'  
		ak.to_dataset(name='AKt').to_netcdf(output_file)
		print(f"Saved {output_file}")			
		# Clean up
		del ak, output_file
		gc.collect()



process_file(ds, xis, eta_selected, eta_targets, out_path)
gc.collect()  # Additional cleanup between files