
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt

out_path = '/data1/roms_dd_waves/analysis_outs/NIW/displacement/'


eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)	

filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc'	
# Open the dataset with Dask, chunked by 'ocean_time' for efficiency
# Open datasets
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})	

ds, xgrid = xroms.roms_dataset(ds1)

for i in range(len(etas)):
	#tprime = xr.open_dataset(theta + f'theta_prime_slice_{i}.nc').theta_prime
	ssh = ds.zeta.isel(eta_rho=slice(etas[i][0],etas[i][-1]),xi_rho=xis).compute()
	ssh.to_dataset(name='ssh').to_netcdf(out_path + f'ssh_slice_{i}.nc')	
	
	gc.collect()

	
print(f"Completed ssh slice {i}.")
