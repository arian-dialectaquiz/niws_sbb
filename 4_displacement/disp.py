
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt

out_path = '/data1/roms_dd_waves/analysis_outs/NIW/displacement/'
theta = '/data1/roms_dd_waves/analysis_outs/NIW/theta/'
dzs = '/data1/roms_dd_waves/analysis_outs/dz/'

eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)	


for i in range(len(etas)):
	#tprime = xr.open_dataset(theta + f'theta_prime_slice_{i}.nc').theta_prime
	t_slice = xr.open_dataset(theta + f'theta_slice_{i}.nc').theta
	dtdz = xr.open_dataset(theta + f'dT_dz_slice_{i}.nc').dT_dz
	dtdz_m = dtdz.mean(dim='ocean_time').compute()	
	eta = (t_slice/dtdz_m).compute()
	eta.to_dataset(name='eta').to_netcdf(out_path + f'eta_full_slice_{i}.nc')	
	del t_slice, dtdz,dtdz_m,eta
	gc.collect()

	
print(f"Completed eta slice {i} (part {t}).")
