import os
import gc
import numpy as np
import xarray as xr
import xroms

filename = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_1km_NIWS_V2.nc'
mld_path = '/data1/roms_dd_waves/analysis_outs/NIWS/MLD/'
speeds   = '/data1/roms_dd_waves/analysis_outs/NIWS/velocities/'
pressure = '/data1/roms_dd_waves/analysis_outs/NIWS/pressure/'
out_path = '/data1/roms_dd_waves/analysis_outs/NIWS/wp_mld/'
os.makedirs(out_path, exist_ok=True)

eta  = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis  = slice(40, None)

NLEV = 39            # number of s_rho levels kept for w_prime / P_prime
METHOD = 'linear'    # 'linear' or 'nearest'
BAND   = 10        # e.g. 10.0 -> also average w'p' over MLD +/- 10 m

# z_rho comes from the raw model file (depends on time through zeta)
ds_raw = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})
ds, xgrid = xroms.roms_dataset(ds_raw)

for i in range(len(etas)):
	print(f'--- eta slice {i} ---')

	eta_sl = slice(etas[i][0], etas[i][-1])   # keep identical to how the inputs were made

	# ---- fluctuations -------------------------------------------------
	w_prime = xr.open_dataset(speeds + f'w_prime_{i}.nc',
							  chunks={'ocean_time': 'auto'}).w_prime.isel(s_rho=slice(0, NLEV))
	P_prime = xr.open_dataset(pressure + f'pprime_slice_{i}.nc',
							  chunks={'ocean_time': 'auto'}).pressure_prime

	w_prime, P_prime = xr.align(w_prime, P_prime, join='override')
	pw = (w_prime * P_prime).rename('w_p')          # (ocean_time, s_rho, eta_rho, xi_rho)

	# ---- depths on exactly the same grid -------------------------------
	z = ds.z_rho.isel(eta_rho=eta_sl, xi_rho=xis, s_rho=slice(0, NLEV))
	z = z.drop_vars([c for c in z.coords if c not in z.dims], errors='ignore')
	z = z.transpose(*pw.dims)                        # same dim order as pw
	pw, z = xr.align(pw, z, join='override')         # force common ocean_time labels

	# ---- MLD ------------------------------------------------------------
	mld = xr.open_dataset(mld_path + f'mld_slice_{i}.nc',
						  chunks={'ocean_time': 'auto'}).mld
	mld = mld.drop_vars([c for c in mld.coords if c not in mld.dims], errors='ignore')
	_, mld = xr.align(pw.isel(s_rho=0, drop=True), mld, join='override')

	# xroms mld is a NEGATIVE depth, like z_rho. Flip it if yours is positive:
	if float(mld.max().compute()) > 0:
		mld = -mld

	# ---- sample w'p' at z = mld -----------------------------------------
	if METHOD == 'nearest':
		k = np.abs(z - mld).argmin(dim='s_rho')      # (ocean_time, eta_rho, xi_rho)
		pw_mld = pw.isel(s_rho=k, drop=True)
		z_used = z.isel(s_rho=k, drop=True)

	else:  # linear interpolation between the two levels bracketing the MLD
		# z increases with s_rho index (bottom -> surface)
		kb = (z <= mld).sum('s_rho') - 1             # last level below the MLD
		kb = kb.clip(0, NLEV - 2)
		ka = kb + 1

		z_b = z.isel(s_rho=kb, drop=True)
		z_a = z.isel(s_rho=ka, drop=True)
		w_b = pw.isel(s_rho=kb, drop=True)
		w_a = pw.isel(s_rho=ka, drop=True)

		frac = ((mld - z_b) / (z_a - z_b)).clip(0.0, 1.0)
		pw_mld = w_b + frac * (w_a - w_b)
		z_used = z_b + frac * (z_a - z_b)

	# mask points where the MLD is outside the resolved column / undefined
	valid = mld.notnull() & (mld >= z.isel(s_rho=0)) & (mld <= z.isel(s_rho=-1))
	pw_mld = pw_mld.where(valid).rename('w_p_mld')
	z_used = z_used.where(valid).rename('z_mld')

	out = xr.Dataset({'w_p_mld': pw_mld,
					  'z_mld':   z_used,
					  'mld':     mld})

	# ---- average over a band around the MLD --------------------
	if BAND is not None:
		band = pw.where((z >= mld - BAND) & (z <= mld + BAND))
		out['w_p_mld_band'] = band.mean('s_rho', skipna=True).where(valid)

	out = out.compute()
	enc = {v: {'zlib': True, 'complevel': 4} for v in out.data_vars}
	out.to_netcdf(out_path + f'w_p_at_mld_slice_{i}.nc', encoding=enc)
	print(f'saved w_p_at_mld_slice_{i}.nc')

	del w_prime, P_prime, pw, z, mld, pw_mld, z_used, out
	gc.collect()

ds_raw.close()
print('All eta slices processed.')