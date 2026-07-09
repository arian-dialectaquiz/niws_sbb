################################################

import xarray as xr
import numpy as np
import dask
import re
import glob
import cartopy
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import ScalarFormatter, MaxNLocator, LogLocator, NullFormatter, FixedLocator
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmocean as cmo
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.dates as mdates
import pandas as pd
import xroms
from scipy import signal
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature
from matplotlib.patheffects import Stroke
import shapely.geometry as sgeom
import matplotlib.patches as mpatches
import os
import warnings
import tropycal.tracks as tracks
import tropycal

#/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km
#avg_1km_NIWS_V2.nc
#paper_2_1km_closed_cropped_smooth_sponge.nc





#################################################################
#########-----> Blended forcing vs quickscat <--#################

era5 = xr.open_dataset('/home/arian/dd_waves/pyroms_tools/data/2004_paper_2/ERA5/single_paper_2.nc')
coarse = xr.open_dataset('/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/3km/inputs/deproas_spongebob_grd_cropped_3km.nc')
finer = xr.open_dataset('/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/inputs/paper_2_1km_closed_cropped_smooth_sponge.nc')


"""
Catarina (March 2004) Figure 1:
  (a) QuikSCAT swath wind speed (one pass) + best track          [map panel]
  (b) Vmax(t): best track vs ERA5 vs blended vs QuikSCAT         [inset]
  (c) Rmax(t): same sources                                      [inset]
  (d) tau_max(t): ROMS bulk stress vs QuikSCAT ESDR L2 stress    [optional]
 
Products (PO.DAAC, both netCDF, one file per orbital rev):
  * Winds : QSCAT_LEVEL_2B_OWV_COMP_12_KUSST_LCRES_4.1  (12.5-km slices)
  * Stress: QUIKSCAT_ESDR_L2_WIND_STRESS_V1.1           (MEaSUREs ESDR,
			equivalent-neutral winds + wind stress on the same 12.5-km swath)
 
CLI download alternative to the earthaccess functions below:
  podaac-data-downloader -c QSCAT_LEVEL_2B_OWV_COMP_12_KUSST_LCRES_4.1 \
	  -d ./qscat_catarina \
	  --start-date 2004-03-23T00:00:00Z --end-date 2004-03-29T00:00:00Z \
	  -b="-60,-35,-35,-20"
  (same for -c QUIKSCAT_ESDR_L2_WIND_STRESS_V1.1 -d ./qscat_esdr_catarina)
 
Requirements: numpy, pandas, xarray, netCDF4, matplotlib, tropycal,
			  earthaccess; cartopy optional (map falls back to plain axes).
 
Caveats encoded in comments:
  * QuikSCAT = 10-m equivalent-neutral wind; best track = 1-min sustained;
	ERA5 = grid-cell mean. Optional averaging conversion factor below.
  * Rain flags usually remove the eyewall: the QC'd QuikSCAT Vmax is a lower
	bound (both QC'd and unflagged maxima are stored).
  * QuikSCAT speeds are uncertain above ~30 m/s; Rmax is the more robust
	quantity near Catarina's peak.
  * ESDR tau uses a neutral-wind drag parameterization; ROMS tau uses your
	bulk formula. Differences in the core reflect drag-coefficient choices
	(e.g. Cd saturation at high winds, Powell et al. 2003) -- which is the
	point of panel (d).
"""




# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
ERA5_FILE       = "/home/arian/dd_waves/pyroms_tools/data/2004_paper_2/ERA5/single_paper_2.nc"       # u10, v10 (time, lat, lon)
QSCAT_DIR       = "qscat_catarina"         # L2B v4.1 wind granules (.nc)
QSCAT_ESDR_DIR  = "qscat_esdr_catarina"    # ESDR L2 wind+stress granules (.nc)
ROMS_FILE       = "/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_1km_NIWS_V2.nc"   # ROMS output with sustr, svstr
OUT_INSET       = "fig1_inset_vmax_rmax"
OUT_FIG1        = "fig1"
 
T0, T1          = "2004-03-24 00:00", "2004-03-28 12:00"   # analysis window
DT_HOURS        = 6                        # cadence for ERA5/blend/ROMS curves
 
SEARCH_RADIUS_KM = 250.0                   # storm-relative search disc
DR_BIN_KM        = 12.5                    # radial bin width (azimuthal mean)
 
MAP_EXTENT = (-54.0, -38.0, -33.5, -22.0)  # lon_min, lon_max, lat_min, lat_max
 
# --- Holland / blend parameters (the \todo tuning targets) -------------------
B_HOLLAND   = 1.4        # shape parameter; tune with fit_holland_B() below
R1_KM       = 100.0      # W=1 inside r1  (pure vortex)
R2_KM       = 250.0      # W=0 outside r2 (pure ERA5)
ALPHA_TRANS = 0.55       # translation-asymmetry weight (Lin & Chavas 2012)
 
# --- best-track auxiliary -----------------------------------------------------
# McTaggart-Cowan et al. (2006) give position/intensity but not RMW.
# Set from your preferred source (satellite eye diameter suggests O(15-30 km)
# near peak); set to None to omit the best-track line in the Rmax panel.
RMW_BT_KM   = 20.0
 
KT2MS = 0.514444
# 1-min sustained -> ~10-min/equivalent-neutral conversion (Harper et al.
# 2010). Set to 1.0 to plot best track as-is.
WIND_AVG_FACTOR = 0.93
 
RHO_AIR = 1.15
OMEGA   = 7.2921e-5
 
 
# ----------------------------------------------------------------------------
# 0. DOWNLOADS (run once each)
# ----------------------------------------------------------------------------
def download_quikscat(outdir=QSCAT_DIR):
	"""QuikSCAT L2B v4.1 wind granules intersecting the Catarina box."""
	import earthaccess
	earthaccess.login()                       # needs (free) Earthdata login
	results = earthaccess.search_data(
		short_name="QSCAT_LEVEL_2B_OWV_COMP_12_KUSST_LCRES_4.1",
		temporal=("2004-03-23", "2004-03-29"),
		bounding_box=(-60.0, -35.0, -35.0, -20.0),   # (W, S, E, N)
	)
	print(f"{len(results)} wind granules found")
	earthaccess.download(results, outdir)
 
 
def download_qscat_esdr(outdir=QSCAT_ESDR_DIR):
	"""ESDR L2 v1.1: equivalent-neutral winds + wind STRESS, same swath."""
	import earthaccess
	earthaccess.login()
	results = earthaccess.search_data(
		short_name="QUIKSCAT_ESDR_L2_WIND_STRESS_V1.1",
		temporal=("2004-03-23", "2004-03-29"),
		bounding_box=(-60.0, -35.0, -35.0, -20.0),
	)
	print(f"{len(results)} ESDR granules found")
	earthaccess.download(results, outdir)
 
 
# ----------------------------------------------------------------------------
# 1. BEST TRACK (tropycal / IBTrACS with the special Catarina entry)
# ----------------------------------------------------------------------------
def get_track():
	"""DataFrame indexed by time: lat, lon, vmax [m/s], utr, vtr [m/s]."""
	from tropycal import tracks
	ibtracs = tracks.TrackDataset(basin="all", source="ibtracs",
								  ibtracs_mode="jtwc_neumann", catarina=True)
	storm = ibtracs.get_storm(("catarina", 2004))
 
	t   = pd.to_datetime(storm.time)
	lat = np.asarray(storm.lat, float)
	lon = np.asarray(storm.lon, float)
	vmx = np.asarray(storm.vmax, float) * KT2MS      # kt -> m/s (1-min)
 
	# translation velocity by centered differences
	ts  = t.view("int64") / 1e9                       # seconds
	dxe = np.gradient(lon) * 111.32e3 * np.cos(np.deg2rad(lat))
	dyn = np.gradient(lat) * 111.32e3
	dts = np.gradient(ts)
	utr, vtr = dxe / dts, dyn / dts
 
	return pd.DataFrame(dict(time=t, lat=lat, lon=lon, vmax=vmx,
							 utr=utr, vtr=vtr)).set_index("time")
 
 
def interp_track(track, when):
	"""Linear interpolation of track quantities to a single datetime."""
	tsec = track.index.view("int64") / 1e9
	w    = pd.Timestamp(when).value / 1e9
	return {c: np.interp(w, tsec, track[c].values) for c in track.columns}
 
 
# ----------------------------------------------------------------------------
# 2. GEOMETRY + STORM-RELATIVE METRICS
# ----------------------------------------------------------------------------
def local_xy_km(lon, lat, clon, clat):
	"""Small-domain equirectangular offsets (km) from storm center."""
	dx = (np.asarray(lon) - clon) * 111.32 * np.cos(np.deg2rad(clat))
	dy = (np.asarray(lat) - clat) * 111.32
	return dx, dy
 
 
def radial_profile(fld, r_km, rmax_km=SEARCH_RADIUS_KM,
				   dr=DR_BIN_KM, min_count=3):
	"""Azimuthal-mean profile of any scalar field in radial bins."""
	edges = np.arange(0.0, rmax_km + dr, dr)
	rc    = 0.5 * (edges[:-1] + edges[1:])
	prof  = np.full(rc.size, np.nan)
	for i in range(rc.size):
		m = (r_km >= edges[i]) & (r_km < edges[i + 1]) & np.isfinite(fld)
		if m.sum() >= min_count:
			prof[i] = np.nanmean(fld[m])
	return rc, prof
 
 
def storm_metrics(fld, r_km):
	"""Return (max_point, r_of_max_point, max_azim, r_of_max_azim)."""
	fld, r_km = np.ravel(fld), np.ravel(r_km)
	inside    = (r_km <= SEARCH_RADIUS_KM) & np.isfinite(fld)
	if inside.sum() < 5:
		return (np.nan,) * 4
	i_pt = np.nanargmax(np.where(inside, fld, -np.inf))
	fmax_pt, rmax_pt = fld[i_pt], r_km[i_pt]
	rc, prof = radial_profile(fld[inside], r_km[inside])
	if np.all(np.isnan(prof)):
		return fmax_pt, rmax_pt, np.nan, np.nan
	j = np.nanargmax(prof)
	return fmax_pt, rmax_pt, prof[j], rc[j]
 
 
def _find_var(ds, candidates):
	"""Return the first candidate variable name present in ds (or None)."""
	low = {v.lower(): v for v in ds.data_vars}
	for name in candidates:
		if name in ds:
			return name
		if name.lower() in low:
			return low[name.lower()]
	return None
 
 
# ----------------------------------------------------------------------------
# 3. HOLLAND VORTEX + BLEND (methods-section equations)
# ----------------------------------------------------------------------------
def holland_V(r_m, vmax, rmax_m, B, f):
	"""Gradient-wind Holland profile V(r) (Eq. 1 of the methods section)."""
	rr = np.maximum(r_m, 1.0)
	x  = (rmax_m / rr) ** B
	V2 = vmax**2 * x * np.exp(1.0 - x) + (rr * f / 2.0) ** 2
	return np.sqrt(V2) - rr * np.abs(f) / 2.0
 
 
def vortex_uv(lon2d, lat2d, clon, clat, vmax_bt, rmax_km, B,
			  utr, vtr, alpha=ALPHA_TRANS):
	"""
	Axisymmetric SH vortex + translation asymmetry.
	- SH cyclone: CLOCKWISE flow -> tangential unit vector (dy, -dx)/r
	- adding alpha*(V/Vmax)*(utr, vtr) puts the wind max on the LEFT of the
	  motion in the SH (tangential wind and translation align there).
	"""
	dx, dy = local_xy_km(lon2d, lat2d, clon, clat)
	r_km   = np.hypot(dx, dy)
	r_m    = np.maximum(r_km, 1e-3) * 1e3
 
	f = 2.0 * OMEGA * np.sin(np.deg2rad(clat))        # < 0 in SH
	trans = np.hypot(utr, vtr)
	vmax_sym = max(vmax_bt - alpha * trans, 5.0)      # remove motion first
 
	V = holland_V(r_m, vmax_sym, rmax_km * 1e3, B, f)
	tx = dy / np.maximum(r_km, 1e-6)
	ty = -dx / np.maximum(r_km, 1e-6)
	decay = np.clip(V / vmax_sym, 0.0, 1.0)
 
	u = V * tx + alpha * decay * utr
	v = V * ty + alpha * decay * vtr
	return u, v, r_km
 
 
def blend_weight(r_km, r1=R1_KM, r2=R2_KM):
	"""Raised-cosine taper W(r): 1 inside r1, 0 outside r2 (Eq. 2)."""
	W   = np.zeros_like(r_km, dtype=float)
	W[r_km <= r1] = 1.0
	mid = (r_km > r1) & (r_km < r2)
	W[mid] = 0.5 * (1.0 + np.cos(np.pi * (r_km[mid] - r1) / (r2 - r1)))
	return W
 
 
# ----------------------------------------------------------------------------
# 4. ERA5 + BLENDED CURVES
# ----------------------------------------------------------------------------
def open_era5(path=ERA5_FILE):
	ds = xr.open_dataset(path)
	ren = {}
	for cand, std in [("latitude", "lat"), ("longitude", "lon"),
					  ("valid_time", "time")]:
		if cand in ds.coords or cand in ds.dims:
			ren[cand] = std
	ds = ds.rename(ren)
	if ds.lon.max() > 180:                            # 0..360 -> -180..180
		ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
	return ds
 
 
def era5_and_blend_series(track):
	ds    = open_era5()
	times = pd.date_range(T0, T1, freq=f"{DT_HOURS}h")
	rows  = []
	for t in times:
		c = interp_track(track, t)
		try:
			sub = ds.sel(time=t, method="nearest",
						 tolerance=pd.Timedelta("3h"))
		except KeyError:
			continue
		box = sub.where(
			(np.abs(sub.lat - c["lat"]) < 4.0) &
			(np.abs(((sub.lon - c["lon"] + 180) % 360) - 180) < 4.5),
			drop=True)
		if box.lat.size < 4:
			continue
		lon2d, lat2d = np.meshgrid(box.lon.values, box.lat.values)
		uE, vE = box.u10.values, box.v10.values
		dx, dy = local_xy_km(lon2d, lat2d, c["lon"], c["lat"])
		r_km   = np.hypot(dx, dy)
 
		# --- raw ERA5 metrics
		spdE = np.hypot(uE, vE)
		eV, eR, eVa, eRa = storm_metrics(spdE, r_km)
 
		# --- blended field (Eq. 2)
		uV, vV, _ = vortex_uv(lon2d, lat2d, c["lon"], c["lat"],
							  c["vmax"] * WIND_AVG_FACTOR,
							  RMW_BT_KM or 25.0, B_HOLLAND,
							  c["utr"], c["vtr"])
		W  = blend_weight(r_km)
		uB = W * uV + (1.0 - W) * uE
		vB = W * vV + (1.0 - W) * vE
		spdB = np.hypot(uB, vB)
		bV, bR, bVa, bRa = storm_metrics(spdB, r_km)
 
		rows.append(dict(
			time=t,
			era5_vmax=eV,  era5_rmax=eRa if np.isfinite(eRa) else eR,
			blend_vmax=bV, blend_rmax=bRa if np.isfinite(bRa) else bR))
	return pd.DataFrame(rows).set_index("time")
 
 
# ----------------------------------------------------------------------------
# 5. QUIKSCAT WIND PASSES (L2B v4.1)
# ----------------------------------------------------------------------------
def _qc_mask(ds, fld, rain_thresh=2.5):
	"""
	QC for JPL L2B v3/v4.x and ESDR files:
	  * rain_impact > ~2.5 => rain-degraded (see product user guide)
	  * bit-flag variables ('flags', 'eflags', 'quality_flag'): drop
		rain/ice/coastal/land/retrieval bits when flag_masks/flag_meanings
		attributes are present (defensive parsing).
	Tighten against the user guide on PO.DAAC once you inspect ds.data_vars.
	"""
	good = np.isfinite(fld)
	if "rain_impact" in ds:
		ri = ds["rain_impact"].values
		good &= ~(np.isfinite(ri) & (ri > rain_thresh))
	for fname in ("flags", "eflags", "quality_flag", "wvc_quality_flag"):
		if fname not in ds:
			continue
		fl    = ds[fname]
		masks = fl.attrs.get("flag_masks", None)
		names = fl.attrs.get("flag_meanings", "")
		if masks is None or not names:
			continue
		names  = names.split()
		bad_kw = ("rain", "ice", "coast", "land", "wind_retrieval")
		fv = fl.values.astype(np.int64)
		for m, n in zip(np.atleast_1d(masks), names):
			if any(k in n.lower() for k in bad_kw):
				good &= (fv & int(m)) == 0
	return good
 
 
def _granule_geo(ds):
	"""(lon[-180..180], lat, mid-time) for a swath granule."""
	lat = ds["lat"].values
	lon = ds["lon"].values
	lon = np.where(lon > 180, lon - 360, lon)
	tvals = pd.to_datetime(np.ravel(ds["time"].values))
	return lon, lat, tvals[tvals.size // 2]
 
 
def qscat_pass_metrics(track, qdir=QSCAT_DIR):
	rows = []
	t0, t1 = pd.Timestamp(T0), pd.Timestamp(T1)
	for f in sorted(glob.glob(os.path.join(qdir, "*.nc"))):
		try:
			ds = xr.open_dataset(f)
		except Exception as e:
			warnings.warn(f"skip {f}: {e}")
			continue
		lon, lat, tmid = _granule_geo(ds)
		if not (t0 - pd.Timedelta("3h") <= tmid <= t1 + pd.Timedelta("3h")):
			ds.close()
			continue
		spd = ds["retrieved_wind_speed"].values
		# (direction is oceanographic "toward", CW from N; if you need u,v:
		#  u = spd*sin(rad(dir)); v = spd*cos(rad(dir)) )
 
		c = interp_track(track, tmid)
		dx, dy = local_xy_km(lon, lat, c["lon"], c["lat"])
		r_km   = np.hypot(dx, dy)
		near   = r_km <= SEARCH_RADIUS_KM
		if near.sum() < 30 or (r_km <= 100).sum() < 5:
			ds.close()
			continue                                   # swath misses the core
 
		good = _qc_mask(ds, spd)
		qV, qR, qVa, qRa = storm_metrics(np.where(good, spd, np.nan), r_km)
		rV_raw = np.nanmax(np.where(near, spd, np.nan))  # unflagged (context)
 
		rows.append(dict(time=tmid, file=os.path.basename(f),
						 vmax=qV, rmax=qRa if np.isfinite(qRa) else qR,
						 vmax_unflagged=rV_raw,
						 n_core=int((r_km <= 100).sum())))
		ds.close()
	out = pd.DataFrame(rows)
	return out.set_index("time").sort_index() if len(out) else out
 
 
# ----------------------------------------------------------------------------
# 6. STRESS: QUIKSCAT ESDR L2 PASSES + ROMS BULK SERIES  (panel d)
# ----------------------------------------------------------------------------
STRESS_MAG_CANDS = ["stress_magnitude", "wind_stress_magnitude",
					"wind_stress", "stress", "tau"]
STRESS_X_CANDS   = ["eastward_stress", "surface_downward_eastward_stress",
					"stress_u", "taux", "tau_x"]
STRESS_Y_CANDS   = ["northward_stress", "surface_downward_northward_stress",
					"stress_v", "tauy", "tau_y"]
 
 
def esdr_pass_metrics(track, qdir=QSCAT_ESDR_DIR):
	"""
	tau_max per overpass from the ESDR L2 wind-stress swaths (N m-2).
	Variable names are sniffed against common candidates; if the guesses
	miss, run `print(xr.open_dataset(f).data_vars)` on one granule and add
	the correct names to STRESS_*_CANDS above.
	"""
	rows = []
	t0, t1 = pd.Timestamp(T0), pd.Timestamp(T1)
	for f in sorted(glob.glob(os.path.join(qdir, "*.nc"))):
		try:
			ds = xr.open_dataset(f)
		except Exception as e:
			warnings.warn(f"skip {f}: {e}")
			continue
		vmag = _find_var(ds, STRESS_MAG_CANDS)
		if vmag is not None:
			tau = ds[vmag].values
		else:
			vx = _find_var(ds, STRESS_X_CANDS)
			vy = _find_var(ds, STRESS_Y_CANDS)
			if vx is None or vy is None:
				warnings.warn(f"{os.path.basename(f)}: no stress variable "
							  f"found; vars = {list(ds.data_vars)[:14]}")
				ds.close()
				continue
			tau = np.hypot(ds[vx].values, ds[vy].values)
 
		lon, lat, tmid = _granule_geo(ds)
		if not (t0 - pd.Timedelta("3h") <= tmid <= t1 + pd.Timedelta("3h")):
			ds.close()
			continue
		c = interp_track(track, tmid)
		dx, dy = local_xy_km(lon, lat, c["lon"], c["lat"])
		r_km   = np.hypot(dx, dy)
		if (r_km <= SEARCH_RADIUS_KM).sum() < 30 or (r_km <= 100).sum() < 5:
			ds.close()
			continue
 
		good = _qc_mask(ds, tau)
		tV, tR, tVa, tRa = storm_metrics(np.where(good, tau, np.nan), r_km)
		rows.append(dict(time=tmid, file=os.path.basename(f),
						 tau_max=tV,
						 tau_rmax=tRa if np.isfinite(tRa) else tR,
						 n_core=int((r_km <= 100).sum())))
		ds.close()
	out = pd.DataFrame(rows)
	return out.set_index("time").sort_index() if len(out) else out
 
 
def roms_tau_series(track, path=ROMS_FILE):
	"""
	|tau|(t) storm-relative metrics from ROMS sustr/svstr (N m-2, on u/v
	points). Components are averaged to interior rho points. NOTE: the
	magnitude |tau| is rotation-invariant, so no grid-angle rotation is
	needed even on a curvilinear/rotated grid.
	"""
	ds = xr.open_dataset(path)
	tname = "ocean_time" if "ocean_time" in ds else "time"
	lonr = ds["lon_rho"].values[1:-1, 1:-1]
	latr = ds["lat_rho"].values[1:-1, 1:-1]
	lonr = np.where(lonr > 180, lonr - 360, lonr)
	landmask = (ds["mask_rho"].values[1:-1, 1:-1]
				if "mask_rho" in ds else np.ones_like(lonr))
 
	times = pd.date_range(T0, T1, freq=f"{DT_HOURS}h")
	rows  = []
	for t in times:
		try:
			sub = ds.sel({tname: t}, method="nearest",
						 tolerance=pd.Timedelta(f"{DT_HOURS // 2}h"))
		except KeyError:
			continue
		su = sub["sustr"].values                # (eta_rho,   xi_rho-1)
		sv = sub["svstr"].values                # (eta_rho-1, xi_rho  )
		taux = 0.5 * (su[1:-1, :-1] + su[1:-1, 1:])   # -> interior rho
		tauy = 0.5 * (sv[:-1, 1:-1] + sv[1:, 1:-1])   # -> interior rho
		tau  = np.hypot(taux, tauy)
		tau  = np.where(landmask == 0, np.nan, tau)
 
		c = interp_track(track, t)
		dx, dy = local_xy_km(lonr, latr, c["lon"], c["lat"])
		tV, tR, tVa, tRa = storm_metrics(tau, np.hypot(dx, dy))
		rows.append(dict(time=t, tau_max=tV,
						 tau_rmax=tRa if np.isfinite(tRa) else tR))
	ds.close()
	return pd.DataFrame(rows).set_index("time")
 
 
# ----------------------------------------------------------------------------
# 7. OPTIONAL: tune Holland B against a QuikSCAT azimuthal-mean profile
# ----------------------------------------------------------------------------
def fit_holland_B(spd, r_km, vmax, rmax_km, clat,
				  fit_range_km=(None, 200.0),
				  Bgrid=np.arange(0.8, 2.51, 0.02)):
	"""Least-squares B against the QC'd azimuthal-mean profile outside RMW."""
	rc, prof = radial_profile(spd, r_km)
	lo = fit_range_km[0] or rmax_km
	m  = np.isfinite(prof) & (rc >= lo) & (rc <= fit_range_km[1])
	if m.sum() < 4:
		return np.nan
	f = 2.0 * OMEGA * np.sin(np.deg2rad(clat))
	costs = [np.nanmean((holland_V(rc[m] * 1e3, vmax, rmax_km * 1e3, B, f)
						 - prof[m]) ** 2) for B in Bgrid]
	return float(Bgrid[int(np.argmin(costs))])
 
 
# ----------------------------------------------------------------------------
# 8. PLOTTING
# ----------------------------------------------------------------------------
def _make_map_axes(fig, spec):
	"""GeoAxes if cartopy is available, plain axes otherwise."""
	try:
		import cartopy.crs as ccrs
		import cartopy.feature as cfeature
		proj = ccrs.PlateCarree()
		ax = fig.add_subplot(spec, projection=proj)
		ax.add_feature(cfeature.LAND, facecolor="0.88", zorder=1)
		ax.coastlines("50m", lw=0.6, zorder=4)
		gl = ax.gridlines(draw_labels=True, lw=0.3, alpha=0.4)
		gl.top_labels = gl.right_labels = False
		gl.xlabel_style = {"size": 7}
		gl.ylabel_style = {"size": 7}
		ax.set_extent(MAP_EXTENT, crs=proj)
		return ax, proj
	except ImportError:
		ax = fig.add_subplot(spec)
		midlat = 0.5 * (MAP_EXTENT[2] + MAP_EXTENT[3])
		ax.set_aspect(1.0 / np.cos(np.deg2rad(midlat)))
		ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
		ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
		ax.tick_params(labelsize=7)
		return ax, None
 
 
def plot_swath_map(track, qscat=None, granule=None, ax=None, proj=None,
				   save=None):
	"""
	Fig. 1a: QuikSCAT swath wind speed for one pass + best track.
	QC-rejected (mostly rain-flagged eyewall) cells are shown in grey.
	  granule : path to a specific L2B .nc file; if None, the pass with the
				best core coverage in `qscat` (from qscat_pass_metrics).
	  ax/proj : pass both to draw into a composed figure.
	"""
	if granule is None:
		if qscat is None or not len(qscat):
			raise ValueError("provide qscat=qscat_pass_metrics(track) "
							 "or granule=<path>")
		granule = os.path.join(QSCAT_DIR,
							   qscat.loc[qscat.n_core.idxmax(), "file"])
	ds = xr.open_dataset(granule)
	lon, lat, tmid = _granule_geo(ds)
	spd  = ds["retrieved_wind_speed"].values
	good = _qc_mask(ds, spd)
 
	created = ax is None
	if created:
		fig = plt.figure(figsize=(3.8, 3.6))
		ax, proj = _make_map_axes(fig, 111)
	else:
		fig = ax.figure
	kw = {"transform": proj} if proj is not None else {}
 
	dom = ((lon > MAP_EXTENT[0]) & (lon < MAP_EXTENT[1]) &
		   (lat > MAP_EXTENT[2]) & (lat < MAP_EXTENT[3]))
	ax.scatter(lon[dom & ~good], lat[dom & ~good], c="0.78", s=2.5,
			   lw=0, zorder=2, **kw)
	sc = ax.scatter(lon[dom & good], lat[dom & good], c=spd[dom & good],
					s=2.5, lw=0, cmap="viridis", vmin=0, vmax=35,
					zorder=3, **kw)
	cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.03)
	cb.set_label(r"QuikSCAT 10-m wind (m s$^{-1}$)", fontsize=7)
	cb.ax.tick_params(labelsize=6)
 
	# best track: line + daily 00Z dots + day labels + centre at pass time
	bt = track.loc["2004-03-20":T1]
	ax.plot(bt.lon, bt.lat, "k-", lw=1.2, zorder=5, **kw)
	daily = bt[bt.index.hour == 0]
	ax.plot(daily.lon, daily.lat, "ko", ms=3, zorder=6, ls="none", **kw)
	for tt, row in daily.iterrows():
		ax.text(row.lon + 0.25, row.lat + 0.15, tt.strftime("%d"),
				fontsize=6, zorder=7, **kw)
	c = interp_track(track, tmid)
	ax.plot(c["lon"], c["lat"], marker="*", ms=12, mfc="crimson",
			mec="k", mew=0.5, ls="none", zorder=8, **kw)
	ax.set_title(f"QuikSCAT pass {tmid:%d %b %Y %H:%M} UTC", fontsize=8)
	ds.close()
	if created and save:
		for ext in ("pdf", "png"):
			fig.savefig(f"{save}.{ext}", dpi=300, bbox_inches="tight")
	return ax
 
 
def _timeseries_panels(axes, track, model, qscat, roms_tau=None, esdr=None):
	"""Draw the Vmax / Rmax / (optional) tau panels into a list of axes."""
	ax1, ax2 = axes[0], axes[1]
 
	# ---- Vmax panel
	bt = track.loc[T0:T1]
	ax1.plot(bt.index, bt.vmax * WIND_AVG_FACTOR, "k-", lw=1.6,
			 label="best track")
	ax1.plot(model.index, model.era5_vmax, "--", color="steelblue",
			 lw=1.3, label="ERA5")
	ax1.plot(model.index, model.blend_vmax, "-", color="crimson",
			 lw=1.3, label="blended")
	if qscat is not None and len(qscat):
		ax1.plot(qscat.index, qscat.vmax, "^", color="seagreen", ms=6,
				 mec="k", mew=0.4, ls="none", label="QuikSCAT")
		ax1.plot(qscat.index, qscat.vmax_unflagged, "^", color="seagreen",
				 ms=6, alpha=0.3, ls="none")      # unflagged (context only)
	ax1.set_ylabel(r"$V_{\max}$ (m s$^{-1}$)", fontsize=8)
	ax1.legend(fontsize=6, frameon=False, ncol=2, loc="upper left")
 
	# ---- Rmax panel
	if RMW_BT_KM is not None:
		ax2.axhline(RMW_BT_KM, color="k", lw=1.2, ls=":")
	ax2.plot(model.index, model.era5_rmax, "--", color="steelblue", lw=1.3)
	ax2.plot(model.index, model.blend_rmax, "-", color="crimson", lw=1.3)
	if qscat is not None and len(qscat):
		ax2.plot(qscat.index, qscat.rmax, "^", color="seagreen", ms=6,
				 mec="k", mew=0.4, ls="none")
	ax2.set_ylabel(r"$R_{\max}$ (km)", fontsize=8)
	ax2.set_ylim(0, 160)
 
	# ---- tau panel (optional)
	if len(axes) > 2:
		ax3 = axes[2]
		if roms_tau is not None and len(roms_tau):
			ax3.plot(roms_tau.index, roms_tau.tau_max, "-",
					 color="darkorange", lw=1.3, label="ROMS (bulk)")
		if esdr is not None and len(esdr):
			ax3.plot(esdr.index, esdr.tau_max, "s", color="seagreen",
					 ms=5, mec="k", mew=0.4, ls="none",
					 label="QuikSCAT ESDR")
		ax3.set_ylabel(r"$\tau_{\max}$ (N m$^{-2}$)", fontsize=8)
		ax3.legend(fontsize=6, frameon=False, loc="upper left")
 
	loc = mdates.DayLocator()
	t0, t1 = pd.Timestamp(T0), pd.Timestamp(T1)
	for i, ax in enumerate(axes):
		ax.set_xlim(t0, t1)
		ax.tick_params(labelsize=7)
		ax.grid(alpha=0.25, lw=0.4)
		ax.xaxis.set_major_locator(loc)
		if i < len(axes) - 1:
			ax.tick_params(labelbottom=False)
		else:
			ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
 
 
def plot_inset(track, model, qscat, roms_tau=None, esdr=None, save=OUT_INSET):
	"""Standalone inset (2 panels; 3 with the stress comparison)."""
	n = 2 + int((roms_tau is not None and len(roms_tau)) or
				(esdr is not None and len(esdr)))
	fig, axes = plt.subplots(n, 1, figsize=(3.4, 1.85 * n + 0.3),
							 constrained_layout=True)
	_timeseries_panels(list(np.atleast_1d(axes)), track, model, qscat,
					   roms_tau, esdr)
	for ext in ("pdf", "png"):
		fig.savefig(f"{save}.{ext}", dpi=300)
	print(f"saved {save}.pdf/.png")
	return fig
 
 
def plot_fig1(track, model, qscat, roms_tau=None, esdr=None, granule=None,
			  save=OUT_FIG1):
	"""Composed Figure 1: (a) swath map + (b,c[,d]) time-series panels."""
	n = 2 + int((roms_tau is not None and len(roms_tau)) or
				(esdr is not None and len(esdr)))
	fig = plt.figure(figsize=(7.2, 3.7 if n == 2 else 4.6))
	gs  = fig.add_gridspec(n, 2, width_ratios=[1.2, 1.0],
						   hspace=0.14, wspace=0.30)
	ax_map, proj = _make_map_axes(fig, gs[:, 0])
	plot_swath_map(track, qscat=qscat, granule=granule, ax=ax_map, proj=proj)
	ts_axes = [fig.add_subplot(gs[i, 1]) for i in range(n)]
	_timeseries_panels(ts_axes, track, model, qscat, roms_tau, esdr)
	for lab, ax in zip("abcd", [ax_map] + ts_axes):
		ax.text(0.02, 0.97, f"({lab})", transform=ax.transAxes,
				fontsize=9, fontweight="bold", va="top", zorder=20)
	for ext in ("pdf", "png"):
		fig.savefig(f"{save}.{ext}", dpi=300, bbox_inches="tight")
	print(f"saved {save}.pdf/.png")
	return fig
 
 
# ----------------------------------------------------------------------------
if __name__ == "__main__":
	# download_quikscat(); download_qscat_esdr()        # <- run once
	track = get_track()
	model = era5_and_blend_series(track)
	qscat = qscat_pass_metrics(track)
	print(model.round(1))
	print(qscat.round(1) if len(qscat) else "no usable QuikSCAT wind passes")
 
	# ---- stress comparison inputs (auto-skip when data are not there yet)
	roms_tau = roms_tau_series(track) if os.path.exists(ROMS_FILE) else None
	esdr = (esdr_pass_metrics(track)
			if os.path.isdir(QSCAT_ESDR_DIR) else None)
	if esdr is not None and not len(esdr):
		esdr = None
	if esdr is not None:
		print(esdr.round(2))
 
	plot_inset(track, model, qscat, roms_tau, esdr)     # inset alone
	plot_fig1(track, model, qscat, roms_tau, esdr)      # (a) map + (b)-(d)
 
