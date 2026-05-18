######################################################################################################################
################### ----- Plots -----############################


fig = plt.figure(figsize=(6, 7))
gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[50,50], height_ratios=[1,1,1,1])
gs.update(left=0.08, right=0.96, wspace=0.22, hspace=0.2, top=0.98, bottom=0.08)

####################-----> Pot energy at MLD MAP
ax = plt.subplot(gs[0:2, 0], projection=ccrs.PlateCarree())
# Define colormap and normalization
cmap = plt.cm.bwr
#cmap = cmo.cm.balance
vmin = -3
vmax = 3
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
bar_title = r'W.m$^{-2}$'
ax.set_ylim(bottom=-30.5, top=-22)
ax.set_xlim(left=-49.2, right=-40.9)
# Contourf with normalization
ax.contourf(z.lon_rho, z.lat_rho, EP_at_MLD*1e3, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)

ax.scatter(z.lon_rho[905,380].values.mean(), z.lat_rho[905,380].values.mean(), zorder=5, s=30, marker='*', color='green', label='A')
ax.scatter(z.lon_rho[500,238].values.mean() , z.lat_rho[500,238].values.mean(), zorder=5, s=30, marker='+', color='k', label='B')
ax.scatter(z.lon_rho[280,221].values.mean(), z.lat_rho[280,221].values.mean(), zorder=5, s=30, marker='s', color='orange', label='C')
ax.scatter(z.lon_rho[120,275].values.mean(), z.lat_rho[120,275].values.mean(), zorder=5, s=30, marker='D', color='cyan', label='D')

legend = ax.legend(loc=4, fontsize='xx-small', markerscale=0.7)

# Depth contour levels
levels_1 = [-50]
levels_2 = [-200]
levels_3 = [-1000]
levels_4 = [-2000]
# Contour lines for isobaths
ax.contour(lon_rho, lat_rho, z, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax.contour(lon_rho, lat_rho, z, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax.contour(lon_rho, lat_rho, z, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax.contour(lon_rho, lat_rho, z, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]
labels = ['50 m', '200 m', '1000 m', '2000 m']

fig.legend(
	legend_lines,
	labels,
	title='Isobaths',
	fontsize='xx-small',
	title_fontsize='xx-small',
	loc='center',
	bbox_to_anchor=(0.42, 0.72)
)

# Coastlines and gridlines
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax.patch.set_edgecolor('black')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}


ax.text(0.94, 0.97, 'a', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left',
		 bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgray'))
# Colorbar
cbar_1 = inset_axes(ax, width="50%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb1.set_label(bar_title, size='x-small')
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)
cb1.set_label(fr'{bar_title} $\times$ 10$^{{-3}}$', size='x-small', labelpad=2)

###############---> contourf per time and z
##_A
cmap_N = plt.cm.bwr
vmin_N = -0.5
vmax_N = 0.5
norm_N = mpl.colors.Normalize(vmin=vmin_N, vmax=vmax_N)
bar_title = r'W.m$^{-3}$'


axa = plt.subplot(gs[0, 1])
axa.set_ylim(-250, 0) 
axa.contourf(time, zcross_A[:,380], EP_A_p.T*1e3, cmap=cmap_N, levels=50,norm=norm_N)
axa.plot(time,-1*ml_A_p,lw=1,ls=':',color='k', label='Mixed Layer')
axa.legend(loc=4,fontsize='x-small')
axa.set_xlabel(None,size='x-small')
axa.set_ylabel('m',size='x-small')
axa.tick_params(axis='both', labelsize='x-small')
axa.set_ylim(-200, 0) 
#ax.tick_params(axis='x', labelsize='x-small', rotation=25)
#ax.tick_params(axis='y', labelsize='x-small')
axa.set_xticks([])  # Remove x-axis ticks

axa.text(0.02, 0.95, 'c', transform=axa.transAxes, fontsize='x-small', fontweight='demibold', va='top', ha='left',
		 bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgray'))
# Colorbar
cbar_3 = axa.inset_axes([0.035, 0.14, 0.4, 0.06])
cbar_3.set_facecolor('lightgrey')
cb3 = mpl.colorbar.ColorbarBase(cbar_3, cmap=cmap_N, norm=norm_N, extend='both', orientation='horizontal')
cb3.set_label(bar_title, size='x-small')
cbar_3.xaxis.set_ticks_position('bottom')
cbar_3.tick_params(axis='x', labelsize='x-small')
#cb3.set_label(fr'{bar_title} $\times$ 10$^{{-3}}$', size='x-small', labelpad=2)
cb3.set_label(fr'{bar_title} $\times$ 10$^{{-3}}$', size='x-small', labelpad=2)

##_B
axb = plt.subplot(gs[1, 1])
axb.set_ylim(-250, 0) 
axb.contourf(time, zcross_B[:,238], EP_B_p.T*1e3, cmap=cmap_N, levels=50,norm=norm_N)
axb.plot(time,-1*ml_B_p,lw=1,ls=':',color='k')

axb.set_xlabel(None,size='x-small')
axb.set_ylabel('m',size='x-small')
axb.tick_params(axis='both', labelsize='x-small')
axb.set_ylim(-200, 0) 
#ax.tick_params(axis='x', labelsize='x-small', rotation=25)
#ax.tick_params(axis='y', labelsize='x-small')
axb.set_xticks([])  # Remove x-axis ticks

axb.text(0.02, 0.95, 'd', transform=axb.transAxes, fontsize='x-small', fontweight='demibold', va='top', ha='left',
		 bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgray'))
##_C
axc = plt.subplot(gs[2, 1])
axc.set_ylim(-250, 0) 
axc.contourf(time, zcross_C[:,221], EP_C_p.T*1e3, cmap=cmap_N, levels=50,norm=norm_N)
axc.plot(time,-1*ml_C_p,lw=1,ls=':',color='k')

axc.set_xlabel(None,size='x-small')
axc.set_ylabel('m',size='x-small')
axc.tick_params(axis='both', labelsize='x-small')
axc.set_ylim(-200, 0) 
#ax.tick_params(axis='x', labelsize='x-small', rotation=25)
#ax.tick_params(axis='y', labelsize='x-small')
axc.set_xticks([])  # Remove x-axis ticks

axc.text(0.02, 0.95, 'e', transform=axc.transAxes, fontsize='x-small', fontweight='demibold', va='top', ha='left',
		 bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgray'))

##_D
axd = plt.subplot(gs[3, 1])
axd.set_ylim(-250, 0) 
axd.contourf(time, zcross_D[:,275], EP_D_p.T*1e3, cmap=cmap_N, levels=50,norm=norm_N)
axd.plot(time,-1*ml_D_p,lw=1,ls=':',color='k')

axd.set_xlabel(None,size='x-small')
axd.set_ylabel('m',size='x-small')
axd.tick_params(axis='both', labelsize='x-small')
axd.set_ylim(-180, 0) 
axd.xaxis.set_major_locator(mdates.DayLocator(interval=12))
axd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.setp(axd.get_xticklabels(), rotation=15, ha='center', fontsize='x-small')

#axd.tick_params(axis='x', labelsize='x-small', rotation=25)
axd.tick_params(axis='y', labelsize='x-small')
#axd.set_xticks([])  # Remove x-axis ticks

axd.text(0.02, 0.95, 'f', transform=axd.transAxes, fontsize='x-small', fontweight='demibold', va='top', ha='left',
		 bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgray'))

###########-----> Ratios at MLD

axr = plt.subplot(gs[2:4, 0], projection=ccrs.PlateCarree())
# Define colormap and normalization
#cmap = plt.cm.bwr
cmap = cmo.cm.curl
vmin = -0.1
vmax = 0.1
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
bar_title = r'W.m$^{-2}$'
axr.set_ylim(bottom=-30.5, top=-22)
axr.set_xlim(left=-49.2, right=-40.9)
# Contourf with normalization
axr.contourf(z.lon_rho, z.lat_rho, leak_map, levels=200, cmap=cmap,extend='both', norm=norm, zorder=0)

axr.scatter(z.lon_rho[905,380].values.mean(), z.lat_rho[905,380].values.mean(), zorder=5, s=30, marker='*', color='green', label='A')
axr.scatter(z.lon_rho[500,238].values.mean() , z.lat_rho[500,238].values.mean(), zorder=5, s=30, marker='+', color='k', label='B')
axr.scatter(z.lon_rho[280,221].values.mean(), z.lat_rho[280,221].values.mean(), zorder=5, s=30, marker='s', color='orange', label='C')
axr.scatter(z.lon_rho[120,275].values.mean(), z.lat_rho[120,275].values.mean(), zorder=5, s=30, marker='D', color='cyan', label='D')

legend = axr.legend(loc=4, fontsize='xx-small', markerscale=0.7)

# Depth contour levels
levels_1 = [-50]
levels_2 = [-200]
levels_3 = [-1000]
levels_4 = [-2000]
# Contour lines for isobaths
axr.contour(lon_rho, lat_rho, z, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
axr.contour(lon_rho, lat_rho, z, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
axr.contour(lon_rho, lat_rho, z, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
axr.contour(lon_rho, lat_rho, z, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]
labels = ['50 m', '200 m', '1000 m', '2000 m']

# Coastlines and gridlines
axr.coastlines()
axr.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
axr.patch.set_edgecolor('black')
gl = axr.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}


axr.text(0.94, 0.97, 'b', transform=axr.transAxes, fontsize=10, fontweight='bold', va='top', ha='left',
		 bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgray'))
# Colorbar
cbar_r = inset_axes(axr, width="50%", height="3%", loc=2)
cbar_r.set_facecolor('lightgray')
cbr = mpl.colorbar.ColorbarBase(cbar_r, cmap=cmap, norm=norm,extend='both',  orientation='horizontal')
cbr.set_label(bar_title, size='x-small')
cbar_r.xaxis.set_ticks_position('bottom')
cbar_r.tick_params(axis='x', labelsize='x-small', rotation=25)
cbr.set_label('ratio', size='x-small', labelpad=2)


plt.savefig("MLD_fig.png", dpi=300)