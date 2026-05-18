
##################-----> Plotting <------###################################
colors = ['Orange', 'fuchsia', 'Lime', 'dodgerblue']  
labels = ['A', 'B', 'C','D']

name = f"fig_wm_comps.png"  # Create a unique filename

fig = plt.figure(figsize=(7, 6))
gs = gridspec.GridSpec(nrows=3, ncols=4, width_ratios=[33,33,33,1], height_ratios=[1,1,1])
gs.update(left=0.1, right=0.92, wspace=0.1, hspace=0.1, top=0.96, bottom=0.07)


#----> maps horizontal
vmin = -1
vmax = 1
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
bar_title = r'x10$^{-3}$ W.m$^{-2}$'
cmap = plt.cm.bwr_r

cbar_0 = plt.subplot(gs[0, 3])
cb_0 = mpl.colorbar.ColorbarBase(cbar_0, cmap=cmap, norm=norm, extend='both', orientation='vertical')
cb_0.set_label(bar_title, size='x-small', labelpad=5)
cbar_0.yaxis.set_ticks_position('left')  # Ticks on the left side
cbar_0.tick_params(axis='y', labelsize='x-small', rotation=25)

######----> Normal
ax = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax.set_title('Normal', loc='left', fontsize=10)
ax.text(0.03, 0.9, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
# Define colormap and normalization
ax.text(-0.25, 0.5, 'Horizontal', transform=ax.transAxes, 
		rotation='vertical', va='center', ha='right', fontsize=10, fontweight='bold')

ax.set_ylim(bottom=-32.6, top=-22)
ax.set_xlim(left=-51, right=-39.2)
# Contourf with normalization
ax.contourf(lon_rho, lat_rho, hor_normal*100, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Depth contours
levels = [50, 200, 1000, 2000]
styles = ['dotted', 'dotted', 'dashed', 'solid']
colors = ['gray', 'silver', 'dimgrey', 'gray']
for l, s, c in zip(levels, styles, colors):
	ax.contour(h.lon_rho, h.lat_rho, h, levels=[l], zorder=3, colors=c, linestyles=s, linewidths=1)

# --- NEW LEGEND LOCATION (Axis Coordinates) ---
legend_lines = [Line2D([0], [0], linestyle=s, linewidth=1, color=c) for s, c in zip(styles, colors)]
labels_iso = ['50 m', '200 m', '1000 m', '2000 m']

# This places the legend inside the axes at the lower right
ax.legend(
	legend_lines,
	labels_iso,
	title='Isobaths',
	fontsize='xx-small',
	title_fontsize='xx-small',
	loc='lower right',           # Automatically anchors to bottom-right of AXIS
	framealpha=0.9,              # Opaque box so lines don't clash with map
	edgecolor='gray'
)
# Coastlines and gridlines
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax.patch.set_edgecolor('black')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = False
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}


######----> CF
ax1 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax1.set_title('Cold Fronts', loc='left', fontsize=10)
ax1.text(0.03, 0.9, '(b)', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
# Define colormap and normalization

ax1.set_ylim(bottom=-32.6, top=-22)
ax1.set_xlim(left=-51, right=-39.2)
# Contourf with normalization
ax1.contourf(lon_rho, lat_rho, hor_cf*100, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax1.contour(h.lon_rho, h.lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax1.contour(h.lon_rho, h.lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax1.contour(h.lon_rho, h.lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax1.contour(h.lon_rho, h.lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]

# Coastlines and gridlines
ax1.coastlines()
ax1.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax1.patch.set_edgecolor('black')
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = False
gl.right_labels = False
gl.bottom_labels = False
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}

######----> Hurricane
ax2 = plt.subplot(gs[0, 2], projection=ccrs.PlateCarree())
ax2.set_title('Hurricane', loc='left', fontsize=10)
ax2.text(0.03, 0.9, '(c)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
# Define colormap and normalization

ax2.set_ylim(bottom=-32.6, top=-22)
ax2.set_xlim(left=-51, right=-39.2)
# Contourf with normalization
ax2.contourf(lon_rho, lat_rho, hor_hurricane*100, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax2.contour(h.lon_rho, h.lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax2.contour(h.lon_rho, h.lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax2.contour(h.lon_rho, h.lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax2.contour(h.lon_rho, h.lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)

# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]

# Coastlines and gridlines
ax2.coastlines()
ax2.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax2.patch.set_edgecolor('black')
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = False
gl.right_labels = False
gl.bottom_labels = False
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}



#----> maps vert
vmin = -0.5
vmax = 0.5
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
bar_title = r'x10$^{-3}$ W.m$^{-2}$'
cmap = plt.cm.bwr_r

cbar = plt.subplot(gs[1, 3])
cb = mpl.colorbar.ColorbarBase(cbar, cmap=cmap, norm=norm, extend='both', orientation='vertical')
cb.set_label(bar_title, size='x-small', labelpad=5)
cbar.yaxis.set_ticks_position('left')  # Ticks on the left side
cbar.tick_params(axis='y', labelsize='x-small', rotation=25)

######----> Normal
ax = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax.text(0.03, 0.9, '(d)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
# Define colormap and normalization
ax.text(-0.25, 0.5, 'Vertical', transform=ax.transAxes, 
		rotation='vertical', va='center', ha='right', fontsize=10, fontweight='bold')
ax.set_ylim(bottom=-32.6, top=-22)
ax.set_xlim(left=-51, right=-39.2)
# Contourf with normalization
ax.contourf(lon_rho, lat_rho, vert_normal*100, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Depth contours
levels = [50, 200, 1000, 2000]
styles = ['dotted', 'dotted', 'dashed', 'solid']
colors = ['gray', 'silver', 'dimgrey', 'gray']
for l, s, c in zip(levels, styles, colors):
	ax.contour(h.lon_rho, h.lat_rho, h, levels=[l], zorder=3, colors=c, linestyles=s, linewidths=1)

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




######----> CF
ax1 = plt.subplot(gs[1, 1], projection=ccrs.PlateCarree())
ax1.text(0.03, 0.9, '(e)', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
# Define colormap and normalization

ax1.set_ylim(bottom=-32.6, top=-22)
ax1.set_xlim(left=-51, right=-39.2)
# Contourf with normalization
ax1.contourf(lon_rho, lat_rho, vert_cf*100, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax1.contour(h.lon_rho, h.lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax1.contour(h.lon_rho, h.lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax1.contour(h.lon_rho, h.lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax1.contour(h.lon_rho, h.lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]

# Coastlines and gridlines
ax1.coastlines()
ax1.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax1.patch.set_edgecolor('black')
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = False
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}


######----> Hurricane
ax2 = plt.subplot(gs[1, 2], projection=ccrs.PlateCarree())
ax2.text(0.03, 0.9, '(f)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
# Define colormap and normalization

ax2.set_ylim(bottom=-32.6, top=-22)
ax2.set_xlim(left=-51, right=-39.2)
# Contourf with normalization
ax2.contourf(lon_rho, lat_rho, vert_hurricane*100, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax2.contour(h.lon_rho, h.lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax2.contour(h.lon_rho, h.lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax2.contour(h.lon_rho, h.lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax2.contour(h.lon_rho, h.lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)

colors_c = ['Orange', 'cyan', 'Lime', 'grey']  
labels_c = ['A', 'B', 'C', 'D']

ax2.scatter(h.lon_rho[A[1],:], h.lat_rho[A[1],:],c=colors_c[0],s=0.05,marker='.')
ax2.scatter(h.lon_rho[B[1],:], h.lat_rho[B[1],:],c=colors_c[1],s=0.05,marker='.')
ax2.scatter(h.lon_rho[C[1],:], h.lat_rho[C[1],:],c=colors_c[2],s=0.05,marker='.')
ax2.scatter(h.lon_rho[D[1],:], h.lat_rho[D[1],:],c=colors_c[3],s=0.05,marker='.')

ax2.text(h.lon_rho[A[1], 609]+0.2, h.lat_rho[A[1], 609]-1, 'A', color=colors_c[0],zorder=5, fontsize='x-small', verticalalignment='bottom')
ax2.text(h.lon_rho[[B[1]], 609]+0.2, h.lat_rho[[B[1]], 609]-1, 'B', color=colors_c[1], zorder=5,fontsize='x-small', verticalalignment='bottom')
ax2.text(h.lon_rho[C[1], 609]+0.2, h.lat_rho[C[1], 609]-1, 'C', color=colors_c[2], zorder=5,fontsize='x-small', verticalalignment='bottom')
ax2.text(h.lon_rho[D[1], 609]+0.2, h.lat_rho[D[1], 609]-1, 'D', color=colors_c[3], zorder=5,fontsize='x-small', verticalalignment='bottom')

# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]

# Coastlines and gridlines
ax2.coastlines()
ax2.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax2.patch.set_edgecolor('black')
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = False
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}

km_cross = np.arange(0,608,1)

#################################################################################
#######-----> Cross-sections <----####################

cmap_ke = plt.cm.bwr_r

vmin_ke = -3
vmax_ke = 3
norm_ke = mpl.colors.Normalize(vmin=vmin_ke, vmax=vmax_ke)
bar_title_ke = r'x10$^{-5}$ W.m$^{-3}$'

cbar_ke = plt.subplot(gs[2, 3])
cbke = mpl.colorbar.ColorbarBase(cbar_ke, cmap=cmap_ke, norm=norm_ke, extend='both', orientation='vertical')
cbke.set_label(bar_title_ke, size='x-small', labelpad=5)
cbar_ke.yaxis.set_ticks_position('left')  # Ticks on the left side
cbar_ke.tick_params(axis='y', labelsize='x-small', rotation=25)

#---> A
ax2_A = plt.subplot(gs[2, 0])
for i in range(len(km_cross)):
	ax2_A.contourf(km_cross, zcross_A[:,i], vert_A*1e5, levels=100,cmap=cmap_ke, norm=norm_ke)

ax2_A.fill_between(km_cross, np.min(zcross_A, axis=0), y2=ax2_A.get_ylim()[0], color='grey', zorder = 2)

ax2_A.set_xlabel('km from coast',size='x-small')
ax2_A.set_ylabel('m',size='x-small')
ax2_A.tick_params(axis='both', labelsize='x-small')
ax2_A.set_ylim(-600, 0) 
ax2_A.set_xlim(230, 609) 
ax2_A.text(0.02, 0.05, '(g)', transform=ax2_A.transAxes, fontsize=8, fontweight='bold')

#---> B
ax2_B = plt.subplot(gs[2, 1])
for i in range(len(km_cross)):
	ax2_B.contourf(km_cross, zcross_B[:,i], vert_B*1e5, levels=100,cmap=cmap_ke, norm=norm_ke)

ax2_B.fill_between(km_cross, np.min(zcross_B, axis=0), y2=ax2_B.get_ylim()[0], color='grey', zorder = 2)
ax2_B.set_yticks([])
ax2_B.set_ylabel('')
ax2_B.set_xlabel('km from coast',size='x-small')
ax2_B.tick_params(axis='both', labelsize='x-small')
ax2_B.set_ylim(-600, 0) 
ax2_B.set_xlim(70, 609) 
ax2_B.text(0.02, 0.05, '(h)', transform=ax2_B.transAxes, fontsize=8, fontweight='bold')

#---> D
ax2_C = plt.subplot(gs[2, 2])
for i in range(len(km_cross)):
	ax2_C.contourf(km_cross, zcross_D[:,i], vert_D*1e5, levels=100,cmap=cmap_ke, norm=norm_ke)

ax2_C.fill_between(km_cross, np.min(zcross_D, axis=0), y2=ax2_C.get_ylim()[0], color='grey', zorder = 2)
ax2_C.set_yticks([])
ax2_C.set_ylabel('')
ax2_C.set_xlabel('km from coast',size='x-small')
ax2_C.tick_params(axis='both', labelsize='x-small')
ax2_C.set_ylim(-600, 0) 
ax2_C.set_xlim(250, 609) 
ax2_C.text(0.02, 0.05, '(i)', transform=ax2_C.transAxes, fontsize=8, fontweight='bold')

plt.savefig(name,dpi=300)