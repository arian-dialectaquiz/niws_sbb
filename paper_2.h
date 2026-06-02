/*
** svn $Id: upwelling.h 1001 2020-01-10 22:41:16Z arango $
*******************************************************************************
** Copyright (c) 2002-2020 The ROMS/TOMS Group                               **
**   Licensed under a MIT/X style license                                    **
**   See License_ROMS.txt                                                    **
*******************************************************************************
**
** Options for Upwelling Test.
**
** Application flag:   PAPAER_2
** Input script:       
*/

#define UV_ADV		/*to turn ON advection terms */

#define UV_COR		/*to turn ON Coriolis term  */
#define UV_VIS2	/*to turn ON harmonic horizontal mixing Laplacian viscosity*/
#define UV_SMAGORINSKY	/* to turn ON Smagorinsky-like viscosity*/

#define VISC_GRID /* to scale viscosity coefficient by grid size*/
#define SPLINES_VVISC	/*splines reconstruction of vertical viscosity*/

#define UV_QDRAG /* coeficient of quadritic bottom friction*/
#undef UV_LOGDRAG /* logaritimic bottom drag */


#define DIFF_GRID /*to scale diffusion coefficients by grid size */


#define  MIX_GEO_UV /*if mixing on geopotential (constant Z) surfaces*/
#define MIX_GEO_TS /* if mixing on geopotential (constant Z) surfaces*/


#define TS_DIF2	/*to turn ON or OFF harmonic horizontal mixing Laplacian diffusion*/
#undef  TS_DIF4	/*to turn ON or OFF biharmonic horizontal mixing*/
#define TS_SMAGORINSKY	/* to turn ON Smagorinsky-like diffusion */ 


#define SALINITY	/*if having salinity*/
#define DJ_GRADPS	/*if splines density Jacobian (Shchepetkin, 2000)*/
#define SOLVE3D	/*if solving 3D primitive equations*/
#define MASKING	/*if land/sea masking */
#define DIAGNOSTICS_UV	/*if writing out momentum diagnostics*/
#define DIAGNOSTICS_TS	/*if writing out tracer diagnostics*/

#define SSH_TIDES   /*if imposing tidal elevation*/
#define UV_TIDES	/*if imposing tidal currents*/
#define RAMP_TIDES	/*if ramping (over one day) tidal forcing*/

#define WET_DRY	/*to activate wetting and drying*/
#define MASKING	/*if land/sea masking */
#define AVERAGES	/*if writing out NLM time-averaged data*/

/*analitical VERTICAL BOUNDARIES FLUXES*/
#define ANA_BTFLUX
#define ANA_BSFLUX

/* lateral BOUNDARIES FLUXES*/
#define RADIATION_2D

/* surface FLUXES*/
#define LONGWAVE_OUT
#define ANA_SRFLUX /* if analytical surface shortwave radiation flux  */
#define ANA_STFLUX /* if analytical surface net heat flux */

/*ana test to the grid */

#undef ANA_FSOBC
#undef ANA_INITIAL
#undef ANA_M2OBC
#undef ANA_M3OBC






/* mixing */

#define MY25_MIXING	/*Mellor/Yamada Level-2.5 closure*/
#if defined GLS_MIXING || defined MY25_MIXING
# define KANTHA_CLAYSON	/*Kantha and Clayson stability function*/
# define N2S2_HORAVG		/*horizontal smoothing of buoyancy/shear*/
# define RI_SPLINES		/*splines reconstruction for vertical sheer*/
#else
# define ANA_VMIX
#endif
