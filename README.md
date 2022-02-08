# Photometric Redshifts with Pan-STARRS, (un)WISE, and SDSS

Photometric redfhit calculations using machine learning (random forest) techniques, with data from the panSTARRS and WISE surveys.

This project will include data cleaning of the unWISE and panSTARRS data, combining the data to match sources, and engineering colour features from the various filter magnitudes. These data will then be split into a training and test dataset to train, optimize and evaluate machine learning models.

# The Data
Data from unWISE  : https://catalog.unwise.me/catalogs.html.  
Data from SDSS data release 16  : LINK.  
Data from PanSTARRS : https://outerspace.stsci.edu/display/PANSTARRS/PS1+Source+extraction+and+catalogs.  

### unWISE column headers
`unwise_decl` : declination(degree)  
`unwise_primary`  : the center of this source is in the primary region of its coadd  
`unwise_ra` : right ascension (degree)  
`unwise_unwise_objid` : unique object id  
`unwise_w1/w2_decl12` : positions from individual-image catalogs  
`unwise_w1/w2_dflux` : uncertainty in flux (statistical only)  
`unwise_w1/w2_dfluxlbs`  : uncertainty in local-background-subtracted flux  
`unwise_w1/w2_dspread_model` : uncertainty in spread_model  
`unwise_w1/w2_dx`  : uncertainty in x  
`unwise_w1/w2_dy`  : uncertainty in y  
`unwise_w1/w2_flags_info`  : additional informational flags at central pixel  
`unwise_w1/w2_flags_unwise`  : unWISE Coadd Flags flags at central pixel  
`unwise_w1/w2_flux`  : flux (Vega nMgy)  
`unwise_w1/w2_fluxlbs` : local-background-subtracted flux (Vega nMgy)  
`unwise_w1/w2_fracflux`  : fraction of flux in this object's PSF that comes from this object  
`unwise_w1/w2_fwhm` : full-width at half-maximum of PSF (pixels)  
`unwise_w1/w2_nm`  : number of single-exposure images of this part of sky in coadd  
`unwise_w1/w1_primary12`  : 'primary' status from individual-image catalogs  
`unwise_w1/w2_qf` : "quality factor"  
`unwise_w1/w2_ra12` : Positions from individual-image catalogs  
`unwise_w1/w2_rchi2`  : average chi-square per pixel, weighted by PSF  
`unwise_w1/w2_sky`  : sky (Bega nMgy)  
`unwise_w1/w2_spread_model` : SExtractor spread_model parameter  
`unwise_w1/w2_unwise_detid` : unique ID  
`unwise_w1/w2_x`  : x coordinate (pix)  
`unwise_w1/w2_y`  : y coordinate (pix)  

### SDSS column headers
`z` : redshift  
`z_err` : redshift error  
`z_warning` : Bad redshift if this is non-zero -- see Schlegel data model  
`class` : Spectroscopic class (GALAXY, QSO, or STAR)  
`subclass`  : Spectroscopic subclass   
`survey`  : Survey name  
`type`  : Morphological type classification of the object  
`target_obj_id` : ID of target PhotoObj  
`spec_obj_id` : Unique ID  
`sdss_ra` : Right ascension  
`sdss_decl` : Declination

### Pan-STARRS colum headers (* indicates further info on catalog page)
`objid` : Unique object identifier  : DEFAULT=NA  
`qualityFlag` : Subset of objinfoFlag denoting whether this object is real of a likely false positive : DEFAULT=0  
`raMean`  : Right ascension from single epoch detections (weighted mean) in equinox J2000 at the mean epoch given by epochMEAN  : DEFAULT=-999    
`decMean` : Declination from single epoch detections (weighted mean) in equiniox J2000 at the mean epoch given by epochMean : DEFAULT=-999  
`raMeanErr` : Right ascension standard deviation from single epoch detections : DEFAULT=-999  
`decMeanErr`  : Declination standard deviation from single epoch detections : DEFAULT=-999  
`epochMean` : Modified Julian Date of the mean epoch corresponding to raMEAN, decMean (equinox J2000)* : DEFAULT=-999  
`nDetections` : Number of single epoch detections in all filters  : DEFAULT=-999  
`ng/r/i/z/y`  : Number of single epoch detections in g/r/i/z/y filter : DEFAULT=-999  
`primaryDetection`  : Identifies if this row is the primary stack detection*  : DEFAULT=255  
`g/r/i/z/yPSFMag` : PSF magnitude from g/r/i/z/y filter stack detection : DEFAULT=-999  
`g/r/i/z/yPSFMagErr` : Error in PSF magnitude from g/r/i/z/y filter stack detection : DEFAULT=-999  
`g/r/i/z/yKronMag` : Kron magnitude from g/r/i/z/y filter stack detection : DEFAULT=-999  
`g/r/i/z/yKronMagErr` : Kron magnitude from g/r/i/z/y filter stack detection : DEFAULT=-999  
`g/r/i/z/ymomentXX` : Second moment M_xx for g/r/i/z/y filter stack detection : DEFAULT=-999  
`g/r/i/z/ymomentXY` : Second moment M_xy for g/r/i/z/y filter stack detection : DEFAULT=-999  
`g/r/i/z/ymomentYY` : Second moment M_yy for g/r/i/z/y filter stack detection : DEFAULT=-999  
`g/r/i/z/ymomentR1` : First radial moment for g/r/i/z/y filter stack detection : DEFAULT=-999  

### Added column headers
`ps1dr2_angular_separation` : Angular separation between ?? and ??

### Unknown column headers (where I have not yet found documentation describing the column explicitely)
`photo_ra`
`photo_decl`
`ps1dr2_baldeschi_quality`
`ps1dr2_decl`
`ps1dr2_id`
`ps1dr2_p_hsff`
`ps1dr2_p_spiral`
`ps1dr2_p_star`
`ps1dr2_ra`
`unwise_angular_separation`
`unwise_coadd_id`
`unwise_w1/w2_mag_ab`
`unwise_w1/w2_mag_vega`


