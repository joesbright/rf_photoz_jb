# Photometric Redshifts with Pan-STARRS, (un)WISE, and SDSS

Photometric redfhit calculations using machine learning (random forest) techniques, with data from the panSTARRS and WISE surveys.

This project will include data cleaning of the unWISE and panSTARRS data, combining the data to match sources, and engineering colour features from the various filter magnitudes. These data will then be split into a training and test dataset to train, optimize and evaluate machine learning models.

# The Data
## unWISE/SDSS
The first data catalog used in this project is the unWISE catalog (https://catalog.unwise.me/catalogs.html) crossmatched with the SDSS source catalog ().

### unWISE column headers
`unwise_decl` : declination(degree)  
`unwise_primary`: the center of this source is in the primary region of its coadd

### SDSS column headers

### Unknown column headers (where I have not yet found documentation describing the column explicitely)
`spec_obj_id`
`target_obj_id`
`objid`
`sdss_ra`
`sdss_decl`
`photo_ra`
`photo_decl`
`z`
`z_err`
`class`
`subclass`
`z_warning`
`type`
`survey`
`ps1dr2_angular_separation`
`ps1dr2_baldeschi_quality`
`ps1dr2_decl`
`ps1dr2_id`
`ps1dr2_p_hsff`
`ps1dr2_p_spiral`
`ps1dr2_p_star`
`ps1dr2_ra`
`unwise_angular_separation`
`unwise_coadd_id`
