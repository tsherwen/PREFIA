"""
Functions to setup and process output for GEOS-Chem model runs for PREFIA
"""
import os
import xarray as xr
import glob
import numpy as np
import AC_tools as AC
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime as datetime_
from time import gmtime, strftime
import datetime as datetime
import matplotlib.pyplot as plt
import seaborn as sns
import gc


def main():
    """
    Driver to run processing, analysis, and setup of GEOS-Chem output for PREFIA
    """
    # - set resolution to use
    res = '2x2.5'

    # - Setup the models runs
    # Print the lines needed to get the HISTORY.rc output setup
#    print_lines4HISTORY_rc_config()

    # - Process output to files for submission
    processing4PREFIA_IC( res=res )

    # - Do checks on output
#    get_global_overview_stats_on_model_runs()
#    check_values4file()
#    check_units_in_outputted_files()

    # Update the NetCDF variables following NetCDF checker.
#    update_files_following_CF_checker()


def processing4PREFIA_IC(res='4x5'):
    """
    Process output for PREFIA runs
    """
    # - Get a dictionary of the names and locations output for runs
    res = '2x2.5'  # production run resolution.
    run_dict = get_dictionary_of_IC_runs(res=res)

    # - make one file per dat with all the values in it
    # Run for a single model run
#    combine_output2_1_file_per_day( ['GFED.DICE'],  run_dict=run_dict, res=res)
    # Run for all model output sets at once by pooling the task
    from multiprocessing import Pool
    from functools import partial
#    runs2use = list( run_dict.keys() )[:2] # Just run for the 1st two
    runs2use = list(run_dict.keys())
    runs2use = [[i] for i in runs2use]
#    runs2use = [ ['GFAS.PP'], ] # just run for Eloise's powerplant emissions
    print('Running file processing for:', runs2use)
    p = Pool(len(runs2use))
    # Use Pool to concurrently process all file sets
    p.map(partial(combine_output2_1_file_per_day, run_dict=run_dict, res=res),
          runs2use)


def combine_output2_1_file_per_day(runs2use, run_dict=None, res='4x5', version="0.1.1"):
    """
    Combine the PREFIA African AQ files into daily files for PREFIA inter-comparison
    """
    # format of outputted file
    FileStr = 'PREFIA_York_GEOSChem_{}_{}_{}_v{}'
    # file prefixes to use
    prefixes = ['HEMCO_diagnostics', ]
    GCprefixs = [
        'StateMet', 'LevelEdgeDiags', 'SpeciesConc', 'WetLossLS', 'Aerosols',
        'ConcAfterChem', 'DryDep', 'WetLossConv',
        ]
    prefixes += ['GEOSChem.'+i for i in GCprefixs]
    REFprefix = 'GEOSChem.StateMet'
    # Set ordering of prefixes
    # NOTE: use Met for base as it has all of the extra dimensions
    prefixes = [REFprefix]+[i for i in prefixes if i != REFprefix]
    # Loop by runs
    if isinstance(runs2use, type(None)):
        runs2use = run_dict.keys()
    for run in runs2use:
        folder = run_dict[run]
        print(run, folder)
        # Loop by prefix
        files4prefix = {}
        for prefix in prefixes:
            print(prefix)
            files4prefix[prefix] = get_files_in_folder4prefix(folder=folder,
                                                              prefix=prefix)
        # Check the number of files for each prefix.
        N_files = [len(files4prefix[i]) for i in files4prefix.keys()]
        if len(set(N_files)) == 1:
            print('WARNING: Different numbers of files for prefixes')
        # Get a list of dates
        dates = get_files_in_folder4prefix(folder=folder, prefix=REFprefix,
                                           rtn_dates4files=True)
        for day in dates:
            # Check all the files are present and that there is one of them
            #            day_str = day.strftime(format='%Y%m%d_%H%M')
            day_str = day.strftime(format='%Y%m%d')
            print(day_str)
            # Loop to get file names by prefix
            files2combine = []
            for prefix in prefixes:
                files = [i for i in files4prefix[prefix] if day_str in i]
                if len(files) != 1:
                    print('WARNING: more than one file found for prefix')
                print(day_str, prefix, files)
#                files2combine += [ xr.open_dataset( file[0] ) ]
                files2combine += [files]
            # Remove the doubled up data variables
            vars2rm = ['AREA', 'P0', 'hybm', 'hyam', 'hyai', 'hybi', '']
            # Open first fileset as a single xarray dataset
            ds = xr.open_mfdataset(files2combine[0])
            # Now add other file sets to this
            for files in files2combine[1:]:
                print(files)
                dsNew = xr.open_mfdataset(files)
                # Make sure indices for levels are coordinate variables
                var2add = 'ilev'
                if var2add not in dsNew.coords:
                    print('Adding ilev to dataset')
                    if var2add not in dsNew.data_vars:
                        dsNew[var2add] = ds[var2add].values
                    dsNew = dsNew.set_coords(var2add)
                    dsNew = dsNew.assign_coords(ilev=ds[var2add].copy())
                var2add = 'lev'
                if var2add not in dsNew.coords:
                    print('Adding lev to dataset')
                    if var2add not in dsNew.data_vars:
                        dsNew[var2add] = ds[var2add].values
                    dsNew = dsNew.set_coords(var2add)
                    dsNew = dsNew.assign_coords(lev=ds[var2add])
                # Update the timestamp for the HEMCO files
                if all(['HEMCO_diagnostics' in i for i in files]):
                    dt = [AC.add_hrs(i, 0.5)
                                     for i in AC.dt64_2_dt(dsNew.time.values)]
                    dsNew.time.values = dt
                # Combine new data into a core file
                vars2add = [i for i in dsNew.data_vars if i not in vars2rm]
                ds = xr.merge([ds, dsNew[vars2add]])
                del dsNew
            # Remove OH and HO2 from SpeciesConc diag.
            ds = remove_OH_HO2_from_SpeciesConc(ds=ds)
            # Remove any unneeded spatial dimensions
            ds = remove_unneeded_vertical_dimensions(ds=ds)
            # Update any units to those requested for PREFIA...
            ds = convert_IC_ds_units(ds=ds)  # loosing units here
            # Add combined variables
            ds = add_combined_vars(ds=ds)
            # Only include requested parameters
            ds = only_inc_requested_vars(ds=ds)
            # Change any names
            ds = update_names_in_ds(ds=ds)
            # Update extents to be over Africa
            # (19.9W, 39.9S), and north-eastern cell centre is placed at (54.9W, 39.9N)
            ds = only_consider_domain_over_Africa(ds)
            # Update the global attributes
            ds = add_global_attributes4run(ds, run=run)
            # remove all the unneeded coordinates
            ds = ds.squeeze()
            # Now save the file for the given day
            filename = FileStr.format(res, run, day_str, version)
            ds.to_netcdf('{}/{}.nc'.format(folder, filename))
            # Remove the used files from memory
            del ds
            gc.collect()


def remove_OH_HO2_from_SpeciesConc(ds):
    """
    Remove OH and HO2 from SpeciesConc diag (use ConcAfterChem instead)
    """
    vars2rm = ['SpeciesConc_OH', 'SpeciesConc_HO2']
    for var in vars2rm:
        del ds[var]
        print("Removed variable from data set ('{}')".format(var))
    return ds


def check_units_in_outputted_files(ds):
    """
    Check the units in the outputted files
    """
    specs2check = ['O3']
    for spec in specs2check:
        # Check the data variables for  ozone
        data_vars = [i for i in ds.data_vars if 'O3' in i]
        # Check the deposition units
        DryDep = 'DryDep_{}'.format(spec)
        print(spec, ds[DryDep].values.sum(), ds[DryDep].values.mean())
        ds[DryDep] = ds[DryDep]*ds['AREA'] / 1E6 * 30 * 12
        print(spec, ds[DryDep].values.sum(), ds[DryDep].values.mean())
        # Also check wet deposition (not for O3)
#        WetDep = [ i for i in data_vars if 'Wet' in i]


def only_consider_domain_over_Africa(ds):
    """
    Return dataset, but just for domain over Africa
    """
    # 20W-55E, 40S-40N,
    bool1 = ((ds.lon >= -22) & (ds.lon <= 57)).values
    bool2 = ((ds.lat >= -50) & (ds.lat <= 40)).values
    # cut by lon, then lat
    ds = ds.isel(lon=bool1)
    ds = ds.isel(lat=bool2)
    return ds


def only_inc_requested_vars(ds):
    """
    Only include the requested variables
    """
    # Met variables to include
    vars = ['Met_PBLH', 'Met_PRECTOT', 'Met_TSKIN', 'Met_CLDFRC']
    # include the AOD numbers
    vars += ['AOD_ALL_550nm', ]
    # Include the total emissions for the species
    vars += [i for i in ds.data_vars if ('Emis' in i) and ('Total' in i)]
    # Include the total drydep for the species
    vars += [i for i in ds.data_vars if ('Emis' in i) and ('Total' in i)]
    # Include the total dry dep for the species
    vars += [i for i in ds.data_vars if ('DryDep' in i)]
    # Include the total wet dep for the species
    vars += [i for i in ds.data_vars if ('WetDep' in i)]
    # Include the species concentrations
    vars += [i for i in ds.data_vars if ('SpeciesConc_' in i)]
    vars += [i for i in ds.data_vars if ('AfterChem' in i)]
    # Include area
    vars += ['AREA']
    # Don't include any of the values for expanded SOA scheme.
    vars2rm = 'SOAMG', 'SOAGX', 'SOAME', 'SOAIE'
    vars = [i for i in vars if all([(ii not in i) for ii in vars2rm])]
    # Remove variables that will not be considered by the intercomparisons
    vars2rm = 'CH4', 'GLYX', 'MGLY'
    vars = [i for i in vars if all([(ii not in i) for ii in vars2rm])]

    return ds[vars]


def remove_unneeded_vertical_dimensions(ds=None, verbose=False, debug=False):
    """
    Remove or sum the extra vertical levels not needed for analysis
    """
    # - Sum AOD over all levels
    prefix = 'AOD'
    vars2use = [i for i in ds.data_vars if i.startswith(prefix)]
    for var in vars2use:
        if debug:
            print(var, ds[var])
        # Save the attributes for later
        attrs = ds[var].attrs.copy()
        # Sum over altitude
        try:
            ds[var] = ds[var].sum(dim='lev',
                                  #                        keep_attrsbool=True # Not in version in use
                                  )
            # Re add the attributes
            ds[var].attrs = attrs
        except ValueError:
            if verbose:
                print("WARNING: Not summed over 'lev' for '{}'".format(var))

    # - Sum emissions over all levels
    prefix = 'Emis'
    vars2use = [i for i in ds.data_vars if i.startswith(prefix)]
    for var in vars2use:
        if debug:
            print(var, ds[var])
        # Save the attributes for later
        attrs = ds[var].attrs.copy()
        # Sum over altitude
        try:
            ds[var] = ds[var].sum(dim='lev',
                                  #                        keep_attrsbool=True # Not in version in use
                                  )
            # Re add the attributes
            ds[var].attrs = attrs
        except ValueError:
            if verbose:
                print("WARNING: Not summed over 'lev' for '{}'".format(var))
    # - Sum wet depositional loss over all levels
    prefix = 'WetLossLS_'
    vars2use = [i for i in ds.data_vars if i.startswith(prefix)]
    for var in vars2use:
        if debug:
            print(var, ds[var])
        # Save the attributes for later
        attrs = ds[var].attrs.copy()
        # Sum over altitude
        try:
            ds[var] = ds[var].sum(dim='lev',
                                  #                        keep_attrsbool=True # Not in version in use
                                  )
            # Re add the attributes
            ds[var].attrs = attrs
        except ValueError:
            if verbose:
                print("WARNING: Not summed over 'lev' for '{}'".format(var))

    # - Sum wet depositional loss over all levels
    prefix = 'WetLossConv_'
    vars2use = [i for i in ds.data_vars if i.startswith(prefix)]
    for var in vars2use:
        if debug:
            print(var, ds[var])
        # Save the attributes for later
        attrs = ds[var].attrs.copy()
        # Sum over altitude
        try:
            ds[var] = ds[var].sum(dim='lev',
                                  #                        keep_attrsbool=True # Not in version in use
                                  )
            # Re add the attributes
            ds[var].attrs = attrs
        except ValueError:
            if verbose:
                print("WARNING: Not summed over 'lev' for '{}'".format(var))

    # - Only include species concentrations for the surface
    prefix = 'SpeciesConc'
    Species_vars = [i for i in ds.data_vars if i.startswith(prefix)]
    for var in Species_vars:
        print(var, ds[var])
        attrs = ds[var].attrs.copy()
        # Select surface
        ds[var] = ds[var].sel(lev=float(ds['lev'][-1].values))
        # Add the attributes back into dataset
        ds[var].attrs = attrs


    # - Only include species concentrations for the surface
    suffix = 'concAfterChem'
    vars2use = [i for i in ds.data_vars if i.endswith(suffix)]
    for var in vars2use:
        print(var, ds[var])
        attrs = ds[var].attrs.copy()
        # Select surface
        ds[var] = ds[var].sel(lev=float(ds['lev'][-1].values))
        # Add the attributes back into dataset
        ds[var].attrs = attrs

    # - Return the updated dataset
    return ds


def convert_IC_ds_units(ds=None):
    """
    Convert the units into those required for PREFIA
    """
    # - Local variables
    C12vars = ['ISOP', 'C3H8', 'C2H6', 'ALK4', 'ALD2', 'ACET']
    Cmasses = {
        'ISOP': 68.12, 'C3H8': 44.10, 'C2H6': 30.07, 'ALK4': 58.12, 'ALD2': 44.05,
        'ACET': 58.08,
    }

    # - Convert the Emissions into require units
    # Starting units are: kg/m2/s
    # requested units are: kg/m2/hr
    prefix = 'Emis'
    Emission_vars = [i for i in ds.data_vars if i.startswith(prefix)]
    for var in Emission_vars:
        spec = var.split(prefix)[-1].split('_')[0]
        print(var, spec)
        # store the current attributes
        attrs = ds[var].attrs.copy()
        # Update the units
        ds[var] = ds[var] / 60
        # Update to grams of species instead carbon
        if spec in C12vars:
            scale = 1/AC.species_mass('C') / \
                AC.spec_stoich(spec, C=True)*Cmasses[spec]
            ds[var] = ds[var]*scale
        # update the units
        attrs['units'] = 'kg/m2/hr'
        ds[var].attrs = attrs

    # - Convert the species concentrations
    # 'Met_AIRVOL'
    AIRVOL = 'Met_AIRVOL'  # Grid box volume, dry air (m3)
#    AD = 'Met_AD'  # AD: Air mass in grid box (kg)
    AIRDEN = 'Met_AIRDEN'  # Dry air density - kg/m3
#    AREA = 'AREA'
    prefix = 'SpeciesConc'
    RMM_air = AC.constants('RMM_air')
    Species_vars = [i for i in ds.data_vars if i.startswith(prefix)]
    # starting units are v/v
    # requested unit: ug<subst>/m3
    # Get moles of air at surface (kg => g, then mols)
#    mols = ds[AD].sel(lev=float(ds['lev'][0].values) )*1E3/RMM_air
    # Get moles of air at surface (kg/m3 => mols/m3 => g, then mols)
#    mols = ds[AIRDEN].sel(lev=float(ds['lev'][0].values) ) *1E3
    # RMM in g/mol, therefore (1/(g/mol)) = (mol/g) ; (mol/g) * (g/m3) = mol/m3
    mols_per_m3 = ds[AIRDEN].sel(
        lev=float(ds['lev'].values[-1])) * 1E3 * (1/RMM_air)
    # Get the volume at the surface (units: m3)
#    VOL = ds[AIRVOL].sel(lev=float(ds['lev'][0].values) )
    # Loop by species and convert to ug/m3
    for var in Species_vars:
        spec = var.split(prefix+'_')[-1]
        print(var, spec)
        # store the current attributes
        attrs = ds[var].attrs.copy()
        # Adjust to mols/m3, then mass, then micrograms
        ds[var] = ds[var]*mols_per_m3*AC.species_mass(spec) * 1E6
        # Update to grams of species instead carbon
        if spec in C12vars:
            scale = 1/AC.species_mass('C') / \
                AC.spec_stoich(spec, C=True)*Cmasses[spec]
            ds[var] = ds[var]*scale
        # update the units
        attrs['units'] = 'μg/m3'
        # Update the unit string
        attrs['long_name'] = 'Dry mass concentration of species {}'.format(spec)
        ds[var].attrs = attrs

    # - Convert Wet loss
    # kg s-1 to μg <Species name> m-2 hr-1
    prefix = 'WetLossLS'
    RMM_air = AC.constants('RMM_air')
    Species_vars = [i for i in ds.data_vars if i.startswith(prefix)]
    # Loop by species and convert to ug/m2/hr
    for var in Species_vars:
        spec = var.split(prefix+'_')[-1]
        print(var, spec)
        # store the current attributes
        attrs = ds[var].attrs.copy()
        # Update the units (/ m2, sec=>min=>hour)
        ds[var] = ds[var] / ds['AREA'] * 60 * 60 * 1E3 * 1E6
        # Update to grams of species instead carbon
        if spec in C12vars:
            scale = 1/AC.species_mass('C') / \
                AC.spec_stoich(spec, C=True)*Cmasses[spec]
            ds[var] = ds[var]*scale
        # update the units + add averaging method
        attrs['units'] = 'μg/m2/hr'
        attrs['averaging_method'] = 'time-averaged'
        ds[var].attrs = attrs

    # - Convert Wet loss
    # kg s-1 to μg <Species name> m-2 hr-1
    prefix = 'WetLossConv'
    RMM_air = AC.constants('RMM_air')
    Species_vars = [i for i in ds.data_vars if i.startswith(prefix)]
    # Loop by species and convert to ug/m2/hr
    for var in Species_vars:
        spec = var.split(prefix+'_')[-1]
        print(var, spec)
        # store the current attributes
        attrs = ds[var].attrs.copy()
        # Update the units
        ds[var] = ds[var] / ds['AREA'] * 60 * 60 * 1E3 * 1E6
        # Update to grams of species instead carbon
        if spec in C12vars:
            scale = 1/AC.species_mass('C') / \
                AC.spec_stoich(spec, C=True)*Cmasses[spec]
            ds[var] = ds[var]*scale
        # update the units + add averaging method
        attrs['units'] = 'μg/m2/hr'
        attrs['averaging_method'] = 'time-averaged'
        ds[var].attrs = attrs

    # - Convert Dry dep
    # molec cm-2 s-1 to μg <Species name> m-2 hr-1
    prefix = 'DryDep'
    RMM_air = AC.constants('RMM_air')
    Species_vars = [i for i in ds.data_vars if i.startswith(prefix)]
    # Loop by species and convert to ug/m2/hr
    for var in Species_vars:
        print(var)
        spec = var.split(prefix)[-1][1:]
        # store the current attributes
        attrs = ds[var].attrs.copy()
        # /cm2 to /m2
        ds[var] = ds[var] * 10000
        # molecules to mols to uq
        ds[var] = ds[var] / AC.constants('AVG') * AC.species_mass(spec) * 1E6
        # /s to /hr
        ds[var] = ds[var] * 60 * 60
#        / ds['AREA'] * 60 * 60 * 1e+9
        # Update to grams of species instead carbon
        if spec in C12vars:
            scale = 1/AC.species_mass('C') / \
                AC.spec_stoich(spec, C=True)*Cmasses[spec]
            ds[var] = ds[var]*scale
        # update the units
        attrs['units'] = 'μg/m2/hr'
        ds[var].attrs = attrs


    # - Convert the concAfterChem concentrations (OH, HO2)
    suffix = 'concAfterChem'
    AVG = AC.constants('AVG')
    vars2use = [i for i in ds.data_vars if i.endswith(suffix)]
    # requested unit: ug<subst>/m3
    # Loop by species and convert molec/cm3	to ug/m3
    for var in vars2use:
        spec = var.split('concAfterChem')[0]
        print(var, spec)
        # store the current attributes
        attrs = ds[var].attrs.copy()
        # Adjust to mol/cm3=> g/cm3, then ug/cm3 , then ug/m3
        ds[var] = ds[var] / AVG * AC.species_mass(spec) * 1E6 * 1E6
        # update the units
        attrs['units'] = 'μg/m3'
        # Update the unit string
        attrs['long_name'] = 'Dry mass concentration of species {}'.format(spec)
        ds[var].attrs = attrs

    # - Convert precipitation
    # from kg m-2 s-1  to mm hr-1
    # NOTE: HISTORY.rc descriptor must be wrong. - Now updated this in docs.
    # assume mm/day as on wiki
    # http://wiki.seas.harvard.edu/geos-chem/index.php/List_of_diagnostics_archived_to_bpch_format#ND67:_GMAO_2-D_met_fields
    var2use = 'Met_PRECTOT'
    attrs = ds[var2use].attrs.copy()
#    ds[var2use] = ds[var2use] * 60 * 60
    ds[var2use] = ds[var2use] / 24
    attrs['units'] = 'mm/hr'
    ds[var2use].attrs = attrs

    # - Then just return the dataset
    return ds


def check_values4file():
    """
    Check values in the outputted PREFIA NetCDF file(s)
    """
    res = '4x5'  # 4x5 for testing
#    res = '2x2.5' # for production runs
    filestr = '*{}*v12.5*UT*PREFIA*'.format(res)
    folder = '/users/ts551/scratch/GC/rundirs/'
    # Setup the run directories
    folders = glob.glob(folder + filestr)
    names = [i.split('UT.')[-1] for i in folders]
    run_dict = dict(zip(names, folders))
    runs = list(run_dict.keys())

    # -- Perform checks on emisisons
    # Read in the HEMCO files into a dataset for each run.
    ds_d = {}
    for run in runs:
        print(run, run_dict[run])
        # Get the HEMCO files as a dataset
        folder2use = '{}/OutputDir/'.format(run_dict[run])
        # Check the number of files
        files2use = glob.glob(folder2use+'*HEMCO*')
#        print(run, len(files2use), len(files2use)/30/24 )
        # Open the HEMCO files as a dataset
        ds = AC.get_HEMCO_diags_as_ds(wd=folder2use)
        # Only consider the production year
        ds = ds.sel(time=ds['time.year'] > 2016)
        # Resample this to monthly
        # NOTE: 4x5 output is only monthly (2x2.5 is daily)
        if res == '2x2.5':
            ds = ds.resample({'time': 'M'})
        # Store dataset
        ds_d[run] = ds.copy()
        del ds

    # Now process dataset
    for run in runs:
        # Select dataset
        ds = ds_d[run]
        # Sum over all levels
        ds = ds.sum(dim='lev')
        # Remove the area unit and convert time to per month
        EmisVars = [i for i in ds.data_vars if 'Emis' in i]
        for var in EmisVars:
            # Convert to grams
            ds[var].values = ds[var] * 1E3
            # multiple by AREA
            ds[var].values = ds[var] * ds['AREA']
            # convert /s to / month
            ds[var].values = ds[var] * 60 * 60*24 * 31
        # Save the updated dataset
        ds_d[run] = ds.copy()
        del ds

    # Print out the totals
    df = pd.DataFrame()
    TotalVars = [i for i in ds_d[runs[0]].data_vars if '_Total' in i]
    for run in runs:
        # select dataset and variables for totals
        ds = ds_d[run]
        #
        S = pd.Series()
        for var in TotalVars:
            S[var] = ds[var].values.sum()
        df[run] = S

    # Print out the totals
    df = df / 1E12
    df.to_csv('PREFIA_Emissions_totals_Global_{}.csv'.format(res))
    # Also save in percentage terms
    REF = 'PREFIA.GFAS'
    for col in df.columns:
        df[col] = (df[col]-df[REF])/df[REF]*100
    df.to_csv('PREFIA_Emissions_totals_Global_pcent_{}.csv'.format(res))

    # Only consider over Africa
    df = pd.DataFrame()
    TotalVars = [i for i in ds_d[runs[0]].data_vars if '_Total' in i]
    for run in runs:
        # select dataset and variables for totals
        ds = ds_d[run]
        # Exact values by species
        S = pd.Series()
        for var in TotalVars:
            # select African region
            # 20W-55E, 40S-40N,
            bool1 = ((ds.lon >= -22) & (ds.lon <= 57)).values
            bool2 = ((ds.lat >= -50) & (ds.lat <= 40)).values
            # cut by lon, then lat
#            ds_tmp = ds.sel(lon=(ds['lon'].values > -22)))
            ds_tmp = ds.isel(lon=bool1)
            ds_tmp = ds_tmp.isel(lat=bool2)
            # No save data
            S[var] = ds_tmp[var].values.sum()
        df[run] = S

    # Print out the totals
    df = df / 1E12
    df.to_csv('PREFIA_Emissions_totals_Africa_{}.csv'.format(res))
    # Also save in percentage terms
    REF = 'PREFIA.GFAS'
    for col in df.columns:
        df[col] = (df[col]-df[REF])/df[REF]*100
    df.to_csv('PREFIA_Emissions_totals_Africa_pcent_{}.csv'.format(res))


def add_combined_vars(ds=None):
    """
    Add variables that are combinations of others
    """
    # - Add a single AOD variable
    var2use = 'AOD_ALL_550nm'
    prefix = 'AOD'
    Species_vars = [i for i in ds.data_vars if i.startswith(prefix)]
    # Don't include the total for dust to avoid double counting
    Species_vars = [i for i in Species_vars if i != 'AODDust']
#    Species_vars = [i for i in Species_vars if ('550nm_bin' not in i) ]
    # copy values from first species
    ds[var2use] = ds[Species_vars[0]].copy()
    attrs = ds[var2use].attrs.copy()
#    attrs['long_name'] = 'Optical depth for hygroscopic and dust aerosol at 550 nm'
    # Sum up the components of AOD
    for var in Species_vars[1:]:
        print(var)
        ds[var2use] = ds[var2use] + ds[var].values
    # Add units back into attribute dictionary
    attrs['long_name'] = 'Optical depth for aerosol at 550 nm'
    attrs['averaging_method'] = 'time-averaged'
    attrs['units'] = '1'
    ds[var2use].attrs = attrs

    # - compute PM2.5
    var2use = 'SpeciesConc_PM25'
    specs2use = [
        'BCPI', 'NH4', 'SO4', 'BCPO', 'SALA', 'DST1', 'DST2', 'NIT', 'OCPO',
        'OCPI',  # Exclude OCPI in initial test (but not in finel output)
    ]
    prefix = 'SpeciesConc'
    Species_vars = [prefix+'_'+i for i in specs2use]
    # copy values and units from first species
    ds[var2use] = ds[Species_vars[0]].copy()
    attrs = ds[var2use].attrs.copy()
    # do scaling for OC and partial inclusion of dust
    for var in Species_vars[1:]:
        print(var)
        # Scale OM/OC
        scale = 1
        if ('OCPO' in var) or ('OCPI' in var):
            scale = 2.1
        # only include 38% of DST2 in PM2.5
        if ('DST2' in var):
            scale = 0.38
        # Add to total
        ds[var2use] = ds[var2use] + (ds[var].values * scale)
    # Add units back into attribute dictionary
    attrs['long_name'] = 'Dry mass concentration of species PM2.5'
    attrs['units'] = 'μg/m3'
    attrs['averaging_method'] = 'time-averaged'
    ds[var2use].attrs = attrs

    # - compute PM10
    var2use = 'SpeciesConc_PM10'
    specs2use = [
        'BCPI', 'NH4', 'SO4', 'BCPO', 'SALA', 'DST1', 'DST2', 'NIT', 'OCPO',
        'DST2', 'DST3', 'DST4',
        'OCPI',  # Exclude OCPI in initial test (but not in final output)
    ]
    prefix = 'SpeciesConc'
    Species_vars = [prefix+'_'+i for i in specs2use]
    # copy values and units  from first species
    ds[var2use] = ds[Species_vars[0]].copy()
    attrs = ds[var2use].attrs.copy()
    for var in Species_vars[1:]:
        print(var)
        # Scale OM/OC
        scale = 1
        if ('OCPO' in var) or ('OCPI' in var):
            scale = 2.1
        # Add to total
        ds[var2use] = ds[var2use] + (ds[var].values * scale)
    # Add units back into attribute dictionary
    attrs['long_name'] = 'Dry mass concentration of species PM10'
    attrs['units'] = 'μg/m3'
    attrs['averaging_method'] = 'time-averaged'
    ds[var2use].attrs = attrs

    # - compute OC
    var2use = 'SpeciesConc_EC'
    # copy values from first species
    ds[var2use] = ds['SpeciesConc_OCPO'].copy()
    # Update the long_name attributes
    ds[var2use] = ds[var2use] + ds['SpeciesConc_OCPI'].values
    attrs = ds[var2use].attrs.copy()
    attrs['long_name'] = 'Dry mass concentration of species EC'
    attrs['units'] = 'μg/m3'
    attrs['averaging_method'] = 'time-averaged'
    ds[var2use].attrs = attrs

    # - Combine wet deposition routes
    NewPrefix = 'WetDep_'
    WetConv = 'WetLossConv_'
    WetLossLS_ = 'WetLossLS_'
    vars2use = [i.split(WetConv)[-1] for i in ds.data_vars if WetConv in i]
    # copy values from first species
    for var in vars2use:
        # setup a new variable name
        var2use = NewPrefix+var
        # copy the convective tive flux value
        ds[var2use] = ds[WetConv+var].copy()
        # copy the convective tive flux value
        ds[var2use] = ds[var2use] + ds[WetLossLS_+var].values
        # Update the long name
        attrs = ds[var2use].attrs.copy()
        attrs['long_name'] = 'Wet deposition flux of species {}'.format(var)
        attrs['units'] = 'μg/m2/hr'
        attrs['averaging_method'] = 'time-averaged'
        ds[var2use].attrs = attrs
    return ds


def update_names_in_ds(ds=None):
    """
    Update the names used in the dataset
    """
    # Update names from SpeciesConc_<> to  cnc_<substance>
    prefix = 'SpeciesConc'
    Species_vars = [i for i in ds.data_vars if i.startswith(prefix)]
    newprefix = 'cnc'
    NewVars = [newprefix+i.split(prefix)[-1] for i in Species_vars]
    name_dict = dict(zip(Species_vars, NewVars))
    # Now rename all varibles at once
    ds = ds.rename(name_dict=name_dict)

    # Update names from <>concAfterChem to  cnc_<substance>
    suffix = 'concAfterChem'
    vars = [i for i in ds.data_vars if i.endswith(suffix)]
    newprefix = 'cnc_'
    NewVars = [newprefix+i.split(suffix)[0] for i in vars]
    name_dict = dict(zip(vars, NewVars))
    # Now rename all varibles at once
    ds = ds.rename(name_dict=name_dict)

    # - Return dataset
    return ds


def get_files_in_folder4prefix(folder=None, prefix=None,  rtn_dates4files=False):
    """
    Get all the files with a prefix in a directory
    """
    # Get the dates of model output files
    files = glob.glob('/{}/{}*'.format(folder, prefix))
    files = sorted(list(files))
    # return the dates of the list of files
    if rtn_dates4files:
        dates = [i.split(prefix)[-1][1:-5] for i in files]
        dates = pd.to_datetime(dates, format='%Y%m%d_%H%M')
        return dates
    else:
        return files


def get_global_overview_stats_on_model_runs(res='4x5'):
    """
    Process bpch files to NetCDF, then get general stats on runs
    """
    # Get the model run locations, then process to NetCDF from bpch
    run_dict = get_dictionary_of_IC_runs(res=res, NetCDF=False)
    runs = list(sorted(run_dict.keys()))
    for run in runs:
        folder = run_dict[run]
        print(run, run_dict[run])
        a = AC.get_O3_burden_bpch(folder)
        print(run, run_dict[run], a.sum())
    # Get summary stats on model runs.
    df = AC.get_general_stats4run_dict_as_df_bpch(
        run_dict=run_dict, REF1='GFAS.DICE', res=res)
    # Setup a name for the csv file
    filestr = 'PREFIA_summary_stats_{}{}.csv'
    if res == '4x5':
        filename = filestr.format(res, '')
    elif res == '2x2.5':
        filename = filestr.format(
            res, '_VALUES_ARE_APROX_bpch_files_incomplete')
    else:
        print("WARNING: resolution ('{}') not known".format(res))
    # Save summary stats to disk
    df.T.to_csv(filename)


def get_dictionary_of_IC_runs(res='4x5', NetCDF=True):
    """
    Get a dictionary of the Africa PREFIA inter-comparison runs and their locations
    """
    RunRoot = '/users/ts551/scratch/GC/rundirs/'
    # Get the NetCDF or bpch file location
    ExStr = ''
    if NetCDF:
        ExStr = '/OutputDir/'
    # Get the model run names and directories for a given resolution
    if res == '4x5':
        RunStr = 'geosfp_4x5_tropchem.v12.5.0.UT.PREFIA{}{}'
        d = {
            # Initial testing
            #    'TEST' : RunRoot +'/geosfp_4x5_standard.v12.4.0.IC.2months/',
            # Expanded testing
            #    'TEST.II' : RunRoot +'/geosfp_4x5_standard.v12.4.0.IC.2months.repeat/',
            # Production output locations and names
            'GFED.DICE': RunRoot + RunStr.format('/', ExStr),
            'GFAS.DICE': RunRoot + RunStr.format('.GFAS/', ExStr),
            'GFAS.DICE.PP': RunRoot + RunStr.format('.GFAS.EloisePP/', ExStr),
            'DACCIWA': RunRoot + RunStr.format('.GFAS.DACCIWA/', ExStr),
            'CEDS': RunRoot + RunStr.format('.GFAS.CEDS/', ExStr),
        }
    elif res == '2x2.5':
        RunStr = 'geosfp_2x25_tropchem.v12.5.0.UT.PREFIA{}{}'
        d = {
            'GFED.DICE': RunRoot + RunStr.format('.repeat/', ExStr),
            'GFAS.DICE': RunRoot + RunStr.format('.GFAS.repeat/', ExStr),
            'GFAS.DICE.PP': RunRoot + RunStr.format('.EloisePP/', ExStr),
            'DACCIWA': RunRoot + RunStr.format('.DACCIWA/', ExStr),
            'CEDS': RunRoot + RunStr.format('.CEDS/', ExStr),
        }
    else:
        print("WARNING: Resolution ('{}') not found".format(res))
        sys.exit()
    return d


def print_lines4HISTORY_rc_config():
    """
    Print the lines to setup the NetCDF diagnostics in HISTORY.rc for PREFIA
    """
    # Analysis species
    # NOTE: species names updated (HCHO=>CH2O)
    GasSpecs = [
        'SO2', 'SO4', 'NO', 'NO2', 'NO3', 'HNO3', 'O3', 'NH3', 'CH2O', 'CO', 'NH4', 'OH',
        'HO2'
    ]
    # Aeros
    AerosolSpecs = [
        'NIT', 'NITs', 'BCPI', 'BCPO', 'OCPO', 'DST1', 'DST2', 'DST3', 'DST4', 'SALA',
        'SALC',
        # 'SOA',
        'SOAP', 'SOAS', 'SOAIE', 'SOAME', 'SOAGX', 'SOAMG',
    ]
    # Analysis species
    AnalSpecs = GasSpecs + AerosolSpecs
    # - Print lines for wet deposition
    specs2use = AnalSpecs
    # Drop species not wet deposited
    NotDep = ['NO', 'NO3', 'CO', 'OH', 'HO2', 'SOAP']
    NotWetDep = NotDep + ['NO2', 'O3']
    specs2use = [i for i in specs2use if (i not in NotWetDep)]
    # Now print species
    pstr1 = "  WetLossLS.fields:           'WetLossLS_{:<10}               ', 'GIGCchem',"
    pstr2 = "                              'WetLossLS_{:<10}               ', 'GIGCchem',"
    # print out the
    print(pstr1.format(specs2use[0]))
    for spec in specs2use[1:]:
        print(pstr2.format(spec))

    # - Print lines for dry deposition
    specs2use = AnalSpecs
    # Drop species not dry deposited
    NotDryDep = NotDep + []
    specs2use = [i for i in specs2use if (i not in NotDryDep)]
    # Now print species
    pstr1 = "  DryDep.fields:              'DryDep_{:<10}                 ', 'GIGCchem',"
    pstr2 = "                              'DryDep_{:<10}                 ', 'GIGCchem',"
    # print out the lines to paste into the HISTORY.rc
    print(pstr1.format(specs2use[0]))
    for spec in specs2use[1:]:
        print(pstr2.format(spec))

    # - Print lines for species concentration
    specs2use = AnalSpecs
    pstr1 = "  SpeciesConc.fields:         'SpeciesConc_{:<10}             ', 'GIGCchem',"
    pstr2 = "                              'SpeciesConc_{:<10}             ', 'GIGCchem',"
    # print out the lines to paste into the HISTORY.rc
    print(pstr1.format(specs2use[0]))
    for spec in specs2use[1:]:
        print(pstr2.format(spec))


def update_files_following_CF_checker(version='0.1.1'):
    """
    Update NetCDF files following Puma CF checker
    """
    # Get the model run names and locations
    res = '2x2.5' # production run resolution.
    run_dict = get_dictionary_of_IC_runs(res=res)
    # Loop by model run and update the datasets
    runs = list(run_dict.keys())
    print(runs)
    for run in runs:
        # Get the location of the data
        folder = run_dict[run]
        # Get the files
        files = list(sorted(glob.glob('{}/{}*{}*'.format(folder, 'PREFIA', version))))
        # Loop the files
#        for file in files:
        for file in files:
            filename = file.split('/')[-1]
            print(run, filename)
            # open as a dataset
            ds = xr.open_dataset(file)
            # Update the variables
#            ds = add_missing_globl_and_variables_attrs(ds=ds, run=run)
            # Update the variable names
            prefix = 'SpeciesConc'
            Species_vars = [i for i in ds.data_vars if i.startswith(prefix)]
            newprefix = 'cnc'
            NewVars = [newprefix+i.split(prefix)[-1] for i in Species_vars]
            name_dict = dict(zip(Species_vars, NewVars))
            # Now rename all varibles at once
            ds = ds.rename(name_dict=name_dict)
            # Save the updated file
            UseNewFilename = False
            if UseNewFilename:
                Filestr = '{}_v{}.nc'
                Newfilename = Filestr.format(filename.split('.nc')[0], version)
                ds.to_netcdf(folder+Newfilename)
            else:
                NewFolder = folder + '/SC_name_update/'
                print( NewFolder, filename )
                ds.to_netcdf(NewFolder+filename) # Testing
            del ds
            gc.collect()


def add_global_attributes4run(ds, run='GFED.DICE', conventions='CF-1.3'):
    """
    Add global attributes for single day file for PREFIA
    """
    attrs = ds.attrs
    # Add title
    title_str = "GEOS-Chem model run output ('{}') for PREFIA intercomparison"
    attrs['title'] = title_str.format(run)
    # Add 'Conventions' to global meta data
    attrs['Conventions'] = conventions
    # Add history attribute
    history_str = 'Last modified on'
    attrs['history'] = history_str.format(strftime("%B %d %Y", gmtime()))
    # Add reference
    attrs['reference'] = 'The International GEOS-Chem User Community. (2019, September 9). geoschem/geos-chem: GEOS-Chem 12.5.0 release (Version 12.2.0). Zenodo. http://doi.org/10.5281/zenodo.3403111'
    # Contact
    attrs['contact'] = 'tomas.sherwen@york.ac.uk'
    # Explicitly state the default format
    attrs['format'] = 'NETCDF4'
    # Store updates
    ds.attrs = attrs
    return ds


if __name__ == "__main__":
    main()
