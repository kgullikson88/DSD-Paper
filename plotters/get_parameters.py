"""
Get the parameters of the stars in the known sample which I detect.
"""

import os
import pandas as pd
import numpy as np
import StarData
import Correct_CCF_Temperatures
import HDF5_Helpers

SAMPLE_HIP = [1366, 3300, 12719, 13165, 15338, 17563, 22840, 22958, 24902, 
              26063, 26563, 28691, 33372, 44127, 58590, 65477, 76267, 77516, 
              77858, 79199, 79404, 81641, 84606, 85385, 88290, 89156, 91118, 
              92027, 92728, 98055, 100221, 106786, 113788, 116247, 116611]
SAMPLE_STARS = ['HIP {}'.format(hip) for hip in SAMPLE_HIP]

home = os.environ['HOME']
OBSERVED_FNAME = '{}/Dropbox/School/Research/AstarStuff/TargetLists/Observed_Targets3.xls'.format(home)

def read_full_sample(fname=OBSERVED_FNAME):
    """
    Read and do some initial processing on my sample
    """
    sample_names = ['identifier', 'RA/DEC (J2000)', 'plx', 'Vmag', 'Kmag', 'vsini', 'configuration', 'Instrument', 'Date', 
                    'Temperature', 'Velocity', 'vsini_sec', '[Fe/H]', 'Significance', 'Sens_min', 'Sens_any', 'Comments', 
                    'Rank', 'Keck', 'VLT', 'Gemini', 'Imaging_Detecton']
    sample = pd.read_excel(fname, sheetname=0, na_values=['     ~'], names=sample_names)
    sample = sample.reset_index(drop=True)[1:]
    sample.dropna(subset=['RA/DEC (J2000)', 'Instrument'], how='any', inplace=True)

    return sample


def get_corrected_temperatures(sample):
    # Get the names right
    sample['Star'] = sample['identifier']
    sample['vsini_prim'] = sample['vsini']
    sample['vsini'] = sample['vsini_sec']
    corrected = Correct_CCF_Temperatures.get_real_temperature(sample)

    return corrected.dropna(subset=['Temperature_tex'])


def make_tex_string(row, cen, low_err, up_err, fmt='${:.0f}^{{+{:.0f}}}_{{-{:.0f}}}$'):
    """
    Make a LaTex string out of values in the dataframe row.
    cen: the column name of the central value
    low_err: the column name of the lower error
    up_err: the column name of the upper error
    """
    if pd.notnull(row[cen]):
        return fmt.format(row[cen], row[low_err], row[up_err])
    return '\\nodata'


def old_main():
    # Read the full sample
    full_sample = read_full_sample()

    # Read the (previously processed) wds and sb9 databases
    wds = pd.read_csv('WDS_Sample.csv')
    sb9 = pd.read_csv('SB9_Sample.csv')

    # Read the known sample
    known_stars = pd.read_csv('Known_Sample.csv')

    # Pull the information from sample out of known stars
    sample = pd.merge(known_stars, full_sample, on='identifier')

    # Run through the pre-tabulated MCMC fits to measured --> actual temperature
    corrected = get_corrected_temperatures(sample.copy())
    corrected['identifier'] = corrected['Star']

    # Merge the wds and sb9, only keeping the values where there is a secondary temperature estimate
    known_data = pd.merge(wds[['identifier', 'pri_teff', 'pri_teff_err', 'sec_teff', 'sec_teff_lowerr', 'sec_teff_uperr']], 
                          sb9[['identifier', 'sec_teff', 'sec_teff_lowerr', 'sec_teff_uperr']], 
                          on='identifier', suffixes=['_wds', '_sb9'], how='outer').dropna(subset=['sec_teff_wds', 'sec_teff_sb9'], how='all').drop_duplicates()
    
    # Some of the wds low_err values are negative. Set those equal to the up_err (make error bars symmetric)
    known_data['sec_teff_lowerr_wds'] = known_data.apply(lambda r: r['sec_teff_lowerr_wds'] if r['sec_teff_lowerr_wds'] > 0 else r['sec_teff_uperr_wds'], axis=1)

    # Convert the known temperatures to latex
    known_data['wds_teff_tex'] = known_data.apply(lambda r: make_tex_string(r, 'sec_teff_wds', 'sec_teff_lowerr_wds', 'sec_teff_uperr_wds'), axis=1)
    known_data['sb9_teff_tex'] = known_data.apply(lambda r: make_tex_string(r, 'sec_teff_sb9', 'sec_teff_lowerr_sb9', 'sec_teff_uperr_sb9'), axis=1)
    known_data['exp_tex'] = known_data.apply(lambda r: r['sb9_teff_tex'] if r['sb9_teff_tex'] != '\\nodata' else r['wds_teff_tex'], axis=1)

    # Add the known data to my measurements
    total = pd.merge(known_data, corrected.dropna(subset=['Temperature_tex']), on='identifier', how='outer')
    
    # Save to table
    outfile='Known_Properties_Table.tex'
    table = total[['identifier', 'Temperature_tex', '[Fe/H]', 'vsini', 'exp_tex']].sort('identifier')
    table.to_latex(outfile, index=False, na_rep='\\nodata', escape=False, float_format=lambda f: '{:.2f}'.format(f))
    print(table)


def convert_dates(d):
    sd = str(d)
    year = sd[:4]
    month = sd[4:6]
    day = int(sd[6:])
    return '{}-{}-{:02d}'.format(year, month, day+1)


def main(star_list=SAMPLE_STARS):
    # Read the full sample
    full_sample = read_full_sample()

    # Fix the dates
    full_sample['Parsed_date'] = full_sample.Date.map(convert_dates)
    full_sample.replace(to_replace={'Instrument': {'HRS': 'HET'}}, inplace=True)

    # Get the detected stars that are in the star list
    detected = full_sample.loc[full_sample.Temperature.notnull()]
    matches = detected.loc[detected.identifier.isin(star_list)][['identifier', 'Instrument', 'Parsed_date', 'Temperature', '[Fe/H]', 'vsini_sec']].copy()

    # Get the measured values from the cross-correlation values
    hdf_int = HDF5_Helpers.Full_CCF_Interface()
    m_list = []
    for _, row in matches.iterrows():
        measured = hdf_int.get_measured_temperature(row['identifier'],
                                                    row['Parsed_date'],
                                                    row['Temperature'],
                                                    instrument=row['Instrument'],
                                                    feh=row['[Fe/H]'],
                                                    vsini=row['vsini_sec'])
        m_list.append(measured)
    measurements = pd.concat(m_list, ignore_index=True)

    # Convert to true values for each individual star
    corrected = hdf_int.convert_measured_to_actual(measurements)
    corrected['T_tex'] = corrected.apply(lambda r: make_tex_string(r, 'Corrected_Temperature', 'T_lowerr', 'T_uperr'), axis=1)

    print(corrected)
    return corrected



if __name__ == '__main__':
    #star_list = ['HIP 2912', 'HIP 13165', 'HIP 3881', 'HIP 14576', 'HIP 18724', 'HIP 22958', 'HIP 24902', 
    #'HIP 33372', 'HIP 65477', 'HIP 76267', 'HIP 77516', 'HIP 79199', 'HIP 79404', 'HIP 81641', 
    #'HIP 84606', 'HIP 88290', 'HIP 89156', 'HIP 91118', 'HIP 92027', 'HIP 98055', 'HIP 100221', 
    #'HIP 113788', 'HIP 116247', 'HIP 116611']
     
    star_list = SAMPLE_STARS 
    main(star_list)

