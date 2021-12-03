# ==================================================================================
# Imp: Demo
# Computes demographic difference between CTIS and Census (by week or overall using time_grain)
# ==================================================================================

import argparse
import configparser
import datetime
import numpy as np
import pandas as pd
import sys
import time
from IPython.display import display_html
import warnings
warnings.filterwarnings('ignore')
# warnings.resetwarnings()
# warnings.simplefilter('default')


def average(lst):
    return sum(lst) / len(lst)


def round_lst(lst, precision):
    return [round(num, precision) for num in lst]


def abs_lst(lst):
    return [abs(num) for num in lst]


def compute_value_counts(df_name, col_list):
    list_dfs = []
    for col in col_list:
        list_dfs.append('{}.{}.value_counts().to_frame()'.format(df_name, col))
    return list_dfs


def gen_df_demo_gov(
        data_file='../data/population-estimate/IHME_GBD_2019_POP_2019_Y2020M10D15.CSV',
        test=False):
    # # Step 1. read in Cesus data

    dict_gov_to_CTIS_country_name_map = {
        "Bolivia (Plurinational State of)": "Bolivia",
        "Democratic Republic of the Congo": 'Congo',
        "Democratic People's Republic of Korea": "Korea",
        "Lao People's Democratic Republic": "Lao",
        "Republic of Moldova": "Moldova",
        "Taiwan (Province of China)": "Taiwan",
        "United Arab Emirates": 'UAE',
        "United Republic of Tanzania": "Tanzania",
        "Venezuela (Bolivarian Republic of)": "Venezuela"
    }

    # l_countries_not_in_gov_data = ['Hong Kong', 'Congo']

    list_fb_age_group_name = ['15 to 19', '20 to 24', '25 to 29', '30 to 34',
                              '35 to 39', '40 to 44', '45 to 49', '50 to 54',
                              '55 to 59', '60 to 64', '65 to 69', '70 to 74', '75 plus']

    df_demo_gov = pd.read_csv(data_file)

    print("Original:", df_demo_gov.shape, end=' ')
    print(df_demo_gov.info())

    if test:
        print(df_demo_gov.head())
        value_count_string = ', '.join(compute_value_counts('df_demo_gov', df_demo_gov.columns.to_list()))
        print(value_count_string)

    # display_side_by_side(
    #     df_demo_gov.sex_name.value_counts().to_frame(),
    #     df_demo_gov.age_group_name.value_counts().to_frame(),
    #     df_demo_gov.year_id.value_counts().to_frame(),
    #     df_demo_gov.measure_name.value_counts().to_frame(),
    #     df_demo_gov.metric_name.value_counts().to_frame(),
    # )

    # # Step 2: reindex data to agegroup and filter (year==2019, measure_name==Prevalence)
    df_demo_gov.set_index('age_group_name', drop=False, inplace=True)

    df_demo_gov = df_demo_gov[
        (df_demo_gov.age_group_name.isin(list_fb_age_group_name)) & (df_demo_gov.measure_name == 'Population') & (
                df_demo_gov.sex_name != 'both') & (
                df_demo_gov.metric_name == 'Number')].copy()  ## 1350 regions and countries* 13 age-groups

    df_demo_gov.sort_index(inplace=True)
    print("Filtered:", df_demo_gov.shape)
    # display_side_by_side(df_demo_gov.age_group_name.value_counts().to_frame())
    print(df_demo_gov.head())

    # Step 3: preprocess
    # country names mapping when name doesn't match in gov and CTIS data
    if test: print('Country names before preprocess:', sorted(df_demo_gov.location_name.unique().tolist()))

    df_demo_gov['location_name'] = pd.np.where(
        df_demo_gov['location_name'].isin(list(dict_gov_to_CTIS_country_name_map.keys())),
        df_demo_gov['location_name'].map(dict_gov_to_CTIS_country_name_map), df_demo_gov['location_name'])

    # # Location name in Govt. data
    if test: print('Country names after preprocess:', sorted(df_demo_gov.location_name.unique().tolist()))

    return df_demo_gov


def gen_file_name(country_iso3, directory='.', source='raw'):
    """ Generate data file name to process data by country"""
    prefix = ''
    suffix = '.csv'
    # provide common prefix of survey data file name by country
    if source == 'raw':
        prefix = 'fbsurvey_intl_raw_all_rows_all_cols_country'

    elif source == 'demo_filtered':
        prefix = 'fbsurvey_intl_demo_filtered_rows_selected_cols_country'

    file_name = directory + '/' + prefix + '_' + country_iso3 + suffix
    return file_name


def read_data_col_list(country_iso3, list_cols, directory, source):
    """ Read the survey data in csv format and return dataframe for processing"""
    if len(list_cols) > 0:
        df = pd.read_csv(gen_file_name(country_iso3, directory, source=source), na_filter=False, skipinitialspace=True,
                         usecols=list_cols)
    else:
        df = pd.read_csv(gen_file_name(country_iso3, directory, source=source), na_filter=False, skipinitialspace=True)

    l_incomplete_value_identifiers = ['-99', -99, -99.0, '-99.0', '-77', -77, -77.0, '-77.0', '', ' ']
    df = df.replace(l_incomplete_value_identifiers, np.NaN)

    return df


def gen_date_index_and_filter(df, start_date, end_date, survey_version=False):
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    df['index_old'] = df.index
    df['recordeddate'] = pd.to_datetime(df['recordeddate']).dt.normalize()
    df.set_index('recordeddate', drop=False, inplace=True)

    if survey_version:
        df = df[(df.index.date >= start_date) & (df.index.date <= end_date) & (df.survey_version == survey_version)]
    else:
        df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]

    df.sort_index(inplace=True)
    return df


def display_side_by_side(*args):
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table', 'table style="display:inline; border-style:hidden; align:center"'), raw=True)


def write_dict_to_csv(dict_name, output_dir, output_file):
    df_demo_diff = pd.DataFrame.from_dict(dict_name, orient='index')
    df_demo_diff.index.name = 'country'
    df_demo_diff.to_csv(output_dir+'/'+output_file, index=True, header=True)


# ---------------------------- main() begins ----------------------------#


def main():
    print("\nProgram starts")
    start_time_all = time.time()

    print('''===== Generate (weekly) demographic differences between CTIS and Census data ======''')

    # # define code argument to accept the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True)
    args = parser.parse_args()

    # Define config parser object
    config = configparser.ConfigParser()
    config.read(args.file)

    # Read in all the config/global variables from the config file
    TEST = config['GLOBAL'].getboolean('TEST')
    TEST_WITH_NO_VERBOSE = config['GLOBAL'].getboolean('TEST_WITH_NO_VERBOSE')
    WEEKLY_ANALYSIS = config['GLOBAL'].getboolean('WEEKLY_ANALYSIS')
    WEIGHTED_COUNTRIES = config['GLOBAL'].getboolean('WEIGHTED_COUNTRIES')
    SOURCE = config['GLOBAL']['SOURCE']
    PRECISION = config['GLOBAL'].getint('PRECISION')

    START_DATE = config['GLOBAL'][
        'START_DATE']  ##Typically, a start date of a wave; Technically, a closest Monday (forward) from the date
    END_DATE = config['GLOBAL'][
        'END_DATE']  ##Typically, an open date; Technically, a closest Sunday (backward) from the date.

    INP_DIRECTORY = config['IO']['INP_DIRECTORY']

    # # gender response map
    dict_e3 = {
        1: 'Male',
        2: 'Female',
        3: 'Other',
        4: 'Prefer not to answer'
    }

    # # age-group response map
    dict_e4 = {
        1: '18-24',
        2: '25-34',
        3: '35-44',
        4: '45-54',
        5: '55-64',
        6: '65-74',
        7: '75+'}

    # # important columns
    imp_cols = ['iso_3', 'recordeddate', 'count_symp_endorsed', 'q_totalduration',
                'weight', 'survey_version', 'country_agg', 'region_agg', 'name_1']
    demo_cols = ['e3', 'e4']  # ==> e2: area; e3:gender, e4: age

    list_cols = imp_cols + demo_cols  # imp_cols+demo_cols+pmc_cols
    if TEST: print(list_cols)

    if TEST:
        output_dir = config['IO']['OUTPUT_DIR']  #'test'
        if WEIGHTED_COUNTRIES:
            l_countries = ['AFG']  # ['THA','LAO']
            l_names_countries = ['Afghanistan']  # ['Thailand','Lao']
        else:
            l_countries = ['ALA', 'AND']
            l_names_countries = ['Aland Islands', 'Andorra']
    else:
        output_dir = config['IO']['OUTPUT_DIR']
        if WEIGHTED_COUNTRIES:
            country_matadata_file = config['IO']['COUNTRY_MATADATA_FILE']
        else:
            country_matadata_file = config['IO']['COUNTRY_MATADATA_FILE_UNWEIGHTED']

        l_countries, l_names_countries = read_country_metadata(country_matadata_file, names=True)

        if TEST_WITH_NO_VERBOSE: l_countries, l_names_countries = ['THA', 'LAO'], ['Thailand', 'Lao']

    # #check
    print(len(l_names_countries), l_countries)
    # sys.exit(0)

    dict_demo_diff_weekly = {}
    for country, name_country in zip(l_countries, l_names_countries):
        start_time_country = time.time()

        # # ================ CTIS data begins ======================
        print("{}({})".format(name_country, country), end='  ')

        # # Step 1. read in data
        df_demo_fb = read_data_col_list(country, list_cols, INP_DIRECTORY, SOURCE)

        print("Original:", df_demo_fb.shape, end=' ')
        if TEST: print(df_demo_fb.info())

        # # Step 2: date-index and filter
        df_demo_fb = gen_date_index_and_filter(df_demo_fb, START_DATE, END_DATE)
        print("Filtered:", df_demo_fb.shape)

        # #  Step 3: preprocess
        # # to float type
        df_demo_fb.rename(columns={'e3': 'gender', 'e4': 'age_group'}, inplace=True)

        df_demo_fb.age_group = df_demo_fb.age_group.map(dict_e4)  ## int to age-group string
        df_demo_fb.gender = df_demo_fb.gender.map(dict_e3)  ## int to gender string

        if TEST:
            print(df_demo_fb.info())
            # display_side_by_side(
            #     df_demo_fb.age_group.value_counts().to_frame(),
            #     df_demo_fb.gender.value_counts().to_frame(),
            # )
            print(df_demo_fb.age_group.value_counts())
            print(df_demo_fb.gender.value_counts())

        if WEEKLY_ANALYSIS:
            df_demo_fb['iso_week'] = df_demo_fb.recordeddate.apply(lambda x: str(x.isocalendar()[1]).zfill(2))
        else:
            # # create overall age_gender counts
            df_demo_fb['iso_week'] = 100

        # # create weekly age_gender counts
        df_demo_fb_byAG = df_demo_fb[(df_demo_fb['age_group'].notnull()) & (df_demo_fb['gender'].isin(['Male', 'Female']))][
            ['age_group', 'gender', 'recordeddate', 'iso_week']].groupby(
            ['iso_week', 'age_group', 'gender']).count()  # .reset_index()

        if TEST: print(df_demo_fb_byAG)

        # # Weekly demo counts by bins and difference
        dict_demo_diff_this_country = {}
        # # multiindex level 0 = iso_week, level 1 = age_group, level 2 = gender
        for iso_week in df_demo_fb_byAG.index.levels[0].unique():
            if TEST: print("iso_week:", iso_week)

            # # Filter an iso_week data when reset_index() was NOT used
            df_demo_fb_byAG_this_week = df_demo_fb_byAG.iloc[
                df_demo_fb_byAG.index.get_level_values('iso_week') == iso_week].copy()

            if TEST:
                print(df_demo_fb_byAG_this_week)
                print(set(df_demo_fb_byAG_this_week.index.levels[1]))

            for age_group in ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']:
                for gender in ['Male', 'Female']:
                    if ~df_demo_fb_byAG_this_week.index.isin([(iso_week, age_group, gender)]).any():
                        df_demo_fb_byAG_this_week.loc[(iso_week, age_group, gender)] = [0]

            df_demo_fb_byAG_this_week = df_demo_fb_byAG_this_week.reset_index()
            df_demo_fb_byAG_this_week = df_demo_fb_byAG_this_week[['age_group', 'gender', 'recordeddate']]

            df_demo_fb_byAG_this_week.rename(columns={'recordeddate': 'val'}, inplace=True)
            df_demo_fb_byAG_this_week.set_index('age_group', drop=False, inplace=True)
            df_demo_fb_byAG_this_week.sort_index(inplace=True)

            # # initialize missing age_gender bins to zero
            if TEST: print(df_demo_fb_byAG_this_week)

            total = len(df_demo_fb_byAG_this_week[(df_demo_fb_byAG_this_week['age_group'].notnull()) & (
                df_demo_fb_byAG_this_week['gender'].isin(['Male', 'Female']))])

            if TEST:
                print(total)
                print(df_demo_fb_byAG_this_week)

            df_demo_fb_this_male = df_demo_fb_byAG_this_week[(df_demo_fb_byAG_this_week.gender == 'Male')].copy()
            df_demo_fb_this_female = df_demo_fb_byAG_this_week[(df_demo_fb_byAG_this_week.gender == 'Female')].copy()

            age_index = ['18-34', '35-54', '55-75+']

            total_demo_fb_male = df_demo_fb_this_male.at['18-24', 'val'] \
                                 + df_demo_fb_this_male.at['25-34', 'val'] \
                                 + df_demo_fb_this_male.at['35-44', 'val'] \
                                 + df_demo_fb_this_male.at['45-54', 'val'] \
                                 + df_demo_fb_this_male.at['55-64', 'val'] \
                                 + df_demo_fb_this_male.at['65-74', 'val'] \
                                 + df_demo_fb_this_male.at['75+', 'val']

            total_demo_fb_female = df_demo_fb_this_female.at['18-24', 'val'] \
                                   + df_demo_fb_this_female.at['25-34', 'val'] \
                                   + df_demo_fb_this_female.at['35-44', 'val'] \
                                   + df_demo_fb_this_female.at['45-54', 'val'] \
                                   + df_demo_fb_this_female.at['55-64', 'val'] \
                                   + df_demo_fb_this_female.at['65-74', 'val'] \
                                   + df_demo_fb_this_female.at['75+', 'val']

            total_demo_fb = total_demo_fb_male + total_demo_fb_female
            if TEST: print("total_demo_fb_male:{}, total_demo_fb_female:{}, total_demo_fb:{}".format(total_demo_fb_male,
                                                                                                     total_demo_fb_female,
                                                                                                     total_demo_fb))

            demo_fb_male = [
                (df_demo_fb_this_male.at['18-24', 'val'] + df_demo_fb_this_male.at['25-34', 'val']) / total_demo_fb,
                (df_demo_fb_this_male.at['35-44', 'val'] + df_demo_fb_this_male.at['45-54', 'val']) / total_demo_fb,
                (df_demo_fb_this_male.at['55-64', 'val'] + df_demo_fb_this_male.at['65-74', 'val'] +
                 df_demo_fb_this_male.at['75+', 'val']) / total_demo_fb
            ]

            demo_fb_female = [
                (df_demo_fb_this_female.at['18-24', 'val'] + df_demo_fb_this_female.at['25-34', 'val']) / total_demo_fb,
                (df_demo_fb_this_female.at['35-44', 'val'] + df_demo_fb_this_female.at['45-54', 'val']) / total_demo_fb,
                (df_demo_fb_this_female.at['55-64', 'val'] + df_demo_fb_this_female.at['65-74', 'val'] +
                 df_demo_fb_this_female.at['75+', 'val']) / total_demo_fb
            ]

            #     print(demo_fb_male, demo_fb_female)
            #     sys.exit(0)

            # # ======== Census data Begins =============
            CENSUS_DATA_FILE = config['IO']['CENSUS_DATA_FILE']
            df_demo_gov = gen_df_demo_gov(data_file=CENSUS_DATA_FILE, test=TEST)

            # # Get male and female counts separately
            df_demo_gov_this_male = df_demo_gov[
                (df_demo_gov.location_name == name_country) & (df_demo_gov.sex_name == 'male')].copy()
            df_demo_gov_this_female = df_demo_gov[
                (df_demo_gov.location_name == name_country) & (df_demo_gov.sex_name == 'female')].copy()

            if TEST:
                print(df_demo_gov_this_male[['val']], df_demo_gov_this_female[['val']])

            total_demo_gov_male = df_demo_gov_this_male.at['15 to 19', 'val'] \
                                  + df_demo_gov_this_male.at['20 to 24', 'val'] \
                                  + df_demo_gov_this_male.at['25 to 29', 'val'] \
                                  + df_demo_gov_this_male.at['30 to 34', 'val'] \
                                  + df_demo_gov_this_male.at['35 to 39', 'val'] \
                                  + df_demo_gov_this_male.at['40 to 44', 'val'] \
                                  + df_demo_gov_this_male.at['45 to 49', 'val'] \
                                  + df_demo_gov_this_male.at['50 to 54', 'val'] \
                                  + df_demo_gov_this_male.at['55 to 59', 'val'] \
                                  + df_demo_gov_this_male.at['60 to 64', 'val'] \
                                  + df_demo_gov_this_male.at['65 to 69', 'val'] \
                                  + df_demo_gov_this_male.at['70 to 74', 'val'] \
                                  + df_demo_gov_this_male.at['75 plus', 'val']

            total_demo_gov_female = df_demo_gov_this_female.at['15 to 19', 'val'] \
                                    + df_demo_gov_this_female.at['20 to 24', 'val'] \
                                    + df_demo_gov_this_female.at['25 to 29', 'val'] \
                                    + df_demo_gov_this_female.at['30 to 34', 'val'] \
                                    + df_demo_gov_this_female.at['35 to 39', 'val'] \
                                    + df_demo_gov_this_female.at['40 to 44', 'val'] \
                                    + df_demo_gov_this_female.at['45 to 49', 'val'] \
                                    + df_demo_gov_this_female.at['50 to 54', 'val'] \
                                    + df_demo_gov_this_female.at['55 to 59', 'val'] \
                                    + df_demo_gov_this_female.at['60 to 64', 'val'] \
                                    + df_demo_gov_this_female.at['65 to 69', 'val'] \
                                    + df_demo_gov_this_female.at['70 to 74', 'val'] \
                                    + df_demo_gov_this_female.at['75 plus', 'val']

            total_demo_gov = total_demo_gov_male + total_demo_gov_female

            if TEST:
                print("total_demo_gov_male:{}, total_demo_gov_female:{}, total_demo_gov:{}".format(total_demo_gov_male,
                                                                                                   total_demo_gov_female,
                                                                                                   total_demo_gov))

            demo_gov_male = [
                (df_demo_gov_this_male.at['15 to 19', 'val'] + df_demo_gov_this_male.at['20 to 24', 'val'] +
                 df_demo_gov_this_male.at['25 to 29', 'val'] + df_demo_gov_this_male.at[
                     '30 to 34', 'val']) / total_demo_gov,
                (df_demo_gov_this_male.at['35 to 39', 'val'] + df_demo_gov_this_male.at['40 to 44', 'val'] +
                 df_demo_gov_this_male.at['45 to 49', 'val'] + df_demo_gov_this_male.at[
                     '50 to 54', 'val']) / total_demo_gov,
                (df_demo_gov_this_male.at['55 to 59', 'val'] + df_demo_gov_this_male.at['60 to 64', 'val'] +
                 df_demo_gov_this_male.at['65 to 69', 'val'] + df_demo_gov_this_male.at['70 to 74', 'val'] +
                 df_demo_gov_this_male.at['75 plus', 'val']) / total_demo_gov
            ]

            demo_gov_female = [
                (df_demo_gov_this_female.at['15 to 19', 'val'] + df_demo_gov_this_female.at['20 to 24', 'val'] +
                 df_demo_gov_this_female.at['25 to 29', 'val'] + df_demo_gov_this_female.at[
                     '30 to 34', 'val']) / total_demo_gov,
                (df_demo_gov_this_female.at['35 to 39', 'val'] + df_demo_gov_this_female.at['40 to 44', 'val'] +
                 df_demo_gov_this_female.at['45 to 49', 'val'] + df_demo_gov_this_female.at[
                     '50 to 54', 'val']) / total_demo_gov,
                (df_demo_gov_this_female.at['55 to 59', 'val'] + df_demo_gov_this_female.at['60 to 64', 'val'] +
                 df_demo_gov_this_female.at['65 to 69', 'val'] + df_demo_gov_this_female.at['70 to 74', 'val'] +
                 df_demo_gov_this_female.at['75 plus', 'val']) / total_demo_gov
            ]

            # # demo_gov - demo_fb; leads to negative value when CTIS over represented a demographic bin
            #     demo_gov_male_diff_fb_male = [i-j for i,j in zip(demo_gov_male, demo_fb_male)]
            #     demo_gov_female_diff_fb_female = [i-j for i,j in zip(demo_gov_female, demo_fb_female)]

            # # demo_fb - demo_gov; leads to positive value when CTIS over represented a demographic bin
            demo_gov_male_diff_fb_male = [i - j for i, j in zip(demo_fb_male, demo_gov_male)]
            demo_gov_female_diff_fb_female = [i - j for i, j in zip(demo_fb_female, demo_gov_female)]

            # # raw
            #     dict_demo_diff[name_country] = {'male': [round(num, 1) for num in a_list] demo_gov_male_diff_fb_male,
            #                                    'female': demo_gov_female_diff_fb_female,
            #                                     'total_diff':sum(demo_gov_male_diff_fb_male+demo_gov_female_diff_fb_female),
            #                                     'mean_diff': average(demo_gov_male_diff_fb_male+demo_gov_female_diff_fb_female)
            #                                    }
            # # rounded
            dict_demo_diff_this_country[iso_week] = {'male': round_lst(demo_gov_male_diff_fb_male, PRECISION),
                                                     'female': round_lst(demo_gov_female_diff_fb_female, PRECISION),
                                                     'total_diff': round(sum(round_lst(abs_lst(
                                                         demo_gov_male_diff_fb_male + demo_gov_female_diff_fb_female), PRECISION)),
                                                         PRECISION),
                                                     'mean_diff': round(average(round_lst(abs_lst(
                                                         demo_gov_male_diff_fb_male + demo_gov_female_diff_fb_female), PRECISION)),
                                                         PRECISION)
                                                     }

            if TEST and country == 'AFG':
                print('Census', end=':')
                print(age_index)
                print("Male:{} Female:{}".format(demo_gov_male, demo_gov_female))

                print('CITS', end=':')
                print(age_index)
                print("Male:{} Female:{}".format(demo_fb_male, demo_fb_female))

                # print('demo_diff[AFG]:', dict_demo_diff_this_country[name_country])

        dict_demo_diff_weekly[name_country] = dict_demo_diff_this_country

    # print(dict_demo_diff_weekly)
    OUTPUT_FILE = config['IO']['OUTPUT_FILE']
    write_dict_to_csv(dict_name=dict_demo_diff_weekly, output_dir=output_dir, output_file=OUTPUT_FILE)

    print(" [All Done][{} sec]".format(round(time.time() - start_time_all, 1)), end='\n')
    print("\nProgram ends")


if __name__ == "__main__": main()
