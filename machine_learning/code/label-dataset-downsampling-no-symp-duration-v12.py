import csv, sys
import datetime, time
from IPython.display import display_html

import pickle

pickle.HIGHEST_PROTOCOL = 4

import warnings

warnings.filterwarnings('ignore')
# warnings.resetwarnings() 
# warnings.simplefilter('default')

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas import DataFrame
from IPython.display import HTML
import datetime
import pandas as pd
import matplotlib.pyplot as plt
# import xgboost as xgb
import lightgbm as lgbm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix
from sklearn.utils import resample
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import shap
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV

dict_global_fb_variables_selected = {
    # survey variables
    'survey_version': 'survey_version',
    'weight': 'weight',
    'recordeddate': 'recordeddate',
    'q_totalduration': 'q_totalduration',
    # geography
    'country_agg': 'country_agg',
    'region_agg': 'region_agg',
    'gid_0': 'gid_0',
    'gid_1': 'gid_1',
    'iso_3': 'iso_3',
    # demography
    'e3': 'demo_sex',
    'e4': 'demo_age',
    'e6': 'demo_edu',
    'e2': 'demo_live',
    'e5': 'demo_nsleep',
    'e7': 'demo_nroom',
    'd7': 'wk_current',
    'd10': 'wk_type',
    # behaviors
    'd3': 'bx_worrycovid',
    'd5': 'bx_worrymoney',
    'b3': 'bx_knowsick',
    'b4': 'bx_knowsickn',
    'b5': 'ex_conknowsick',
    # exposures
    'c1_m': 'ex_condirect',
    'c2': 'ex_condirctn',
    'c3': 'ex_conhealth',
    'c6': 'ex_conexhouse',
    'c14': 'ex_conavoid',
    'c0_1': 'ex_conwork',
    'c0_2': 'ex_conessen',
    'c0_3': 'ex_conshop',
    'c0_4': 'ex_consocial',
    'c0_5': 'ex_congroup',
    'c5': 'ex_mskpublic',
    # symptoms
    'b1_1': 'sx_fever',
    'b1_2': 'sx_cough',
    'b1_3': 'sx_diffbr',
    'b1_4': 'sx_fatigue',
    'b1_5': 'sx_runnose',
    'b1_6': 'sx_achepain',
    'b1_7': 'sx_throat',
    'b1_8': 'sx_chestpain',
    'b1_9': 'sx_nausea',
    'b1_10': 'sx_smelltaste',
    'b1_11': 'sx_eyepain',
    'b1_12': 'sx_headache',
    'b1_13': 'sx_chills',
    'b1_14': 'sx_sleep',
    'count_symp_endorsed': 'count_symp_endorsed',
    'b1b_x1': 'sx_feverun',
    'b1b_x2': 'sx_coughun',
    'b1b_x3': 'sx_diffbrun',
    'b1b_x4': 'sx_fatigueun',
    'b1b_x5': 'sx_runnoseun',
    'b1b_x6': 'sx_achepainun',
    'b1b_x7': 'sx_throatun',
    'b1b_x8': 'sx_chestpainun',
    'b1b_x9': 'sx_nauseaun',
    'b1b_x10': 'sx_smelltasteun',
    'b1b_x11': 'sx_eyepainun',
    'b1b_x12': 'sx_headacheun',
    'b1b_x13': 'sx_chillsun',
    'b1b_x14': 'sx_sleepun',
    # testing and access
    'b6': 'ts_ever',
    'b7': 'ts_recent',
    'b8': 'ts_pos',
    'b2': 'symp_days',
    'b3': 'ex_sick_know',
    'b4': 'ex_num',
    'b5': 'ex_sick_contact'
}


def gen_file_name(country_iso3, directory='.'):
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


def read_data_col_list(country_iso3, list_cols, directory):
    if len(list_cols) > 0:
        df = pd.read_csv(gen_file_name(country_iso3, directory), na_filter=False, skipinitialspace=True,
                         usecols=list_cols)
    else:
        df = pd.read_csv(gen_file_name(country_iso3, directory), na_filter=False, skipinitialspace=True)

    l_incomplete_value_identifiers = ['-99', -99, -99.0, '-99.0', '-77', -77, -77.0, '-77.0', '', ' ']
    df = df.replace(l_incomplete_value_identifiers, np.NaN)

    return df


def rename_survey_cols(df, dict_rename):
    return df.rename(columns=dict_rename)


# # Swapped dictionary for reversing the named columns to the original columns
def gen_swapped_col_list(col_list):
    dict_global_fb_variables_selected_swapped = dict([(value, key) for key,
                                                                       value in
                                                      dict_global_fb_variables_selected.items()])
    #     print(dict_global_fb_variables_selected_swapped)
    new_col_list = [v for k, v in dict_global_fb_variables_selected_swapped.items() if k in col_list]
    return new_col_list


def read_country_metadata(file_name, names=False):
    df_FB_countries = pd.read_csv(
        file_name)  # ; df_FB_countries=df_FB_countries.loc[~df_FB_countries['ISO3'].isin(['AFG', 'KHM', 'BGD', 'BLR', 'ALB', 'MMR', 'CRI', 'MDA', 'BRA', 'ARM'])]
    #     print (df_FB_countries.head())
    l_countries = list(df_FB_countries['ISO3'].values)
    # l_countries = [country for country in l_countries if country not in ['CHN', 'CUB', 'IRN', 'PRK', 'SYR', 'TKM', np.NaN]]
    l_countries = [country for country in l_countries if
                   country not in ['CHN', 'CUB', 'IRN', 'PRK', 'SYR', 'TKM', np.NaN]]
    l_name_countries = list(df_FB_countries['NAME'].values)
    if names:
        return l_countries, l_name_countries
    else:
        return l_countries


def pre_prep_data(df, outcome_var, na_logic=False):
    CHECK = False

    df["young_male"] = np.nan
    df["old_male"] = np.nan
    df["young_female"] = np.nan
    df["old_female"] = np.nan

    if not na_logic:
        ## === missing_demo = 0
        df['demo_sex'] = pd.to_numeric(df['demo_sex'], errors='coerce').fillna(0).astype(int)
        df['demo_age'] = pd.to_numeric(df['demo_age'], errors='coerce').fillna(0).astype(int)
    else:
        ## === missing_demo = np.nan
        df['demo_sex'] = pd.to_numeric(df['demo_sex'], errors='coerce')
        df['demo_age'] = pd.to_numeric(df['demo_age'], errors='coerce')

    df.loc[((df['demo_sex'] == 1) & (df['demo_age'] <= 4)), 'young_male'] = '1'
    df.loc[((df['demo_sex'] == 1) & (df['demo_age'] >= 5)), 'old_male'] = '1'
    df.loc[((df['demo_sex'] == 2) & (df['demo_age'] <= 4)), 'young_female'] = '1'
    df.loc[((df['demo_sex'] == 2) & (df['demo_age'] >= 5)), 'old_female'] = '1'

    if CHECK: print(df[['demo_sex', 'demo_age', 'young_male', 'old_male', 'young_female', 'old_female']].head().T)

    #     df['male']=df['demo_sex'].apply(lambda x: '1' if x==1 else ('0' if x==2 else np.nan))
    #     df['young']=df['demo_age'].apply(lambda x: '1' if x<=4 else ('0' if x>=5 else np.nan))

    #     df["symp_short"] = np.nan
    #     df["symp_medium"] = np.nan
    #     df["symp_high"] = np.nan
    #     df["symp_over14"] = np.nan
    #     df['symp_days'] = pd.to_numeric(df['symp_days'], errors='coerce').fillna(-1).astype(int)
    #     df[Falseo_age']=pd.to_numeric(df['demo_age'], errors='coerce').fillna(-1).astype(int).apply(lambda x: 'Young' if x>=1 and x<=4 else('Old' if x>=5 and x<=7 else 'NA'))

    #     df.loc[((df['symp_days']>=0)&(df['symp_days']<=4)),'symp_short'] = '1'
    #     df.loc[((df['symp_days']>=5)&(df['symp_days']<=9)),'symp_medium'] = '1'
    #     df.loc[((df['symp_days']>=10)&(df['symp_days']<=14)),'symp_high'] = '1'
    #     df.loc[((df['symp_days']>=15)&(df['symp_days']<=21)),'symp_over14'] = '1'

    if outcome_var == 'ts_pos':
        df['ts_pos'] = pd.to_numeric(df['ts_pos'], errors='coerce').fillna(0).astype(int)
        df["ts_positive"] = np.nan
        df.loc[df['ts_pos'] == 1, 'ts_positive'] = 1
        df.loc[df['ts_pos'] == 2, 'ts_positive'] = 0

    elif outcome_var == 'ts_recent':
        ### NOTE: For the least amount of code change, I added the logic to get the ts_14 in the same sudo variable ts_positive. Hence, you may see ourcome_varable as ts_recent but 'ts_positive' in dataframe info() after preprep step

        df['ts_recent'] = pd.to_numeric(df['ts_recent'], errors='coerce').fillna(0).astype(int)

        #         df["ts_14"] = np.nan
        #         df.loc[df['ts_recent']==1,'ts_14'] = 1
        #         df.loc[df['ts_recent']==2,'ts_14'] = 0

        df["ts_positive"] = np.nan
        df.loc[df['ts_recent'] == 1, 'ts_positive'] = 1
        df.loc[df['ts_recent'] == 2, 'ts_positive'] = 0

    df2 = df[
        ['ts_positive', 'young_male', 'old_male', 'young_female', 'old_female', 'sx_fever', 'sx_cough', 'sx_diffbr',
         'sx_fatigue', 'sx_runnose', 'sx_achepain', 'sx_throat', 'sx_chestpain', 'sx_nausea', 'sx_smelltaste',
         'sx_eyepain',
         'sx_headache', 'recordeddate', 'iso_3', 'weight', 'count_symp_endorsed', 'symp_days', 'ex_sick_know',
         'q_totalduration']]

    sx = [col for col in df2.columns.to_list() if col.startswith('sx_') or col.startswith('symp_')]

    df2[sx] = df2[sx].apply(pd.to_numeric, errors='coerce')  ## <=== missing_sx = NA
    # df2[sx] = df2[sx].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int) ## <=== missing_sx = 0

    df2 = df2.replace(2, 0)

    #     df2[['symp_short','symp_medium','symp_high','symp_over14','young_male','old_male',
    #     'young_female','old_female']] = df2[['symp_short','symp_medium','symp_high',
    #     'symp_over14','young_male','old_male','young_female','old_female']].replace(np.nan,0)

    if not na_logic:
        df2[['young_male', 'old_male', 'young_female', 'old_female']] = df2[
            ['young_male', 'old_male', 'young_female', 'old_female']].replace(np.nan, 0)

    return df2


def run_models(df, iso_3, ct_name, stratas, models, output_folder='', gen_labeled_df=False, output_file_prefix='',
               gen_labeled_df_remaining=False):
    VERBOSE = True

    selected_features = ['young_male', 'old_male', 'young_female', 'old_female', 'sx_fever', 'sx_cough', 'sx_diffbr',
                         'sx_fatigue', 'sx_runnose', 'sx_achepain', 'sx_throat', 'sx_chestpain', 'sx_nausea',
                         'sx_smelltaste', 'sx_eyepain', 'sx_headache']

    for selected_strata in stratas:
        start_time_strata = time.time()

        if selected_strata != 'demo_as_covariate':
            df = df.loc[df[selected_strata] == '1']
            df = df.drop(['old_male', 'young_male', 'old_female', 'young_female'], axis=1)
        else:
            df['old_male'] = df['old_male'].apply(pd.to_numeric, errors='coerce')
            df['young_male'] = df['young_male'].apply(pd.to_numeric, errors='coerce')
            df['old_female'] = df['old_female'].apply(pd.to_numeric, errors='coerce')
            df['young_female'] = df['young_female'].apply(pd.to_numeric, errors='coerce')

        print("  - Strata={}, n=({:,})".format(selected_strata, df.shape[0]))
        n_total = df.shape[0]

        if gen_labeled_df_remaining:
            print('Entire df')
            print(df.shape)
            print(df.info())
            print('-' * 40)

            df_prime = df[df['ts_positive'].isna()]
            print('ts_positive isna df:- ')
            print(df_prime.shape)
            print(df_prime.info())
            print('-' * 40)
        ## consider samples for training-testing using ts_pos or ts_positive 
        #         df_om = df_om.dropna() ##<==== Need to investigate
        #         df_om = df_om[df_om['ts_positive'].notna()]
        df = df.dropna(subset=['ts_positive'])

        print('ts_positive notna df:-')
        print(df.shape)
        print(df.info())
        print('-' * 40)
        ##sys.exit(0)

        ## Prepare train and test sets

        y = df.ts_positive
        X = df.drop(['ts_positive'], axis=1)
        print("y={}, X={}".format(y.shape, X.shape))

        # evaluate each model in turn
        for model_name, model in models:
            print(" ===== Data labeling using model: {} =======".format(model_name))
            start_time_model = time.time()
            print('=' * 50)
            print('===== IMPORTANT: Num. of features used in the model:{} ====='.format(model.n_features_))
            print('=' * 50)
            ## predicting lables
            print(' - Predicting lables on Part 1')
            # y_pred_train_pre_p = model.predict(X_train)
            y_pred_X = model.predict(X[selected_features])

            if gen_labeled_df:
                start_time_gen_labeled_df = time.time()
                labeled_data_file = output_folder + '/' + '_'.join(
                    ['labeled_df', output_file_prefix, model_name, selected_strata]) + '.csv'

                print(' - Generating labeled df_notna_ts_pos', end='...')
                df_l = X.copy()
                df_l['ts_pos_fb'] = y
                df_l.reset_index(inplace=True)
                df_l['ts_pos_pred'] = y_pred_X

                df_l.to_csv(labeled_data_file, index=False, na_rep='NA')
                print("  [Done][{} sec]".format(round(time.time() - start_time_gen_labeled_df, 1)), end='\n')
                print("   - df_l.shape:", df_l.shape)
                print('X-' * 20)

            if gen_labeled_df_remaining:
                print(" - Generating labeled 'df_na_ts_pos' dataframe", end='...')
                start_time_gen_labeled_df_remaining = time.time()
                labeled_remaining_data_file = output_folder + '/' + '_'.join(
                    ['labeled_df_remaining', output_file_prefix, model_name, selected_strata]) + '.csv'
                df_na_ts_pos = df_prime.drop(['ts_positive'], axis=1)
                df_na_ts_pos['ts_pos_fb'] = 'NA'
                df_na_ts_pos['ts_pos_pred'] = model.predict(df_na_ts_pos[selected_features])
                df_na_ts_pos.to_csv(labeled_remaining_data_file, index=True, index_label='index', na_rep='NA')
                print("  [Done][{} sec]".format(round(time.time() - start_time_gen_labeled_df_remaining, 1)), end='\n')
                print("   - df_na_ts_pos.shape:", df_na_ts_pos.shape)
                print('X-' * 20)

            print('-' * 40)

            print("    - [{} Done][{} sec]".format(model_name, round(time.time() - start_time_model, 1)))
        print("  - [Done][{} sec]".format(round(time.time() - start_time_strata, 1)), end='\n')


# #---------------------------- main() begins ----------------------------#
def main():
    start_time_all = time.time()
    print("\nProgram starts")

    print('''
    =======================-==================
    =========== Data Labeling ================
    ==========================================
    ''')

    # # define code argument to accept the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True)
    args = parser.parse_args()

    # Define config parser object
    config = configparser.ConfigParser()
    config.read(args.file)

    # Read in all the config/global variables from the config file

    TEST = True  # False #True
    TEST = config['GLOBAL'].getboolean('TEST')

    READ_FROM_DATA_PICKLE = False  # True #False
    READ_FROM_DATA_PICKLE = config['GLOBAL'].getboolean('READ_FROM_DATA_PICKLE')
    DATA_PICKLE_FILE = config['GLOBAL']['DATA_PICKLE_FILE'] if READ_FROM_DATA_PICKLE else ''
    #DATA_PICKLE_FILE = './run/Global-lightGBM-downsample-v12-ts_pos-hpt/run_model_three_LightGBM_demosx_NA/df_global-ts_pos-downsampled_data-v12_LightGBM_LightGBM_usemissing_false_LightGBM_tunned.zip'

    GEN_DATA_PICKEL_FOR_FUTURE = False if READ_FROM_DATA_PICKLE else True
    
    VERBOSE = bool(config['GLOBAL']['VERBOSE'])
    WEIGHTED_COUNTRIES = config['GLOBAL'].getboolean('WEIGHTED_COUNTRIES')
    SOURCE = config['GLOBAL']['SOURCE']
    
    # 'ts_recent' #'ts_pos'
    OUTCOME_VAR = config['GLOBAL']['OUTCOME_VAR']

    GEN_LABELED_DF = True  # True  ## <== Used for generating labeled data that comprised of: train,test, and held-out set data (all rows)
    GEN_LABELED_DF_REMAINING = True  # True ## <== Used for generating labled data comprised of: All data with demo and at least one symptom but no ts_pos i.e, All rows - (train+test+held-out set)

    # stratas = ['old_male','young_male','old_female','young_female','demo_as_covariate']
    SELECTED_STRATAS = config['GLOBAL']['SELECTED_STRATAS'].split(',')

    start_date = datetime.date(2020, 4, 26)  ## <=== actual_date=this_date + 1 day
    # start_date= datetime.date(2020,6,28) ## <=== actual_date=this_date + 1 day
    end_date = datetime.date(2020, 12, 21)  ## <=== actual_date=this_date - 1 day
    
    start_date_for_printing = start_date + datetime.timedelta(days=1)
    end_date_for_printing = end_date - datetime.timedelta(days=1)

    ## -------- models ------------

    EXTERNAL_MODEL_PICKLE_FILE = config['GLOBAL']['EXTERNAL_MODEL_PICKLE_FILE'] if RUN_EXTERNAL_MODEL else ''
    # ./run/Global-lightGBM-downsample-v12-ts_pos/clf_LightGBM_demo_as_covariate_global_v12_downsampled.zip'

    # EXTERNAL_MODEL_PICKLE_FILE_2 can be empty
    EXTERNAL_MODEL_PICKLE_FILE_2 = config['GLOBAL']['EXTERNAL_MODEL_PICKLE_FILE_2'] if RUN_EXTERNAL_MODEL else ''
    #./run/Global-lightGBM-downsample-v12-ts_pos-hpt/run_tuned_model_with_labeled_data_with_labeled_df_ts_pos_na/clf_LightGBM_tunned_demo_as_covariate_global_v12_downsampled.zip'
    
    # ['LightGBM', 'LightGBM_tuned']
    SELECTED_MODELS = config['GLOBAL']['SELECTED_MODELS'].split(',')

    models = []
    start_time_load_pickle = time.time()
    print(' - Loading the pickled models', end='...')
    loaded_model_1 = pickle.load(open(EXTERNAL_MODEL_PICKLE_FILE_2, 'rb'))
    loaded_model_2 = pickle.load(open(EXTERNAL_MODEL_PICKLE_FILE_2, 'rb'))
    print("  [Done][{} sec]".format(round(time.time() - start_time_load_pickle, 1)), end='\n')

    print('===== Unpickled model analysis =====')
    print(loaded_model_1)
    print(' - Num. of features used in the LightGBM:{} ====='.format(loaded_model_1.n_features_))
    print(loaded_model_1.feature_name_)
    print(loaded_model_1.feature_importances_)
    # for feat_name, wt in zip (loaded_model_1.feature_name_, loaded_model_1.feature_importances_):
    #    print('{} - {}'.format(feat_name, wt))
    print('Adding the new param(s) in the LightGBM model...')
    params_model_1 = loaded_model_1.get_params()
    params_model_1['use_missing'] = False
    loaded_model_1.set_params(**params_model_1)
    print('The new model is: -')
    print(loaded_model_1)

    print()
    print(loaded_model_2)
    print(' - Num. of features used in the LightGBM_tuned:{} ====='.format(loaded_model_2.n_features_))
    print(loaded_model_2.feature_name_)
    print(loaded_model_2.feature_importances_)
    # for feat_name, wt in zip (loaded_model_2.feature_name_, loaded_model_2.feature_importances_):
    #    print('{} - {}'.format(feat_name, wt))
    print('Adding the new param(s) in the LightGBM_tuned  model...')
    params_model_2 = loaded_model_2.get_params()
    params_model_2['use_missing'] = False
    loaded_model_2.set_params(**params_model_2)
    print('The new model is: -')
    print(loaded_model_2)
    print()

    models.append(('LightGBM', loaded_model_1))  ##<== with added use_missing=False
    models.append(('LightGBM_tuned', loaded_model_2))  ##<== with added use_missing=False
    selected_models = [m_tuple for m_tuple in models if m_tuple[0] in SELECTED_MODELS]
    # sys.exit(0)
    ## -------------------------------

    output_file_prefix = 'global-' + OUTCOME_VAR + '-downsampled_data-v12'  ## <== v12:Changed df_om.dropna() to df_om[df_om['ts_positive'].notna()]
    # output_metric_table = 'metric_table_'+output_file_prefix +'_' + '_'.join(SELECTED_MODELS)+'.csv'
    output_df_pickle = 'df_' + output_file_prefix + '_' + '_'.join(SELECTED_MODELS) + '.zip'

    print("- Test:", TEST)
    print("- Weighted countries:", WEIGHTED_COUNTRIES)
    print("- Data source:", SOURCE)
    print("- Date range:{} to {}".format(start_date_for_printing.strftime('%m/%d/%Y'),
                                         end_date_for_printing.strftime('%m/%d/%Y')))
    print('--------')
    print('- Read data from pickle:', READ_FROM_DATA_PICKLE)
    if READ_FROM_DATA_PICKLE:
        print('Data pickle file:', DATA_PICKLE_FILE)
    else:
        print(' - Generate the data pickel for future use: ', GEN_DATA_PICKEL_FOR_FUTURE)
    print('- Outcome variable:', OUTCOME_VAR)
    print('- Generate labeled dataframe:', GEN_LABELED_DF)
    print('--------')

    if TEST:
        #     warnings.resetwarnings()
        #     warnings.simplefilter('default'):w
        warnings.filterwarnings('ignore')
        output_dir = '.'
        if WEIGHTED_COUNTRIES:
            # l_countries = ['COD']#['CHE']
            # l_names_countries = ['Congo']#['Switzerland']
            l_countries = ['AFG', 'CHE']  # ['CHE'] #['COD']
            l_names_countries = ['Afganistan', 'Switzerland']  # ['Switzerland'] ['Congo']#
            weighted_for_read_file = True
        else:
            l_countries = ['ALA', 'AND']
            l_names_countries = ['Aland Islands', 'Andorra']
            weighted_for_read_file = False
    else:
        warnings.filterwarnings('ignore')

        output_dir = config['IO']['output_dir']

        if WEIGHTED_COUNTRIES:
            country_matadata_file = config['IO']['COUNTRY_MATADATA_FILE']
            weighted_for_read_file = True
        #         l_countries_skip = []
        else:
            country_matadata_file = 'fb_countries_unweighted.csv'
            weighted_for_read_file = False
        #         l_countries_skip = ['CHN', 'CUB', 'IRN', 'PRK', 'SYR', 'TKM', 'NA']

        l_countries, l_names_countries = read_country_metadata(country_matadata_file, names=True)

    print('------------')
    print("- Num of countries data to label:", len(l_countries))
    print('------------')

    ## 1.read in the data 
    df_all = pd.DataFrame()
    start_readingin = time.time()

    if not READ_FROM_DATA_PICKLE:
        print("- Readingin data from files", end='...')

        # req_cols = ['iso_3', 'recordeddate', 'count_symp_endorsed','q_totalduration','weight','survey_version']
        req_cols = ['iso_3', 'recordeddate', 'count_symp_endorsed', 'q_totalduration', 'weight', 'survey_version', 'b2',
                    'b3', 'q_totalduration']  # ,'symp_days','ex_sick_know']
        symp_cols = [col for col in dict_global_fb_variables_selected.keys() if
                     (col.startswith('b1_') or col in ['b2']) and col not in ['b1_13', 'b1_14']]  # <== get symptom cols
        demo_cols = [k for k, v in dict_global_fb_variables_selected.items() if v.startswith('demo_')]
        test_cols = [k for k, v in dict_global_fb_variables_selected.items() if v.startswith('ts_')]
        list_cols = req_cols + symp_cols + demo_cols + test_cols

        for country, name_country in zip(l_countries, l_names_countries):
            start_time_country = time.time()
            print("{}({})".format(name_country, country), end='  ')

            ## 1. read data
            df_r1 = read_data_col_list(country, list_cols, directory=CTIS_DATA_DIR)
            df_r1 = rename_survey_cols(df_r1, dict_global_fb_variables_selected)

            ## 2. select data from only study period
            df_r1['recordeddate'] = pd.to_datetime(
                df_r1['recordeddate']).dt.normalize()  ##<== Ignore time from the date-time
            df_r1['recordeddate'] = pd.to_datetime(df_r1['recordeddate'])
            df_r1.set_index('recordeddate', drop=False, inplace=True)
            df_r1 = df_r1[(df_r1.index.date > start_date) & (df_r1.index.date < end_date)]

            print("n={:,}".format(df_r1.shape[0]), end='|')

            df_all = pd.concat([df_all, df_r1], ignore_index=True)
        print("[Done][{} sec]".format(round(time.time() - start_readingin, 1)), end='\n')

        if VERBOSE:
            print(df_all.info())

        ## 2. prepare-data
        print("- Preparing data", end='...')
        start_prepare = time.time()
        print('===== Before pre_prep() =====')
        # print(df_all.loc[(df_all['demo_sex'].notna()) & df_all['sx_fever'].isnull()].head(5).T)
        print(df_all.loc[[44, 66]].T)
        df = pre_prep_data(df_all, outcome_var=OUTCOME_VAR, na_logic=True)
        print('===== After pre_prep() =====')
        # print(df.loc[(df['young_male']=='1') & df['sx_fever'].isnull()].head(5).T)
        print(df.loc[[44, 66]].T)
        print("[Done][{} sec]".format(round(time.time() - start_prepare, 1)), end='\n')
        if GEN_DATA_PICKEL_FOR_FUTURE:
            print('=' * 40)
            print("- Pickleing Dataframe", end='...')
            start_pickel = time.time()
            df.to_pickle(output_dir + '/' + output_df_pickle)
            print("[Done][{} sec]".format(round(time.time() - start_pickel, 1)), end='\n')
            print('=' * 40)
    else:
        print("- Readingin selected and prepared data from pickle", end='...')
        df = pd.read_pickle(DATA_PICKLE_FILE)
        print("n={:,}".format(df.shape[0]), end=' ')
        print("[Done][{} sec]".format(round(time.time() - start_readingin, 1)), end='\n')

    if VERBOSE:
        print('After preparing')
        print(df.info())
        # sys.exit(0)  ## breakpoint-check

    ## 5. running models
    print("- Running models...")

    clf_file_name = output_dir  ## <-- this get updated inside run_models()
    run_models(df, 'global', 'global', SELECTED_STRATAS, selected_models, output_folder=output_dir,
               gen_labeled_df=GEN_LABELED_DF, output_file_prefix=output_file_prefix,
               gen_labeled_df_remaining=GEN_LABELED_DF_REMAINING)

    print(" [All Done][{} sec]".format(round(time.time() - start_time_all, 1)), end='\n')

    print("\nProgram ends")


if __name__ == "__main__": main()
