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


def display_side_by_side(*args):
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table', 'table style="display:inline; border-style:hidden; align:center"'), raw=True)


# # Swapped dictionary for reversing the named columns to the original columns
def gen_swapped_col_list(col_list):
    dict_global_fb_variables_selected_swapped = dict([(value, key) for key,
                                                                       value in
                                                      dict_global_fb_variables_selected.items()])
    #     print(dict_global_fb_variables_selected_swapped)
    new_col_list = [v for k, v in dict_global_fb_variables_selected_swapped.items() if k in col_list]
    return new_col_list


def read_country_metadata(file_name, names=False):
    df_FB_countries = pd.read_csv(file_name)
    #     print (df_FB_countries.head())
    l_countries = list(df_FB_countries['ISO3'].values)
    l_countries = [country for country in l_countries if
                   country not in ['CHN', 'CUB', 'IRN', 'PRK', 'SYR', 'TKM', np.NaN]]
    l_name_countries = list(df_FB_countries['NAME'].values)
    if names:
        return l_countries, l_name_countries
    else:
        return l_countries


def downscale_majority_class(X_train, y_train, outcome_var):
    # concatenate our training data back together
    df_train = pd.concat([X_train, y_train], axis=1)

    print('Training data value count for outcome variable before down sampling')
    print(df_train.ts_positive.value_counts())

    # separate minority and majority classes
    class_0 = df_train[df_train[outcome_var] == 0]
    class_1 = df_train[df_train[outcome_var] == 1]

    # upsample minority
    if class_0.shape[0] > class_1.shape[0]:
        class_0_downsampled = resample(class_0,
                                       replace=False,  # sample with replacement
                                       n_samples=len(class_1),  # match number in minority class
                                       random_state=10)  # reproducible results
    #								  random_state=5) # reproducible results
    else:
        class_0_downsampled = class_0

    # combine majority and upsampled minority
    return pd.concat([class_0_downsampled, class_1])


def pre_prep_data(df, outcome_var):
    df["young_male"] = np.nan
    df["old_male"] = np.nan
    df["young_female"] = np.nan
    df["old_female"] = np.nan
    df['demo_sex'] = pd.to_numeric(df['demo_sex'], errors='coerce').fillna(0).astype(int)
    df['demo_age'] = pd.to_numeric(df['demo_age'], errors='coerce').fillna(0).astype(int)
    df.loc[((df['demo_sex'] == 1) & (df['demo_age'] <= 4)), 'young_male'] = '1'
    df.loc[((df['demo_sex'] == 1) & (df['demo_age'] >= 5)), 'old_male'] = '1'

    df.loc[((df['demo_sex'] == 2) & (df['demo_age'] <= 4)), 'young_female'] = '1'
    df.loc[((df['demo_sex'] == 2) & (df['demo_age'] >= 5)), 'old_female'] = '1'

    #     df['male']=df['demo_sex'].apply(lambda x: '1' if x==1 else ('0' if x==2 else np.nan))
    #     df['young']=df['demo_age'].apply(lambda x: '1' if x<=4 else ('0' if x>=5 else np.nan))

    #     df["symp_short"] = np.nan
    #     df["symp_medium"] = np.nan
    #     df["symp_high"] = np.nan
    #     df["symp_over14"] = np.nan
    #     df['symp_days'] = pd.to_numeric(df['symp_days'], errors='coerce').fillna(-1).astype(int)
    #     df['demo_age']=pd.to_numeric(df['demo_age'], errors='coerce').fillna(-1).astype(int).apply(lambda x: 'Young' if x>=1 and x<=4 else('Old' if x>=5 and x<=7 else 'NA'))

    #     df.loc[((df['symp_days']>=0)&(df['symp_days']<=4)),'symp_short'] = '1'
    #     df.loc[((df['symp_days']>=5)&(df['symp_days']<=9)),'symp_medium'] = '1'
    #     df.loc[((df['symp_days']>=10)&(df['symp_days']<=14)),'symp_high'] = '1'
    #     df.loc[((df['symp_days']>=15)&(df['symp_days']<=21)),'symp_over14'] = '1'

    if outcome_var == 'ts_pos':
        df['ts_pos'] = pd.to_numeric(df['ts_pos'], errors='coerce').fillna(-1).astype(int)
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
         'sx_headache']]

    sx = [col for col in df2.columns.to_list() if col.startswith('sx_') or col.startswith('symp_')]
    df2[sx] = df2[sx].apply(pd.to_numeric, errors='coerce')

    df2 = df2.replace(2, 0)

    #     df2[['symp_short','symp_medium','symp_high','symp_over14','young_male','old_male',
    #     'young_female','old_female']] = df2[['symp_short','symp_medium','symp_high',
    #     'symp_over14','young_male','old_male','young_female','old_female']].replace(np.nan,0)

    df2[['young_male', 'old_male', 'young_female', 'old_female']] = df2[
        ['young_male', 'old_male', 'young_female', 'old_female']].replace(np.nan, 0)

    df2 = df2.replace(2, 0)
    return df2


def run_models_by_country(df, iso_3, ct_name, stratas, models, op_file, use_global_model=True, global_model_file='',
                          RESAMPLE_DATA=False):
    VERBOSE = True

    #     clf_file_name = 'clf_LightGBM_demo_as_covariate_global.zip'
    clf_file_name = global_model_file  # 'clf_LightGBM_demo_as_covariate_global_up_sample.zip'

    for selected_strata in stratas:
        start_time_strata = time.time()

        if selected_strata != 'demo_as_covariate':
            df_om = df.loc[df[selected_strata] == '1']
            df_om = df_om.drop(['old_male', 'young_male', 'old_female', 'young_female'], axis=1)
        else:
            df_om = df
            df_om['old_male'] = df_om['old_male'].apply(pd.to_numeric, errors='coerce')
            df_om['young_male'] = df_om['young_male'].apply(pd.to_numeric, errors='coerce')
            df_om['old_female'] = df_om['old_female'].apply(pd.to_numeric, errors='coerce')
            df_om['young_female'] = df_om['young_female'].apply(pd.to_numeric, errors='coerce')

        print("  - Strata={}, n=({:,})".format(selected_strata, df_om.shape[0]), end=' ')
        n_total = df_om.shape[0]

        ## consider samples for training-testing using ts_pos or ts_positive
        #         df_om = df_om.dropna() ##<==== Need to investigate
        #         df_om = df_om[df_om['ts_positive'].notna()]
        df_om = df_om.dropna(subset=['ts_positive'])

        ## Prepare train and test sets
        y = df_om.ts_positive
        X = df_om.drop(['ts_positive'], axis=1)
        print("y={}, X={}".format(y.shape, X.shape), end='\n')

        if VERBOSE:
            print('Support: n_selected_0 and n_selected_1:-')
            print(y.value_counts())
        n_selected_support = y.value_counts().to_dict()
        n_selected_0 = n_selected_support[0] if 0 in n_selected_support.keys() else 0
        n_selected_1 = n_selected_support[1] if 1 in n_selected_support.keys() else 0

        if X.shape[0] < 50:
            op_file.write(
                "{},{},{},{},{},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA\n".format(iso_3, ct_name, selected_strata, n_total,
                                                                           X.shape[0]))
            print(" [Skipped][{} sec]".format(round(time.time() - start_time_strata, 1)), end='\n')
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        if VERBOSE:
            print('Support for n_selected_train_0 and n_selected_train_1:', y_train.value_counts())
            print('Support for n_selected_test_0 and n_selected_test_1:-')
            print(y_test.value_counts())

        n_train_support = y_train.value_counts().to_dict()
        n_train_0 = n_train_support[0] if 0 in n_train_support.keys() else 0
        n_train_1 = n_train_support[1] if 1 in n_train_support.keys() else 0

        n_test_support = y_test.value_counts().to_dict()
        n_test_0 = n_test_support[0] if 0 in n_test_support.keys() else 0
        n_test_1 = n_test_support[1] if 1 in n_test_support.keys() else 0

        if RESAMPLE_DATA:
            print('-' * 34)
            print('-- Up/down sampling  --')
            print('-' * 34)
            #             df_up = upscale_minority_class(X_train, y_train, 'ts_positive')
            df_up = downscale_majority_class(X_train, y_train, 'ts_positive')

            if VERBOSE:
                print('Training data value count for outcome variable post-up/down sampling')
                print(df_up.ts_positive.value_counts())

            y_train = df_up.ts_positive
            X_train = df_up.drop(['ts_positive'], axis=1)
            print("y_train={}, X_train={}".format(y_train.shape, X_train.shape), end='\n')

            #             n_train = X_train.shape[0]

            print('-' * 34)

            ## --------------
            ## IMPORTANT: recompute n_train, n_train_0, and n_train_1 after sample-up/sample-down
            n_train = y_train.shape[0]

            n_train_support = y_train.value_counts().to_dict()
            n_train_0 = n_train_support[0] if 0 in n_train_support.keys() else 0
            n_train_1 = n_train_support[1] if 1 in n_train_support.keys() else 0

            if VERBOSE:
                print('Support for n_train_support', y_train.shape[0])
                print('Support for n_selected_train_0 and n_selected_train_1:-')
                print(y_train.value_counts())
            ## ---------------

        #         if VERBOSE:
        #             print("\nX_train:{}, y_train:{}\nX_test:{}, y_test:{}"
        #                   .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
        # #             display_side_by_side(X_train.head().T, y_train.to_frame().head().T,
        # #                                  X_test.head().T, y_test.to_frame().head().T)

        # evaluate each model in turn
        for model_name, model in models:
            start_time_model = time.time()

            print("model_name:{}, model:{}".format(model_name, model))

            if not use_global_model:
                print(' - Running stratified 10-fold cross validation...', end=' ')
                #                 scoring='f1'
                scoring = {'accuracy': make_scorer(accuracy_score),
                           'precision': make_scorer(precision_score),
                           'recall': make_scorer(recall_score),
                           'f1_score': make_scorer(f1_score)}
                #    kfold = model_selection.KFold(n_splits=5)
                skf = StratifiedKFold(n_splits=10)
                #                 cvresults = model_selection.cross_val_score(model, X_train, y_train.ravel(), cv=skf, scoring=scoring)
                #                 op_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(iso_3,
                #                                                                     ct_name,
                #                                                                     selected_strata,
                #                                                                     n_total,
                #                                                                     X.shape[0],
                #                                                                     X_train.shape[0],
                #                                                                     X_test.shape[0],
                #                                                                     model_name,
                #                                                                     cvresults.mean(),
                #                                                                     cvresults.std()))

                cvresults = model_selection.cross_validate(model, X_train, y_train.ravel(), cv=skf, scoring=scoring)
                op_file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    iso_3,
                    ct_name,
                    selected_strata,
                    n_total,
                    X.shape[0],
                    n_selected_0, n_selected_1,
                    n_train,
                    n_test,
                    model_name,
                    #                                                                     cvresults.mean(),
                    #                                                                     cvresults.std()),
                    np.mean(cvresults['test_accuracy']),
                    np.std(cvresults['test_accuracy']),
                    np.mean(cvresults['test_precision']),
                    np.std(cvresults['test_precision']),
                    np.mean(cvresults['test_recall']),
                    np.std(cvresults['test_recall']),
                    np.mean(cvresults['test_f1_score']),
                    np.std(cvresults['test_f1_score'])
                ))

                if VERBOSE:
                    print("- F1 socre using 'model_selection.cross_val_score' and running stratified 10-fold CV:")
                    print(" - %s %f (%f)" % (model_name, cvresults.mean(), cvresults.std()))


            else:
                print('-' * 40)
                ## Added new strats
                model.fit(X_train, y_train.ravel())
                ## predicting lables
                print(' - Fitting Country-model and predicting lables for training set...')
                y_pred_train_country_model = model.predict(X_train)
                print(classification_report(y_train, y_pred_train_country_model))
                ## Added new ends
                print('-' * 40)
                print(' - Fitting Country-model and predicting lables for held-out set...')
                # model.fit(X_train, y_train.ravel())
                y_pred_test_country_model = model.predict(X_test)
                print(classification_report(y_test, y_pred_test_country_model))
                #                 plot_confusion_matrix(confusion_matrix(y_test, y_pred_test_country_model), show_absolute=True, show_normed=True,colorbar=True)

                accuracy_country_model = accuracy_score(y_test, y_pred_test_country_model)
                precision_country_model = precision_score(y_test, y_pred_test_country_model, average=None)
                recall_country_model = recall_score(y_test, y_pred_test_country_model, average=None)
                f1_country_model = f1_score(y_test, y_pred_test_country_model, average=None)
                print("- f1_scores:", f1_country_model)

                # 3. load the model from disk
                print(' - Loading pickled Global-model and predicting lables for held-out set...')
                loaded_model = pickle.load(open(clf_file_name, 'rb'))
                y_pred_test_global_model = loaded_model.predict(X_test)

                print(classification_report(y_test, y_pred_test_global_model))
                #                 plot_confusion_matrix(confusion_matrix(y_test, y_pred_test_global_model), show_absolute=True, show_normed=True,colorbar=True)

                accuracy_global_model = accuracy_score(y_test, y_pred_test_global_model)
                precision_global_model = precision_score(y_test, y_pred_test_global_model, average=None)
                recall_global_model = recall_score(y_test, y_pred_test_global_model, average=None)
                f1_global_model = f1_score(y_test, y_pred_test_global_model, average=None)
                print(" -f1_scores:", f1_global_model)

                print('-' * 40)

                if len(f1_country_model) < 2: f1_country_model = np.append(f1_country_model, 'NA')
                if len(f1_global_model) < 2: f1_global_model = np.append(f1_global_model, 'NA')

                #                 op_file.write("iso_3,name,strata,n_total,n_selected,n_train,n_test,model,f1_0_country,f1_1_country,f1_0_global,f1_1_global\n")
                op_file.write(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        iso_3,
                        ct_name,
                        selected_strata,
                        n_total,
                        X.shape[0],
                        n_selected_0, n_selected_1,
                        #                             X_train.shape[0],
                        #                             X_test.shape[0],
                        n_train,
                        n_test,
                        n_train_0, n_train_1, n_test_0, n_test_1,
                        model_name,
                        accuracy_country_model,
                        accuracy_global_model,

                        precision_country_model[0],
                        precision_country_model[1],
                        precision_global_model[0],
                        precision_global_model[1],

                        recall_country_model[0],
                        recall_country_model[1],
                        recall_global_model[0],
                        recall_global_model[1],

                        f1_country_model[0],
                        f1_country_model[1],
                        f1_global_model[0],
                        f1_global_model[1]
                    ))

            # # flush buffer for instantly updating the csv file
            op_file.flush()  ## <== Need to test this

            print("    - [{} Done][{} sec]".format(model_name, round(time.time() - start_time_model, 1)))
        print("  - [Done][{} sec]".format(round(time.time() - start_time_strata, 1)), end='\n')


# # ---------------------------- main() begins ----------------------------#
def main():
    print("\nProgram starts")

    print('''=========== ML building ===============
    ======== Global Model evaluation ======
    ======== conuntry-by-country ==========
    =======================================''')

    start_time_all = time.time()
    # # define code argument to accept the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True)
    args = parser.parse_args()

    # Define config parser object
    config = configparser.ConfigParser()
    config.read(args.file)

    # Read in all the config/global variables from the config file

    TEST = config['GLOBAL'].getboolean('TEST')
    
    VERBOSE = bool(config['GLOBAL']['VERBOSE'])
    
    WEIGHTED_COUNTRIES = config['GLOBAL'].getboolean('WEIGHTED_COUNTRIES')
    
    SOURCE = config['GLOBAL']['SOURCE']

    USE_GLOBAL_MODEL = config['GLOBAL'].getboolean('USE_GLOBAL_MODEL')

    GLOBAL_MODEL_FILE = config['GLOBAL']['EXTERNAL_MODEL_PICKLE_FILE'] if RUN_EXTERNAL_MODEL else ''
    #  GLOBAL_MODEL_FILE = './run/Global-lightGBM-downsample_majority_class-model-v12/clf_LightGBM_demo_as_covariate_global_v12_downsampled.zip'
   
    RESAMPLE_DATA = config['GLOBAL'].getboolean('RESAMPLE_DATA')
    
    # 'ts_recent' #'ts_pos'
    OUTCOME_VAR = config['GLOBAL']['OUTCOME_VAR']

    SELECTED_STRATAS = config['GLOBAL']['SELECTED_STRATAS'].split(',')

    SELECTED_MODELS = config['GLOBAL']['SELECTED_MODELS'].split(',')


    
    added_name = 'downsampled_data' if RESAMPLE_DATA else ''
    model_str = '_'.join(SELECTED_MODELS)
    output_metric_table = '-'.join(
        ['metric_table', 'country_and_global', added_name, OUTCOME_VAR, model_str, 'v12']) + '.csv'

    if TEST:
        #     warnings.resetwarnings()
        #     warnings.simplefilter('default')
        warnings.filterwarnings('ignore')
        output_dir = '.'
        if WEIGHTED_COUNTRIES:
            l_countries = ['COD']  # ['AFG', 'CHE', 'NZL']
            l_names_countries = ['Congo']  # ['Afganistan','Switzerland', 'New Zealand']
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

        REMAINING_COUNTRIES = []
        if len(REMAINING_COUNTRIES):
            l_countries, l_names_countries = zip(
                *[[c, n] for c, n in zip(l_countries, l_names_countries) if c in REMAINING_COUNTRIES])
            l_countries = list(l_countries)
            l_names_countries = list(l_names_countries)

    print("- Test:", TEST)
    print("- Weighted countries:", WEIGHTED_COUNTRIES)
    print("- Data source:", SOURCE)
    print('- Upsample the minority class:', RESAMPLE_DATA)
    print('- Use Global model:', USE_GLOBAL_MODEL)
    if USE_GLOBAL_MODEL:
        print('- Pickled global model to use:', GLOBAL_MODEL_FILE)
    print("- output metric table: {}/{}".format(output_dir, output_metric_table))
    print('------------')

    op_file = open(output_dir + '/' + output_metric_table, "a")

    if USE_GLOBAL_MODEL:
        op_file.write(
            "iso_3,name,strata,n_total,n_selected,n_train,n_test,model,accuracy_country,accuracy_global,precision_0_country,precision_1_country,precision_0_global,precision_1_global,recall_0_country,recall_1_country,recall_0_global,recall_1_global,f1_0_country,f1_1_country,f1_0_global,f1_1_global\n")
    else:
        op_file.write("iso_3,name,strata,n_total,n_selected,n_train,n_test,model,f1_mean,f1_sd\n")

    start_date = datetime.date(2020, 4, 26)  ## <=== actual_date=this_date + 1 day
    end_date = datetime.date(2020, 12, 21)  ## <=== actual_date=this_date - 1 day

    req_cols = ['iso_3', 'recordeddate', 'count_symp_endorsed', 'q_totalduration', 'weight', 'survey_version']
    symp_cols = [col for col in dict_global_fb_variables_selected.keys() if
                 (col.startswith('b1_') or col in ['b2']) and col not in ['b1_13', 'b1_14']]  # <== get symptom cols
    demo_cols = [k for k, v in dict_global_fb_variables_selected.items() if v.startswith('demo_')]
    test_cols = [k for k, v in dict_global_fb_variables_selected.items() if v.startswith('ts_')]
    list_cols = req_cols + symp_cols + demo_cols + test_cols

    for country, name_country in zip(l_countries, l_names_countries):
        start_time_country = time.time()
        print('=' * 80)
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
        print("n={:,}".format(df_r1.shape[0]))

        if VERBOS: print(df_r1.info())

        ## 3. prepare-data
        # df = pre_prep_data(df_r1)
        df = pre_prep_data(df_r1, outcome_var=OUTCOME_VAR)

        if VERBOS: print(df.info())

        # 4. prepare models
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        models.append(('RF', RandomForestClassifier()))
        models.append(('LightGBM', lgbm.LGBMClassifier()))
        # models.append(('KNN', KNeighborsClassifier()))
        # models.append(('CART', DecisionTreeClassifier()))
        # models.append(('LDA', LinearDiscriminantAnalysis()))
        # models.append(('XGBoost', xgb.XGBClassifier(scale_pos_weight= 5)))
        selected_models = [m_tuple for m_tuple in models if m_tuple[0] in SELECTED_MODELS]

        #     run_models(df, country, name_country, op_file)
        #     run_models(df, country, name_country, SELECTED_STRATAS, selected_models, op_file)
        run_models_by_country(df, country, name_country, SELECTED_STRATAS, selected_models, op_file,
                              use_global_model=USE_GLOBAL_MODEL, global_model_file=GLOBAL_MODEL_FILE,
                              RESAMPLE_DATA=RESAMPLE_DATA)

        print(" - [Done][{} sec]".format(round(time.time() - start_time_country, 1)), end='\n')

    #     sys.exit(0)
    op_file.close()
    print(" [All Done][{} sec]".format(round(time.time() - start_time_all, 1)), end='\n')

    print("\nProgram ends")


if __name__ == "__main__": main()
