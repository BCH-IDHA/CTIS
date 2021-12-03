import argparse
import configparser
import csv
import datetime
import pickle
import sys
import time
pickle.HIGHEST_PROTOCOL = 4

import warnings
warnings.filterwarnings('ignore')
# warnings.resetwarnings()
# warnings.simplefilter('default')

# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas import DataFrame
import datetime
import pandas as pd
import matplotlib.pyplot as plt

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


def read_data_col_list(country_iso3, list_cols, directory):
    """ Read the survey data in csv format and return dataframe for processing"""
    if len(list_cols) > 0:
        df = pd.read_csv(gen_file_name(country_iso3, directory, source=SOURCE), na_filter=False, skipinitialspace=True,
                         usecols=list_cols)
    else:
        df = pd.read_csv(gen_file_name(country_iso3, directory, source=SOURCE), na_filter=False, skipinitialspace=True)

    l_incomplete_value_identifiers = ['-99', -99, -99.0, '-99.0', '-77', -77, -77.0, '-77.0', '', ' ']
    df = df.replace(l_incomplete_value_identifiers, np.NaN)

    return df


def rename_survey_cols(df, dict_rename):
    """Rename columns using dictionary """
    return df.rename(columns=dict_rename)


def read_country_metadata(file_name, names=False):
    """ Read the metadata of the countries that were surveyed.
    It may contain info such as country_name, iso_3, iso_2, HDI_levels etc. for each country"""
    df_FB_countries = pd.read_csv(
        file_name)
    # df_FB_countries=df_FB_countries.loc[~df_FB_countries['ISO3'].isin(['AFG', 'KHM', 'BGD', 'BLR', 'ALB', 'MMR', 'CRI', 'MDA', 'BRA', 'ARM'])]
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


def resample_majority_class(df_train, outcome_var):
    """ Resample surveys in classes for model training """

    ## concatenate our training data back together
    # df_train = pd.concat([X_train, y_train], axis=1)

    print('Training data value count for outcome variable pre up/down sampling')
    print(df_train.ts_positive.value_counts())

    # separate minority and majority classes
    class_0 = df_train[df_train[outcome_var] == 0]
    class_1 = df_train[df_train[outcome_var] == 1]

    # upsample minority
    class_0_downsampled = resample(class_0,
                                   replace=False,  # sample with replacement
                                   n_samples=len(class_1),  # match number in minority class
                                   random_state=10)  # reproducible results

    # combine majority and upsampled minority
    return pd.concat([class_0_downsampled, class_1])


def pre_prep_data(df, outcome_var, na_logic=False):
    """Prepare data before providing it for building the model"""
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
        ### NOTE: For the least amount of code change, we added the logic to get the ts_14 in the same sudo variable ts_positive. Hence, you may see ourcome_varable as ts_recent but 'ts_positive' in dataframe info() after preprep step

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
         'sx_headache', 'recordeddate', 'iso_3', 'weight', 'count_symp_endorsed']]

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


def run_models(df, iso_3, ct_name, stratas, models, op_file, resample_data=False, gen_pickle_model=False,
               output_folder='', gen_feature_importance=False, perform_grid_search=False, gen_labeled_df=False,
               output_file_prefix='', gen_labeled_df_remaining=False,
               run_external_model=False, external_model_file='', external_model_file_2=''):
    """ Workhorse that builds ML model and generate output files:
    confusion matrix, roc curve, feature importance csv, performance metric table etc"""

    VERBOSE = True
    validate_pickle = True
    gen_feature_csv = True
    op_file.write(
        "iso_3,name,strata,n_total,n_selected,n_selected_0,n_selected_1,n_train,n_test,n_train_0,n_train_1,n_test_0,n_test_1,model,"
        "acc_mean,acc_sd,precision_mean,precision_sd,recall_mean,recall_sd,f1_mean,f1_sd,"
        "accuracy_global,precision_0_global,precision_1_global,recall_0_global,recall_1_global,f1_0_global,f1_1_global\n")

    selected_features = ['young_male', 'old_male', 'young_female', 'old_female', 'sx_fever', 'sx_cough', 'sx_diffbr',
                         'sx_fatigue', 'sx_runnose', 'sx_achepain', 'sx_throat', 'sx_chestpain', 'sx_nausea',
                         'sx_smelltaste', 'sx_eyepain', 'sx_headache']

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

        if gen_labeled_df_remaining:
            print('Entire df')
            print(df_om.shape)
            print(df_om.info())
            print('-' * 40)

            df_na_ts_pos = df_om[df_om['ts_positive'].isna()]
            print('ts_positive isna df:- ')
            print(df_na_ts_pos.shape)
            print(df_na_ts_pos.info())
            print('-' * 40)
        ## consider samples for training-testing using ts_pos or ts_positive 
        #         df_om = df_om.dropna() ##<==== Need to investigate
        #         df_om = df_om[df_om['ts_positive'].notna()]
        df_om = df_om.dropna(subset=['ts_positive'])

        print('ts_positive notna df:-')
        print(df_om.shape)
        print(df_om.info())
        print('-' * 40)
        ##sys.exit(0)

        # # Prepare train and test sets

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
                "{},{},{},{},{},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA\n".format(iso_3,
                                                                                                                  ct_name,
                                                                                                                  selected_strata,
                                                                                                                  n_total,
                                                                                                                  X.shape[
                                                                                                                      0]))
            print(" [Skipped][{} sec]".format(round(time.time() - start_time_strata, 1)), end='\n')
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state=5,
                                                            shuffle=True,  ##<== default is True
                                                            stratify=y)
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

        if resample_data:
            print('-' * 34)
            # print ('-- Upscalling minority class --')
            print('-- Down sampling majority class --')
            print('-' * 34)

            df_train = pd.concat([X_train, y_train], axis=1)
            # df_up = upscale_minority_class(X_train, y_train, 'ts_positive')
            df_up = resample_majority_class(df_train, 'ts_positive')
            print('-' * 34)
            print(' - Obtaining down/up-sampled df along with the dropped-samaple df due to resampling...')
            df_train_dropped = pd.concat([df_train, df_up]).loc[df_train.index.symmetric_difference(df_up.index)]
            print("df_train shape:", df_train.shape)
            print('   - First 10 indexes:', df_train.sort_index().head(10).index.tolist())
            print("df_downsampled shape:", df_up.shape)
            print('   - First 10 indexes:', df_up.sort_index().head(10).index.tolist())
            print("df_train_dropped shape:", df_train_dropped.shape)
            print('   - First 10 indexes:', df_train_dropped.sort_index().head(10).index.tolist())
            print('-' * 34)
            if VERBOSE:
                #                 print('Training data value count for outcome variable post-upscalling')
                print('Down/up-sampled training data value count for outcome variable post resampling')
                print(df_up.ts_positive.value_counts())
                print('Dropped training data value count for outcome variable post resampling')
                print(df_train_dropped.ts_positive.value_counts())

            y_train = df_up.ts_positive
            X_train = df_up.drop(['ts_positive'], axis=1)
            print("y_train={}, X_train={}".format(y_train.shape, X_train.shape), end='\n')

            y_train_dropped = df_train_dropped.ts_positive
            X_train_dropped = df_train_dropped.drop(['ts_positive'], axis=1)
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

            print('-' * 34)

        if VERBOSE:
            print("\nX_train:{}, y_train:{}\nX_test:{}, y_test:{}"
                  .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

        # evaluate each model in turn
        for model_name, model in models:
            start_time_model = time.time()
            #             scoring='f1'
            scoring = {'accuracy': make_scorer(accuracy_score),
                       'precision': make_scorer(precision_score),
                       'recall': make_scorer(recall_score),
                       'f1_score': make_scorer(f1_score)}
            print("model_name: ", model_name)

            # 1. k-fold cross validation (used for model evaluation)
            skf = StratifiedKFold(n_splits=2)

            # Perform grid search for Hyper-parameter tuning
            if perform_grid_search:  # and 'LightGBM' in model_name:
                print("Performing grid search...")
                gsresults = perform_random_grid_search_2(X_train, y_train, X_test, y_test, skf, scoring)
                if gsresults: print("Grid search completed!")
                sys.exit(0)

            # 1. k-fold cross validation (used for model evaluation)
            #    kfold = model_selection.KFold(n_splits=5)
            skf = StratifiedKFold(n_splits=10)
            #             cvresults = model_selection.cross_val_score(model, X_train, y_train.ravel(), cv=skf, scoring=scoring)
            cvresults = model_selection.cross_validate(model, X_train[selected_features], y_train.ravel(), cv=skf,
                                                       scoring=scoring)

            # 2. Fit the model on training set (for pickling the model)
            print(' - Fitting model and pickeling(if needed)', end='...')
            # model.fit(X_train, y_train.ravel())
            model.fit(X_train[selected_features], y_train.ravel())
            print('=' * 50)
            print('===== IMPORTANT: Num. of features used in the model:{} ====='.format(model.n_features_))
            print('=' * 50)
            ## predicting lables
            print(' - Predicting lables on training set')
            # y_pred_train_pre_p = model.predict(X_train)
            y_pred_train_pre_p = model.predict(X_train[selected_features])
            print(classification_report(y_train, y_pred_train_pre_p))

            # pickeling model
            if gen_pickle_model:
                start_time_pickle = time.time()
                clf_file_name = ''
                if gen_pickle_model:
                    if resample_data:
                        #                         clf_file_name = output_folder+'/'+ '_'.join(['clf', model_name,selected_strata,iso_3,'v12','upsampled'])+'.zip'
                        clf_file_name = output_folder + '/' + '_'.join(
                            ['clf', model_name, selected_strata, iso_3, 'v12', 'downsampled']) + '.zip'
                    else:
                        clf_file_name = output_folder + '/' + '_'.join(
                            ['clf', model_name, selected_strata, iso_3, 'v12', 'scale_positive']) + '.zip'

                print('*' * 50)
                print("- output model pickle file: {}".format(clf_file_name))
                print('*' * 50)

                print(' - Pickeling the model', end='...')
                pickle.dump(model, open(clf_file_name, 'wb'))
                print("  [Done][{} sec]".format(round(time.time() - start_time_pickle, 1)), end='\n')

            # 3. load the model from disk ## <=== Not needed as we veriffied that the 
            # results are same for pre and post pickling
            if validate_pickle:
                start_time_load_pickle = time.time()
                print(' - Loading the pickled model', end='...')
                loaded_model = pickle.load(open(clf_file_name, 'rb'))
                print("  [Done][{} sec]".format(round(time.time() - start_time_load_pickle, 1)), end='\n')

                print(' - Predicting lables on training set (post-pickling)')
                # y_pred_train = loaded_model.predict(X_train)
                y_pred_train = loaded_model.predict(X_train[selected_features])
                print(classification_report(y_train, y_pred_train))

            print('X-' * 20)
            print(' - Predicting lables on heldout set')
            #             result = model.score(X_test, y_test) ## <=== outputs 1 value that compares y_test and Y_pred
            #             print('result:', result)
            # y_pred_test = model.predict(X_test)
            y_pred_test = model.predict(X_test[selected_features])

            print(classification_report(y_test, y_pred_test))

            # obtain metrics' values for held-out set
            accuracy_global_model = accuracy_score(y_test, y_pred_test)
            precision_global_model = precision_score(y_test, y_pred_test, average=None)
            recall_global_model = recall_score(y_test, y_pred_test, average=None)
            f1_global_model = f1_score(y_test, y_pred_test, average=None)
            print(" -f1_scores:", f1_global_model)

            # confusion matrix
            fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred_test), show_absolute=True, show_normed=True,
                                            colorbar=True)
            plt.savefig(output_folder + '/' + '_'.join(
                ['confusion_matrix', model_name, selected_strata, iso_3, 'v12']) + '.png')
            print('-' * 20)

            # ------------------------------
            if run_external_model:
                print('-' * 40)
                # 4.1 load the ext model from disk
                print(' - Loading pickled model and predicting labels for held-out set...')
                loaded_ext_model = pickle.load(open(external_model_file, 'rb'))
                y_pred_test_ext_model = loaded_ext_model.predict(X_test[selected_features])

                print(classification_report(y_test, y_pred_test_ext_model))
                plot_confusion_matrix(confusion_matrix(y_test, y_pred_test_ext_model), show_absolute=True,
                                      show_normed=True, colorbar=True)

                # accuracy_ext_model = accuracy_score(y_test, y_pred_test_ext_model)
                # precision_ext_model = precision_score(y_test, y_pred_test_ext_model, average=None)
                # recall_ext_model = recall_score(y_test, y_pred_test_ext_model, average=None)
                f1_ext_model = f1_score(y_test, y_pred_test_ext_model, average=None)
                print(" -f1_scores:", f1_ext_model)

                # confusion matrix
                fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred_test_ext_model), show_absolute=True,
                                                show_normed=True,
                                                colorbar=True)
                plt.savefig(output_folder + '/' + '_'.join(
                    ['confusion_matrix', 'ext_model_1', iso_3, 'v12']) + '.png')

                print('-' * 40)
                # 4.2 load the ext model from disk
                print(' - Loading 2nd pickled model and predicting labels for held-out set...')
                loaded_ext_model_2 = pickle.load(open(external_model_file_2, 'rb'))
                y_pred_test_ext_model_2 = loaded_ext_model_2.predict(X_test[selected_features])

                print(classification_report(y_test, y_pred_test_ext_model_2))
                plot_confusion_matrix(confusion_matrix(y_test, y_pred_test_ext_model_2), show_absolute=True,
                                      show_normed=True, colorbar=True)

                # accuracy_ext_model = accuracy_score(y_test, y_pred_test_ext_model)
                # precision_ext_model = precision_score(y_test, y_pred_test_ext_model, average=None)
                # recall_ext_model = recall_score(y_test, y_pred_test_ext_model, average=None)
                f1_ext_model_2 = f1_score(y_test, y_pred_test_ext_model_2, average=None)
                print(" -f1_scores:", f1_ext_model_2)

                # confusion matrix
                fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred_test_ext_model_2), show_absolute=True,
                                                show_normed=True,
                                                colorbar=True)
                plt.savefig(output_folder + '/' + '_'.join(
                    ['confusion_matrix', 'ext_model_2', iso_3, 'v12']) + '.png')

                print('-' * 40)
            # ------------------------------

            # writing to metric table
            if VERBOSE:
                print(cvresults)
            #                 print("- F1 socre using 'model_selection.cross_val_score' and running stratified 10-fold CV:")
            #                 print(" - %s %f (%f)" % (model_name, np.mean(cvresults['f1_score']), np.std(cvresults['f1_score'])))

            op_file.write("{},{},{},{},{},{},{},{},{},{},"
                          "{},{},{},{},{},{},{},{},{},{},"
                          "{},{},{},{},{},{},{},{},{}\n".format(
                iso_3,
                ct_name,
                selected_strata,
                n_total,
                X.shape[0],
                n_selected_0,
                n_selected_1,
                n_train,
                n_test,
                n_train_0,
                n_train_1,
                n_test_0,
                n_test_1,
                model_name,

                np.mean(cvresults['test_accuracy']),
                np.std(cvresults['test_accuracy']),
                np.mean(cvresults['test_precision']),
                np.std(cvresults['test_precision']),
                np.mean(cvresults['test_recall']),
                np.std(cvresults['test_recall']),
                np.mean(cvresults['test_f1_score']),
                np.std(cvresults['test_f1_score']),

                accuracy_global_model,
                precision_global_model[0],
                precision_global_model[1],
                recall_global_model[0],
                recall_global_model[1],
                f1_global_model[0],
                f1_global_model[1]
            ))

            print('-' * 40)
            print(' - Generating roc curve...')
            # y_pred_probs_test = model.predict_proba(X_test)
            y_pred_probs_test = model.predict_proba(X_test[selected_features])
            # keep probabilities for the positive outcome only
            y_pred_probs_test = y_pred_probs_test[:, 1]
            y_pred_probs_test_auc = roc_auc_score(y_test, y_pred_probs_test)
            false_positive_rate, true_positive_rate, threshold1 = roc_curve(y_test, y_pred_probs_test)
            print('roc_auc_score: ', roc_auc_score(y_test, y_pred_probs_test))
            plt.figure()
            plt.plot(false_positive_rate, true_positive_rate,
                     label="auc=" + str(round(roc_auc_score(y_test, y_pred_probs_test), 3)))
            plt.legend(loc=4)
            plt.xlabel('1 – specificity (or false positive rate)')
            plt.ylabel('sensitivity (or true positive rate)')

            roc_file_name = output_folder + '/' + '_'.join(
                ['roc', model_name, selected_strata, iso_3, 'v12', 'downsampled']) + '.png'
            plt.savefig(roc_file_name)
            print('X-' * 20)

            if gen_labeled_df:
                start_time_gen_labeled_df = time.time()
                labeled_data_file = output_folder + '/' + '_'.join(
                    ['labeled_df', output_file_prefix, model_name, selected_strata]) + '.csv'

                print(' - Predicting lables on samples that were dropped due to resampling')
                # y_pred_train_dropped = model.predict(X_train_dropped)
                y_pred_train_dropped = model.predict(X_train_dropped[selected_features])

                print(' - Generating labeled dataframe', end='...')
                df_l_test = X_test.copy()
                df_l_test['sample_type'] = 'test'
                df_l_test['ts_pos_fb'] = y_test
                df_l_test.reset_index(inplace=True)
                df_l_test['ts_pos_pred'] = y_pred_test

                df_l_train = X_train.copy()
                df_l_train['sample_type'] = 'train'
                df_l_train['ts_pos_fb'] = y_train
                df_l_train.reset_index(inplace=True)
                df_l_train['ts_pos_pred'] = y_pred_train

                df_l_train_dropped = X_train_dropped.copy()
                df_l_train_dropped['sample_type'] = 'train_dropped'
                df_l_train_dropped['ts_pos_fb'] = y_train_dropped
                df_l_train_dropped.reset_index(inplace=True)
                df_l_train_dropped['ts_pos_pred'] = y_pred_train_dropped

                df_l = pd.concat([df_l_test, df_l_train, df_l_train_dropped])
                df_l.to_csv(labeled_data_file, index=False, na_rep='NA')
                print("  [Done][{} sec]".format(round(time.time() - start_time_gen_labeled_df, 1)), end='\n')
                print("   - df_l.shape:", df_l.shape)
                print('X-' * 20)

                if gen_labeled_df_remaining:
                    print(" - Generating labeled 'remaining' dataframe", end='...')
                    start_time_gen_labeled_df_remaining = time.time()
                    labeled_remaining_data_file = output_folder + '/' + '_'.join(
                        ['labeled_df_remaining', output_file_prefix, model_name, selected_strata]) + '.csv'
                    df_na_ts_pos = df_na_ts_pos.drop(['ts_positive'], axis=1)
                    df_na_ts_pos['ts_pos_fb'] = 'NA'
                    df_na_ts_pos['ts_pos_pred'] = model.predict(df_na_ts_pos[selected_features])
                    df_na_ts_pos.to_csv(labeled_remaining_data_file, index=True, index_label='index', na_rep='NA')
                    print("  [Done][{} sec]".format(round(time.time() - start_time_gen_labeled_df_remaining, 1)),
                          end='\n')
                    print("   - df_na_ts_pos.shape:", df_na_ts_pos.shape)
                    print('X-' * 20)

            print('-' * 40)
            ## Gen feature importance
            #            if gen_feature_importance:
            #                shap_values = shap.TreeExplainer(model).shap_values(X_train)
            #    #             shap.summary_plot(shap_values, X_train, plot_type="bar")
            #                f = plt.figure()
            #                shap.summary_plot(shap_values, X_test)
            #                f.savefig("./feature_importance_1-"+iso_3+"-downsampled_"+str(resample_data)+'-X_test'+".png", bbox_inches='tight', dpi=600)
            #
            #                shap.summary_plot(shap_values, X_train)
            #                f.savefig("./feature_importance_2-"+iso_3+"-downsampled_"+str(resample_data)+'-X_train'+".png", bbox_inches='tight', dpi=600)
            #                shap.summary_plot(shap_values[1], X_train,show=False)
            #                f.savefig("./feature_importance_3-"+iso_3+"-downsampled_"+str(resample_data)+'-X_train'+".png", bbox_inches='tight', dpi=600)
            if gen_feature_importance:
                # shap_values = shap.TreeExplainer(model).shap_values(X_trains)
                shap_values = shap.TreeExplainer(model).shap_values(X_train[selected_features])
                #             shap.summary_plot(shap_values, X_train, plot_type="bar")
                if gen_feature_csv:
                    df_mean_shap_feature_values = pd.DataFrame(shap_values[1], columns=X[
                        selected_features].columns.to_list()).abs().mean(axis=0).sort_values(ascending=False)
                    print(df_mean_shap_feature_values.shape)
                    print(df_mean_shap_feature_values)
                    shap_csv_file_name = output_folder + '/' + '_'.join(
                        ['feature_importance', model_name, selected_strata, iso_3, 'v12', 'downsampled',
                         str(resample_data)]) + '.csv'
                    df_mean_shap_feature_values.to_csv(shap_csv_file_name)
                ## sys.exit(0)
                f = plt.figure()
                # shap.summary_plot(shap_values, X_test)
                shap.summary_plot(shap_values, X_test[selected_features])
                f.savefig(
                    output_folder + '/' + "feature_importance_1-" + iso_3 + '-strata_' + selected_strata + "-downsampled_" + str(
                        resample_data) + '-X_test' + ".png", bbox_inches='tight', dpi=600)
                f = plt.figure()
                # shap.summary_plot(shap_values, X_train)
                shap.summary_plot(shap_values, X_train[selected_features])
                f.savefig(
                    output_folder + '/' + "feature_importance_2-" + iso_3 + '-strata_' + selected_strata + "-downsampled_" + str(
                        resample_data) + '-X_train' + ".png", bbox_inches='tight', dpi=600)
                f = plt.figure()
                # shap.summary_plot(shap_values[1], X_train,show=False)
                shap.summary_plot(shap_values[1], X_train[selected_features], show=False)
                f.savefig(
                    output_folder + '/' + "feature_importance_3-" + iso_3 + '-strata_' + selected_strata + "-downsampled_" + str(
                        resample_data) + '-X_train' + ".png", bbox_inches='tight', dpi=600)

            ## flush buffer for instantly updating the csv file
            op_file.flush()  ## <== Need to test this

            print("    - [{} Done][{} sec]".format(model_name, round(time.time() - start_time_model, 1)))
        print("  - [Done][{} sec]".format(round(time.time() - start_time_strata, 1)), end='\n')


def lgb_f1_score(y_true_new, y_pred_new):
    """ Gen f1 as an eval object for the grid search """
    #     y_true = data.get_label()
    y_pred_new = np.round(y_pred_new)  # scikits f1 doesn't like probabilities
    return 'f1_for_eval', f1_score(y_true_new, y_pred_new), True


## 4*5*6*3*3*4*5*10*2/(24*60*16)
def perform_random_grid_search_2(X_train, y_train, X_test, y_test, a_cv, a_scoring):
    """ Perform grid search for tuning the hyperparameter of the model"""
    train_data = lgbm.Dataset(X_train, label=y_train)
    test_data = lgbm.Dataset(X_test, label=y_test)

    ## All parameters: https://sites.google.com/view/lauraepp/parameters; https://lightgbm.readthedocs.io/en/latest/Parameters.html
    # Select Hyper-Parameters
    params = {
        'nthread': 16,  # 8, ## equals logical core for best spead up. This Mac:Physical cores=4; logical core = 8
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'max_depth': -1,
        ## Typical: 6, usually [3, 12]. Note:Each model trained at each iteration will have that maximum depth and cannot bypass it.
        'num_leaves': 64,  ## Alias: num_leaf; Defaults to 31.
        'learning_rate': 0.07,
        ## identified using np.arange(0.01, 0.5, 0.01). Note:Once learning rate is fixed, do not change it.
        'max_bin': 255,
        ## Number of maximum unique values per feature. Note: Histogram binning small number of bins may reduce training accuracy but may increase general power.
        'subsample_for_bin': 200,
        'subsample': 1,
        ##  Percentage of rows used per iteration frequency similar to feature_fraction (randomly select part of data without resampling). Used: for speed up training, to deal with over-fitting. Alias: bagging_fraction(Row Sampling)
        'subsample_freq': 1,
        # Iteration frequency to update the selected rows. Alias: bagging_freq (Row Sampling Frequency)
        'colsample_bytree': 0.8,
        ## Percentage of columns used per iteration. Used: for speed up training, to deal with over-fitting. By training over random partitions of the data, abusing the stochastic nature of the process, the resulting model might fit better the data. This is the third most sensible hyperparameter for gradient boosting: tune it with the row sampling. Alias: feature_fraction, sub_feature.
        ## Regularization basically adds the penalty as model complexity increases.
        'reg_alpha': 1.2,
        ## L1 Regularization for boosting; adds “absolute value of magnitude” of coefficient as penalty term to the loss function. Works well for feature selection in case we have a huge number of features. Alias: lambda_l1
        'reg_lambda': 1.2,
        ## L2 Regularization for boosting; Adds “squared magnitude” of coefficient as penalty term to the loss function. Alias: lambda_l2
        'min_split_gain': 0.5,
        'min_child_weight': 1,  ## Prune by minimum hessian requirement. Alias: min_sum_hessian_in_leaf
        'min_child_samples': 5,
        ## Prune by minimum number of observations requirement per leaf. Default=20. Can be used to deal with over-fitting. It is recommended to lower that value for small datasets (like 100 observations), and to increase it (if needed) on large datasets. Alias: min_data_in_leaf,
        'scale_pos_weight': 1,
        #         'metric' : 'f1',
        'valid_data': [test_data],
        'early_stopping_round': 30,

    }

    # Create parameters to search
    gridParams = {
        'learning_rate': [0.07],  # np.arange(0.01, 0.5, 0.01),
        'n_estimators': [500],  # [8,16,32,64], ## Alias=num_iterations; default 100
        'num_leaves': [16, 32, 64],  # 128, 255],
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
        'random_state': [501],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0.0, 1.0, 1.2],
        'reg_lambda': [0.0, 1.0, 1.2],
        'min_split_gain': [0.0, 0.25, 0.5, 0.75],
        'min_child_samples': [10, 20, 50, 100]  # ,200],
        ##         'subsample' : [0.7,0.75],

    }

    # Create classifier to use
    mdl = lgbm.LGBMClassifier(boosting_type='gbdt',
                              objective='binary',
                              n_jobs=16,  # 8,
                              silent=True,
                              max_depth=params['max_depth'],
                              max_bin=params['max_bin'],
                              subsample_for_bin=params['subsample_for_bin'],
                              subsample=params['subsample'],
                              subsample_freq=params['subsample_freq'],
                              min_child_weight=params['min_child_weight'],
                              scale_pos_weight=params['scale_pos_weight'],
                              early_stopping_rounds=params['early_stopping_round']
                              )

    # View the default model params:
    print(mdl.get_params().keys())

    # Create the grid
    #     grid = GridSearchCV(mdl, gridParams, verbose=2, cv=10, n_jobs=-1)
    #     grid = GridSearchCV(mdl, gridParams, verbose=10, cv=2, error_score=1) #, n_jobs=4)
    grid = GridSearchCV(estimator=mdl, param_grid=gridParams, verbose=10,
                        cv=a_cv, scoring=a_scoring, refit='f1_score', n_jobs=16)

    try:
        # Run the grid
        grid.fit(X_train,
                 y_train,
                 # eval_set=[(X_train,y_train),(X_test, y_test)],
                 # eval_names = ['train', 'held-out'],
                 eval_set=[(X_test, y_test), (X_train, y_train)],
                 eval_names=['held-out', 'train'],
                 eval_metric=lgb_f1_score
                 )

        df_gridsearch = pd.DataFrame(grid.cv_results_)
        #         df_gridsearch = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["score"])],axis=1)

        df_gridsearch.to_csv('grid_search_param.csv')

        # Print the best parameters found
        print(grid.best_params_)
        print(grid.best_score_)
    except (KeyboardInterrupt, OSError) as e:
        print(e)
        # Code to "save"
        df_gridsearch = pd.DataFrame(grid.cv_results_)
        #         df_gridsearch = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["score"])],axis=1)
        df_gridsearch.to_csv('grid_search_param.csv')
    return True


# ---------------------------- main() begins ----------------------------#
def main():
    print("\nProgram starts")

    print('''=========== ML building ==================
    ======== Global Model evaluation and Pickling model(and data) ========
    ==========================================''')

    # # define code argument to accept the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True)
    args = parser.parse_args()

    # Define config parser object
    config = configparser.ConfigParser()
    config.read(args.file)

    # Read in all the config/global variables from the config file
    TEST = config['GLOBAL'].getboolean('TEST')
    READ_FROM_DATA_PICKLE = config['GLOBAL'].getboolean('READ_FROM_DATA_PICKLE')

    DATA_PICKLE_FILE = config['GLOBAL']['DATA_PICKLE_FILE'] if READ_FROM_DATA_PICKLE else ''

    CTIS_DATA_DIR = config['IO']['CTIS_DATA_DIR'] if not READ_FROM_DATA_PICKLE else ''

    RUN_EXTERNAL_MODEL = config['GLOBAL'].getboolean('RUN_EXTERNAL_MODEL')
    EXTERNAL_MODEL_PICKLE_FILE = config['GLOBAL']['EXTERNAL_MODEL_PICKLE_FILE'] if RUN_EXTERNAL_MODEL else ''
    # EXTERNAL_MODEL_PICKLE_FILE_2 can be empty
    EXTERNAL_MODEL_PICKLE_FILE_2 = config['GLOBAL']['EXTERNAL_MODEL_PICKLE_FILE_2'] if RUN_EXTERNAL_MODEL else ''

    VERBOSE = bool(config['GLOBAL']['VERBOSE'])
    WEIGHTED_COUNTRIES = config['GLOBAL'].getboolean('WEIGHTED_COUNTRIES')
    SOURCE = config['GLOBAL']['SOURCE']

    PICKLE_MODEL = config['GLOBAL'].getboolean('PICKLE_MODEL')

    # 'ts_pos' or ts_recent
    OUTCOME_VAR = config['GLOBAL']['OUTCOME_VAR']
    RESAMPLE_DATA = config['GLOBAL'].getboolean('RESAMPLE_DATA')

    # for generating feature imporance csv
    GEN_FEATURE_IMPORTANCE = config['GLOBAL'].getboolean('GEN_FEATURE_IMPORTANCE')

    PERFORM_GRID_SEARCH = config['GLOBAL'].getboolean('PERFORM_GRID_SEARCH')

    # Used for generating labeled data that comprised of: train,test, and held-out set data (all rows)
    GEN_LABELED_DF = config['GLOBAL'].getboolean('GEN_LABELED_DF')

    # Used for generating labeled data comprised of: All data with demo and at least one symptom but no ts_pos i.e, All rows - (train+test+held-out set)
    GEN_LABELED_DF_REMAINING = config['GLOBAL'].getboolean('GEN_LABELED_DF_REMAINING')

    # demo_as_covariate or old_male,young_male,old_female,young_female,demo_as_covariate
    SELECTED_STRATAS = config['GLOBAL']['SELECTED_STRATAS'].split(',')

    # LightGBM or LightGBM_tunned
    # or LightGBM,LightGBM_tunned,LightGBM_usemissing_false,LightGBM_tuned_usemissing_false
    # or 'LR','NB','SVM',LightGBM
    SELECTED_MODELS = config['GLOBAL']['SELECTED_MODELS'].split(',')

    # Regular timeline
    # start_date = datetime.date(2020, 4, 26)  ## <=== actual_date=this_date + 1 day
    # end_date = datetime.date(2020, 12, 21)  ## <=== actual_date=this_date - 1 day

    # Timeline for sensitivity analysis
    # start_date = datetime.date(2020, 6, 28)  ## <=== actual_date=this_date + 1 day
    # end_date = datetime.date(2020, 12, 21)  ## <=== actual_date=this_date - 1 day

    # Timeline for early-full model analysis (2-week model)
    start_date = datetime.date(2020, 4, 26)  ## <=== actual_date=this_date + 1 day
    end_date = datetime.date(2020, 5, 11)  ## <=== actual_date=this_date - 1 day

    # Timeline for early-full model analysis (12-week model)
    # start_date = datetime.date(2020, 4, 26)  ## <=== actual_date=this_date + 1 day
    # end_date = datetime.date(2020, 10, 20)  ## <=== actual_date=this_date - 1 day

    start_date_for_printing = start_date + datetime.timedelta(days=1)
    end_date_for_printing = end_date - datetime.timedelta(days=1)

    # output_file_prefix = 'model-evaluate-global-ts_positive-v2'
    # output_file_prefix = 'global-ts_positive-downsampled_data-v12'## <== v12:Changed df_om.dropna() to df_om[df_om['ts_positive'].notna()]
    output_file_prefix = 'global-' + OUTCOME_VAR + '-downsampled_data-v12'  ## <== v12:Changed df_om.dropna() to df_om[df_om['ts_positive'].notna()]
    output_metric_table = 'metric_table_' + output_file_prefix + '_' + '_'.join(SELECTED_MODELS) + '.csv'
    output_df_pickle = 'df_' + output_file_prefix + '_' + '_'.join(SELECTED_MODELS) + '.zip'

    start_time_all = time.time()
    print("- Test:", TEST)
    print("- Weighted countries:", WEIGHTED_COUNTRIES)
    print("- Data source:", SOURCE)
    print("- Date range:{} to {}".format(start_date_for_printing.strftime('%m/%d/%Y'),
                                         end_date_for_printing.strftime('%m/%d/%Y')))
    print('--------')
    print('- Read data from pickle:', READ_FROM_DATA_PICKLE)
    if READ_FROM_DATA_PICKLE:
        print('\t\t-Data pickle file:', DATA_PICKLE_FILE)
    else:
        print("\t\t-CTIS_DATA_DIR:", CTIS_DATA_DIR)
    print('- Pickel the model:', PICKLE_MODEL)
    print('- Up/down sample the minority class:', RESAMPLE_DATA)
    print('- Outcome variable:', OUTCOME_VAR)
    print('--------')
    print('- Generate feature importance plot:', GEN_FEATURE_IMPORTANCE)
    print('- Perform grid search:', PERFORM_GRID_SEARCH)
    print('- Generate labeled dataframe:', GEN_LABELED_DF)
    print('--------')
    print('SELECTED_STRATAS:', SELECTED_STRATAS)
    print('SELECTED_MODELS:', SELECTED_MODELS)

    if TEST:
        #     warnings.resetwarnings()
        #     warnings.simplefilter('default'):w
        warnings.filterwarnings('ignore')
        output_dir = '.'

        if WEIGHTED_COUNTRIES:
            l_countries = ['AFG', 'THA']
            l_names_countries = ['Afghanistan', 'Thailand']
        else:
            l_countries = ['ALA', 'AND']
            l_names_countries = ['Aland Islands', 'Andorra']
    else:
        warnings.filterwarnings('ignore')
        output_dir = config['IO']['output_dir']
        country_matadata_file = config['IO']['COUNTRY_MATADATA_FILE']

        l_countries, l_names_countries = read_country_metadata(country_matadata_file, names=True)

    print('------------')
    print("- Num of countries for model building:", len(l_countries))
    print("- output metric table: {}/{}".format(output_dir, output_metric_table))
    print("- output data pickle file: {}/{}".format(output_dir, output_df_pickle))
    print('------------')

    # sys.exit(0)

    op_file = open(output_dir + '/' + output_metric_table, "a")

    # 1. Read in the data
    df_all = pd.DataFrame()
    start_readingin = time.time()

    if not READ_FROM_DATA_PICKLE:
        print("- Reading in data from files", end='...')

        req_cols = ['iso_3', 'recordeddate', 'count_symp_endorsed', 'q_totalduration', 'weight', 'survey_version']
        symp_cols = [col for col in dict_global_fb_variables_selected.keys() if
                     (col.startswith('b1_') or col in ['b2']) and col not in ['b1_13', 'b1_14']]  # <== get symptom cols
        demo_cols = [k for k, v in dict_global_fb_variables_selected.items() if v.startswith('demo_')]
        test_cols = [k for k, v in dict_global_fb_variables_selected.items() if v.startswith('ts_')]
        list_cols = req_cols + symp_cols + demo_cols + test_cols

        # 1. Read data from file country by country
        for country, name_country in zip(l_countries, l_names_countries):
            start_time_country = time.time()
            print("{}({})".format(name_country, country), end=' ')

            # 1.a read data
            df_r1 = read_data_col_list(country, list_cols, directory=CTIS_DATA_DIR)
            df_r1 = rename_survey_cols(df_r1, dict_global_fb_variables_selected)

            # 1.b select data from only study period
            df_r1['recordeddate'] = pd.to_datetime(
                df_r1['recordeddate']).dt.normalize()  ##<== Ignore time from the date-time
            df_r1['recordeddate'] = pd.to_datetime(df_r1['recordeddate'])
            df_r1.set_index('recordeddate', drop=False, inplace=True)
            df_r1 = df_r1[(df_r1.index.date > start_date) & (df_r1.index.date < end_date)]

            print("n={:,}".format(df_r1.shape[0]), end=' | ')

            df_all = pd.concat([df_all, df_r1], ignore_index=True)
        print("[Done][{} sec]".format(round(time.time() - start_readingin, 1)), end='\n')

        if VERBOSE: print(df_all.info())

        # 2. Prepare data
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

        # 3. Pickle data
        print("- Pickling Dataframe", end='...')
        start_pickel = time.time()
        df.to_pickle(output_dir + '/' + output_df_pickle)
        print("[Done][{} sec]".format(round(time.time() - start_pickel, 1)), end='\n')
    else:
        print("- Reading in selected and prepared data from pickle", end='...')
        df = pd.read_pickle(DATA_PICKLE_FILE)
        print("n={:,}".format(df.shape[0]), end=' ')
        print("[Done][{} sec]".format(round(time.time() - start_readingin, 1)), end='\n')

    if VERBOSE:
        print('After preparing')
        print(df.info())

    # 4. prepare models
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('LightGBM', lgbm.LGBMClassifier()))
    models.append(('LightGBM_usemissing_false', lgbm.LGBMClassifier(use_missing=False)))
    models.append(('LightGBM_tuned', lgbm.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        n_jobs=16,  # 8,
        silent=True,
        max_depth=-1,
        max_bin=255,
        subsample_for_bin=200,
        subsample=1,
        subsample_freq=1,
        min_child_weight=1,
        scale_pos_weight=1,
        # early_stopping_rounds=30,
        colsample_bytree=0.6,
        learning_rate=0.07,
        min_child_samples=50,
        min_split_gain=0.75,
        # n_estimators = 500,
        num_leaves=32,
        # random_state = 501,
        reg_alpha=1.2,
        reg_lambda=1.2
    )))
    models.append(('LightGBM_tuned_usemissing_false', lgbm.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        n_jobs=16,  # 8,
        silent=True,
        max_depth=-1,
        max_bin=255,
        subsample_for_bin=200,
        subsample=1,
        subsample_freq=1,
        min_child_weight=1,
        scale_pos_weight=1,
        # early_stopping_rounds=30,
        colsample_bytree=0.6,
        learning_rate=0.07,
        min_child_samples=50,
        min_split_gain=0.75,
        # n_estimators = 500,
        num_leaves=32,
        # random_state = 501,
        reg_alpha=1.2,
        reg_lambda=1.2,
        use_missing=False
    )))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('XGBoost', xgb.XGBClassifier(scale_pos_weight= 5)))
    selected_models = [m_tuple for m_tuple in models if m_tuple[0] in SELECTED_MODELS]

    # 5. running models
    print("- Running models...")
    run_models(df, 'global', 'global',
               SELECTED_STRATAS,
               selected_models,
               op_file,
               resample_data=RESAMPLE_DATA,
               gen_pickle_model=PICKLE_MODEL,
               output_folder=output_dir,
               gen_feature_importance=GEN_FEATURE_IMPORTANCE,
               perform_grid_search=PERFORM_GRID_SEARCH,
               gen_labeled_df=GEN_LABELED_DF,
               output_file_prefix=output_file_prefix,
               gen_labeled_df_remaining=GEN_LABELED_DF_REMAINING,
               run_external_model=RUN_EXTERNAL_MODEL,
               external_model_file=EXTERNAL_MODEL_PICKLE_FILE,
               external_model_file_2=EXTERNAL_MODEL_PICKLE_FILE_2,
               )

    op_file.close()
    print(" [All Done][{} sec]".format(round(time.time() - start_time_all, 1)), end='\n')

    print("\nProgram ends")


if __name__ == "__main__": main()
