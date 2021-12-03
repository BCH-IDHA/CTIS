# ML-related scripts
## Global model evaluation code
```model-evaluation-global-downsampling-no-symp-duration-v12.py```
### Description: 
This code evaluates different Machine Learning models trained on global data 
(i.e., data from more than one country) for predicting test positive or recently
tested outcome variable. The same code is used for performing Hyper-parameter tuning 
of a selected (LightGBM) model.

<br>This code was used for: -
1. Evaluating ML models trained on Global CTIS data for predicting test positive 
survey responses using limited demographic information and self-reported symptoms.
2. Generating ROC plot shown in Figure 2.a,
3. Producing feature importance data for the global model that was used in
generating Figure 2.b in the manuscript and Figure 3.c-d in the supplement.
4. Generating confusion matrix shown in Figures 3.e-k in the supplement.
5. Generating pickled data and classifiers to be used in production.

### input: 
Inputs to the code is provided using a configuration file.
It helps initialize the code with various global variables, model settings, and I/O variables. 
An example of such a configuration file is shown:
```./figure_2_COVID_test_prediction_model/model_eval_template.inp```
### Usage example: 
```python3.7  ../code/model-evaluation-global-downsampling-no-symp-duration-v12.py --file ./model_eval.inp > console.txt```

### Output: -
* Pickled classifier(s), 
* confusion metric(s), 
* ROC curve plot(s), 
* feature importance(s) csv file(s), 
* feature importance SHAP plot(s), and 
* a metric table with the following columns: - 
<pre>
  * iso_3        # 'global' for global or a country ISO_3 code for a country-based model 
  * name         # 'global' or 'country'
  * strata       # name of the data strata
  * n_total      # total number of CTIS surveys aka samples
  * n_selected   # number of CTIS surveys with a selected criteria. For example, surveys with a proper age-group and a clear gender information.
  * n_selected_0 # number of selected samples with outcome variable (test negative i.e., ```ts_pos``` or tested (i.e., ```ts_recent```) equals zero.
  * n_selected_1 # number of selected samples with outcome variable (test negative i.e., ```ts_pos``` or tested (i.e., ```ts_recent```) equals one.
  * n_train      # number of samples in the model training set. The 10-fold cross-validation is performed on this set.
  * n_test       # number of samples in the model held-out set
  * n_train_0    # number of training samples with outcome variable equals zero
  * n_train_1    # number of training samples with outcome variable equals one
  * n_test_0     # number of held-out samples with outcome variable equals zero
  * n_test_1     # number of held-out samples with outcome variable equals one
  * model           # a self-defined model name to track the model settings. Such as ```LightGBM``` or ```LightGBM_usemissing_false```
  * acc_mean        # model mean accuracy over 10-fold cross-validation
  * acc_sd          # standard deviation of model accuracy over 10-fold cross-validation
  * precision_mean  # model mean precision over 10-fold cross-validation
  * precision_sd    # standard deviation of model precision over 10-fold cross-validation
  * recall_mean     # mean recall of the model over 10-fold cross-validation
  * recall_sd       # standard deviation of recall over 10-fold cross-validation
  * f1_mean	        # mean F1-score over 10-fold cross-validation
  * f1_sd	        # standard deviation of F1-score over 10-fold cross-validation
  * accuracy_global     # model accuracy on the held-out set
  * precision_0_global  # model precision of the negative (i.e., 0 class) on the held-out set
  * precision_1_global  # model precision of the positive (i.e., 1 class) on the held-out set
  * recall_0_global	# model recall of the negative (i.e., 0 class) on the held-out set
  * recall_1_global	# model recall of the positive (i.e., 1 class) on the held-out set
  * f1_0_global	        # model F1-score of the negative (i.e., 0 class) on the held-out set
  * f1_1_global         # model F1-score of the positive (i.e., 1 class) on the held-out set
  </pre>
 


## Global and country-level model evaluation/comparison code
### Description: 
This code compares the performance of a selected global model with different 
country-level models (i.e., learns from one country) with the same model 
settings as global model. It can take raw or pickled data and models as inputs.

<br>This code was used for: -
1. Evaluating 114 ML models trained on individual country's CTIS data 
for predicting test positive survey responses using limited demographic 
information and self-reported symptoms.
2. Generating data for producing supplement Figure 3.b-c and performing 
sensitivity analysis of model performance for individual country/territory models.

### Run template: 
```python3.7  ../code/model-evaluation-country-and-global-downsampling-no-symp-duration-v12.py --file ./model_eval.inp > console.txt```

### Output: -
* Pickled classifier(s), 
* feature importance(s) csv file(s), 
* a metric table with the following columns: -
<pre>
  * iso_3        # country ISO code e.g., 'CHE'
  * name         # name of the country e.g., 'Switzerland'
  * strata       # name of the data strata
  * n_total      # total number of CTIS surveys aka samples available for the country
  * n_selected   # number of CTIS surveys with a selected criteria. For example, surveys with a proper age-group and a clear gender information.
  * n_selected_0 # number of selected samples with outcome variable (test negative i.e., ```ts_pos``` or tested (i.e., ```ts_recent```) equals zero.
  * n_selected_1 # number of selected samples with outcome variable (test negative i.e., ```ts_pos``` or tested (i.e., ```ts_recent```) equals one.
  * n_train      # number of samples in the model training set. The 10-fold cross-validation is performed on this set.
  * n_test       # number of samples in the model held-out set
  * n_train_0    # number of training samples with outcome variable equals zero
  * n_train_1    # number of training samples with outcome variable equals one
  * n_test_0     # number of held-out samples with outcome variable equals zero
  * n_test_1     # number of held-out samples with outcome variable equals one
  * model                # a self-defined model name to track the model settings. Such as ```LightGBM``` or ```LightGBM_usemissing_false```
  * accuracy_country     # country model accuracy on the held-out set
  * accuracy_global      # global model accuracy on the held-out set
  * precision_0_country  # country model precision of the negative (i.e., 0 class) on the held-out set
  * precision_1_country  # country model precision of the positive (i.e., 1 class) on the held-out set
  * precision_0_global   # global model precision of the negative (i.e., 0 class) on the held-out set
  * precision_1_global   # global model precision of the positive (i.e., 1 class) on the held-out set
  * recall_0_country	 # country model recall of the negative (i.e., 0 class) on the held-out set
  * recall_1_country     # country model recall of the positive (i.e., 1 class) on the held-out set
  * recall_0_global	 # global model recall of the negative (i.e., 0 class) on the held-out set
  * recall_1_global	 # global model recall of the positive (i.e., 1 class) on the held-out set
  * f1_0_country         # country model F1-score of the negative (i.e., 0 class) on the held-out set
  * f1_1_country         # country model F1-score of the positive (i.e., 1 class) on the held-out set
  * f1_0_global	         # global model F1-score of the negative (i.e., 0 class) on the held-out set
  * f1_1_global          # global model F1-score of the positive (i.e., 1 class) on the held-out set
  </pre>


## Data labeling code
### Description: 
This code uses pickled model(s) to predict the label of the CTIS surveys. 
It processes each survey using the same data-processing pipeline as the model 
before predicting the label.

<br> This code was used:
1. ADD HERE
2. ADD HERE

### Input: 
Unlabeled data, pickled model(s), data and model settings. Inputs to the 
script is provided using a configuration file. 
### Output: 
Predicted labels for each survey as a csv file. Two labeled survey csv 
files are generated: one for surveys where self-identified outcome variable 
was provided by the subjects and second where self-identified outcome variable 
was not provided by the subjects.

