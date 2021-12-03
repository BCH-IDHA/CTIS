# Demographic measure generation related scripts 
## gen_demo_diff_between_census_and_ctis.py
### Description: <br>
This code is used to compute demographic difference between CTIS and Census (by week or overall). 
It was used to generate data to generate Figures 1.c, 1.d, 1.e and 1.f in the manuscript, 
and Figure 2 in the Supplement.
### Input
A configuration file that initializes many global and input/output variables. An example configuration file:
```config_demo_diff.inp```

### Example usage one:
``python3.7 gen_demo_diff_between_census_and_ctis.py -f config_demo_diff.inp > ./test/console_gen_demo_diff_overall_AFG.txt``
<br><br>Note: Use ```WEEKLY_ANALYSIS = False``` in the config file for computing demo diff across entire study period
#### Output: <br>
A CSV file showing demographic differences for male and female in three age_groups: [18-34, 35-54, 55-75+]
<br>Format of output file for differences across complete study period :
<pre>
country,male,female,total_diff,mean_diff
Afghanistan,"[0.33764, 0.05842, 0.00161]","[-0.26159, -0.11287, -0.0232]",0.79533,0.13256
Albania,"[0.06022, 0.093, -0.0833]","[0.06166, 0.00418, -0.13577]",0.43813,0.07302
</pre>
### Example usage two:
```python3.7 gen_demo_diff_between_census_and_ctis.py -f config_demo_diff.inp > ./test/console_gen_demo_diff_weekly_AFG.txt```
<br><br>Note: Use ```WEEKLY_ANALYSIS = True ``` for computing weekly demo diff<br>

#### Output: <br>
A CSV file showing _weekly_ demographic differences for male and female in three age_groups: [18-34, 35-54, 55-75+]
<br>
Format of output file for weekly difference: -
<pre>
country, 18, 19, ...
Afghanistan,"{'male': [0.4312, 0.01767, -0.01523], 'female': [-0.28235, -0.12423, -0.02706], 'total_diff': 0.89774, 'mean_diff': 0.14962}","{'male': [0.42712, 0.03706, -0.01281], 'female': [-0.29163, -0.12605, -0.03368], 'total_diff': 0.92835, 'mean_diff': 0.15472}",...
</pre>

## json_to_csv.ipynb
### Description: <br>
This script is used to reformat the demographic data that is used to generate Figure1_c_f. <br>
### Input:<br>
demo_diff_weekly_ctis_census_by_country.csv <br>
### Output: <br>
demo_diff_weekly_ctis_census_by_country_TJV_format.csv <br>
## Figure1_c_f.R
### Description: <br>
This script is used to generate Figure 1 c-f in the main text, and outputs the various demographic calculations that are reported in the text. <br>
### Input: <br>
demo_diff_weekly_ctis_census_by_country_TJV_format.csv <br>
demo_diff_ctis_census_by_country.csv <br>
### Output: <br>
Figure1_c_f.png <br>
Demographic calculations
