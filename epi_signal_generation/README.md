# Epidemic signal generation related scripts 
## generate_heatmap_blocks.R
### Description: <br>
This script is used to process the raw line-list data for the various CTIS variables described in the main text. Due to the large size of the line-list dataset, data are stored in individual files for each country. In this script, individual country data are imported, processed, and merged together in an interative process. Data for each variable are stored in a matrix and saved as an RDS file. These matrices are then used for constructing the heatmaps (Figure 3_e, Figure 4). <br>
### Input: <br>
A directory of files with the line-list data. Files must have the following naming convention: fbsurvey_intl_raw_all_rows_all_cols_country_<iso 3 code>.csv <br>
### Output: <br>
20210506_df_block_1.rds (Benchmark signal) <br>
20210506_df_block_2.rds (Positive test signal) <br>
20210506_df_block_3.rds (CCLI signal) <br>
20210506_df_block_4.rds (CLI broad signal) <br>
20210506_df_block_5.rds (CLI narrow signal) <br>
20210506_df_raw.rds (Raw data) <br>
20210506_df_age.rds (Age data) <br>
20210506_df_sex.rds (Gender data) <br>
## Figure3_e.R
### Description: <br>
This script is used for generating the heatmap in Figure 3_e. <br>
### Input: <br>
20210506_df_block_1.rds (Benchmark signal) <br>
20210506_df_block_2.rds (Positive test signal) <br>
20210506_df_block_3.rds (CCLI signal) <br>
20210506_df_block_4.rds (CLI broad signal) <br>
20210506_df_block_5.rds (CLI narrow signal) <br>
20210506_df_raw.rds (Raw data) <br>
20210506_df_age.rds (Age data) <br>
20210506_df_sex.rds (Gender data) <br>
### Output: <br>
Figure3_e.png <br>
## Figure4.R
### Description: <br>
This script is used for generating the heatmap in Figure 4. <br>
### Input: <br>
20210506_df_block_1.rds (Benchmark signal) <br>
20210506_df_block_2.rds (Positive test signal) <br>
20210506_df_block_3.rds (CCLI signal) <br>
20210506_df_block_4.rds (CLI broad signal) <br>
20210506_df_block_5.rds (CLI narrow signal) <br>
20210506_df_raw.rds (Raw data) <br>
20210506_df_age.rds (Age data) <br>
20210506_df_sex.rds (Gender data) <br>
### Output: <br>
Figure4.png
## SI_signal_histogram.R
### Description: <br>
This script is used to generate the histogram of CTIS signal Spearman correlations found in the SI.
### Input: <br>
20210506_df_raw.rds (Raw data) <br>
### Output: <br>
SI_correlation_histogram.png
