# set up -----
# libraries
library(tidyverse) #tidyverse_1.3.0
library(lubridate) #lubridate_1.7.4
library(viridis) #viridis_0.5.1
library(readxl) #readxl_1.3.1
library(zoo)
library(tidyquant)
library(broom)

# session information
sessionInfo() # R version 3.6.3 (2020-02-29)

# set paramters
options(stringsAsFactors = FALSE)

# weekly average function
weeksum <- function(z) sum(z[time(z) > max(time(z)) - 7])

# start and end dates
par_start <- ymd("2020-04-23")
par_end <- ymd("2020-12-20")

# empty tibble file
empty_time <- tibble(
  iso_dt = seq(
    
    # first survey date
    from = par_start,
    
    # data slice date
    to = par_end,
    
    # increment of one day
    by = 1))


# multiplot function -----
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


# directory and file paths -----

# file paths
fp_root <- "../fasst_all/"
#TJV: my method for creating a dataframe of the file and country names
fp_filelist <- list.files('../fasst_all')
df_filelist = data.frame(file_name = fp_filelist) %>%
  mutate(file_country = substr(file_name, nchar(file_name)-6, nchar(file_name)-4))

#TJV: uncomment to subset our list of countries to the two of interest (BOL and ITA)
#df_filelist = df_filelist[(df_filelist$file_country %in% c('ITA', 'BOL')), ]

dim(df_filelist)
str(df_filelist)

# benchmark file
fp_owid <- "../datafolder/OWID/owid-covid-data_20210423.csv"

## combined countries parameter set -----
#TJV: find the subset of indices
var_index_list <- row.names(df_filelist)
var_index_start <- var_index_list[1]

# SIGNALS  -----
start_time <- Sys.time()
df_natwk_all <- tibble()


#TJV: restructured the for loop to list through the subset of indices
for(var_index in var_index_list) {
  
  # index, country code, and file
  print(var_index)
  
  var_iso <- df_filelist[var_index, "file_country"]
  print(var_iso)
  
  fp_file <- df_filelist[var_index, "file_name"]
  fp_country <- paste0(fp_root, fp_file)
  
  
  # read in country file
  df_natwk_index <- read.csv(fp_country, 
                             header=T) %>%
    
    # filter for through survey 6
    dplyr::filter(!survey_version %in% c("7", "7.0", "8")) %>%
    
    # time
    mutate(iso_dt = date(ymd_hms(recordeddate))) %>%
    
    # select minimal variables
    dplyr::select(
      
      # survey variables
      iso_dt,
      weight,
      
      # symptoms
      dur_ill = b2,     
      count_symp_endorsed,
      sx_fever = b1_1, 
      sx_cough = b1_2, 
      sx_smelltaste = b1_10, 
      sx_smelltasteun = b1b_x10,
      
      exp_knowsick = b3, 
      ts_ever = b6, 
      ts_recent = b7, 
      ts_pos = b8) %>%
    
    # cli counts of yes per suvey with missing as 0
    mutate(
      unweight = 1,
      cli_afc_nl = if_else((sx_smelltaste == 1 | sx_fever == 1 | sx_cough == 1),
                           1L, 0L, 0L),
      cli_a14_nl = if_else(sx_smelltaste == 1 & 
                             dur_ill < 14 &
                             dur_ill >= 0,
                           1L, 0L, 0L), 
      cli_a14_un = if_else(sx_smelltasteun == 1 & 
                             dur_ill < 14 &
                             dur_ill >= 0,
                           1L, 0L, 0L),
      
      exp_kno = if_else(exp_knowsick == 1,
                        1L, 0L, 0L),
      
      test_ever = if_else(ts_ever == 1,
                          1L, 0L, 0L),
      test_recent_as = if_else(ts_ever == 1 & ts_recent == 1,
                               1L, 0L, 0L), 
      test_recent_sx = if_else(ts_ever == 1 & ts_recent == 1 & count_symp_endorsed >= 1,
                               1L, 0L, 0L), 
      test_pos_as = if_else(ts_ever == 1 & ts_recent == 1 & ts_pos == 1,
                            1L, 0L, 0L),
      test_pos_sx = if_else(ts_ever == 1 & ts_recent == 1 & count_symp_endorsed >= 1 & ts_pos == 1,
                            1L, 0L, 0L)) %>%
    
    # variables
    dplyr::select(iso_dt, 
                  unweight, weight, 
                  starts_with("cli"), 
                  exp_kno, 
                  starts_with("test")) %>%
    
    # daily estimates
    group_by(iso_dt) %>%
    summarise_all(list(rw=~sum(., na.rm=T), wt=~sum(.*weight, na.rm=T))) %>%
    ungroup() %>%
    
    # fill missing time 
    right_join(empty_time, 
               by = "iso_dt") %>%
    
    # order chronologicall
    arrange(iso_dt) %>%
    
    # add in country indicator
    mutate(country_agg = var_iso) 
  
  
  # combine all countries together in long tibble
  ifelse(var_index == var_index_start,
         df_natwk_all <- df_natwk_index, 
         df_natwk_all <- df_natwk_all %>% bind_rows(df_natwk_index))
  
}  

df_sig_day <- df_natwk_all
str(df_sig_day)

end_time <- Sys.time()
end_time - start_time
# Time difference of 22.45725 mins

# moving average percentages over 7 days
df_sig_temp <- df_sig_day %>%
  
  # select variables
  dplyr::select(country_agg, iso_dt, 
                n_sur = unweight_rw,
                wt_sur = weight_rw,
                contains("cli_"), 
                contains("exp_"), 
                contains("test_")) 

# so headers match in colname argument use temp file
df_sig_wk <- df_sig_temp %>%
  
  # group by country
  group_by(country_agg) %>%
  
  # rolling sum
  tq_mutate(
    
    # variables (all if NULL)
    select     = NULL,
    
    # rolling sum over 7 day width aligned right
    mutate_fun = rollapply, 
    width      = 7,
    align      = "right",
    na.rm = TRUE,
    FUN = sum, 
    
    # add columns to the tibble using all but first column (time)
    col_rename = paste0("m7_", colnames(df_sig_temp)[-c(1,2)])) %>%
  
  # create percentages of weight an unweighted, daily and weekly
  mutate_at(vars(ends_with("_rw") & !starts_with("m7_")), list(pct_r1=~./n_sur)) %>%
  mutate_at(vars(ends_with("_wt") & !starts_with("m7_")), list(pct_w1=~./wt_sur)) %>%
  mutate_at(vars(ends_with("_rw") & starts_with("m7_")), list(pct_r7=~./m7_n_sur)) %>%
  mutate_at(vars(ends_with("_wt") & starts_with("m7_")), list(pct_w7=~./m7_wt_sur))

# DEMOGRAPHICS  -----

# estimate run time
start_time <- Sys.time()

df_natwk_all_sex <- tibble()
df_natwk_all_age <- tibble()

for(var_index in var_index_list) {
  
  # index, country code, and file
  print(var_index)
  
  var_iso <- df_filelist[var_index, "file_country"]
  print(var_iso)
  
  fp_file <- df_filelist[var_index, "file_name"]
  fp_country <- paste0(fp_root, fp_file)
  
  
  # read in country file
  df_natwk_index <- read.csv(fp_country, 
                             header=T) %>%
    
    
    # filter for through survey 6
    dplyr::filter(!survey_version %in% c("7", "7.0", "8")) %>%
    
    # time
    mutate(iso_dt = date(ymd_hms(recordeddate))) %>%
    dplyr::filter(iso_dt >= par_start, 
                  iso_dt <= par_end) %>%
    
    # select key variable
    dplyr::select(
      
      # survey variables
      weight,
      
      # symptoms
      demo_sex = e3,
      demo_age = e4) %>%
    
    # cli
    mutate(
      demo_sex = as.character(if_else(demo_sex %in% c(1:4), as.character(demo_sex), "5", "5")),
      demo_age = as.character(if_else(demo_age %in% c(1:7), as.character(demo_age), "8", "8"))) %>%
    
    # add in country indicator
    mutate(country_agg = var_iso) 
  
  
  # sex
  df_natwk_index_sex <- df_natwk_index %>%
    
    # group estimates
    group_by(demo_sex, country_agg) %>%
    summarise(n_sur_rw = n(), 
              n_sur_wt = sum(weight, na.rm=T)) %>%
    ungroup() %>%
    group_by(country_agg) %>%
    mutate(n_all_rw = sum(n_sur_rw, na.rm=T), 
           n_all_wt = sum(n_sur_wt, na.rm=T),
           pct_gp_rw = n_sur_rw/n_all_rw,
           pct_gp_wt = n_sur_wt/n_all_wt)
  
  
  df_natwk_index_age <- df_natwk_index %>%
    
    # group estimates
    group_by(demo_age, country_agg) %>%
    summarise(n_sur_rw = n(), 
              n_sur_wt = sum(weight, na.rm=T)) %>%
    ungroup() %>%
    group_by(country_agg) %>%
    mutate(n_all_rw = sum(n_sur_rw, na.rm=T), 
           n_all_wt = sum(n_sur_wt, na.rm=T),
           pct_gp_rw = n_sur_rw/n_all_rw,
           pct_gp_wt = n_sur_wt/n_all_wt)
  
  
  # combine all countries together in long tibble
  # replaced var_index_start w/ 25
  ifelse(var_index == var_index_start, 
         df_natwk_all_sex <- df_natwk_index_sex, 
         df_natwk_all_sex <- df_natwk_all_sex %>% bind_rows(df_natwk_index_sex))
  
  ifelse(var_index == var_index_start, 
         df_natwk_all_age <- df_natwk_index_age, 
         df_natwk_all_age <- df_natwk_all_age %>% bind_rows(df_natwk_index_age))
  
} 

df_sex <- df_natwk_all_sex
str(df_sex)

df_age <- df_natwk_all_age
str(df_age)

end_time <- Sys.time()
end_time - start_time
# Time difference of 20.1255 mins

# BENCHMARK DATA -----

df_owid_full <- read.csv(fp_owid, 
                         header=T) %>%
  dplyr::select(
    iso_dt = date, 
    country_agg = iso_code,
    continent,
    population,
    stringency_index,
    bench_case = new_cases_smoothed,
    bench_pp = positive_rate,
    bench_test = new_tests_smoothed) %>%
  mutate(iso_dt = ymd(iso_dt)) %>%
  dplyr::filter(iso_dt >= par_start, 
                iso_dt <= par_end)

df_bench <- df_owid_full %>%
  dplyr::select(country_agg, iso_dt, 
                bench_case, bench_pp, bench_test)


# ENSURE TIME SERIES AND COUNTRY SUBSET -----
# time list (drop first 7 days)
df_time <- empty_time %>%
  dplyr::filter(iso_dt >= par_start + days(x=7)) %>%
  arrange(iso_dt)

# country list (drop HKG, PRI)
df_country_drop <- c("HKG", "PRI")
df_country <- df_filelist %>%
  dplyr::select(country_agg = file_country) %>%
  dplyr::filter(!country_agg %in% df_country_drop) %>%
  arrange(country_agg)


# RAW & NORMALIZED TIME SERIES -----
df_raw <- df_country %>%
  left_join(df_sig_wk) %>%
  left_join(df_bench) %>%
  
  # time filter
  dplyr::filter(iso_dt %in% df_time$iso_dt) %>%
  
  # NA treated as Inf for min step but then NA for ratio
  mutate_at(vars(contains("_un_")),
            ~ifelse(iso_dt >= ymd("2020-06-27"), 
                    ., NA)) %>%
  
  # create pos proportion 
  mutate(m7_test_pp_as_wt_pct_w7 = m7_test_pos_as_wt_pct_w7/m7_test_recent_as_wt_pct_w7) %>%
  mutate(m7_test_pp_sx_wt_pct_w7 = m7_test_pos_sx_wt_pct_w7/m7_test_recent_sx_wt_pct_w7) %>%
  as_tibble()

df_norm <- df_raw %>%
  group_by(country_agg) %>%
  mutate_at(vars(contains("_rw") | contains("_wt") | starts_with("bench")),
            ~((. - min(.,na.rm=T))/
                max((. - min(.,na.rm=T)),na.rm=T))) %>%
  ungroup()


# TIME SERIES MATRICES FOR BLOCKS: MAIN-----
# benchmark
df_block_1 <- df_norm %>% 
  dplyr::select(country_agg, iso_dt, var_sig=bench_case) %>%
  pivot_wider(names_from = iso_dt, 
              values_from = var_sig) %>%
  column_to_rownames(var = "country_agg") %>%
  as.matrix()

# test positive
df_block_2 <- df_norm %>% 
  dplyr::select(country_agg, iso_dt, var_sig=m7_test_pos_sx_wt_pct_w7) %>%
  pivot_wider(names_from = iso_dt, 
              values_from = var_sig) %>%
  column_to_rownames(var = "country_agg") %>%
  as.matrix()

# CLI know 
df_block_3 <- df_norm %>% 
  dplyr::select(country_agg, iso_dt, var_sig=m7_exp_kno_wt_pct_w7) %>%
  pivot_wider(names_from = iso_dt, 
              values_from = var_sig) %>%
  column_to_rownames(var = "country_agg") %>%
  as.matrix()

# CLI broad
df_block_4 <- df_norm %>% 
  dplyr::select(country_agg, iso_dt, var_sig=m7_cli_afc_nl_wt_pct_w7) %>%
  pivot_wider(names_from = iso_dt, 
              values_from = var_sig) %>%
  column_to_rownames(var = "country_agg") %>%
  as.matrix()

df_block_4_series <- df_norm %>% 
  dplyr::select(country_agg, iso_dt, var_sig=m7_cli_afc_nl_wt_pct_w7)

# CLI narrow
df_block_5 <- df_norm %>% 
  dplyr::select(country_agg, iso_dt, var_sig=m7_cli_a14_nl_wt_pct_w7) %>%
  pivot_wider(names_from = iso_dt, 
              values_from = var_sig) %>%
  column_to_rownames(var = "country_agg") %>%
  as.matrix()

# SAVE FILES -----
# clean up
rm(df_natwk_index)
#rm(df_natwk_index_new) #TJV: this object isn't defined
rm(df_natwk_all)


# RDS for files (if only running script for ITA and BOL, add "sub_" to the beginning of file names)
saveRDS(df_block_1,
        "../datafolder/benchmark/sub_20210506_df_block_1.rds")

saveRDS(df_block_2,
        "../datafolder/benchmark/sub_20210506_df_block_2.rds")

saveRDS(df_block_3,
        "../datafolder/benchmark/sub_20210506_df_block_3.rds")

saveRDS(df_block_4,
        "../datafolder/benchmark/sub_20210506_df_block_4.rds")

saveRDS(df_block_5,
        "../datafolder/benchmark/sub_20210506_df_block_5.rds")

saveRDS(df_raw,
        "../datafolder/benchmark/sub_20210506_df_raw.rds")

saveRDS(df_age,
        "../datafolder/benchmark/sub_20210506_df_age.rds")

saveRDS(df_sex,
        "../datafolder/benchmark/sub_20210506_df_sex.rds")

