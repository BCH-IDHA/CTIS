library(tidyverse) #tidyverse_1.3.0
library(lubridate) #lubridate_1.7.4
library(viridis) #viridis_0.5.1
library(readxl) #readxl_1.3.1
# Run the following two lines to download the ComplexHeatmap library
#library(devtools)
#install_github("jokergoo/ComplexHeatmap")
library(ComplexHeatmap) #spatialEco::explode
library(circlize)
library(zoo)
library(tidyquant)
library(broom)

par_start <- ymd("2020-04-23")
par_end <- ymd("2020-12-20")

empty_time <- tibble(
  iso_dt = seq(
    
    # first survey date
    from = par_start,
    
    # data slice date
    to = par_end,
    
    # increment of one day
    by = 1))

df_time <- empty_time %>%
  dplyr::filter(iso_dt >= par_start + days(x=7)) %>%
  arrange(iso_dt)

#TJV: my method for creating a dataframe of the file and country names
fp_filelist <- list.files('../fasst_all')
df_filelist = data.frame(file_name = fp_filelist) %>%
  mutate(file_country = substr(file_name, nchar(file_name)-6, nchar(file_name)-4))
df_filelist <- df_filelist[(df_filelist$file_country %in% c('ITA', 'BOL')), ]
df_country_drop <- c("HKG", "PRI")

df_country <- df_filelist %>%
  dplyr::select(country_agg = file_country) %>%
  dplyr::filter(!country_agg %in% df_country_drop) %>%
  arrange(country_agg)

# TIME SERIES MATRICES FOR BLOCKS: MAIN-----
df_block_1 <- readRDS("../datafolder/benchmark/sub_20210506_df_block_1.rds")

df_block_2 <- readRDS("../datafolder/benchmark/sub_20210506_df_block_2.rds")

df_block_3 <- readRDS("../datafolder/benchmark/sub_20210506_df_block_3.rds")

df_block_4 <- readRDS("../datafolder/benchmark/sub_20210506_df_block_4.rds")

df_block_5 <- readRDS("../datafolder/benchmark/sub_20210506_df_block_5.rds")

df_raw <- readRDS("../datafolder/benchmark/sub_20210506_df_raw.rds")

df_age <- readRDS("../datafolder/benchmark/sub_20210506_df_age.rds")

df_sex <- readRDS("../datafolder/benchmark/sub_20210506_df_sex.rds")

fp_owid <- '../datafolder/OWID/owid-covid-data_20210423.csv'

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


# Calculate time to peak
df_norm <- df_raw %>%
  group_by(country_agg) %>%
  mutate_at(vars(contains("_rw") | contains("_wt") | starts_with("bench")),
            ~((. - min(.,na.rm=T))/
                max((. - min(.,na.rm=T)),na.rm=T))) %>%
  ungroup()
var_list <- c('m7_test_pos_sx_wt_pct_w7', 'm7_exp_kno_wt_pct_w7', 'm7_cli_afc_nl_wt_pct_w7', 'm7_cli_a14_nl_wt_pct_w7')
var_labels <-c('test_pos', 'CCLI', 'bCLI', 'nCLI')
delta_time_df <- data.frame()
time_diff_dist <- data.frame()
for (country in df_country$country_agg){
  for (i in 1:length(var_list)){
    country_sub <- df_raw[(df_raw$country_agg==country), ]
    max_t_bench_list <- country_sub[(country_sub$bench_case==max(country_sub$bench_case)), ]$iso_dt
    max_t_var <- country_sub[(country_sub[var_list[i]]==max(country_sub[var_list[i]])), ]$iso_dt
    max_t_bench <- ifelse(length(max_t_bench_list)>1, max_t_bench_list[1], max_t_bench_list)
    #print(max_t_var, max_t_bench)
    day_diff <- abs(julian(max_t_var, max_t_bench)[1])
    delt_df <- data.frame(var=var_labels[i], delta_day=day_diff, country=country)
    delta_time_df <- rbind(delta_time_df, delt_df)
  }
  delt_country_sub <- delta_time_df[(delta_time_df$country==country), ]
  country_min_var <- delt_country_sub[(delt_country_sub$delta_day==min(delt_country_sub$delta_day)), ]$var
  #print(country_min_var)
  for (var in country_min_var){
    tp_num <- ifelse(var=='test_pos', 1, 0)
    ccli_num <- ifelse(var=='CCLI', 1, 0)
    bcli_num <- ifelse(var=='nCLI', 1, 0)
    ncli_num <- ifelse(var=='bCLI', 1, 0)
    num_df <- data.frame(country=country, test_num=tp_num, ccli_num=ccli_num, bcli_num=bcli_num, ncli_num=ncli_num)
    time_diff_dist <- rbind(time_diff_dist, num_df)
  }
  
}

# ANNOTATION PEAK TIMING BY BLOCK -----

df_max1 <- which(df_block_1 == max(df_block_1, na.rm=T), arr.ind = TRUE) %>% 
  as_tibble(rownames = "country_agg") %>% 
  group_by(country_agg) %>%
  slice_min(order_by = col) %>%
  ungroup %>%
  dplyr::select(country_agg, peak_sig_1 = col) 

df_max2 <- which(df_block_2 == max(df_block_2, na.rm=T), arr.ind = TRUE) %>% 
  as_tibble(rownames = "country_agg") %>% 
  group_by(country_agg) %>%
  slice_min(order_by = col) %>%
  ungroup %>%
  dplyr::select(country_agg, peak_sig_2 = col) 

df_max3 <- which(df_block_3 == max(df_block_3, na.rm=T), arr.ind = TRUE) %>% 
  as_tibble(rownames = "country_agg") %>% 
  group_by(country_agg) %>%
  slice_min(order_by = col) %>%
  ungroup %>%
  dplyr::select(country_agg, peak_sig_3 = col) 

df_max4 <- which(df_block_4 == max(df_block_4, na.rm=T), arr.ind = TRUE) %>% 
  as_tibble(rownames = "country_agg") %>% 
  group_by(country_agg) %>%
  slice_min(order_by = col) %>%
  ungroup %>%
  dplyr::select(country_agg, peak_sig_4 = col)  

df_max5 <-  which(df_block_5 == max(df_block_5, na.rm=T), arr.ind = TRUE) %>% 
  as_tibble(rownames = "country_agg") %>% 
  group_by(country_agg) %>%
  slice_min(order_by = col) %>%
  ungroup %>%
  dplyr::select(country_agg, peak_sig_5 = col)

df_anno_peak <- df_country %>%
  left_join(df_max1) %>%
  #left_join(df_max2) #%>%
  left_join(df_max3) %>%
  #left_join(df_max4) %>%
  #left_join(df_max5) %>%
  column_to_rownames(var = "country_agg") %>%
  as.matrix()



# ANNOTATION DEMOGRAPHICS -----
# demographics
df_anno_sex <- df_country %>%
  left_join(df_sex) %>%
  dplyr::select(demo_sex, pct_gp_rw, country_agg) %>%
  pivot_wider(names_from = demo_sex, 
              values_from = pct_gp_rw) %>%
  column_to_rownames(var = "country_agg") %>%
  as.matrix()

df_anno_age <- df_country %>%
  left_join(df_age) %>%
  dplyr::select(demo_age, pct_gp_rw, country_agg) %>%
  pivot_wider(names_from = demo_age, 
              values_from = pct_gp_rw) %>%
  column_to_rownames(var = "country_agg") %>%
  as.matrix()

df_anno_sex_wt <- df_country %>%
  left_join(df_sex) %>%
  dplyr::select(demo_sex, pct_gp_wt, country_agg) %>%
  pivot_wider(names_from = demo_sex, 
              values_from = pct_gp_wt) %>%
  column_to_rownames(var = "country_agg") %>%
  as.matrix()

df_anno_age_wt <- df_country %>%
  left_join(df_age) %>%
  dplyr::select(demo_age, pct_gp_wt, country_agg) %>%
  pivot_wider(names_from = demo_age, 
              values_from = pct_gp_wt) %>%
  column_to_rownames(var = "country_agg") %>%
  as.matrix()

df_anno_rep <- df_country %>%
  left_join(df_sex) %>%
  group_by(country_agg) %>%
  summarise(n_tot_sur_rw = sum(n_sur_rw,na.rm=T)) %>%
  ungroup() %>%
  left_join(df_owid_full %>% 
              dplyr::select(country_agg, population) %>%
              unique()) %>%
  mutate(pop_rel = n_tot_sur_rw/population, 
         pop_sur = n_tot_sur_rw, 
         pop_pct = 100*pop_rel) %>%
  dplyr::select(country_agg, pop_rel, pop_sur, pop_pct) %>%
  column_to_rownames(var = "country_agg") 



# ANNOTATION REGIONS  -----
df_anno_reg <- df_country %>% 
  left_join(df_owid_full %>%
              dplyr::select(country_agg, continent)) %>%
  unique() %>%
  mutate(region = recode(continent, 
                         "Africa" = "Africa",
                         "North America" = "Americas", 
                         "South America" = "Americas",
                         "Asia" = "Asia and Oceania",
                         "Oceania" = "Asia and Oceania",
                         "Europe" = "Europe")) %>%
  dplyr::select(country_agg, region)


# ANNOTATION MONTHS ----
df_anno_months <- df_time %>%
  mutate(row_num = row_number(),
         iso_dt_mo = floor_date(iso_dt,unit = "months"), 
         iso_mo_name = month(iso_dt_mo, label = TRUE, abbr = TRUE)) %>%
  dplyr::filter(iso_dt == iso_dt_mo) 






# CORRELATION -----
df_anno_corr_1_2 <- df_raw %>% 
  nest(data = -country_agg) %>% 
  mutate(test = map(data,
                    ~ cor.test(.x$m7_test_pos_sx_wt_pct_w7, .x$bench_case, 
                               method="spearman", exact=FALSE)),
         tidied = map(test, tidy)) %>% 
  unnest(tidied) %>%
  mutate(mod_type = "sig1to2")

df_anno_corr_1_3 <- df_raw %>% 
  nest(data = -country_agg) %>% 
  mutate(test = map(data,
                    ~ cor.test(.x$m7_exp_kno_wt_pct_w7, .x$bench_case, 
                               method="spearman", exact=FALSE)),
         tidied = map(test, tidy)) %>% 
  unnest(tidied) %>%
  mutate(mod_type = "sig1to3")

df_anno_corr_1_4 <- df_raw %>% 
  nest(data = -country_agg) %>% 
  mutate(test = map(data,
                    ~ cor.test(.x$m7_cli_afc_nl_wt_pct_w7, .x$bench_case, 
                               method="spearman", exact=FALSE)),
         tidied = map(test, tidy)) %>% 
  unnest(tidied) %>%
  mutate(mod_type = "sig1to4")

df_anno_corr_1_5 <- df_raw %>% 
  nest(data = -country_agg) %>% 
  mutate(test = map(data,
                    ~ cor.test(.x$m7_cli_a14_nl_wt_pct_w7, .x$bench_case, 
                               method="spearman", exact=FALSE)),
         tidied = map(test, tidy)) %>% 
  unnest(tidied) %>%
  mutate(mod_type = "sig1to5")


df_anno_corr_all <- df_anno_corr_1_3 %>% 
  #bind_rows(df_anno_corr_1_3, df_anno_corr_1_4, df_anno_corr_1_5) %>%
  mutate(nlp = -log10(p.value)) %>%
  dplyr::select(country_agg,mod_type,nlp) %>%
  pivot_wider(names_from = mod_type, 
              values_from = nlp) %>%
  column_to_rownames(var = "country_agg") %>%
  as.matrix()


df_anno_corr_est <- df_anno_corr_1_3 %>% 
  #bind_rows(df_anno_corr_1_3, df_anno_corr_1_4, df_anno_corr_1_5) %>%
  dplyr::select(country_agg,mod_type,estimate) %>%
  pivot_wider(names_from = mod_type, 
              values_from = estimate) %>%
  column_to_rownames(var = "country_agg") %>%
  as.matrix()



df_anno_corr_cp <- df_anno_corr_1_3 %>%
  #bind_rows(df_anno_corr_1_3, df_anno_corr_1_4, df_anno_corr_1_5) %>%
  dplyr::select(country_agg,mod_type,estimate) %>%
  mutate(estimate_d = if_else(estimate <= -0.8, -1, 
                              if_else(estimate <= -0.7, -1,
                                      if_else(estimate <= -0.4, -1, 
                                              if_else(estimate < 0.4, 0, 
                                                      if_else(estimate < 0.7, 0.4, 
                                                              if_else(estimate < 0.9, 0.7, 
                                                                      if_else(estimate >=0.9, 1, NA_real_,NA_real_), 
                                                                      NA_real_), 
                                                              NA_real_), 
                                                      NA_real_), 
                                              NA_real_), 
                                      NA_real_))) %>%
  mutate(corP = estimate_d) %>%
  dplyr::select(country_agg,mod_type,corP) %>%
  pivot_wider(names_from = mod_type, 
              values_from = corP) %>%
  column_to_rownames(var = "country_agg") %>%
  as.matrix()


df_anno_corr_cp_series <- df_anno_corr_1_3 %>%
  #df_anno_corr_1_2 %>% 
  #bind_rows(df_anno_corr_1_3, df_anno_corr_1_4, df_anno_corr_1_5) %>%
  dplyr::select(country_agg,mod_type,estimate, p.value) %>%
  mutate(estimate_d = if_else(estimate <= -0.8, -1, 
                              if_else(estimate <= -0.7, -1,
                                      if_else(estimate <= -0.4, -1, 
                                              if_else(estimate < 0.4, 0, 
                                                      if_else(estimate < 0.7, 0.4, 
                                                              if_else(estimate < 0.9, 0.7, 
                                                                      if_else(estimate >=0.9, 1, NA_real_,NA_real_), 
                                                                      NA_real_), 
                                                              NA_real_), 
                                                      NA_real_), 
                                              NA_real_), 
                                      NA_real_)))


#Final Figure

par_row_split <- factor(df_anno_reg$region, levels=c("Americas", "Europe"))
# size of panels
par_width <- 3
par_height <- 1


# colors -----
# region
p_reg_africa = "green4"
p_reg_asia = "red2"
p_reg_europe = "yellow"
p_reg_americas = "navy"

# demographics
p_age1 = "blue3"
p_age2 = "yellow"

p_female = "orange"
p_male = "green4"
p_nonbin = "purple4"
p_noans = "snow3"
p_missing = "grey9"

# base colors
p_null = "white"

# signals
p_sig_1 = "black"
p_sig_2 = "navy"
p_sig_3 = "deeppink4"
p_sig_4 = "red2"
p_sig_5 = "orange"


# color schemes -----
# blocks
col_sig_1 = colorRamp2(c(0, 1), c(p_null, p_sig_1))
col_sig_2 = colorRamp2(c(0, 1), c(p_null, p_sig_2))
col_sig_3 = colorRamp2(c(0, 1), c(p_null, p_sig_3))
col_sig_4 = colorRamp2(c(0, 1), c(p_null, p_sig_4))
col_sig_5 = colorRamp2(c(0, 1), c(p_null, p_sig_5))

# annotations
col_cor = colorRamp2(c(-1,0, 1), c(p_age2, p_null, p_male))

col_reg = c("Africa" = p_reg_africa, 
            "Asia and Oceania" = p_reg_asia, 
            "Europe" = p_reg_europe, 
            "Americas" = p_reg_americas)

col_sex = c("1" = p_male, 
            "2" = p_female, 
            "3" = p_nonbin, 
            "4" = p_noans, 
            "5" = p_missing)

col_sex_list = c(p_male,
                 p_female,
                 p_nonbin,
                 p_noans,
                 p_missing)

col_age = colorRamp2(c(1, 7, 8), c(p_age1,  p_age2, p_missing))



# annotations -----
# legend annotation
anno_lgd = list(
  # Age
  Legend(labels = c("18-24", 
                    "25-34", 
                    "35-44",
                    "45-54",
                    "55-64",
                    "65-74",
                    "75+",
                    "Missing"), 
         title = "Age (years)", type = "points", pch = 15,
         legend_gp = gpar(col = col_age(1:8))),
  
  # Gender
  Legend(labels = c("Male", 
                    "Female", 
                    "Non-Binary", 
                    "Prefer Not to Answer", 
                    "Missing"), 
         title = "Gender", type = "points", pch = 15, 
         legend_gp = gpar(col = col_sex_list)),
  
  # Peak
  Legend(labels = c("Benchmark Case", 
                    "CCLI"), 
         title = "Signal Peak", type = "points",
         pch = c(19, 5),
         legend_gp = gpar(col = c(
           p_sig_1, 
           p_sig_3))),
  
  
  # legend
  Legend(at = c(-1, -0.7, -0.4, 0, 0.4, 0.7, 1, NaN),
         title = "Benchmark Correlation",
         
         labels = c("Strong (0.9 to 1)", 
                    "Moderate (0.7 to 0.9)", 
                    "Weak (0.4 to 0.7)", 
                    "None (-0.4 to 0.4)",
                    "Negative (-1 to -0.4)"
         ), 
         legend_gp = gpar(fill = c(col_cor(c(1, 0.7, 0.4, 0, -1))))))


# top annotation
anno_top= columnAnnotation(
  Months_survey = anno_mark(at = df_anno_months$row_num,
                            labels = df_anno_months$iso_mo_name))

anno_top_corr= columnAnnotation(
  Months_survey = anno_mark(at = c(1:4),
                            labels = c("CCLI")))


# left annotation 
anno_left_far = rowAnnotation(
  # data for annotation
  Region = par_row_split,
  
  # color for annotation
  col = list(
    Region = col_reg))

anno_left = rowAnnotation(
  
  # rotation of text
  
  # position of text
  gap = unit(2, "mm"),
  
  # data for annotation
  LogN = anno_barplot(log10(df_anno_rep$pop_sur), gp = gpar(fill = 1),axis_param = list(direction = "reverse")),
  Pct = anno_barplot(df_anno_rep$pop_pct, gp = gpar(fill = 0), ylim=c(0,6)),
  Age =  anno_barplot(df_anno_age, gp = gpar(fill = col_age(1:8), col=FALSE)),
  Gender =  anno_barplot(df_anno_sex, gp = gpar(fill = col_sex_list, col=FALSE))
)


# middle annotation
anno_middle = rowAnnotation(
  
  
  # data for annotation
  Peak = anno_points(df_anno_peak,
                     ylim = c(0, 240),
                     width= unit(par_width/2, "cm"),
                     pch = c(19,5),
                     gp = gpar(col = c(
                       p_sig_1, 
                       p_sig_3)),
                     add_points = TRUE,
                     pt_gp = gpar(col = c(
                       p_sig_1, 
                       p_sig_3))) 
)


# primary block -----
h1 <- Heatmap(df_block_1, 
              
              # color scheme
              col = col_sig_1, 
              
              # cluster rows only, dendrogram on left
              cluster_columns = FALSE,
              show_column_dend = FALSE,
              
              # split for the rows
              row_split = par_row_split,
              cluster_row_slices = FALSE,
              
              # annotation
              top_annotation = anno_top,
              
              # titles
              name = "Benchmark Case Signal", 
              column_title = "Benchmark\nCase Signal",
              column_title_side = "bottom",
              row_title = "Country/Territory", 
              #row_title_side = "left",
              show_column_names = FALSE,
              # change all to FALSE for 2 country ex.
              show_row_dend = FALSE,
              show_row_names = FALSE,
              show_heatmap_legend = FALSE,
              # size
              width = unit(par_width, "cm"), 
              height = unit(par_height, "cm"))

h2 <- Heatmap(df_block_2, 
              name = "Positive Test Signal", 
              column_title = "Positive Test\nSignal",
              
              # color scheme
              col = col_sig_2, 
              
              
              # cluster
              cluster_columns = FALSE, 
              cluster_rows = FALSE,
              show_column_dend = FALSE, 
              
              # annotation
              top_annotation = anno_top,
              left_annotation = anno_left,
              
              # size
              width = unit(par_width, "cm"), 
              show_column_names = FALSE,
              column_title_side = "bottom")


h3 <- Heatmap(df_block_3, 
              name = "CCLI Signal", 
              column_title = "CCLI Signal",
              
              # color scheme
              col = col_sig_3, 
              
              # cluster
              cluster_columns = FALSE, 
              cluster_rows = FALSE,
              show_column_dend = FALSE, 
              
              # annotation
              top_annotation = anno_top,
              # Comment out for full list of countries
              left_annotation = anno_left,
              right_annotation = anno_middle,
              
              # size
              width = unit(par_width, "cm"), 
              show_column_names = FALSE,
              show_heatmap_legend = FALSE,
              column_title_side = "bottom")



h4 <- Heatmap(df_block_4, 
              name = "CLI Signal (Broad)", 
              column_title = "CLI Signal\n(Broad)",
              
              # color scheme
              col = col_sig_4, 
              
              # cluster
              cluster_columns = FALSE, 
              cluster_rows = FALSE,
              show_column_dend = FALSE, 
              
              # annotation
              top_annotation = anno_top,
              
              # size
              width = unit(par_width, "cm"), 
              show_column_names = FALSE,
              column_title_side = "bottom")


h5 <- Heatmap(df_block_5, 
              
              name = "CLI Signal (Narrow)", 
              column_title = "CLI Signal\n(Narrow)",
              
              # color scheme
              col = col_sig_5, 
              
              # cluster
              cluster_columns = FALSE, 
              cluster_rows = FALSE,
              show_column_dend = FALSE, 
              
              # annotation
              top_annotation = anno_top,
              right_annotation = anno_middle,
              
              # size
              width = unit(par_width, "cm"), 
              show_column_names = FALSE,
              column_title_side = "bottom")

hC <- Heatmap(df_anno_corr_cp, 
              name = "Spearman Correlation", 
              column_title = "Correlation",
              
              
              # color scheme
              col = col_cor, 
              top_annotation = anno_top_corr,
              show_heatmap_legend = FALSE,
              
              # cluster
              cluster_columns = FALSE, 
              cluster_rows = FALSE,
              show_column_dend = FALSE, 
              show_row_dend = FALSE,
              
              
              # size
              width = unit(par_width/4, "cm"), 
              show_column_names = FALSE,
              column_title_side = "bottom")



png("../ITA_BOL_hm_TJV_11_29_21_test.png", 
    width=16, height=4, units="cm", res=500)
ht_list =  h1 + h3 + hC 

draw(ht_list,
     ht_gap = unit(c(3,7), "mm"))

dev.off()


