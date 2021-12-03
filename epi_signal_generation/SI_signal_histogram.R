library(tidyverse) #tidyverse_1.3.0
library(lubridate) #lubridate_1.7.4
library(viridis) #viridis_0.5.1
library(readxl) #readxl_1.3.1
library(zoo)
library(tidyquant)
library(broom)
library(ggpubr)
library(grid)

df_raw <- readRDS("../datafolder/benchmark/20210506_df_raw.rds")

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

plot_corr_df <- df_anno_corr_1_2 %>% 
  bind_rows(df_anno_corr_1_3, df_anno_corr_1_4, df_anno_corr_1_5)

final_df <- plot_corr_df

sub1 = final_df[(final_df$mod_type=='sig1to2'), ]
sub2 = final_df[(final_df$mod_type=='sig1to3'), ]
sub3 = final_df[(final_df$mod_type=='sig1to4'), ]
sub4 = final_df[(final_df$mod_type=='sig1to5'), ]

med1_2 <- median(sub1$estimate)

med2_2 <- median(sub2$estimate)

med3_2 <- median(sub3$estimate)

med4_2 <- median(sub4$estimate)

# create lists with the median values to match the structure of the final_df dataframe
med_list2 = c()
for (i in 1:nrow(final_df)){
  if (final_df$mod_type[i]=='sig1to2'){
    med_list2 <- append(med_list2, med1_2)
  }
  if (final_df$mod_type[i]=='sig1to3'){
    med_list2 <- append(med_list2, med2_2)
  }
  if (final_df$mod_type[i]=='sig1to4'){
    med_list2 <- append(med_list2, med3_2)
  }
  if (final_df$mod_type[i]=='sig1to5'){
    med_list2 <- append(med_list2, med4_2)
  }
}
#change the name of the sig's
final_df$mod_type[final_df$mod_type=="sig1to2"] <- 'Test positive'
final_df$mod_type[final_df$mod_type=="sig1to3"] <- 'CCLI'
final_df$mod_type[final_df$mod_type=="sig1to4"] <- 'CLI (broad)'
final_df$mod_type[final_df$mod_type=="sig1to5"] <- 'CLI (narrow)'
# add the medians to the dataframe
final_df$all_median <- med_list2

#subset the data 
tp <- final_df[(final_df$mod_type=='Test positive'), ]
ccli <- final_df[(final_df$mod_type=='CCLI'), ]
bcli <- final_df[(final_df$mod_type=='CLI (broad)'), ]
ncli <- final_df[(final_df$mod_type=='CLI (narrow)'), ]

tp_fig <- ggplot(data = tp) +
  geom_histogram(aes(estimate, fill=mod_type), binwidth=0.05, boundary=0, show.legend=FALSE) +
  #geom_vline(aes(xintercept=sig_median), linetype='dashed', color='black') +
  geom_vline(aes(xintercept=all_median), linetype='dashed', color='black') +
  #annotate(geom="text",x=-0.6, y=17, label=tp_lab, size=3) +
  scale_fill_manual(values=c("navy")) +
  scale_x_continuous(breaks = c(-1, -.8, -.6,-.4, -.2, 0, 0.2, 0.4, 0.6, 0.8, 1), limits=c(-1, 1)) +
  ylab('Number of countries') +
  xlab('Correlation estimate') +
  labs(fill='') +
  theme_minimal() +
  rremove("ylab") + 
  rremove("xlab") +
  ylim(0, 20) +
  theme(panel.grid.minor = element_blank())
#xlim(-1, 1)
#png('tp_hist_TJV.png', width=10, height=10, units="cm", res=500)
#print(tp_fig)
#dev.off()

ccli_fig <- ggplot(data = ccli) +
  geom_histogram(aes(estimate, fill=mod_type), binwidth=0.05, boundary=0, show.legend=FALSE) +
  #geom_vline(aes(xintercept=sig_median), linetype='dashed', color='black') +
  geom_vline(aes(xintercept=all_median), linetype='dashed', color='black') +
  #annotate(geom="text",x=-0.6, y=17, label=ccli_lab, size=3) +
  scale_fill_manual(values=c("deeppink4")) +
  scale_x_continuous(breaks = c(-1, -.8, -.6,-.4, -.2, 0, 0.2, 0.4, 0.6, 0.8, 1), limits=c(-1, 1)) +
  ylab('Number of countries') +
  xlab('Correlation estimate') +
  labs(fill='') +
  theme_minimal() +
  rremove("ylab") + 
  rremove("xlab") +
  ylim(0, 20) +
  theme(panel.grid.minor = element_blank())
#xlim(-1, 1)
#png('ccli_hist_TJV.png', width=10, height=10, units="cm", res=500)
#print(ccli_fig)
#dev.off()

bcli_fig <- ggplot(data = bcli) +
  geom_histogram(aes(estimate, fill=mod_type), binwidth=0.05, boundary=0, show.legend=FALSE) +
  #geom_vline(aes(xintercept=sig_median), linetype='dashed', color='black') +
  geom_vline(aes(xintercept=all_median), linetype='dashed', color='black') +
  #annotate(geom="text",x=-0.6, y=17, label=bcli_lab, size=3) +
  scale_fill_manual(values=c("red2")) +
  scale_x_continuous(breaks = c(-1, -.8, -.6,-.4, -.2, 0, 0.2, 0.4, 0.6, 0.8, 1), limits=c(-1, 1)) +
  ylab('Number of countries') +
  xlab('Correlation estimate') +
  labs(fill='') +
  theme_minimal() +
  rremove("ylab") + 
  rremove("xlab") +
  ylim(0, 20) +
  theme(panel.grid.minor = element_blank())
#xlim(-1, 1)
#png('bcli_hist_TJV.png', width=10, height=10, units="cm", res=500)
#print(bcli_fig)
#dev.off()

ncli_fig <- ggplot(data = ncli) +
  geom_histogram(aes(estimate, fill=mod_type), binwidth=0.05, boundary=0, show.legend=FALSE) +
  #geom_vline(aes(xintercept=sig_median), linetype='dashed', color='black') +
  geom_vline(aes(xintercept=all_median), linetype='dashed', color='black') +
  #annotate(geom="text",x=-0.6, y=17, label=ncli_lab, size=3) +
  scale_fill_manual(values=c("orange")) +
  scale_x_continuous(breaks = c(-1, -.8, -.6,-.4, -.2, 0, 0.2, 0.4, 0.6, 0.8, 1), limits=c(-1, 1)) +
  ylab('Number of countries') +
  xlab('Correlation estimate') +
  labs(fill='') +
  theme_minimal() +
  rremove("ylab") + 
  rremove("xlab") +
  ylim(0, 20) +
  theme(panel.grid.minor = element_blank())
#xlim(-1, 1)
#png('ncli_hist_TJV.png', width=10, height=10, units="cm", res=500)
#print(ncli_fig)
#dev.off()


final_figure <- ggarrange(tp_fig, ccli_fig, bcli_fig, ncli_fig, ncol=1)
final_fig <- annotate_figure(final_figure, left = textGrob("Number of countries", rot = 90, vjust = 1, gp = gpar(fontsize=10)),
                             bottom = textGrob("Correlation estimate", gp = gpar(fontsize = 10)))
#png('../Corr_hist_TJV_11_17_21.png', width=10, height=10, units="cm", res=500)
#print(final_fig)
#dev.off()

tp_check <- ggplot_build(tp_fig)$data[[1]]
sum(tp_check[(tp_check$fill=='navy') & (tp_check$xmin >= -1) & (tp_check$xmin < -0.4), ]$count)
sum(tp_check[(tp_check$fill=='navy') & (tp_check$xmin > -0.4) & (tp_check$xmin < 0.4), ]$count)
sum(tp_check[(tp_check$fill=='navy') & (tp_check$xmin > 0.4) & (tp_check$xmin < 0.7), ]$count)
sum(tp_check[(tp_check$fill=='navy') & (tp_check$xmin > 0.7) & (tp_check$xmin < 0.9), ]$count)
sum(tp_check[(tp_check$fill=='navy') & (tp_check$xmin > 0.9) & (tp_check$xmin <= 1), ]$count)

