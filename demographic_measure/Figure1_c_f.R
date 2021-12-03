library(ggplot2)
library(tidyverse) #tidyverse_1.3.0
library(lubridate) #lubridate_1.7.4
library(viridis)
library(ggpubr)


demo_diff_ctis_census_by_country <- read.csv("../datafolder/demographics/demo_diff_ctis_census_by_country.csv")
demo_diff_weekly_ctis_census_by_country_TJV_format <- read.csv("../datafolder/demographics/demo_diff_weekly_ctis_census_by_country_TJV_format.csv")

demo_sub <- demo_diff_ctis_census_by_country[(demo_diff_ctis_census_by_country$country!='Congo'), ]

male <- tibble(demo_sub$male)
male_test <- male %>% 
  gather(var, value) %>%
  separate(value, c("YoungMale", "MiddleAgeMale", "ElderlyMale"), sep = ",") %>% 
  mutate(YoungMale = str_remove_all(YoungMale, "\\["), ElderlyMale = str_remove_all(ElderlyMale, "\\]")) 
male_test
female <- tibble(demo_sub$female)
female_test <- female %>% 
  gather(var, value) %>%
  separate(value, c("YoungFemale", "MiddleAgeFemale", "ElderlyFemale"), sep = ",") %>% 
  mutate(YoungFemale = str_remove_all(YoungFemale, "\\["), ElderlyFemale = str_remove_all(ElderlyFemale, "\\]")) 
female_test

demo_sub$YoungMale <- as.numeric(male_test$YoungMale)
demo_sub$MiddleAgeMale <- as.numeric(male_test$MiddleAgeMale)
demo_sub$ElderlyMale <- as.numeric(male_test$ElderlyMale)

demo_sub$YoungFemale <- as.numeric(female_test$YoungFemale)
demo_sub$MiddleAgeFemale <- as.numeric(female_test$MiddleAgeFemale)
demo_sub$ElderlyFemale <- as.numeric(female_test$ElderlyFemale)

df_new <- demo_diff_weekly_ctis_census_by_country_TJV_format %>%
  dplyr::filter(!Group %in% c("total_diff", "mean_diff")) %>%
  
  # median for each age-gender-country over study period weeks
  group_by(Country, Group) %>%
  mutate(Value_median = median(Value, na.rm=T)) %>%
  ungroup() %>%
  
  # normalize to the country-group median
  mutate(Value_norm = Value - Value_median) %>%
  
  # collapse over group and week for time series across CTIS
  group_by(Group, Week) %>%
  
  # metrics for plotting
  summarise(value_min = min(Value_norm, na.rm=T), 
            value_max = max(Value_norm, na.rm=T), 
            value_median = median(Value_norm, na.rm=T), 
            value_q1 = quantile(Value_norm, 0.25, na.rm=T),
            value_q3 = quantile(Value_norm, 0.75, na.rm=T)) %>%
  ungroup() %>%
  
  # not ideal method but gets job done
  mutate(Gender = if_else(Group %in% c("YoungMale", "MiddleMale", "ElderlyMale"),
                          "Male",
                          if_else(Group %in% c("YoungFemale", "MiddleFemale", "ElderlyFemale"), 
                                  "Female", "None"))) %>%
  mutate(Age = if_else(Group %in% c("YoungMale", "YoungFemale"), 
                       "Young",
                       if_else(Group %in% c("MiddleMale", "MiddleFemale"), 
                               "Middle", 
                               if_else(Group %in% c("ElderlyMale", "ElderlyFemale"),
                                       "Elderly", "None", "None")))) 


fig1_e <- ggplot(df_new %>% 
                   dplyr::filter(!Group %in% c("total_diff", "mean_diff"))) +
  facet_grid(Age ~ Gender) +
  geom_hline(mapping=aes(yintercept = 0),lwd=0.4) +
  geom_ribbon(mapping=aes(x=Week,
                          ymin=value_q1, ymax=value_q3), alpha=0.5, show.legend=FALSE) +
  #fill=Gender), 
  #alpha =0.15) +
  geom_ribbon(mapping=aes(x=Week,
                          ymin=value_min, ymax=value_max), alpha=0.2, show.legend=FALSE) +
  #fill=Gender), 
  #alpha =0.15) +
  
  geom_line(mapping=aes(x=Week, value_median), lwd=0.2) +
  theme_minimal(base_size=6) +
  ylab(expression("\u0394 D"["g,w"])) +
  #scale_color_manual(values = c("orange", "green4")) +
  ylim(c(-0.2,0.2)) +
  guides(color=F, fill=F) +
  theme(axis.text.x = element_text(angle = 90),
        plot.margin = margin(t = 5.5, r = 8.5, b = 5.5, l = 5.5, unit = "pt"))

# same as above, only for the mean_diff, normalized to overall country mean_diff
df_next <- demo_diff_weekly_ctis_census_by_country_TJV_format %>%
  dplyr::filter(Group %in% c("mean_diff")) %>%
  group_by(Country, Group) %>%
  mutate(Value_median = median(Value, na.rm=T)) %>%
  ungroup() %>%
  mutate(Value_norm = Value - Value_median) %>%
  group_by(Group, Week) %>%
  summarise(value_min = min(Value_norm, na.rm=T), 
            value_max = max(Value_norm, na.rm=T), 
            value_median = median(Value_norm, na.rm=T), 
            value_q1 = quantile(Value_norm, 0.25, na.rm=T),
            value_q3 = quantile(Value_norm, 0.75, na.rm=T)) %>%
  ungroup() 

fig1_f <- ggplot(df_next ) +
  geom_hline(mapping=aes(yintercept = 0),lwd=0.2) +
  geom_ribbon(mapping=aes(x=Week,
                          ymin=value_min, ymax=value_max), 
              alpha =0.2) +
  geom_ribbon(mapping=aes(x=Week,
                          ymin=value_q1, ymax=value_q3), 
              alpha =0.3) +
  geom_line(mapping=aes(x=Week, y=value_median), lwd=0.2) +
  theme_minimal(base_size=6) +
  ylim(c(-0.2,0.2)) +
  guides(color=F, fill=F) +
  ylab(expression('\u0394'['\u03B4,w'])) +
  theme(axis.text.x = element_text(angle = 90))

demo_sub$mean_diff <- as.numeric(demo_sub$mean_diff)
fig1_d <- ggplot(demo_sub, aes(x=mean_diff)) +
  geom_boxplot(outlier.size=0.5) +
  theme_minimal(base_size=6) +
  coord_flip() +
  xlab('\u03B4') +
  theme(axis.ticks.y=element_blank(),axis.text.x=element_blank(),
        panel.grid.minor.x=element_blank(), panel.grid.major.x=element_blank(),
        plot.margin = margin(t = 5.5, r = 8.5, b = 5.5, l = 5.5, unit = "pt")) +
  #xlim(-0.3, .5) +
  #scale_x_continuous(labels = c("", 0, 0.2, 0.4), limits = c(-.3, .5)) +
  geom_vline(xintercept=0.0, linetype = "dashed", lwd=0.2)

col_names <- c(
  "ElderlyFemale" = "Elderly\nFemale",
  "ElderlyMale" = "Elderly\nMale",
  "MiddleAgeFemale" = "Middle\nFemale",
  "MiddleAgeMale" = "Middle\nMale",
  "YoungFemale" = "Young\nFemale",
  "YoungMale" = "Young\nMale"
)

gend_sub <- demo_sub %>% dplyr::select(country, YoungMale:ElderlyFemale) %>% pivot_longer(-c(country))
gend_sub$name <- factor(gend_sub$name, levels = c('YoungMale', 'YoungFemale', 'MiddleAgeMale', 'MiddleAgeFemale', 'ElderlyMale', 'ElderlyFemale'))

fig1_c <- ggplot(gend_sub, aes(x=value)) +
  facet_grid(.~name, labeller = as_labeller(col_names)) +
  geom_boxplot(show.legend=FALSE, outlier.size=0.5) +
  theme_minimal(base_size=6) +
  coord_flip() +
  geom_vline(xintercept=0.0, linetype = "dashed", lwd=0.2) +
  theme(axis.ticks.y=element_blank(),axis.text.x=element_blank(),
        panel.grid.minor.x=element_blank(), panel.grid.major.x=element_blank(),
        plot.margin = margin(t = 5.5, r = 8.5, b = 5.5, l = 5.5, unit = "pt")) +
  xlab(expression(D["g"]))

final_fig <- ggarrange(fig1_c, fig1_d, fig1_e, fig1_f,
                       nrow = 1, widths = c(2,1, 2, 1))
#png("../Fig1_c_f_11_15_21_v1.png",
#    width=7, height=2, units="in", res=500)
#print(final_fig)
#dev.off()

df_last <- demo_diff_weekly_ctis_census_by_country_TJV_format %>%
  dplyr::filter(Group %in% c("mean_diff")) %>%
  group_by(Country, Group) %>%
  mutate(Value_median = median(Value, na.rm=T)) %>%
  ungroup() %>%
  mutate(Value_norm = Value - Value_median)

country_fig <- ggplot(df_last ,
                      aes(x=Week,
                          y = Value_norm)) +
  facet_wrap("Country") +
  geom_hline(mapping=aes(yintercept = 0)) +
  geom_linerange(aes(x=Week, ymax=Value_norm, ymin=0),
                 color="red4",
                 position = position_jitter(height = 0L, seed = 1L)) +
  theme_minimal(base_size=8) +
  ylab(expression("\u0394 D"["g,w"])) +
  ylim(c(-0.06, 0.06)) +
  guides(color=F, fill=F)

yf_med <- median(gend_sub[(gend_sub$name=='YoungFemale'),]$value)
mf_med <- median(gend_sub[(gend_sub$name=='MiddleAgeFemale'),]$value)
ef_med <- median(gend_sub[(gend_sub$name=='ElderlyFemale'),]$value)

ym_med <- median(gend_sub[(gend_sub$name=='YoungMale'),]$value)
mm_med <- median(gend_sub[(gend_sub$name=='MiddleAgeMale'),]$value)
em_med <- median(gend_sub[(gend_sub$name=='ElderlyMale'),]$value)

med_delt <- median(demo_sub$mean_diff)
n_maj_delt <- nrow(demo_sub[(demo_sub$mean_diff<0.1),])
n_min_delt <- nrow(demo_sub[(demo_sub$mean_diff>=0.1),])
n_total <- n_maj_delt + n_min_delt

delt_delt_w_min <- min(df_next$value_min)
delt_delt_w_max <- max(df_next$value_max)
most_variable_w <- df_next[(df_next$value_median==max(df_next$value_median)), ]

