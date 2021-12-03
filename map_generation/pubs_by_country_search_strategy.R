setwd("../COVID/fb_pub_search") #<== set working directory
.libPaths(c("../../../R/win-library/3.6","../R/R-3.6.2/library")) #<== path of R libraries

library(tidyverse)
library(medrxivr)
library(easyPubMed)

df1<-read.csv("fb_n.csv")

#download preprints
preprint_data <- mx_api_content(from_date = "2019-11-01", to_date = "2021-07-01")
preprint_datab <- mx_api_content(server = "biorxiv", from_date = "2019-11-01", to_date = "2021-07-01")

#base query pre loop for pubmed
base_query <- '(COVID-19 OR SARS-CoV-2 OR 2019-nCoV OR ncov2019) AND '
#base query pre loop for preprints
topic1  <- c("COVID-19","SARS-CoV-2 ","2019-nCoV","ncov2019")  # Combined with Boolean OR

results<-data.frame()
for(i in 1:nrow(df1)){
  name1<-as.vector(df1[i,3])
  alt_name<-as.vector(df1[i,4])
  name2<-ifelse(nchar(alt_name)>=1,alt_name,name1)
    
  #pubmed query
  my_query <- paste0(base_query,"(",name1," OR ",name2,")")
  
  #preprint query
  topic2 <- c(name1,name2)
  preprint_query <- list(topic1,topic2) # Combined with Boolean AND

  #search pubmed
  my_entrez_id <- get_pubmed_ids(my_query)
  pubmedCount <- as.numeric(my_entrez_id$Count)

  #search preprints medxriv
  results_medxriv <- mx_search(data = preprint_data,
                       query=preprint_query)
  
  #search preprints bioxriv
  results_bioxriv <- mx_search(data = preprint_datab,
                      query=preprint_query)

  medrxivCount <- nrow(results_medxriv)
  biorxivCount <- nrow(results_bioxriv)

  #counts added
  total_count <- sum(pubmedCount,medrxivCount,biorxivCount)
  
  results_int<-data.frame(ISO3=as.vector(df1[i,1]),total_count)
  
  results<-rbind(results,results_int)
}

df2 <- merge(df1,results,by="ISO3")