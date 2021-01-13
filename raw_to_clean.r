options(warn=-1)
suppressPackageStartupMessages({
library(tidyverse); library(dplyr); library(ggplot2); library(data.table);
library(lubridate); library(readr); library(stringr); library(ggcorrplot);
library(caret)    ; library(e1071); library(foreach); library(caretEnsemble);
library(MLeval)   ; library(ISLR) ; library(plyr)   ; library(randomForest);
})

links = list(link.cl="http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/cleveland.data",
             link.hu="http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/hungarian.data",
             link.lb="http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/long-beach-va.data",
             link.sw="http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/switzerland.data"
            )
link.name = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names"

pattern_blank_bf = "^(^\\s*)(?=\\d{1,2}\\s{1,}(?!=))(?!=)"
pattern_attrib = "(?<=\\d\\s)(?!\\w{2,}\\s{1}\\d+)(?!=)(?!\\w+\\))(?!\\w+\\/)(?!\\w+\\s\\w)\\w+"

info.data <- readLines(url(link.name))
info.data <- info.data[135:length(info.data) - 10]

attrib.name <- str_replace_all(info.data, pattern_blank_bf, "")
attrib.name <- str_extract_all(attrib.name, pattern_attrib, simplify = TRUE)
attrib.name <- attrib.name[!apply(attrib.name == "", 1, all), ]

# options(warn=2)
files <- list()
for (link in links) {
    print(paste("The link is", link, " "))
    
    u <- url(link)
    d <- read_file(u)
    
    s <- str_split(d, "(?<=name)\\n", simplify = FALSE)
    s <- str_replace_all(s[[1]], "\n", " ")
    
    if (grepl("cleveland", link)) {
        s <- s[1:length(s) - 1]
    } else {
        s <- s[1:length(s)]
    }
    
    files <- append(files, s)
}
files <- unlist(files)
df_full <- read.table(header = F, stringsAsFactors = F, text = files, fill = FALSE, 
    col.names = attrib.name)

# print full info on data / uncomment the one your are interested in to show
# print(info.data) # only info on 76 variables
# print(readLines(url(link.name))) # full info on dataset etc..

df_full$num <- ifelse(df_full$num == 0, "No", "Yes")
df_full$num <- relevel(as.factor(df_full$num), "Yes")
colnames(df_full)[colnames(df_full) == "num"] <- "target"

# vector of variables/attributes we will drop (not interesting)
dump1 <- c('id','ccf','pncaden','thaltime','rldv5','rldv5e','restckm','exerckm',
           'thalsev','thalpul','earlobe','lmt','ladprox','laddist','diag','cxmain',
           'ramus','om1', 'om2', 'rcaprox', 'rcadist', 'lvx1', 'lvx2', 'lvx3',
           'lvx4', 'lvf', 'cathef', 'junk', 'name')

# value we drop as well but might take into considerations for further analysis
dump2 <- c('htn','proto','dummy',# 'oldpeak', # check
           'ekgmo','ekgday','ekgyr', "dig", "prop", "nitr", "pro", "diuretic",
           'cmo','cday','cyr'
          )

dump <- ( colnames(df_full) %in% dump1 ) | ( colnames(df_full) %in% dump2 )

# remove selected columns
df_reduced <- df_full[,!dump]# <- list(NULL)

# replace -9 by NA
df_reduced <- df_reduced %>% mutate_all(~na_if(., -9))
# sapply(df_reduced , function(x) sum(is.na(x)))
       
# suppress variables with more than 500 NA
sub.transf <- as.vector(sapply(df_reduced , function(x) sum(is.na(x))<610))
df_reduced <- df_reduced[sub.transf]

attrib.cat <- c('sex','painloc','painexer','relrest','cp','smoke','fbs',
                'famhist','restecg','exang','xhypo','slope','ca','thal',
                 'dm','restwm' # more than 500 NA
               )

attrib.cont <- c('trestbps','chol','cigs','years','thaldur','met',
                 'thalach','thalrest','tpeakbps','tpeakbpd','trestbpd',
                  'restef','exeref','exerwm' # more than 500 NA
                )
# select attribute still in dataframe that are categorical
cat.attrib <- colnames(df_reduced) %in% attrib.cat

# convert to factors categorical features
df_convert <- data.frame(df_reduced[!cat.attrib], sapply(df_reduced[cat.attrib], function(x) as.factor(as.character(x))))

# replace NA by random value around mean in interval of std err
NA_distri <- function(x) {
    x_mean <- mean(x, na.rm = T)
    x_sd <- sd(x, na.rm = T)
    n <- sum(is.na(x))
    x[is.na(x)] <- round(runif(n, x_mean - x_sd, x_mean + x_sd), digits = 1)
    return(x)
}

df_convert$ageinterval <- cut(df_convert$age,
                   breaks = c(-Inf, median(df_convert$age), Inf),
                   labels = c('adult', 'elder'), right = FALSE)

set.seed(13)
df_clean <- df_convert %>% 
    group_by(ageinterval) %>% 
    mutate_if(is.numeric, .funs = funs(NA_distri(.)) ) %>% ungroup %>% subset(select = -c(ageinterval) )

# set.seed(13)
# train2 <- ddply(train2, .(ageinterval), transform, met= NA_distri(met)) %>% subset(select = -c(ageinterval) )


# replace NA with median in numerical features
df_clean <- df_clean %>% mutate_if(is.numeric, ~ifelse(is.na(.), median(., na.rm = TRUE), 
    .))

# replace NA with mode in categorical features
Mode <- function(x, na.rm = TRUE) {
    xtab <- table(x)
    xmode <- names(which(xtab == max(xtab)))
    x <- fct_explicit_na(x, xmode)
    return(x)
}

df_clean <- df_clean %>% mutate_if(is.factor, .funs = funs(Mode(.)))
df_clean <- df_clean %>% mutate_if(is.numeric, scale )


levels(df_clean$slope) <- c(0,0,1,2)
levels(df_clean$ca) <- c(0,1,2,3,3)
levels(df_clean$thal) <- c(0,0,0, 1,1,1, 2)

# saveRDS(df_clean, file = "heart_disease2-1.rds")
