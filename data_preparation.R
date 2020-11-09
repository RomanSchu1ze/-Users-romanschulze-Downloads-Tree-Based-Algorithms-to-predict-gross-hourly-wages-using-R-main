# Tree Based Gross Hourly Wage Prediction                  
# Preparing data for modeling                        
# Datasource: SOEP data 2015                              
# Author: Roman Schulze                               

# Set working directory 
setwd("/Users/romanschulze/Desktop/Roman/Uni/Master/4.Semester/MachineLearning/SOEP/SOEPv33")

# Load relevant libraries and data
library(dplyr)
library(plyr)
library(readstata13)


# -------------------------------------

# 1. Load data

# -------------------------------------


# read data
pequiv15 <- read.dta13("bfpequiv.dta")
pequiv16 <- read.dta13("bgpequiv.dta")
pgen15 <- read.dta13("bfpgen.dta")

# Pequiv 2016 (All specifications refer to previous year (i.e. 2015))
pequiv_16 <- subset(pequiv16, select = c(persnr, i1111016, e1110116))
colnames(pequiv_16) <- c("persnr" , "Income" , "AnnualWorkHours")

# Pequiv 2015, filter relevant data
pequiv_15 <- subset(pequiv15, select = c(persnr, d1110715, d1110115, e1110315, d11102ll,
                                         l1110215))
colnames(pequiv_15) <- c("persnr" , "Children", "Age", "Employment",
                         "Gender", "Origin")

# Pgen 2015, filter relevant data 
pgen_15 <- subset(pgen15, select = c(persnr, bffamstd, oeffd15, betr15, bftatzeit, 
                                     bfbilzeit, erljob15, egp88_15, bferwzeit))
colnames(pgen_15) <- c("persnr", "Married" , "Public" , "Size" , "Hours", 
                       "Education", "LearnedJob","Erikson", "Tenure")

# Merge dataframes
data <- merge(pequiv_16, pequiv_15, by = "persnr")
data <- merge(data, pgen_15, by = "persnr")



# -------------------------------------

# 2. Cleansing Data

# -------------------------------------


# Remove unobservables/Missings

# create a list with strings that will be replaced by Na´s
to_be_replaced = list("[-1] keine Angabe", "[-2] trifft nicht zu", "[-3] nicht valide",
                      "[-4] Unzulaessige Mehrfachantwort",  "[-5] In Fragebogenversion nicht enthalten", 
                      "[-6] Fragebogenversion mit geaenderter Filterfuehrung")


# create a function to repalce elements in a dataframe by Na´s
replace_pattern_by_NA <- function(df, text_element) {
  df[] <- lapply(df, function(x) if(is.factor(x)) droplevels(replace(x, x == text_element, NA)) 
                 else x)
  return(df)
}


# loop over the to_be_replaced list and apply the predefined function
for (var in to_be_replaced) {
  data <- replace_pattern_by_NA(data, var)
}


# Remove observations who are currently in training
levels(data$LearnedJob)[levels(data$LearnedJob) == "[3] In Ausbildung"] <- NA

# Remove NA´s 
any(is.na(data))
sum(is.na(data))
data <- na.omit(data)


# Converting categorical Variables into binarys by summarizing certain categories

# LearnedJob
levels(data$LearnedJob)[levels(data$LearnedJob) == "[4] Keinen erlernten Beruf"] <- "[2] Nein"

# Creating married dummy
levels(data$Married) 

# Adjust to two categories
levels(data$Married) <- c("Married", "Married", "Single", "Single", "Single",
                          "Married", "Single", "Single")


# Remove Non working observations 
data <- dplyr::filter(data, Employment!= "Not Working") 
# Age restriction, keep observations who are 18 - 65
data <- dplyr::filter(data, Age > 17, Age <= 65)       

# define a vector of varibles that will be filtered for values > 0
filter_var <- c("Income", "AnnualWorkHours", "Hours", "Education", "Tenure")

# Keep if individuals show values larger than one w.r.t. filter_var
for (var in filter_var) {
  # Filter rows by acessing rows via df[rows, columns]
  data <- data[data[, var] > 0, ]                
}

                 
# Hourly Salary for 2015
data[, "Salary"] <- data[, "Income"] / data[, "AnnualWorkHours"]

# Remove lower and upper percentile to exclude extreme values
data <- dplyr::filter(data, Salary > quantile(data$Salary , 0.01),
                      Salary < quantile(data$Salary, 0.99))                         

# Generate log_Salary
data$Salary <- log(data$Salary)

# Keep relevant Variables 
data <- subset(data, select = c(Salary, Gender, Age, Origin, Married,
                                Children, Hours, Public, Education, 
                                Size, LearnedJob, Erikson, Tenure))

# Switch the order of the levels  
# I.e., currently the variable Gender has the levels:
# [1] "[1] Male           1" "[2] Female         2"
# After conversion it should be:
#[1] "[2] Female         2" "[1] Male           1"
vars <- c("Married", "Public", "LearnedJob", "Gender", "Origin")
new_label <- c("Single", "[2] Nein", "[2] Nein", "[2] Female         2", 
               "[2] East-Germany   2")

# access elemetns of vars and new_label simultaneously in each iteration
for (i in seq_along(relevel_var)) {
  data[, vars[i]] <- relevel(data[, vars[i]], new_label[i])
}


# Function to convert Factor´s to numeric
convert_factor_to_numeric <- function(var) {
  if (is.factor(var) == TRUE) {
    var <- as.numeric(var)
  }
  else {
    var <- var
  }
}

# Convert Factors to Numeric
data1 <- data
data1[] <- sapply(data, convert_factor_to_numeric)

# Converting following binary variables into 0/1
sub_one <- c("Gender", "Origin", "Married", "Public", "LearnedJob")

# loop over variables to subtract one for each
for (var in sub_one) {
  data1[, var] <- data1[, var] - 1
}


# Adjust Colnames, due to converting the Binarys 
# (columnlabel of binarys represents the 1 in corresponding column,
#  e.g. Male: Male=1, Female=0)
colnames(data1) <- c("Salary", "Male" , "Age" , "West" , "Married", 
                     "Children", "Hours", "Civil", "Education", "Size",
                     "LearnedJob", "Occupation", "Tenure")



# -------------------------------------

# 3. Clean Global Enviroment and save final dataframe as csv file
                 
# -------------------------------------
                 
                 
# Remove dataframes that are not needed anymore
rm(a, b, c, pequiv15, pequiv16, pgen15, data)

# Save final dataframe as csv file
write.csv(data1, file = "MyData.csv")

               
              



