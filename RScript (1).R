#Name: Group 1 Project Script
#Date: 

rm(list=ls()) #Clear environment
cat("\014")   #Clear Console
options(scipen = 999) #Setting high value for scientific notation
install.packages("gbm")
# Load required libraries
library(tidyverse)
library(naniar)
library(outliers)
library(reshape2)
library(ggplot2)
library(reshape2)
library(psych)
library(DataExplorer)
library(gridExtra)
library(caret)
library(dplyr)
library(rpart) #Decision Tree
library(randomForest) #Random Forest
library(MASS)   # For LDA
library(gbm)

# Read the CSV file
credit_card_data <- read.csv("application_record.csv")
str(credit_card_data)

#1. Exploring the data set
describe(credit_card_data)

#2. Number of entries 
glimpse(credit_card_data)

#3. Identifying missing values
plot_missing(credit_card_data)

#A Detailed report on data set
create_report(credit_card_data)

# 3. Check for duplications
num_duplicates <- sum(duplicated(credit_card_data))
cat("Number of duplicate rows:", num_duplicates, "\n")

#New variable AGE is created
credit_card_data$AGE <- round(abs(credit_card_data$DAYS_BIRTH) / 365)

#New variable EXPERIENCE is created
credit_card_data$EXPERIENCE <- round(abs(credit_card_data$DAYS_EMPLOYED) / 365)
total_experience <- sum(credit_card_data$EXPERIENCE == "1001")

# 4. Check for outliers in numerical columns
calculate_outliers <- function(credit_card_data) {
  q25 <- quantile(credit_card_data, 0.25)
  q75 <- quantile(credit_card_data, 0.75)
  iqr <- q75 - q25
  lower <- q25 - 1.5 * iqr
  upper <- q75 + 1.5 * iqr
  return(c("lower"=lower, "upper"=upper))
}

# Applying the outlier detection method to relevant numeric columns in the credit card dataset
# Example assuming the dataset is named 'credit_card_data'
outlier_bounds <- lapply(credit_card_data[c("AMT_INCOME_TOTAL", "DAYS_BIRTH", 
                                            "DAYS_EMPLOYED", "CNT_CHILDREN", 
                                            "CNT_FAM_MEMBERS")], calculate_outliers)

# Print the lower and upper bounds for outliers for each column
print(outlier_bounds)

# Create a boxplot for each numerical variable to visualize outliers
melted_data <- reshape2::melt(credit_card_data, 
                              measure.vars = names(credit_card_data)
                              [sapply(credit_card_data, is.numeric)])
ggplot(melted_data, aes(x = variable, y = value)) +
  geom_boxplot(outlier.color = "red", outlier.shape = 1) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Boxplot of Variables", x = "Variables", y = "Values")

# 6. Data cleaning steps
# Remove duplicates
df_cleaned <- credit_card_data[!duplicated(credit_card_data), ]

# Check for missing values
missing_values <- colSums(is.na(credit_card_data))

# Display columns with missing values
missing_columns <- missing_values[missing_values > 0]

if (length(missing_columns) > 0) {
  cat("Columns with missing values:\n")
  print(missing_columns)
} else {
  cat("No missing values found in the dataset.\n")
}

# Handle missing values (if any)
df_cleaned <- df_cleaned %>% 
  mutate(across(everything(), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Remove outliers from AMT_INCOME_TOTAL
q1 <- quantile(df_cleaned$AMT_INCOME_TOTAL, 0.25)
q3 <- quantile(df_cleaned$AMT_INCOME_TOTAL, 0.75)
iqr <- q3 - q1
lower_bound <- q1 - 1.5 * iqr
upper_bound <- q3 + 1.5 * iqr
df_cleaned <- df_cleaned %>% 
  filter(AMT_INCOME_TOTAL >= lower_bound & AMT_INCOME_TOTAL <= upper_bound)

# Print number of rows removed during cleaning
rows_removed <- nrow(credit_card_data) - nrow(df_cleaned)
cat("Number of rows removed during cleaning:", rows_removed, "\n")

# 7. Correlation Analysis
numerical_cols <- df_cleaned %>% select(where(is.numeric))
correlation_matrix <- cor(numerical_cols, use = "complete.obs")

correlation_melted <- melt(correlation_matrix)
ggplot(correlation_melted, aes(x = Var1, y = Var2, fill = value)) + 
  geom_tile() + 
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name="Correlation") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Correlation Matrix")

# 11. Min-Max Scaling for numerical columns
# Function to apply Min-Max scaling
min_max_scaling <- function(x) {
  return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}

# Select only numerical columns from the dataframe
numerical_cols <- df_cleaned %>% select(where(is.numeric)) # This selects only numerical columns

# Apply Min-Max scaling to the numerical columns
df_scaled <- df_cleaned

# Apply the scaling to numerical columns by their column names
df_scaled[colnames(numerical_cols)] <- lapply(numerical_cols, min_max_scaling)

# Display summary of the scaled dataset
cat("Summary of scaled data (Min-Max Scaling):\n")
summary(df_scaled)

# Optionally, visualize scaled data for a numerical variable (example: AMT_INCOME_TOTAL)
ggplot(df_scaled, aes(x = AMT_INCOME_TOTAL)) +
  geom_histogram(bins = 30, fill = "lightgreen", color = "black") +
  ggtitle("Histogram of Scaled AMT_INCOME_TOTAL (Min-Max Scaling)") +
  labs(title = "Distribution of Total Income Scaled", x = "Count", y = "Income") +
  theme_minimal()

#Additional visualizations to support EDA
# Distribution of income across the dataset
p1 <- ggplot(credit_card_data, aes(x = AMT_INCOME_TOTAL)) + 
  geom_histogram(bins = 100, fill = "lightblue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Income (0 - 2,000,000)", x = "Total Income", y = "Frequency") +
  xlim(0, 4000000)  # Set x-axis limits for the first plot

# Histogram for the range 2,000,000 - 6,000,000
p2 <- ggplot(credit_card_data, aes(x = AMT_INCOME_TOTAL)) + 
  geom_histogram(bins = 100, fill = "lightgreen", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Income (2,000,000 - 6,000,000)", x = "Total Income", y = "Frequency") +
  xlim(2000000, 6000000)  # Set x-axis limits for the second plot
      
# Arrange the plots in a grid
grid.arrange(p1, p2, ncol = 2)

#Distibution of age and work experience in years
p3 <- ggplot(credit_card_data, aes(x = EXPERIENCE)) +
  geom_histogram(bins = 30, fill = "lightcoral", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Experience(upto 50 years)", x = "Experience(years)", y = "Frequency") +
  xlim(0, 50)

p4 <- ggplot(credit_card_data, aes(x = AGE)) +
  geom_histogram(bins = 30, fill = "lightgreen", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Age (years)", x = "Age(Years)", y = "Frequency")

grid.arrange(p3, p4, ncol = 2)

# Gender distribution in the dataset
ggplot(credit_card_data, aes(x = CODE_GENDER)) +
  geom_bar(fill = "grey", color = "black") +
  labs(title = "Distribution of Gender", x = "Gender", y = "Count") +
  theme_minimal()

# Scatter Plot of AMT_INCOME_TOTAL vs. EXPERIENCE
ggplot(credit_card_data, aes(x = EXPERIENCE, y = AMT_INCOME_TOTAL)) +
  geom_point(alpha = 0.5, color = "blue") +
  labs(title = "Scatter Plot of Income vs. Experience", x = "Experience (years)", y = "Total Income") +
  theme_minimal() +
  xlim(0, 80)

# Density Plot of AMT_INCOME_TOTAL
ggplot(credit_card_data, aes(x = AMT_INCOME_TOTAL)) +
  geom_density(fill = "lightgreen", color = "black", alpha = 0.6) +
  labs(title = "Density Plot of Total Income", x = "Total Income", y = "Density") +
  theme_minimal() + 
  xlim(0, 2000000)

# Facet Grid for AMT_INCOME_TOTAL by Gender
ggplot(credit_card_data, aes(x = AMT_INCOME_TOTAL)) +
  geom_histogram(bins = 30, fill = "lightblue", color = "black", alpha = 0.7) +
  facet_wrap(~ CODE_GENDER) +
  labs(title = "Histogram of Total Income by Gender", x = "Total Income", y = "Frequency") +
  theme_minimal() +
  xlim(0, 2000000)

# Violin Plot of AMT_INCOME_TOTAL by Gender
ggplot(credit_card_data, aes(x = CODE_GENDER, y = AMT_INCOME_TOTAL)) +
  geom_violin(fill = "pink", color = "black") +
  labs(title = "Violin Plot of Total Income by Gender", x = "Gender", y = "Total Income") +
  theme_minimal() +
  ylim(0, 2000000)

# Scatter Plot for AMT_INCOME_TOTAL vs. DAYS_BIRTH
ggplot(credit_card_data, aes(x = AGE, y = AMT_INCOME_TOTAL)) +
  geom_point(alpha = 0.5, color = "blue") +
  labs(title = "Scatter Plot of Income vs. Age",
       x = "Age",
       y = "Total Income") +
  theme_minimal() +
  xlim(20, 50) +
  ylim(0, 4000000) +
  geom_smooth(method = "lm", color = "red", se = FALSE) # Adding a linear regression line

# Facet Grid for AMT_INCOME_TOTAL by CODE_GENDER
ggplot(credit_card_data, aes(x = AMT_INCOME_TOTAL)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Total Income by Gender",
       x = "Total Income",
       y = "Frequency") +
  facet_wrap(~ CODE_GENDER) +
  theme_minimal()

# Violin Plot of DAYS_EMPLOYED by NAME_HOUSING_TYPE
ggplot(credit_card_data, aes(x = NAME_HOUSING_TYPE, y = EXPERIENCE)) +
  geom_violin(fill = "lightgreen", color = "black") +
  labs(title = "Violin Plot of Employment Experience by Housing Type",
       x = "Housing Type",
       y = "Experience (years)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability

# Bar Plot for NAME_FAMILY_STATUS
ggplot(credit_card_data, aes(x = NAME_FAMILY_STATUS)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Count of Applicants by Family Status", 
       x = "Family Status", 
       y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability

# Scatter Plot of CNT_CHILDREN vs. CNT_FAM_MEMBERS
ggplot(credit_card_data, aes(x = CNT_CHILDREN, y = CNT_FAM_MEMBERS)) +
  geom_point(alpha = 0.5, color = "blue") +
  labs(title = "Scatter Plot of Children vs. Family Members", 
       x = "Number of Children", 
       y = "Number of Family Members") +
  theme_minimal() +
  geom_smooth(method = "lm", color = "red", se = FALSE)  # Adding a linear regression line

#Data Mining Techniques
#Question 01
# Create a synthetic 'CREDIT_APPROVAL' column based on assumptions
# For example, assume approval if income is above median and they have a stable job
credit_card_data$CREDIT_APPROVAL <- ifelse(credit_card_data$AMT_INCOME_TOTAL > median(credit_card_data$AMT_INCOME_TOTAL) & 
                                             credit_card_data$EXPERIENCE > 3, 1, 0)

# Data Preprocessing: Convert categorical variables into factors
credit_card_data$CODE_GENDER <- as.factor(credit_card_data$CODE_GENDER)
credit_card_data$FLAG_OWN_CAR <- as.factor(credit_card_data$FLAG_OWN_CAR)
credit_card_data$FLAG_OWN_REALTY <- as.factor(credit_card_data$FLAG_OWN_REALTY)
credit_card_data$NAME_INCOME_TYPE <- as.factor(credit_card_data$NAME_INCOME_TYPE)
credit_card_data$NAME_EDUCATION_TYPE <- as.factor(credit_card_data$NAME_EDUCATION_TYPE)
credit_card_data$NAME_FAMILY_STATUS <- as.factor(credit_card_data$NAME_FAMILY_STATUS)
credit_card_data$NAME_HOUSING_TYPE <- as.factor(credit_card_data$NAME_HOUSING_TYPE)
credit_card_data$OCCUPATION_TYPE <- as.factor(credit_card_data$OCCUPATION_TYPE)

# Split the data into training and test sets
set.seed(123)
train_index <- createDataPartition(credit_card_data$CREDIT_APPROVAL, p = 0.7, list = FALSE)
training_set <- credit_card_data[train_index, ]
test_set <- credit_card_data[-train_index, ]

# Logistic Regression Model
#logistic_model <- glm(CREDIT_APPROVAL ~ CODE_GENDER + AGE + AMT_INCOME_TOTAL + NAME_EDUCATION_TYPE + 
#                        NAME_FAMILY_STATUS + FLAG_OWN_CAR + FLAG_OWN_REALTY + CNT_CHILDREN + 
#                        DAYS_EMPLOYED, family = binomial(link = "logit"), data = training_set)

# Logistic Regression Model
logistic_model <- glm(CREDIT_APPROVAL ~ CODE_GENDER + AMT_INCOME_TOTAL +
                        NAME_FAMILY_STATUS + CNT_CHILDREN, 
                        family = binomial(link = "logit"), data = training_set)

# Predictions
predictions_logistic <- predict(logistic_model, test_set, type = "response")
predictions_logistic <- ifelse(predictions_logistic > 0.5, 1, 0)

# Confusion Matrix and Accuracy
confusion_matrix <- confusionMatrix(as.factor(predictions_logistic), as.factor(test_set$CREDIT_APPROVAL))
print(confusion_matrix)


#Question 02
#Decision Tree  - Technique 2
# Build Decision Tree Model
decision_tree_model <- rpart(CREDIT_APPROVAL ~ CODE_GENDER + AMT_INCOME_TOTAL + FLAG_OWN_REALTY +
                                 FLAG_OWN_CAR,
                             data = training_set, method = "class")

# Plot the Decision Tree for visual interpretation
plot(decision_tree_model)
text(decision_tree_model)

# Predict on test set
predictions_tree <- predict(decision_tree_model, test_set, type = "class")

# Evaluate model performance with Confusion Matrix and Accuracy
confusion_matrix <- confusionMatrix(as.factor(predictions_tree), as.factor(test_set$CREDIT_APPROVAL))
confusion_matrix


#Random Forest  - Technique 3
# Build Random Forest model
random_forest_model <- randomForest(CREDIT_APPROVAL ~  CODE_GENDER + AMT_INCOME_TOTAL + FLAG_OWN_REALTY +
                                      FLAG_OWN_CAR,
                                      
                                    data = training_set, ntree = 100)

# View the importance of variables
importance(random_forest_model)
varImpPlot(random_forest_model)

# Predict on the test set
predictions_rf <- predict(random_forest_model, newdata = test_set)

# Adjust levels in predictions to match the actual test set levels
predictions_rf <- factor(predictions_rf, levels = levels(test_set$CREDIT_APPROVAL))


# Evaluate model performance with Confusion Matrix and Accuracy
confusion_matrix <- confusionMatrix(as.factor(predictions_rf), as.factor(test_set$CREDIT_APPROVAL))
confusion_matrix



#Question 3
# Train an SVM model - Technique 4
model_svm <- svm(FLAG_OWN_REALTY ~ CODE_GENDER + AMT_INCOME_TOTAL + CNT_CHILDREN + 
                   NAME_INCOME_TYPE + NAME_EDUCATION_TYPE + NAME_FAMILY_STATUS +
                   DAYS_BIRTH + DAYS_EMPLOYED + CNT_FAM_MEMBERS, 
                 data = training_set, kernel = "linear")
# Summary of the SVM model
summary(model_svm)

# Predict on test data
svm_predictions <- predict(model_svm, newdata = test_set)

# Model evaluation
confusionMatrix(svm_predictions, test_set$FLAG_OWN_REALTY)


# Clustering Technique
# Select relevant numeric columns for clustering
data_clustering <- dplyr::select(df_scaled, AMT_INCOME_TOTAL, DAYS_BIRTH, DAYS_EMPLOYED, CNT_CHILDREN, CNT_FAM_MEMBERS)

# Normalize the data to standardize the scale of each feature
data_normalized <- as.data.frame(scale(data_clustering))

# Determine the optimal number of clusters using the Elbow method
set.seed(123)
wss <- (nrow(data_normalized) - 1) * sum(apply(data_normalized, 2, var))
for (i in 2:15) {
  wss[i] <- sum(kmeans(data_normalized, centers = i, nstart = 20)$tot.withinss)
}

# Plot the Elbow curve
plot(1:15, wss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares", main = "Elbow Method for K-Means")

# Apply K-Means with an appropriate number of clusters (e.g., k = 4 based on Elbow curve)
set.seed(123)
kmeans_model <- kmeans(data_normalized, centers = 4, nstart = 20)

# Add the cluster labels to the original data
df_scaled$Cluster <- as.factor(kmeans_model$cluster)

# Visualize the clusters based on two dimensions (e.g., Income vs. Employment)
ggplot(df_scaled, aes(x = AMT_INCOME_TOTAL, y = DAYS_EMPLOYED, color = Cluster)) +
  geom_point() +
  labs(title = "K-Means Clustering of Loan Applicants", x = "Total Income", y = "Days Employed") +
  theme_minimal()

# Analyze cluster centers to interpret characteristics of each cluster
print(kmeans_model$centers)

#Q3 Gradient Boosting for Employment Type Patterns

# Build Gradient Boosting Model
gbm_model_employment <- gbm(CREDIT_APPROVAL ~ OCCUPATION_TYPE + DAYS_EMPLOYED,
                            distribution = "bernoulli", data = training_set, n.trees = 100)

# Predict on test set
predictions_employment <- predict(gbm_model_employment, test_set, n.trees = 100, type = "response")

# Convert probabilities to binary outcomes
predictions_employment_binary <- ifelse(predictions_employment > 0.5, 1, 0)

# Confusion Matrix
confusion_matrix_employment <- table(predictions_employment_binary, test_set$CREDIT_APPROVAL)
confusion_matrix_employment


# enhansing model's perfomance
# Adding interaction terms
training_set$income_marital_interaction <- training_set$AMT_INCOME_TOTAL * as.numeric(training_set$NAME_FAMILY_STATUS)
test_set$income_marital_interaction <- test_set$AMT_INCOME_TOTAL * as.numeric(test_set$NAME_FAMILY_STATUS)

# Rebuild the model with new features
gbm_model_enhanced <- gbm(CREDIT_APPROVAL ~ OCCUPATION_TYPE + DAYS_EMPLOYED + income_marital_interaction,
                          distribution = "bernoulli", data = training_set, n.trees = 100)
threshold <- 0.5  
# Evaluate the enhanced model
predictions_enhanced <- predict(gbm_model_enhanced, test_set, n.trees = 100, type = "response")
predictions_enhanced_binary <- ifelse(predictions_enhanced > threshold, 1, 0)
confusion_matrix_enhanced <- table(predictions_enhanced_binary, test_set$CREDIT_APPROVAL)
print(confusion_matrix_enhanced)

