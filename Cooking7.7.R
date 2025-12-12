#========================================================
# Libraries
#========================================================
library(tidyverse)
library(tidymodels)
library(textrecipes)
library(vroom)
library(jsonlite)
library(stopwords)
library(glmnet) # Make sure glmnet is installed for the engine

#========================================================
# 1. Load JSON data
#========================================================
# Use the correct way to read the JSON structure for typical recipe data
trainSet <- read_file("./GitHub/WhatsCooking/train.json") |> fromJSON(flatten = TRUE)
testSet  <- read_file("./GitHub/WhatsCooking/test.json")  |> fromJSON(flatten = TRUE)

#========================================================
# 2. Convert ingredient list → single text column
#========================================================
train_docs <- trainSet %>%
  mutate(
    text   = sapply(ingredients, paste, collapse = " "),
    # Assuming 'cuisine' is the actual outcome variable in your JSON
    # We convert it to a factor for classification
    target = as.factor(cuisine) 
  ) %>%
  select(id, text, target)

test_docs <- testSet %>%
  mutate(text = sapply(ingredients, paste, collapse = " ")) %>%
  select(id, text)

# Check the target variable summaries (now showing factor levels/cuisines)
print("Summary of target variable (cuisines):")
print(summary(train_docs$target))
print(paste("Are there any NAs in target:", any(is.na(train_docs$target))))

#========================================================
# 3. Recipe: tokenize → stopwords → tfidf → normalize
#========================================================
ingredient_recipe <- recipe(target ~ text, data = train_docs) %>%
  step_tokenize(text) %>%
  step_stopwords(text) %>%
  # Filter rare words before TFIDF to prevent model matrix issues with classification
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text) %>%
  step_normalize(all_numeric_predictors())

#========================================================
# 4. Model: Multinomial Lasso regression (glmnet)
#========================================================
# Use multinom_reg() for classification tasks with factor outcomes
lin_model <- multinom_reg(
  penalty = 0.001, 
  mixture = 1       # LASSO
) %>%
  set_engine("glmnet")
# set_mode("classification") is handled automatically by multinom_reg()


#========================================================
# 5. Workflow
#========================================================
wf <- workflow() %>%
  add_recipe(ingredient_recipe) %>%
  add_model(lin_model)

#========================================================
# 6. Fit model
#========================================================
# The fit should now work as the target is correctly formatted
fit_lin <- fit(wf, data = train_docs)

#========================================================
# 7. Predict on test
#========================================================
# Changed prediction type to predict the class label (cuisine)
test_preds <- predict(fit_lin, new_data = test_docs, type = "class") %>%
  bind_cols(test_docs) %>%
  rename(prediction = .pred_class) %>% # Rename to match output format
  select(id, prediction)

#========================================================
# 8. Write to CSV
#========================================================
vroom_write(
  test_preds,
  file = "ingredient_linear_predictions.csv",
  delim = ","
)
df <- vroom("ingredient_linear_predictions.csv") %>%
  rename(cuisine = prediction)

vroom_write(df, "ingredient_linear_predictions.csv", delim = ",")

cat("DONE! CSV written: ingredient_linear_predictions.csv\n")
