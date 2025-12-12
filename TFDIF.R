library(vroom)
library(tidyverse)
library(tidymodels)
library(jsonlite)
library(tidytext)
library(textrecipes)

train_data <- read_file("./GitHub/WhatsCooking/train.json") %>%
  fromJSON()
test_data <- read_file("./GitHub/WhatsCooking/test.json") %>%
  fromJSON()

rec <- recipe(cuisine ~ ingredients, data = train_data) %>%
  step_mutate(ingredients_chr = as.character(ingredients),
    ingredient_list = str_split(ingredients_chr, ",\\s*"),
    ingredient_count = sapply(ingredient_list, length)) %>% 
  step_mutate(ingredients = tokenlist(ingredients)) %>%
  step_tokenfilter(ingredients, max_tokens=1000) %>%
  step_tfidf(ingredients) %>% 
  step_rm(ingredients_chr, ingredient_list)
  

prepped_recipe <- prep(rec)
baked_data <- bake(prepped_recipe, new_data=train_data)
ncol(baked_data)

forest_mod <- rand_forest(mtry = tune(),
                          min_n=tune(),
                          trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

forest_workflow <- workflow() %>% 
  add_recipe(rec) %>%
  add_model(forest_mod) 

tuning_grid <- grid_regular(mtry(range(c(1,9))),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- forest_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

final_workflow <- forest_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

Cook_predictions <- predict(final_workflow,
                              new_data=test_data,
                              type="class") # "class" or "prob"

kaggle_sub <- bind_cols(test_data$id, Cook_predictions) %>% 
  rename(cuisine = .pred_class) %>% 
  rename(id = ...1)

vroom_write(kaggle_sub,
            file="GitHub/WhatsCooking/TFDIF1000.csv", delim=",")
