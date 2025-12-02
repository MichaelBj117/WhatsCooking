library(tidyverse)
library(tidymodels)
library(jsonlite)
library(tidytext)

train_data <- read_file("./GitHub/WhatsCooking/train.json") %>%
  fromJSON()
test_data <- read_file("./GitHub/WhatsCooking/test.json") %>%
  fromJSON()

train_data %>%
  unnest(ingredients)
test_data %>%
  unnest(ingredients)


train_data_clean <- train_data_clean %>%
  mutate(
    ingredients_text = sapply(ingredients, paste, collapse = " ")
  )

test_data_clean <- train_data_clean %>%
  mutate(
    ingredients_text = sapply(ingredients, paste, collapse = " ")
  )

unique(train_data$ingredients)

CookRecipe <- recipe(cuisine ~ ., data=train_data_clean) %>% 
  step_mutate(cheese_amount = str_count(ingredients_text, "cheese")) %>% 
  step_mutate(contains_cilantro = str_detect(ingredients_text, "cilantro")) %>% 
  step_mutate(num_of_ing = map_int(ingredients, length)) %>% 
  step_rm(ingredients_text, id)

prepped_recipe <- prep(CookRecipe)
baked_data_train <- bake(prepped_recipe, new_data=train_data_clean)
