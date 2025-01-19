rm(list=ls());graphics.off();cat("\014")

setwd("C:/Users/anton/OneDrive - Università degli Studi di Milano-Bicocca/Desktop/UNI/MAGISTRALE/EXPERIMENT DESIGN FOR DATA SCIENCE/project")


#############################################
#### loading of all the useful libraries ####
#############################################

library(dplyr)
library(mlrMBO)
library(randomForest)
library(caret)


##############################
#### datasets preparation ####
##############################


# loading the data sets in a list

file_list <- list.files(pattern = "*.csv", full.names = TRUE)
file_info <- file.info(file_list)
file_list <- file_list[order(file_info$mtime)]
file_list <- basename(file_list)
file_list <- grep("8", file_list, value = TRUE)


data_list <- lapply(file_list, function(file) read.csv(file, header = FALSE))

file_list <- gsub("_1000.csv$", "", file_list)
names(data_list) <- file_list


# adding the group variable 

for(i in 1:length(data_list)) {
  data_list[[i]] <- data_list[[i]] %>% 
    mutate(group =  as.factor(rep(1:(nrow(data_list[[i]]) / 1000), each = 1000))) %>%
    select(group, everything())
}


# showing the dimension of the data sets

for(i in 1:length(data_list)) {
  print(dim(data_list[[i]]))
}


# creating training e test sets for each data set

trainings <- list()
tests <- list()

# I randomly selected 80% of the observation in each group for the training set
# and the remaining 20% for the test set

for(i in 1:length(data_list)) {
  set.seed(123)
  
  training_data <- list()
  test_data <- list()
  
  # considering a group at a time
  
  for(g in unique(data_list[[i]]$group)) {
    group_data <- filter(data_list[[i]], group == g)
    
    # sampling traing and test indexes
    training_indices <- sample(1:nrow(group_data), size = 0.8 * nrow(group_data))
    test_indices <- setdiff(1:nrow(group_data), training_indices)
    
    # selection training and test data for the group
    training_data[[g]] <- group_data[training_indices, ]
    test_data[[g]] <- group_data[test_indices, ]
  }
  
  # combining the training and test data of each group for the same data set
  trainings[[i]] <- bind_rows(training_data)
  tests[[i]] <- bind_rows(test_data)
}


# printing the dimension of training and test of each data set

for(i in 1:length(data_list)) {
  print(dim(trainings[[i]]))
  print(dim(tests[[i]]))
}

names(trainings) <- file_list
names(tests) <- file_list


######################
#### rf modelling ####
######################


# creating a list in which the results will be saved
results_rf <- list()
itermax <- 25

# for each data set and and for each iteration, I estimated a decision tree using 
# bayesian optimization for tuning the hyperparameters, and a 5 fold cross
# validation to estimate the accuracy that is used as a measure to optimize thise
# hyperparameters. Then, I saved the results, i.e. the predicted group on the
# test sets

for(i in 1:length(data_list)) {
  
  result <- data.frame(ground_truth = tests[[i]]$group)  
  
  for(j in 1:itermax) {
    
    set.seed(34521869)
    
    task <- makeClassifTask(data = trainings[[i]][, 1:(j+1)], target = "group")
    
    par.set.RF <- makeParamSet(
      makeIntegerParam("ntree", lower = 10, upper = 500), #1000
      makeIntegerParam("nodesize", lower = 5, upper = 100))
    ctrl.RF <- makeMBOControl()
    ctrl.RF <- setMBOControlTermination(ctrl.RF, iters = 12)
    ctrl.RF <- setMBOControlInfill(ctrl.RF, opt.focussearch.points = 15)
    tune.ctrl.RF <- makeTuneControlMBO(mbo.control = ctrl.RF)
    learner <- makeLearner("classif.randomForest")
    
    print(paste(names(data_list)[i], j))
    run.RF <- tuneParams(learner, task, 
                         resampling = makeResampleDesc("CV", iters = 5),
                         measures = acc, par.set = par.set.RF,
                         control = tune.ctrl.RF, show.info = T)
    

    mod.RF <- randomForest(group ~ ., data = trainings[[i]][, 1:(j+1)],
                           ntree = run.RF$x$ntree,
                           nodesize = run.RF$x$nodesize)
    
    preds.RF <- predict(mod.RF, tests[[i]])
    result <- cbind(result, setNames(data.frame(preds.RF), paste0("iter", j)))

  }
  
  results_rf[[names(data_list)[i]]] <- result
}


output_dir <- "C:/Users/anton/OneDrive - Università degli Studi di Milano-Bicocca/Desktop/UNI/MAGISTRALE/EXPERIMENT DESIGN FOR DATA SCIENCE/project/results"  # Sostituisci con il percorso desiderato

for (i in seq_along(results_rf)) {
  file_name <- paste0("result_rf_", names(data_list)[i], ".csv")
  write.csv(results_rf[[i]], file = file_name, row.names = FALSE)
}


# plotting the accuracy evolution for each data set. The results are similar to 
# ones in the paper, but the accuracies seem a bit lower

par(mfrow=c(2,2))
for(i in 1:length(results_rf)){
  accuracy <- apply(results[[i]][,-1], 2,
                    function(x) mean(results_rf[[i]]$ground_truth==x))
  plot(accuracy, type = "l", ylab = "accuracy",
       xlab = "iteration", main = names(results_rf)[i], 
       ylim = c(0, max(accuracy)+0.01))
}
par(mfrow=c(1,1))
