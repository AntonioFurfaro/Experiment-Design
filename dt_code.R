rm(list=ls());graphics.off();cat("\014")

setwd("C:/Users/anton/OneDrive - Università degli Studi di Milano-Bicocca/Desktop/UNI/MAGISTRALE/EXPERIMENT DESIGN FOR DATA SCIENCE/project")


#############################################
#### loading of all the useful libraries ####
#############################################

library(dplyr)
library(mlrMBO)
library(rpart)
library(rpart.plot)
library(wesanderson)
library(ggplot2)
library(caret)


##############################
#### datasets preparation ####
##############################


# loading the data sets in a list

file_list <- list.files(pattern = "*.csv", full.names = TRUE)
file_info <- file.info(file_list)
file_list <- file_list[order(file_info$mtime)]
file_list <- basename(file_list)

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
#### dt modelling ####
######################


# creating a list in which the results will be saved
results <- list()
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
    
    par.set.DT <- makeParamSet(
      makeIntegerParam("maxdepth", lower = 3, upper = 30),
      makeIntegerParam("minsplit", lower = 1, upper = 50),
      makeNumericParam("cp", lower = -4, upper = -1, trafo = function(x) 10^x))
    ctrl.DT <- makeMBOControl()
    ctrl.DT <- setMBOControlTermination(ctrl.DT, iters = 10)
    tune.ctrl.DT <- makeTuneControlMBO(mbo.control = ctrl.DT)
    
    print(paste(names(data_list)[i], j))
    run.DT <- tuneParams(makeLearner("classif.rpart"), task, 
                         resampling = makeResampleDesc("CV", iters = 5),
                         measures = acc, par.set = par.set.DT,
                         control = tune.ctrl.DT, show.info = T)

    mod.DT <- rpart(group ~ ., data = trainings[[i]][, 1:(j+1)],
                    maxdepth = run.DT$x$maxdepth,
                    minsplit = run.DT$x$minsplit,
                    cp = run.DT$x$cp)

    preds.DT <- predict(mod.DT, tests[[i]], type = "class")
    
    result <- cbind(result, setNames(data.frame(preds.DT), paste0("iter", j)))
    
  }
  
  results[[names(data_list)[i]]] <- result
}


output_dir <- "C:/Users/anton/OneDrive - Università degli Studi di Milano-Bicocca/Desktop/UNI/MAGISTRALE/EXPERIMENT DESIGN FOR DATA SCIENCE/project/results"  # Sostituisci con il percorso desiderato

for (i in seq_along(results)) {
  file_name <- paste0("result_", names(data_list)[i], ".csv")
  write.csv(results[[i]], file = file_name, row.names = FALSE)
}


# plotting the accuracy evolution for each data set. The results are similar to 
# ones in the paper, but the accuracies seem a bit lower

par(mfrow=c(2,2))
for(i in 1:length(results)){
  accuracy <- apply(results[[i]][,-1], 2,
                    function(x) mean(results[[i]]$ground_truth==x))
  plot(accuracy, type = "l", ylab = "accuracy",
       xlab = "iteration", main = names(results)[i], 
       ylim = c(0, max(accuracy)+0.01))
}
par(mfrow=c(1,1))
