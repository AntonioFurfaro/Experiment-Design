# Decision Tree
Since the paper did not provide any specific application of decision trees, I decided to use the rpart function in R for my implementation. To optimize the hyperparameters and maximize accuracy, I employed Bayesian optimization. Specifically, I tuned the following parameters:

*maxdepth*: the maximum depth of the tree,
*minsplit*: the minimum number of observations required in a node to attempt a split, 
*cp*: complexity parameter, any split that does not reduce the overall lack of fit by a factor of cp is avoided.

Using this approach, I achieved results similar to those reported in the paper, though the accuracy is slightly lower.
