Feature selection (FS) is an important step in the process of building machine learning-based models. The goal of the feature selection step is to
find a small subset of features that will provide good prediction results by removing noisy, irrelevant, or repetitive features. Commonly used Wrapper
methods use a machine learning model as a black box and its performance as the goal function for evaluating different features’ subsets and selecting
the best one. To avoid examining all possible subsets (an NP-hard problem), search algorithms are used to find the subsets to be examined, 
in a heuristic way. As exhaustive search methods are computationally complicated, most methods use simple and greedy search methods that yield only 
locally optimal results and are not sensitive to possible features interactions, which means that a feature may be chosen at the expense of two others 
that are more informative together. We analyze the problem of searching the features subset space with reference to two dimensions, namely, memory of past 
selected features subset and future selected features. 

In this work, we propose a new wrapper feature selection method based on the deep artificial curiosity framework, which implements intrinsic reward 
reinforcement learning, with long short-term memory unit (LSTM). This novel algorithm integrates these two elements of memory and future step.
