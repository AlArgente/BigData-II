# BigData-II
"Big Data II" subject of the master's degree in data science at the University of Granada.

In this practice I had to use Spark to solve some problems of Big Data. The dataset we had to work with is the Higg's dataset, but using 1 million of the data to don't overload the servers from the university.

This dataset is unbalanced, containing 90% of the data for the mayority class, and the remaining 10% of the minority class.

To solve this problem we have to use some preprocessing algorithms to oversample the minority class. I used both Random Oversampling (ROS) to create more data for the minority class, and Random Undersampling (RUS) to decrease the amount of data from the mayority class. Also I used HMe_BD to remove noise from the data by using Random Forest, and I used FCNN_MR for feature selection.

The base methods I used were Decision Tree and Random Forest. With the idea of using more tree ensembles, I used PCARD. I could use kNN as the final method, but I chose Naïve Bayes, because we saw knn in the practical classes. 

The results are in the ResultFiles folder, while the code used is in the ScriptUsed folder.
 
