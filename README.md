# Upvotes-prediction-for-an-online-question-and-answer-platform

An online question and answer platform wants to identify the best question authors on the platform. This identification will bring more insight into increasing the user engagement. Given the tag of the question, number of views received, number of answers, username and reputation of the question author, the problem requires us to predict the upvote count that the question will receive.

## Evaluation Metric
The evaluation metric for this competition is RMSE (root mean squared error).

## Data Dictionary
| Variable   | Definition                                      |
| ---------- | ----------------------------------------------- |
| ID         | Question ID                                     |
| Tag        | Anonymised tags  representing question category |
| Reputation | Reputation  score of question author            |
| Answers    | Number  of times question has been answered     |
| Username   | Anonymised  user id of question author          |
| Views      | Number  of times question has been viewed       |
| Upvotes    | (Target)  Number of upvotes for the question    |


## References
https://datahack.analyticsvidhya.com/contest/enigma-codefest-machine-learning-1/#ProblemStatement

https://seaborn.pydata.org/examples/many_pairwise_correlations.html

https://stackoverflow.com/questions/36631163/what-are-the-pros-and-cons-between-get-dummies-pandas-and-onehotencoder-sciki

https://towardsdatascience.com/decision-tree-and-random-forest-explained-8d20ddabc9dd

https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e

https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/

https://stackoverflow.com/questions/36631163/what-are-the-pros-and-cons-between-get-dummies-pandas-and-onehotencoder-sciki
