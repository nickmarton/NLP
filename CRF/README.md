### Conditional Random Field for Entity Detection
---


#### Dependencies and External Files

1. **nlkt** for POS tagging
2. **scikit-learn** for computing precision, recall, and F-Measure
3. **crfsuite** and **pycrfsuite** for computing the CRF
4. I used the files in [this][cluster] github repository to computer pmi clusters.
---
#### Implementation Description
##### Formal Code Description:
The logical order of my code is as follows:
* Set constants the original training and test data input files, for N-gram number, window size, and the index at which to prune clusters from the tree (i.e., where to cut their bitstring).
* Try loading training and test data (with POS tags included), generating POS tags and writing these files in the case where the try block fails.
* Construct the training matrix *X_train* and label vector *y_train*; every word of every sentence is processed one at a time, but the rows of the training matrix are the features of every word in a full sentence.
* Create a trainer object from pycrfsuite and load the training examples into it.
* Set regularization and early stopping parameters of trainer, then train and save model.
* Evaluate model's performance on test data.
##### Base Features:
After computing part of speech tags with python's nltk package, I chose the following as my base features to include:

    1. bias
    2. word.lower(): the lowercase version of the word
    3. word[-3:]: the last three letters of the word
    4. word[-2:]: the last three letters of the word
    5. word.isupper(): a boolean for whether or not the word appears capitalized
    6. word.isdigit(): a boolean for whether or not the word only consists of digits
    7. postag: a string representing the word's part of speech
    8. postag[:2]: the first two characters of postag

##### N-gram Features:
AS my first extension to the base features, I chose to implement N-grams and extract features for each word in a given word's N-gram. The variable *ngram_number* in the code decides the size of the N-gram. The features of these words that I included were precisely the same features included in the base features except for bias, that is, included were base features **[2, 3, 4, 5, 6, 7, 8]**.

##### Window Features:
As one of the extensions to the base features, I chose to look at the first *n* words that came before and after the word in question where *n* is represented by the variable *window_number* in the code. The features of these words that I included were precisely the same features included in the base features except for bias, that is, included were base features **[2, 3, 4, 5, 6, 7, 8]**.

##### pmi Cluster Features:
As my final extension to the base features, I chose to generate the pmi Word Clusters and for every word in the cluster of the word in question, I included the subset of features from the base features **[2, 3, 4, 5, 6, 7, 8]**.
Additionally, in the table below, the rows where an asterisk * is suffixed to a cluster cut denote the *restricted* model where only the words (in a certain word's cluster) that share the same tag as the current word being analyzed have their features contributed (i.e, if a word in a certain word's cluster does not have the same tag as that word, it is ignored).

#### Performance
---
Note: an N-gram number of 1, window size of 0 and cluster cut of 0 denote the experiment in which these features are respectively not included, i.e., they are *turned off* at these values.

| | Base Features  | N-Gram number | Window Size | Cluster cut | Avg. Precision | Avg. Recall. | Avg. F-Measure
|:---:|:--------:|:---:|:---:|:---:|:-----:|:-----:|:-----:|
|  1. | Included |  1  |  0  |  0  |  81%  |  78%  |  79%  |
|  2. | Included |  2  |  0  |  0  |  81%  |  79%  |  80%  |
|  3. | Included |  3  |  0  |  0  |  83%  |  80%  |  81%  |
|  4. | Included |  4  |  0  |  0  |  81%  |  79%  |  80%  |
|  5. | Included |  5  |  0  |  0  |  82%  |  78%  |  80%  |
|  6. | Included |  1  |  1  |  0  |  86%  |  82%  |  84%  |
|  7. | Included |  1  |  2  |  0  |  87%  |  83%  |  84%  |
|  8. | Included |  1  |  3  |  0  |  84%  |  83%  |  84%  |
|  9. | Included |  1  |  0  | 25* |  92%  |  89%  |  90%  |
| 10. | Included |  1  |  0  | 40  |  82%  |  79%  |  80%  |
| 11. | Included |  1  |  0  | 40* |  92%  |  86%  |  89%  |
| 12. | Included |  1  |  0  | 55  |  81%  |  78%  |  79%  |
| 13. | Included |  1  |  0  | 55* |  91%  |  85%  |  87%  |
| 14. | Included |  1  |  0  | 70  |  81%  |  78%  |  80%  |
| 15. | Included |  1  |  0  | 70* |  85%  |  82%  |  83%  |
| 16. | Included |  1  |  2  | 40* |  94%  |  90%  |  91%  |
| 17. | Included |  1  |  2  | 25* |  93%  |  92%  |  93%  |
| 18. | Included |  3  |  2  | 25* |  93%  |  92%  |  92%  |

From the table, row 17 where the N-gram number is 0, the window size is 2 and the restricted cluster cut of 25 performed best, achieving an F-measure score of 93%. I leave this as the default model in the source code.

[cluster]: https://github.com/mheilman/tan-clustering
