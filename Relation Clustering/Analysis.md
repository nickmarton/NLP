# Nicholas Marton
#Word2Vec Relation Clustering
___
### Analysis:
---
During the testing of my implementation, I was unable to achieve an F-score of over 0.45. After analyzing the model, I have uncovered some ideas for why the error is so high and some possible improvements. Also, I am aware that merging the ground truth vocabulary with text8 vocabulary may constitute data snooping, but I didn't have muc htime so I just ripped it.
##### Error:
The error in the model is extremely high; on average I obtained around a 0.39 F-score. However, the recall was, on average, significantly higher than the precision. I believe a couple key things contribute to this outcome:
* The data set was very small relative to the more robust and computationally intensive word2vec models. With more data, pecision along with recall would increase.
* The word2vec phrase training considered only bigrams in its phrase training period. The vast majority of the data in *triggerdata* consists of phrases with more than two words.
* Pre-trained models did not include the majority of the data within *triggerdata* in their vocabulary.

##### Possible Improvements:
The errors in the model made it extremely hard to generalize well, but improvements on these errors can be made.
* The use of a much bigger pre-trained model would definitely improve some of the error, however this is intractable on my machines. Then, perhaps through the use of some distributed paradigms or computing (on AWS perhaps), using a bigger pre-trained model would become tractible and would produce much better results.
* Another improvement that can be made is to allow word2vec to consider higher-order N-grams in its phrase training period. If this were possible, a lot more of the words in *triggerdata* would be present in the vocabulary and the model would more than likely be able to generalize a lot better; this would however be more computationally intensive, but the resulting phrases could be saved to stable memory.
