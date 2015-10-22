"""
Nicholas Marton
Simple Bigram HMM Part-Of-Speech Tagger.
"""

def get_data(filename, path="./wsj_pos/"):
    """Parse datum from training.txt and test.txt."""
    data = []

    #open filename located at path 
    #files are assumed to be within wsj_pos directory(located in cwd)
    with open(path + filename) as f:
        #for every line in training.txt
        for line in f:

            #if line is not an empty newline, strip newline and add to data
            if line != '\r\n':
                data.append(line.strip())

    return data

def get_starting_probability(data):
    """
    Return the starting probability of the words and tags included in the
    provided dataset data.
    """

    start_words = {}
    start_tags = {}

    all_words = []
    all_tags = []

    for datum in data:

        #split each sentence into word/tag pairs and get the first of pair
        word_tag_pairs = datum.split(" ")
        word, tag = word_tag_pairs[0].split("/")
        
        #save a copy of every word and tag in all_words & all_tags respectively
        for pair in word_tag_pairs:
            w,t = pair.split("/")    
            all_words.append(w)
            all_tags.append(t)

        #place into dictionary; counting for now
        if word in start_words:
            start_words[word] += 1.0
        else:
            start_words[word] = 1.0

        if tag in start_tags:
            start_tags[tag] += 1.0
        else:
            start_tags[tag] = 1.0

    #convert count into probability
    for word, count in start_words.iteritems():
        start_words[word] = count / float(len(data))

    for tag, count in start_tags.iteritems():
        start_tags[tag] = count / float(len(data))

    #count how many words(tags) appeared and how many did not
    word_count = len(start_words)
    tag_count = len(start_tags)
    missing_word_count = 0
    missing_tag_count = 0

    for word in all_words:
        if word not in start_words:
            missing_word_count += 1

    for tag in all_tags:
        if tag not in start_tags:
            missing_tag_count += 1

    #calculate a probability differential to take away from words that did
    #appear to balance the extremely small added probability of words which
    #did not
    word_df = (float(missing_word_count) * 1.0e-250) / float(word_count)
    tag_df = (float(missing_tag_count) * 1.0e-250) / float(tag_count)
    
    #modify probabilities of occuring words so sum of all probabilities
    #remains at 1 and all words have an associated probability
    for word, probability in start_words.iteritems():
        start_words[word] = probability - (word_df * probability)

    for tag, probability in start_tags.iteritems():
        start_tags[tag] = probability - (tag_df * probability)
    
    #for all words and tags that didn't appear at start of some sentence, 
    #set probabilities to exceptionally low nonzero value
    for word in all_words:
        if word not in start_words:
            start_words[word] = 1.0e-250

    for tag in all_tags:
        if tag not in start_tags:
            start_tags[tag] = 1.0e-250

    return start_words, start_tags

def get_words_and_tags(data):
    """Return set of tags used in data."""
    tags = []
    words = []

    for datum in data:

        #split each sentence into word/tag pairs
        word_tag_pairs = datum.split(" ")

        for pair in word_tag_pairs:

            #split word and tag
            word, tag = pair.split("/")

            #add words/tags to sets
            if word not in words:
                words.append(word)
            if tag not in tags:
                tags.append(tag)

    return words, tags

def init_tables(data):
    """Make emission and transition probability tables."""
    #get set of words and tags used in training data
    words, tags = get_words_and_tags(data)

    #create an empty emission probability table as nested dict
    emission = {tag: {} for tag in tags}
    #create an empty transition probability table as nested dict
    transition = {tag: {} for tag in tags}

    return emission, transition

def fill_emission(emission, data):
    """Fill the emission probability table."""
    #set up dict to count how many times each tag appears in total
    all_words, all_tags = get_words_and_tags(data)
    tag_counts = {tag: 0.0 for tag in all_tags}

    for datum in data:
        #split datum into word/tag pairs
        word_tag_pairs = datum.split(" ")

        for pair in word_tag_pairs:
            #split word/tag pair
            word, tag = pair.split("/")

            #increment tag in total tag counts
            tag_counts[tag] += 1.0

            #get the inner table for each tag
            word_table = emission[tag]

            if word in word_table:
                #if tag is already in word's table, increment its count
                word_table[word] += 1.0
            else:
                #otherwise, initialize tag in table with count of 1
                word_table[word] = 1.0


    #for each tag and its set of counts of words
    for tag, word_set in emission.iteritems():

        #count how many words appeared in emission and how many did not
        word_count = len(word_set)
        missing_word_count = 0

        for word in all_words:
            if word not in word_set:
                missing_word_count += 1

        #calculate a probability differential to take away from words that did
        #appear to balance the extremely small added probability of words which
        #did not
        word_df = (float(missing_word_count) * 1.0e-250) / float(word_count)

        #for each word assigned to the tag
        for word in word_set:
            #get the modified emission probabiity for word, taking into account
            #the words that did not appear for a tag
            count = word_set[word]
            probability = count/tag_counts[tag]
            word_set[word] = probability - (word_df * probability)

        #for all words that did not preceed a given tag
        for word in all_words:
            if word not in word_set:
                #set probability to exceedingly small nonzero rational
                word_set[word] = 1.0e-250

def fill_transition(transition, data):
    """Fill the emission probability table."""
    all_words, all_tags = get_words_and_tags(data)
    
    for datum in data:
        
        #split datum into word/tag pairs
        word_tag_pairs = datum.split(" ")

        tag_list = []

        for pair in word_tag_pairs:
            #split word/tag pair
            word, tag = pair.split("/")

            #add tag to list of tags for this datum
            tag_list.append(tag)

        #for each pair of adjacent tags
        for i in range(0, len(tag_list)-1):
            t_i_minus_1, t_i = tag_list[i], tag_list[i+1]

            #either increment or initialize entry in t
            if t_i in transition[t_i_minus_1]:
                transition[t_i_minus_1][t_i] += 1.0
            else:
                transition[t_i_minus_1][t_i] = 1.0

    for t_i_minus_1, tag_set in transition.iteritems():

        #get total amount of tags that preceeded t_i_minus_1
        total_count = sum(tag_set.values())

        
        #count how many tags appeared in transition and how many did not
        tag_count = len(tag_set)
        missing_tag_count = 0

        for tag in all_tags:
            if tag not in tag_set:
                missing_tag_count += 1
        

        #set probability of transition from t_(i-1) to t_i
        for t_i, count in tag_set.iteritems():

            #calculate a probability differential to take away from tags that did
            #appear to balance the extremely small added probability of tags which
            #did not and calculate probability for transition with this in mind
            tag_df = (float(missing_tag_count) * 1.0e-250) / float(tag_count)

            probability = count / total_count
            transition[t_i_minus_1][t_i] = probability - (tag_df * probability)

        #for all tags that didn't appear before some tag
        for tag in all_tags:
            if tag not in tag_set:
                #set probability to exceedingly small nonzero rational.
                tag_set[tag] = 1.0e-250

def predict(test_sentence, tags, initial_probabilities, transition, emission):
    """
    Predict a sequence of tags for given test sentence using viterbi algorithm.
    """

    #create argmax table and prediction dict
    table = [{}]
    prediction = {}
    
    #Calculate starting probability emission and record in argmax table
    for tag in tags:
        table[0][tag] = initial_probabilities[tag] * emission[tag][test_sentence[0]]
        prediction[tag] = [tag]

    #if only 1 observation in test sentence, we have a prediction sequence
    #consisting of a single tag, so just take the max of first entry in
    #argmax table, i.e. the max probability in starting column
    if len(test_sentence) == 1:
        start_tag = max((table[0][tag], tag) for tag in tags)[1]
        return prediction[start_tag]

    #We have more than a single word sentence, proceed with dynamic part
    #of viterbi algorithm; i.e. step through each column computing argmax
    #entries as we go along
    for t in range(1, len(test_sentence)):
        
        #append new column in argmax table and another dictionary for the next
        #tag prediction; this dictionary will extend the last via concatenating
        #the prediction sequence for each tag with the most probable next tag
        table.append({})
        next_prediction = {}

        #for all possible tags, i.e. for this particular column in argmax table
        for tag_j in tags:
            
            #initialize a list to hold pairs of probability for the next tag
            #and the next tag itself
            probability_tag_pairs = []

            #for each tag in the current column of argmax table
            for tag_i in tags:
                #compute prbability for each tag tag_i given a tag tag_j
                next_tag_prob = table[t-1][tag_i] * transition[tag_i][tag_j] \
                                    * emission[tag_j][test_sentence[t]]
                
                #append 2-tuple consisting of probabilty for next tag
                #and the tag itself 
                probability_tag_pairs.append((next_tag_prob, tag_i))
            
            #get the probability-tag pair corresponding to the max probability
            max_probability, max_tag = max(probability_tag_pairs)
            
            #assign max probability in argmax table and concatenate
            #new tag prediction to end of existing prediction sequence
            table[t][tag_j] = max_probability
            next_prediction[tag_j] = prediction[max_tag] + [tag_j]

        #replace prediction sequence with the extended sequence
        prediction = next_prediction

    #get starting tag of most probable sequence in last column of argmax table
    #then return that sequence
    start_tag = max((table[t][tag], tag) for tag in tags)[1]
    return prediction[start_tag]

def get_error(sentence, prediction):
    """Determine error rate of test set."""
    #get tags from test sentence
    tags = [pair.split("/")[1] for pair in sentence.split(" ")]
    #get amount of matches between predictions and tags then compute score
    matches = sum([1 for i in range(len(tags)) if tags[i] == prediction[i]])
    score = float(matches) / float(len(tags))
    return score

def get_mft_baseline(sentence, data):
    """
    Generate the most frequent tag baseline for a given sentence.
    Return an accuracy score computed with the most frequent tag prediction.
    """

    #get al words and tags that occur in the data
    words,tags = get_words_and_tags(data)

    #initialize a dict of word keys and tag-set values to count most frequent
    #tag given a word
    mft = {word: {tag: 0 for tag in tags} for word in words}

    #for each datum
    for datum in data:
        #separate words and their tags
        word_tag_pairs = datum.split(" ")

        #for each word-tag pair, split word from tag and record
        for pair in word_tag_pairs:
            word, tag = pair.split("/")
            mft[word][tag] += 1

    #get word-tag pairs in sentence
    test_word_tag_pairs = sentence.split(" ")

    matches = 0

    #for each pair, split test word and tag
    for test_pair in test_word_tag_pairs:
        test_word, test_tag = test_pair.split("/")

        #get most frequent tag prediction for test_word
        tag_prediction = max(mft[test_word], key=mft[test_word].get)

        #count accurate predictions
        if tag_prediction == test_tag:
            matches += 1

    #compute prediction score and return it
    score = float(matches) / float(len(test_word_tag_pairs))
    return score

def main():
    """main method."""

    #store training and test data as lists of strings
    training_data = get_data("training.txt")
    test_data = get_data("test.txt")

    #initialize emission and transition probability tables
    emission, transition = init_tables(training_data)

    #fill the emission table
    fill_emission(emission, training_data)

    #fill the transition table
    fill_transition(transition, training_data)

    #get list of all words and tags along with starting probabilities
    words,tags = get_words_and_tags(training_data)
    sw, st = get_starting_probability(training_data)

    errors = []

    #for each sentence
    for sentence in test_data:
        #get prediction for sentence
        w,t = get_words_and_tags([sentence])
        seq = predict(w, tags, st, transition, emission)

        mft_score = get_mft_baseline(sentence, training_data)

        #compute error on each sentence
        error = get_error(sentence, seq)
        errors.append(error)
        print "bigram HMM tagger accuracy: " + str(error * 100) + "%"
        print "most frequent tag baseline accuracy: " + str(mft_score * 100) + "%"
        print
        print "bigram HMM tagger error rate: " + str((1 - error) * 100) + "%"
        print "most frequent tag baseline error rate: " + str((1 - mft_score) * 100) + "%"

    #compute average error on the test set
    avg_error = sum(errors)/len(errors)
    print "\naverage error: " + str(avg_error * 100) + "%"

if __name__ == "__main__":
    main()