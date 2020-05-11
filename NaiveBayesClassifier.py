import numpy as np
import time
from math import log

def run_train_test(training_file, testing_file):

    # Set the variables, params, dicts, sets
    alpha = 0.5
    stop_words = {'the','and'}  # Stop words
    logic_negation = {'t','not','no','never','dont','didnt','doesnt'} # Words indicating negation


    # Import training dataset
    training_start_time = time.time()
    vocab = set()
    wordcount_class_0 = {}
    wordcount_class_1 = {}
    total_reviews = 0
    reviewscount_0 = 0
    reviewscount_1 = 0
    train_labels = []
    train_reviews = []
    with training_file as f:
        for line in f:
            review, label = line.split(',')
            words = review.split(' ')
            del words[-1]
            label = int(label.strip("\n"))
            total_reviews += 1
            # Implement negation: add NOT_ to words after logical negation
            for i in range(len(words)):
                if words[i] in logic_negation:
                    try:
                        words[i+1] = 'NOT_' + words[i+1]
                    except:
                        continue
                    try:
                        words[i+2] = 'NOT_' + words[i+2]
                    except:
                        continue
                    try:
                        words[i+3] = 'NOT_' + words[i+3]
                    except:
                        continue
            words = set(words)  # Binary NB
            vocab.update(words)
            for word in words:
                if word not in wordcount_class_0.keys():
                    wordcount_class_0[word] = 0
                    wordcount_class_1[word] = 0

            if label==0:
                reviewscount_0 += 1
                for word in words:
                    wordcount_class_0[word] += 1
            if label==1:
                reviewscount_1 += 1
                for word in words:
                    wordcount_class_1[word] += 1

            train_labels.append(label)
            train_reviews.append(words)

    # Compute CPTs
    P_class = [0,0]
    P_class[0] = reviewscount_0 / total_reviews
    P_class[1] = reviewscount_1 / total_reviews

    P_words_class_0 = {}
    P_words_class_1 = {}
    bottom_0 = sum(wordcount_class_0.values()) + alpha*len(vocab)
    bottom_1 = sum(wordcount_class_1.values()) + alpha*len(vocab)
    for word in vocab:
        if word in stop_words:
            P_words_class_0[word] = (0 + alpha) / bottom_0
            P_words_class_1[word] = (0 + alpha) / bottom_1
        else:
            P_words_class_0[word] = (wordcount_class_0[word] + alpha) / bottom_0
            P_words_class_1[word] = (wordcount_class_1[word] + alpha) / bottom_1

    # Inference on the training dataset
    predict_train_labels = []
    for doc in train_reviews:
        log_sum_0 = 0
        log_sum_1 = 0
        bag_of_words = set(doc)
        for word in bag_of_words:
            log_sum_0 += log(P_words_class_0[word])
            log_sum_1 += log(P_words_class_1[word])
        Prob_c0 = log(P_class[0]) + log_sum_0
        Prob_c1 = log(P_class[1]) + log_sum_1
        if Prob_c0 > Prob_c1:
            c = 0
        else:
            c = 1
        predict_train_labels.append(c)

    # Compute training accuracy
    correct = 0
    for i in range(len(train_labels)):
        if predict_train_labels[i] == train_labels[i]:
            correct += 1
    train_accuracy = correct / len(train_labels)

    # Print results
    training_time = time.time() - training_start_time
    # print(training_time)
    # print(train_accuracy)

    # Import testing dataset
    testing_start_time = time.time()
    test_reviews = []
    test_labels = []
    with testing_file as f:
        for line in f:
            review, label = line.split(',')
            words = review.split(' ')
            del words[-1]
            label = int(label.strip("\n"))
            # Implement negation: add NOT_ to words after logical negation
            for i in range(len(words)):
                if words[i] in logic_negation:
                    try:
                        words[i+1] = 'NOT_' + words[i+1]
                    except:
                        continue
                    try:
                        words[i+2] = 'NOT_' + words[i+2]
                    except:
                        continue
                    try:
                        words[i+3] = 'NOT_' + words[i+3]
                    except:
                        continue
            words = set(words)  # Binary NB
            test_labels.append(label)
            test_reviews.append(words)

    # Inference on the testing dataset
    predict_test_labels = []
    for doc in test_reviews:
        log_sum_0 = 0
        log_sum_1 = 0
        bag_of_words = set(doc)
        bag_of_words = vocab.intersection(bag_of_words)
        for word in bag_of_words:
            log_sum_0 += log(P_words_class_0[word])
            log_sum_1 += log(P_words_class_1[word])
        Prob_c0 = log(P_class[0]) + log_sum_0
        Prob_c1 = log(P_class[1]) + log_sum_1
        if Prob_c0 > Prob_c1:
            c = 0
        else:
            c = 1
        print(c)
        predict_test_labels.append(c)

    # Compute testing accuracy
    correct = 0
    for i in range(len(test_labels)):
        if predict_test_labels[i] == test_labels[i]:
            correct += 1
    test_accuracy = correct / len(test_labels)

    # Print results
    testing_time = time.time() - testing_start_time
    print(round(training_time),'seconds (training)')
    print(round(testing_time),'seconds (labeling)')
    print(round(train_accuracy,3),'(training)')
    print(round(test_accuracy,3),'(testing)')

    return



if __name__ == "__main__":

    import sys

    training_file = open(sys.argv[1], "r")
    testing_file = open(sys.argv[2], "r")

    run_train_test(training_file, testing_file)

    training_file.close()
    testing_file.close()
