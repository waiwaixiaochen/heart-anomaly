import csv
import numpy as np 
#import math
#from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
#from sklearn.metrics import confusion_matrics

# preprocess the basic data for the learners
def prepare_basic_data(train_y, train_row_num, train_x):
    train_y_list = list(train_y)
    count1 = train_y_list.count(1)
    count0 = train_row_num - count1
    p_normal = float(count1)/float(train_row_num)
    p_abnormal = 1-p_normal

    # get the row numbers of normal instances
    ind = [i for i, x in enumerate(train_y_list) if x==1]

    # create a new data set for normal instances
    train_normal = train_x[ind, :]
    # delete the first column
    train_normal = np.delete(train_normal,0, axis=1)
    train_normal = np.array(train_normal)

    # create a new data set for abnormal instances
    train_abnormal = np.delete(train_x, ind, 0)
    # delete the first column for the abnormal instances
    train_abnormal = np.delete(train_abnormal, 0, axis=1)
    train_abnormal = np.array(train_abnormal)

    return p_normal, p_abnormal, train_normal, train_abnormal, count1, count0


# preprocess the data for Naive Bayesian Learning
def prepare_data_naive(p_normal, train_normal, count1):
    
    # create  lists to record the probabilities
    # one list for p(xi=1|normal), one list for p(xi=0|normal)
    # one list for p(xi=1|abnormal), one list for p(xi=0|abnormal)
    p1_normal = list()
    p0_normal = list()
    #print(train_normal)
    feature_num = np.size(train_normal, 1)
    for i in range(0,feature_num):
        feature = train_normal[:, i] 
        #print(feature)
        feature = feature.astype(int)
        #print(feature)
        feature_list = list(feature)
        
        #feature_list = np.array(feature_list)
        count_feature_1 = feature_list.count(1)
        count_feature_0 = count1 - count_feature_1 
        
        p_feature_1 = float(count_feature_1 + 0.5)/ float(count1 + 0.5)
        #print(p_feature_1)
        p_feature_1 = np.log(p_feature_1)
        p1_normal = np.array(p1_normal)
        p1_normal = np.append(p1_normal, p_feature_1)
        
        p_feature_0 = float(count_feature_0 + 0.5) / float(count1 + 0.5)
        p_feature_0 = np.log(p_feature_0)
        p0_normal = np.array(p0_normal)
        p0_normal = np.append(p0_normal, p_feature_0)
    return p1_normal, p0_normal

# prepare data for Guassian Naive Bayes
def prepare_data_guassian(train_normal, train_abnormal):
    # calculate the mean for each feature
    train_normal = train_normal.astype(np.float)
    train_normal_mean = np.mean(train_normal, axis=0)

    # calculate the standard deviation for each feature
    train_normal_std = np.std(train_normal, axis = 0)
    train_normal_std += 0.0001

    # the same process with abnormal data
    train_abnormal = train_abnormal.astype(np.float)
    train_abnormal_mean = np.mean(train_abnormal, axis=0)
    train_abnormal_std = np.std(train_abnormal, axis = 0)
    train_abnormal_std += 0.0001

    return train_normal_mean, train_normal_std, train_abnormal_mean, train_abnormal_std

# load data from csv files
def load_data(train_file_name, test_file_name):
    # create and initialize csv files' row number anf input list
    train_row_num = 0
    test_row_num = 0
    train_x = list()
    test_x = list()
    count = 0
    #train_file_name = "spect-orig.train.csv"
    # read from csv train files
    with open(train_file_name, 'r') as train_file:
        train_reader = csv.reader(train_file)
        for row in train_reader:
            train_row_num += 1
            train_x.append(row)
    train_x = np.array(train_x)

    # labels of the train data
    train_y = train_x[:, 0]
    train_y = train_y.astype(int)

    # read from csv test files
    #test_file_name = "spect-orig.test.csv"
    with open(test_file_name, 'r') as test_file:
        test_reader = csv.reader(test_file)
        for row in test_reader:
            test_row_num += 1
            test_x.append(row)
    test_x = np.array(test_x)

    # labels of the test data
    test_y = test_x[:, 0]
    test_y = test_y.astype(int)
    test_list = list(test_y)
    count_test_normal = test_list.count(1)
    count_test_abnormal = test_row_num - count_test_normal
    #print(test_y)
    test_x_new = np.delete(test_x,0,1)
    test_x_new = np.array(test_x_new)
    test_x_new = test_x_new.astype(np.float)
    return train_y, train_row_num, train_x, test_x_new, test_row_num, test_y, count_test_normal, count_test_abnormal 


# Run Naive Bayesian
def run_naive_bayesian(train_file_name, test_file_name):
    train_y, train_row_num, train_x, test_x_new, test_row_num, test_y, count_test_normal, count_test_abnormal = load_data(train_file_name, test_file_name)
    p_normal, p_abnormal, train_normal, train_abnormal, count1, count0 = prepare_basic_data(train_y, train_row_num, train_x)
    p1_normal, p0_normal = prepare_data_naive(p_normal, train_normal, count1)
    p1_abnormal, p0_abnormal = prepare_data_naive(p_abnormal, train_abnormal, count0)
    #print(p1_normal)
    feature_num = np.size(train_normal, 1)
    #count_normal = 0
    #count_abnormal = 0
    count_correct = 0
    true_normal = 0
    true_abnormal = 0
    predicted = list()

    for i in range(0, test_row_num):
        p1 = np.log(p_normal)
        p0 = np.log(p_abnormal)
        for j in range(0,feature_num):
            if test_x_new[i][j] == 1:
                p1 += p1_normal[j]
                p0 += p1_abnormal[j]
            if test_x_new[i][j] == 0:
                p1 += p0_normal[j]
                p0 += p0_abnormal[j]
        if p1 > p0:
           # count_normal += 1
            predicted.append(1)
        else:
           # count_abnormal += 1
            predicted.append(0)
    predicted = np.array(predicted)

    #print(count_normal)
    for i in range(0, test_row_num):
        if predicted[i] == test_y[i]:
            count_correct += 1
        if predicted[i] == 0 and test_y[i] == 0:
            true_abnormal += 1
        if predicted[i] == 1 and test_y[i] == 1:
            true_normal += 1
    return count_correct, test_row_num, true_abnormal, count_test_abnormal, true_normal, count_test_normal

# run Guassian Naive Bayes
def run_guassian(train_file_name, test_file_name):
    train_y, train_row_num, train_x, test_x_new, test_row_num, test_y, count_test_normal, count_test_abnormal = load_data(train_file_name, test_file_name)
    p_normal, p_abnormal, train_normal, train_abnormal, count1, count0 = prepare_basic_data(train_y, train_row_num, train_x)
    train_normal_mean, train_normal_std, train_abnormal_mean, train_abnormal_std = prepare_data_guassian(train_normal, train_abnormal) 
    p_normal_log = np.log(1/(np.sqrt(2*np.pi)*train_normal_std)) + (-np.square(test_x_new-train_normal_mean)/(2*np.square(train_normal_std)))
    p_abnormal_log = np.log(1/(np.sqrt(2*np.pi)*train_abnormal_std)) + (-np.square(test_x_new-train_abnormal_mean)/(2*np.square(train_abnormal_std)))
    p_normal = np.log(p_normal)
    p_abnormal = np.log(p_abnormal)
    p_test_normal_guassian = np.sum(p_normal_log, axis = 1) + p_normal
    p_test_abnormal_guassian = np.sum(p_abnormal_log, axis = 1) + p_abnormal
    predicted = np.where((p_test_normal_guassian > p_test_abnormal_guassian), 1,0)
    count = np.count_nonzero(predicted == 0)
    return predicted, test_y, test_x_new, test_row_num

def display_results_naive(train_file_name, test_file_name, file_short_name):
    count_correct, test_row_num, true_abnormal, count_test_abnormal, true_normal, count_test_normal = run_naive_bayesian(train_file_name, test_file_name)
    accuracy = round(float(count_correct)/float(test_row_num), 2)
    true_negative = round(float(true_abnormal)/float(count_test_abnormal), 2)
    true_positive = round(float(true_normal)/float(count_test_normal), 2)
    print('{} {}/{}({}) {}/{}({}) {}/{}({})'.format(file_short_name, count_correct, test_row_num, accuracy,true_abnormal,count_test_abnormal, true_negative, true_normal, count_test_normal, true_positive))

def display_results_guassian(train_file_name, test_file_name, file_short_name):
    predicted, test_y, test_x_new, test_row_num = run_guassian(train_file_name, test_file_name)
    count_correct = 0
    true_abnormal = 0
    count_test_abnormal = 0
    true_normal = 0
    count_test_normal = 0
    for i in range(0, test_row_num):
        if predicted[i] == test_y[i]:
            count_correct += 1
        if predicted[i] == 0 and test_y[i] == 0:
            true_abnormal += 1
        if test_y[i] == 0:
            count_test_abnormal += 1
        if predicted[i] == 1 and test_y[i] == 1:
            true_normal += 1
        if test_y[i] == 1:
            count_test_normal += 1
    accuracy = round(float(count_correct)/float(test_row_num), 2)
    true_negative = round(float(true_abnormal)/float(count_test_abnormal), 2)
    true_positive = round(float(true_normal)/float(count_test_normal), 2)
    print('{} {}/{}({}) {}/{}({}) {}/{}({})'.format(file_short_name, count_correct, test_row_num, accuracy,true_abnormal,count_test_abnormal, true_negative, true_normal, count_test_normal, true_positive))



# Part 1: Run Naive Bayesian
print("Naive Bayesian Learner:")
display_results_naive("spect-orig.train.csv", "spect-orig.test.csv", "orig")
display_results_naive("spect-resplit.train.csv", "spect-resplit.test.csv", "resplit")
display_results_naive("spect-itg.train.csv", "spect-itg.test.csv", "itg")
display_results_naive("spect-resplit-itg.train.csv", "spect-resplit-itg.test.csv", "resplit-itg")
print("\n")

# Part 2: Run Guassian Naive Bayes
print("Guassian Naive Bayes Learner:")
display_results_guassian("spect-orig.train.csv", "spect-orig.test.csv", "orig")
display_results_guassian("spect-resplit.train.csv", "spect-resplit.test.csv", "resplit")
display_results_guassian("spect-itg.train.csv", "spect-itg.test.csv", "itg")
display_results_guassian("spect-resplit-itg.train.csv", "spect-resplit-itg.test.csv", "resplit-itg")

