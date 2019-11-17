# heart-anomaly

The program is to build machine learners to diagnose heart anomalies from radiology data. 
We will run the learners on some real heart anomaly data, sliced and diced in various ways 
and get the models which will be used to run on the test data and get the predictions. The 
learners will be used are Naive Bayesian and Gaussian Naive Bayes. The results (the accuracy
on all the instances, true negative rate, and true positive rate) from all different datasets
can be compared, and the results from those two learners can also be compared. 

## Pre-reqs, Setup, and Build

Get rerequisites
- Install Python

Git clone
```shell
$ git clone https://github.com/waiwaixiaochen/heart-anomaly.git
```

Run the program
```shell
$ python heart_anomaly.py
```

## Learners used

- First learner: Naive Bayesian-The learning approach involves counting feature occurrences in the learning phase, and then using these counts in the classification phase

- Second learner: Gaussian Naive Bayes-When using Gaussian Naïve Bayes, the learning approch involves calculating the mean and the standard deviation of the features for each class
                

## Results

- Naive Bayesian Learner:
  - orig 142/187(0.76) 10/15(0.67) 132/172(0.77)
  - resplit 78/90(0.87) 17/19(0.89) 61/71(0.86)
  - itg 145/187(0.78) 15/15(1.00) 130/172(0.76)
  - resplit-itg 63/90(0.70) 17/19(0.89) 46/71(0.65)

- Gaussian Naive Bayes Learner:
  - orig 112/187(0.60) 12/15(0.80) 100/172(0.58)
  - resplit 54/90(0.60) 19/19(1.00) 35/71(0.49)
  - itg 132/187(0.71) 14/15(0.93) 118/172(0.69)
  - resplit-itg 61/90(0.68) 16/19(0.84) 45/71(0.63)
