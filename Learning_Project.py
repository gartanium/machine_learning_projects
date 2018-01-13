import random
from datetime import datetime
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split

class HardCodedModel:

    def predict(self, test_data):
        rObj = list()
        for i in range(len(test_data)):
            rObj.append(0)

        return rObj

class HardCodedClassifier:

    def fit(self, data_target, data_train):
        return HardCodedModel()

# Compares the false answers to the correct ones.
def compare_prediction(target, prediction):
    length = len(target)
    false_count = 0
    for i in range(length):
        if target[i] != prediction[i]:
            false_count += 1
    return float(length - false_count) / length

iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)




# Show the actual target names that correspond to each number
print(iris.target_names)

classifier = GaussianNB()

data_train, data_test, targets_train, targets_test =\
    train_test_split(iris.data, iris.target, test_size=0.33, random_state=46)

classifier = GaussianNB()
model = classifier.fit(data_train, targets_train)

targets_predicted = model.predict(data_test)

print("Percent correct: ", compare_prediction(targets_test, targets_predicted), "\n")

classifier2 = HardCodedClassifier()
model_hard = classifier2.fit(data_train, targets_train)

targets_hard_predicted = model_hard.predict(data_test)

print("Hard Coded Test")
print("Percent correct: ", compare_prediction(targets_test, targets_hard_predicted), "\n")

print("Test Finished")

