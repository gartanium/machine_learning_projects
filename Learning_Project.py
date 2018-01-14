import random
from datetime import datetime
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split

# Test class
class HardCodedModel:

    def predict(self, test_data):
        rObj = list()
        for i in range(len(test_data)):
            rObj.append(0)

        return rObj

# Test Classifier
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

# Setups the iris data to be used by other functions
def setup_iris_data():
    iris = datasets.load_iris()
    return iris

# Sets up the test data
def setup_test():
    iris = setup_iris_data()
    data_train, data_test, targets_train, targets_test = \
        train_test_split(iris.data, iris.target, test_size=0.33, random_state=46)

    return iris, data_train, data_test, targets_train, targets_test

# Prints the iris data set data
def print_iris_data():
    iris = setup_iris_data()
    # Show the data (the attributes of each instance)
    print(iris.data)

    # Show the target values (in numeric format) of each instance
    print(iris.target)

    # Show the actual target names that correspond to each number
    print(iris.target_names)

# Runs the hard coded test, which we expect to get very poor results
def run_hardcode_test():
    print("Running Hard Coded Test")
    iris, data_train, data_test, targets_train, targets_test = setup_test()
    classifier2 = HardCodedClassifier()
    model_hard = classifier2.fit(data_train, targets_train)
    targets_hard_predicted = model_hard.predict(data_test)
    print("Percent correct: ", compare_prediction(targets_test, targets_hard_predicted), "\n")
    print("Test Finished")

# Runs the Gussian test, which should result in high results
def run_gussian_test():
    print("Running Gussian Test")
    iris, data_train, data_test, targets_train, targets_test = setup_test()
    classifier = GaussianNB()
    model = classifier.fit(data_train, targets_train)
    targets_predicted = model.predict(data_test)
    print("Percent correct: ", compare_prediction(targets_test, targets_predicted), "\n")
    print("Test Finished")

# Executes a given option
def execute_option(option):
    if option == 'q':
        return 1
    elif option == '0':
        run_hardcode_test()
        return 0
    elif option == '1':
        run_gussian_test()
        return 0
    elif option == 'p':
        print_iris_data()
        return 0
    else:
        print("Choose option [q] to quit, option [0] to run test 0, option [1] to run test 1, option [p] to print "
              "Iris data.")
        return 0

# Gets the option from the user.
def get_option():
    print("Select a one letter option! (enter h for help!)")
    return input()

# Initializes the program
def initialize_program():
    print("Welcome to Iris classifier 1.0!")
    quit = 0

    while quit == 0:
        option = get_option()
        quit = execute_option(option)

# Main entry point for the application.
def main():
    initialize_program()


if __name__ == "__main__":
    main()

