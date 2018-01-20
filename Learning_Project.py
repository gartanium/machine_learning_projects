import random
from datetime import datetime
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KDTree
import numpy as n

# Class for running an option whenever a specified word is passed in.
class Option:
    def __init__(self):
        self.option_dict = dict()

    def add_option(self, option, func):
        self.option_dict[option] = func

    def execute_option(self, option):
        if option in self.option_dict:
            return self.option_dict[option]()
        else:
            return self.option_dict['default']()

    # default option to be executed when an improper option is given
    def set_default_option(self, func):
        self.option_dict['default'] = func


# k-nearest-neighbors
class k_nearest_neighbors_model:

    def __init__(self, data_target, data_train):
        self.data_target = data_target
        self.data_train = data_train
        self.data_points = len(data_train)


    def predict(self, test_data, k):

        tree = KDTree(self.data_train, leaf_size=2)


        test_results = list()

        # Cycle through every single test_data point
        for i in range(len(test_data)):
            # Get the list of k nearest neighbors
            dist, ind = tree.query(test_data, k)

            # ind returns a list of indexes for nearest neighbors.
            # Create a dictionary list of values for target values and number of knn who have it

            # Dictionary for storing possible values as keyes and number of occournces of said value as values
            target_occurrence_dict = dict()
            # Cycle through every single knn
            for j in range(len(ind[i])):
                # Find the data index of the knn and index j
                index = ind[i, j]
                # Find it's test value
                target_value = self.data_target[index]

                # Set the dictionary target_key's value.
                if target_value in target_occurrence_dict:
                    target_occurrence_dict[target_value] += 1
                else:
                    target_occurrence_dict[target_value] = 1

            # Sort the dictionary
            val = max(target_occurrence_dict, key=target_occurrence_dict.get)
            test_results.append(val)

            #index = ind[i, 0]
            #val = self.data_target[index]
            #test_results.append(val)
            # Add up their values
            # set our friend to the closest one
                # resolve ties

        return test_results


# K-Nearest-Neighbors-Classifier
class K_Nearest_Neighbors_Classifier:

    def fit(self, data_target, data_train):
        return k_nearest_neighbors_model(data_target, data_train)


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
    return float(length - false_count) * 100 / length


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
    return 0

# Runs the k nearest neighbors tests
def run_k_nearest_neighbors_test():
    print("Running K Nearest Neighbors Test!")
    k=9
    iris, data_train, data_test, targets_train, targets_test = setup_test()
    classifier = K_Nearest_Neighbors_Classifier()
    model = classifier.fit(targets_train, data_train)
    predicted = model.predict(data_test, k)
    print("Percent correct: ", compare_prediction(targets_test, predicted), "%\n")
    print("Test Finished")
    return 0

# Runs the hard coded test, which we expect to get very poor results
def run_hardcode_test():
    print("Running Hard Coded Test")
    iris, data_train, data_test, targets_train, targets_test = setup_test()
    classifier2 = HardCodedClassifier()
    model_hard = classifier2.fit(data_train, targets_train)
    targets_hard_predicted = model_hard.predict(data_test)
    print("Percent correct: ", compare_prediction(targets_test, targets_hard_predicted), "%\n")
    print("Test Finished")
    return 0


# Runs the Gussian test, which should result in high results
def run_gussian_test():
    print("Running Gussian Test")
    iris, data_train, data_test, targets_train, targets_test = setup_test()
    classifier = GaussianNB()
    model = classifier.fit(data_train, targets_train)
    targets_predicted = model.predict(data_test)
    print("Percent correct: ", compare_prediction(targets_test, targets_predicted), "%\n")
    print("Test Finished")
    return 0


def default_msg():
    print("Improper command! Type the command [help] for help or [quit] to exit the program!")
    return 0


# Executes a given option
def execute_option(options):

    user_input = get_option()
    return options.execute_option(user_input)


# Gets the option from the user.
def get_option():
    print("Select an option! (enter the command help for help!)")
    return input()


def help():
    print("Choose option [quit] to quit, option [hardcoded] to run the hardcoded test, option [gussian] "
          "to run the gussian test, option [knn] to run k-nearest neighbors test, and option [print] to print "
          "Iris data.")
    return 0


def end_program():
    print("Ending program")
    return 1


# Initializes the program
def initialize_program(options):
    print("Welcome to Iris classifier 1.0!")
    options.add_option("hardcoded", run_hardcode_test)
    options.add_option("gussian", run_gussian_test)
    options.add_option("print", print_iris_data)
    options.add_option("help", help)
    options.add_option("quit", end_program)
    options.add_option("knn", run_k_nearest_neighbors_test)
    options.set_default_option(default_msg)


# Main entry point for the application.
def main():
    options = Option()
    initialize_program(options)
    quit = 0

    while quit == 0:
        quit = execute_option(options)


if __name__ == "__main__":
    main()

