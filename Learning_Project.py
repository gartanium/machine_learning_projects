import random
from datetime import datetime
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
import numpy as n
import bisect
from operator import itemgetter
from pandas import read_csv


# Class for running an option whenever a specified word is passed in.
class Tests:
    def __init__(self):
        self.option_dict = dict()

    def add_option(self, option, func):
        self.option_dict[option] = func

    def execute_option(self, option):
        if option in self.option_dict:
            return self.option_dict[option]()
        else:
            return self.option_dict['default']()

    def load_car_data(self):

        headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "values"]

        df = read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
                         names=headers, na_values=" ?")

        cleanup_nums_buying = {"buying": {"vhigh": 1, "high": .75, "med": .5, "low": .25 }}
        cleanup_nums_maint = {"maint": {"vhigh": 1, "high": .75, "med": .5, "low": .25 }}
        cleanup_nums_doors = {"doors": {"2" : .25, "3" : .5, "4" : .75, "5more" : 1}}
        cleanup_nums_persons = {"persons": {"more" : 1, "2" : .5, "4": .75}}
        cleanup_nums_lug_boot = {"lug_boot": {"small" : 0.33, "med": 0.66, "big": 1}}
        cleanup_nums_safety = {"safety": {"low": .33, "med": .66, "high": 1}}
        cleanup_nums_value = {"values": {"vgood": 1, "good": 2, "acc": 3, "unacc": 4}}
        #cleanup_nums = {"education":     {"12k": 1, "Some-college" : 2, "Grad" : 3}}

        df.replace(cleanup_nums_buying, inplace=True)
        df.replace(cleanup_nums_maint, inplace=True)
        df.replace(cleanup_nums_doors, inplace=True)
        df.replace(cleanup_nums_persons, inplace=True)
        df.replace(cleanup_nums_lug_boot, inplace=True)
        df.replace(cleanup_nums_safety, inplace=True)
        df.replace(cleanup_nums_value, inplace=True)


        print(df.dtypes)
        print(df.head(5))

        car_target = df['values']
        car_data = df.drop('values', axis=1)
        print(car_data)

        self.data_train, self.data_test, self.targets_train, self.targets_test = train_test_split(car_data, car_target, test_size=0.33, random_state=46)

        print("Lengths of: ", len(self.data_train), len(self.data_test), len(self.targets_train), len(self.targets_test))



        return 0

    def knn_kdt_test(self):
        print("Running K Nearest Neighbors Test!")
        k = int(input("Select a value for k: "))
        classifier =  KNeighborsClassifier(k)

        print("Lengths of 2: ", len(self.data_train), len(self.targets_train))
        model = classifier.fit( self.data_train, self.targets_train,)
        predicted = model.predict(self.data_test)
        print("Percent correct: ", compare_prediction(self.targets_test.as_matrix() , predicted), "%\n")
        print("Test Finished")
        return 0


    # default option to be executed when an improper option is given
    def set_default_option(self, func):
        self.option_dict['default'] = func


class terrible_knn_classifier:
    def fit(self, data_target, data_train):
        return terrible_knn_model(data_target, data_train)


class terrible_knn_model:

    def __init__(self, data_target, data_train):
        self.data_target = data_target
        self.data_train = data_train
        self.data_points = len(data_train)

    def predict(self, test_data, k):

        test_results = list()
        neighbor_list = list()

        for i in range(len(test_data)):
            neighbor_list.append(list())

        neighbor_list_sorted = list()
        # First calculate the distances from every object
        for i in range(len(test_data)):
            for j in range(len(self.data_train)):
                a = test_data[i]
                b = self.data_train[j]
                distance = n.linalg.norm(a - b)

                pair = (distance, j)
                neighbor_list[i].append(pair)

            sorted_list = sorted(neighbor_list[i], key=itemgetter(0), reverse=False)
            neighbor_list_sorted.append(sorted_list)
        # Sort each list

        # Cycle through k nearest neighbors
        for i in range(len(test_data)):

            dict_results = dict()
            for j in range(k):

                # Get the training values
                target_index = neighbor_list_sorted[i][k][1]
                target_value = self.data_target[target_index]

                if target_value in dict_results:
                    dict_results[target_value] += 1
                else:
                    dict_results[target_value] = 1

            # Return a list of max dictionary results.
            val = max(dict_results, key=dict_results.get)
            test_results.append(val)

        return test_results

# k-nearest-neighbors
class k_nearest_neighbors_model:

    def __init__(self, data_target, data_train):
        self.data_target = data_target
        self.data_train = data_train
        self.data_points = len(data_train)


    def set_data(self, knn_list):
        pass

    def predict(self, test_data, k, kd_tree):

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
    length2 = len(prediction)
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

def run_terrible_knn_test():
    print("Running terrible KNN Test!")
    k = int(input("Select a value for k: "))
    iris, data_train, data_test, targets_train, targets_test = setup_test()
    classifier = terrible_knn_classifier()
    model = classifier.fit(targets_train, data_train)
    predicted = model.predict(data_test, k)
    print("Percent correct: ", compare_prediction(targets_test, predicted), "%\n")
    print("Test Finished")
    return 0


# Runs the k nearest neighbors tests
def run_k_nearest_neighbors_test():
    print("Running K Nearest Neighbors Test!")
    k = int(input("Select a value for k: "))
    iris, data_train, data_test, targets_train, targets_test = setup_test()
    classifier = K_Nearest_Neighbors_Classifier()
    model = classifier.fit(targets_train, data_train)
    predicted = model.predict(data_test, k, 1)
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
          "to run the gussian test, option[terrible_knn] to run the poor knn test, option [knn] to run k-nearest neighbors test, and option [print] to print "
          "Iris data.")
    return 0


def end_program():
    print("Ending program")
    return 1


# Initializes the program
def initialize_program(tests):


    print("Welcome to Iris classifier 1.0!")
    tests.add_option("hardcoded", run_hardcode_test)
    tests.add_option("gussian", run_gussian_test)
    tests.add_option("print", print_iris_data)
    tests.add_option("help", help)
    tests.add_option("quit", end_program)
    tests.add_option("knn", tests.knn_kdt_test)
    tests.add_option("terrible_knn", run_terrible_knn_test)
    tests.add_option("lc", tests.load_car_data)
    tests.set_default_option(default_msg)


# Main entry point for the application.
def main():

    tests = Tests()
    initialize_program(tests)
    quit = 0

    while quit == 0:
        quit = execute_option(tests)


if __name__ == "__main__":
    main()

