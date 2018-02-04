import random
from datetime import datetime
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import numpy as n
import bisect
from operator import itemgetter
from pandas import read_csv
import math
import statistics

# Class for running an option whenever a specified word is passed in.
class Tests:



    def __init__(self):
        self.option_dict = dict()
        self.initialize_program()

    # Initializes the program
    def initialize_program(self):
        print("Welcome to Matthew Brown's Machine Learning Project!")

        self.add_option("gussian", run_gussian_test)
        self.add_option("print", print_iris_data)
        self.add_option("help", self.help)
        self.add_option("quit", end_program)
        self.add_option("knn", self.knn_kdt_test)
        self.add_option("knnbrute", self.knn_brute_force)
        self.add_option("lc", self.load_car_data)
        self.add_option("ld", self.load_diabetes_data)
        self.add_option("knnreg", self.knn_reg)
        self.add_option("lmpg", self.load_mpg_data)
        self.add_option("dtree", self.decision_tree)
        self.add_option("utests", self.run_unit_tests)
        self.set_default_option(self.default_msg)


    def default_msg(self):
        print("Improper command! Type the command [help] for help or [quit] to exit the program!")
        return 0

    def help(self):
        print("[quit]: Quit the program.\n"
            "[lc]: Load the car dataset. NOTE, YOU MUST LOAD A SET BEFORE RUNNING A TEST!!!\n"
            "[ld]: Load diabetes dataset. NOTE, YOU MUST LOAD A SET BEFORE RUNNING A TEST!!!\n"
            "[lmpg]: Load MPG dataset. NOTE, YOU MUST LOAD A SET BEFORE RUNNING A TEST!!!\n"
            "[knnreg]: Run the knn regression test\n"
            "[classification]: Sets the output of the test to classification\n"
            "[knnbrute]: to run the hardcoded test.\n"
            "[gussian]: to run the gussian test.\n"
            "[terrible_knn]: to run the poor knn test.\n"
            "[knn]: to run k-nearest neighbors test.\n"
            "[dtree]: to run the decsion tree test.\n"
            "[utests]: runs all unit tests.\n"
            "[print]: to print the iris data set.\n")
        return 0

    def add_option(self, option, func):
        self.option_dict[option] = func

    def execute_option(self, option):
        if option in self.option_dict:
            return self.option_dict[option]()
        else:
            return self.option_dict['default']()


    def load_mpg_data(self):
        headers = ["mpg", "cylidners", "displacement", "horse_power", "weight", "acceleration", "model_year", "origin", "car_name"]
        df = read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data',
            names=headers, na_values="?", delim_whitespace=True)

        df = df.fillna(df.mean())

        print("Loaded the MPG data Set.")
        print(df.head(5))

        print("Object types: ", df.dtypes)

        d_target = df['mpg']
        d_data = df.drop('mpg', axis=1)
        d_data = df.drop('car_name', axis=1)

        self.data_train, self.data_test, self.targets_train, self.targets_test = \
            train_test_split(d_data, d_target, test_size=0.33, random_state=46)

        return 0


    def load_diabetes_data(self):

        headers = ["Pregnant", "g-conc", "b-pressure", "fold-thickness", "insulin", "bmi", "pedigree", "age", "class"]
        df = read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
                      names=headers, na_values="0")
        df = df.fillna(df.mean())

        print("Loaded the Diabetes Data Set.")
        print(df.head(5))

        d_target = df['class']
        d_data = df.drop('class', axis=1)

        self.data_train, self.data_test, self.targets_train, self.targets_test = \
            train_test_split(d_data, d_target, test_size=0.33, random_state=46)

        return 0



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


        print("Loaded the UCI Car Data Set.")
        print(df.head(5))

        car_target = df['values']
        car_data = df.drop('values', axis=1)

        self.data_train, self.data_test, self.targets_train, self.targets_test = train_test_split(car_data, car_target, test_size=0.33, random_state=46)

        return 0

    def knn_kdt_test(self):
        print("Running K Nearest Neighbors Test!\n")
        k = int(input("Select a value for k: "))
        classifier =  KNeighborsClassifier(k)

        model = classifier.fit( self.data_train, self.targets_train,)
        predicted = model.predict(self.data_test)
        print("Percent correct: ", compare_prediction(self.targets_test.as_matrix() , predicted), "%\n")
        print("Test Finished")
        return 0

    def knn_reg(self):
        print("Running K Nearest Neighbors Regression Test!\n")
        k = int(input("Select a value for k: "))
        variance = float(input("Select an acceptable variance: "))
        classifier = KNeighborsRegressor(k)

        model = classifier.fit(self.data_train, self.targets_train)

        predicted = model.predict(self.data_test)
        print("Percent correct: ", compare_prediction_variance(self.targets_test.as_matrix(), predicted, variance), "%\n")
        print("Test Finished")
        return 0

    def knn_brute_force(self):

        print("Running terrible KNN Test!")
        k = int(input("Select a value for k: "))
        classifier = terrible_knn_classifier()
        model = classifier.fit(self.targets_train.as_matrix(), self.data_train.as_matrix())
        predicted = model.predict(self.data_test, k)
        print("Percent correct: ", compare_prediction(self.targets_test.as_matrix(), predicted), "%\n")
        print("Test Finished")
        return 0


    def decision_tree(self):

        return 0

    def run_unit_tests(self):

        test_array = [[1, 3, 6], [0, 2, 5], [1, 4, 8], [0, 2, 5]]
        test_target = [0, 1, 0, 1]

        print("Running unit tests!")

        print("Running decision tree Entropy Test!")
        testObj = Decision_Tree_Classifier()
        actual = testObj.entropy([5, 9])
        print("Entroy Test P1= 5/14, P2 = 9/14, Expected = 0.9403, actual: ", actual)
        print("Test Finished\n")

        print("Running split_data test!")
        testObj = Decision_Tree_Classifier()
        target_data = [0, 0, 0, 1, 1, 1]
        test_data = [[1], [2], [3], [4], [5], [6]]
        attribute = 4
        left_data, left_target, right_data, right_target = testObj.split_data(0, test_data, target_data, attribute)
        print("Left_Data: ", left_data)
        print("Left_Target: ", left_target)
        print("Right_Data: ", right_data)
        print("Right_Target: ", right_target)
        print("Test Finished\n")

        print("Running Column tree Entropy Test!")
        testObj = Decision_Tree_Classifier()
        actual = testObj.feature_entropy([0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0])
        print("Entroy Test P1= 5/14, P2 = 9/14, Expected = 0.9403, actual: ", actual)
        print("Test Finished\n")

        print("Running Attribute test!")
        testObj = Decision_Tree_Classifier()

        test_data = [[0, 1, 2], [1, 1, 2], [2, 1, 2], [3, 1, 2], [4, 1, 2], [5, 1, 2]]
        expected = 3

        print("Actual: ", testObj.get_attritube(0, test_data))
        print("Expected: ", expected)

        test_data = [[1, 1, 2], [0, 2, 3], [1, 2, 3]]
        print("Actual: ", testObj.get_attritube(0, test_data))
        print("Expected: ", 1)
        print("Test Finished\n")

        print("Running Best Feature Test")
        testObj = Decision_Tree_Classifier()

        entropy, d_l, t_l, d_r, t_r = testObj.find_best_gain(test_array, test_target)
        print("Test Finished\n")
        print("Data Left: ", d_l)
        print("Test Left: ", t_l)
        print("Data Right: ", d_r)
        print("Test Right: ", t_r)
        print("Entropy: ", entropy)

        print("Unit tests finished!/n")
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


class DT_Node:
    def __init__(self, parent):
        self.parent = parent

    def __init__(self, parent):
        self.parent = parent


    def set_left(self, node):
        self.left = node

    def set_right(self, node):
        self.right = node

class Decision_Tree:
    def __init__(self):
        pass


class Decision_Tree_Classifier:



    def fit(self, data_target, data_train):
        return Decision_Tree_Model()

    # Calculates the entropy based on the number of occurences of a class in a list.
    def entropy(self, class_list):
        # Probability of set 1
        # Probability of set 2
        size = len(class_list)
        total = 0
        pop_count = 0

        for i in range(size):
            pop_count += class_list[i]

        for i in range(size):
            p1 = class_list[i]/pop_count
            total += (-1*p1*math.log(p1,2))

        return total

    # Splits data based off an attribute.
    def split_data(self, feature_index, data, target_data, attribute):
        # Step 1. Split the target_data based off the attribute and feature index
            data_list_left = list()
            target_list_left = list()
            data_list_right = list()
            target_list_right = list()

            # Cycle through each feature and target, split based off attribute
            for i in range(len(target_data)):
                if data[i][feature_index] < attribute:
                # List left keeps all less than
                    data_list_left.append(data[i])
                    target_list_left.append(target_data[i])
                # List right keeps all greater than or equal to
                else:
                    data_list_right.append(data[i])
                    target_list_right.append(target_data[i])

            return data_list_left, target_list_left, data_list_right, target_list_right


    # Calculates the entropy of a given column
    def feature_entropy(self, target_data):

        length = len(target_data)
        target_dict = dict()

        for i in range(length):

            value = target_data[i]
            if value in target_dict:
                target_dict[value] += 1
            else:
                target_dict[value] = 1
            class_list = [ v for v in target_dict.values()]

        return self.entropy(class_list)

    # Returns split data of best info gain.
    # Returns in the order of Entropy, data_list_left, target_list_left, data_list_right, target_list_right
    def find_best_gain(self, data, target_data):

        data_list = list()

        # Find Population count
        pop_count = len(target_data)
        entropy_list = list()

        feature_count = len(data[0])
        # Calculate the information gain for each item
        for i in range(feature_count):
            # Split the data
            attribute = self.get_attritube(i, data)
            data_list_left, target_list_left, data_list_right, target_list_right = self.split_data(i, data, target_data, attribute)
            # Calculate Entropy
            entropy_left = self.feature_entropy(target_list_left)
            entropy_right = self.feature_entropy(target_list_right)
            entropy_left_pop = len(target_list_left) / pop_count
            entropy_right_pop = len(target_list_right) / pop_count
            entropy_combined = entropy_left*entropy_left_pop + entropy_right*entropy_right_pop
            entropy_list.append(entropy_combined)
            data_list.append([ data_list_left, target_list_left, data_list_right, target_list_right])
            # Store information gain into a list.

        max_value = max(entropy_list)
        max_entropy_row_index =  entropy_list.index(max_value)

        chosen_row = data_list[max_entropy_row_index]

        # Returns in the order of Entropy, data_list_left, target_list_left, data_list_right, target_list_right
        return entropy_list[max_entropy_row_index], chosen_row[0], chosen_row[1], chosen_row[2], chosen_row[3]


    def get_attritube(self, feature_index, data):

        row_count = len(data)
        column_list = list()

        for i in range(row_count):
            column_list.append(data[i][feature_index])

        #median = statistics.median(column_list)

        s_list = sorted(column_list)

        median = s_list[int(len(column_list)/2)]

        return median

    def build_node(self, parent, data, target_data, entropy):

        if parent is None:
            parent = DT_Node()

        # If the number of features is 1, stop!
        if len(target_data[0]) == 1:
            return None
        else:
            combined_entropy, data_list_left, target_list_left, data_list_right, target_list_right = \
                self.find_best_gain(data, target_data)





class Decision_Tree_Model:
    def predict(self, test_data):
        pass


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

def compare_prediction_variance(target, prediction, variance):
    length = len(target)
    length2 = len(prediction)
    false_count = 0
    for i in range(length):
        max = target[i] + variance
        min = target[i] - variance
        if prediction[i] < min or prediction[i] > max:
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




# Executes a given option
def execute_option(options):

    user_input = get_option()
    return options.execute_option(user_input)


# Gets the option from the user.
def get_option():
    print("Select an option! (enter the command [help] for help or [quit] to quit!)")
    return input("-> ")




def end_program():
    print("Ending program")
    return 1



# Main entry point for the application.
def main():

    tests = Tests()
    quit = 0

    while quit == 0:
        quit = execute_option(tests)


if __name__ == "__main__":
    main()

