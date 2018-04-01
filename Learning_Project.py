import random
from datetime import datetime
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import multilayer_perceptron
from sklearn.neighbors import KNeighborsRegressor
import numpy as n
import bisect
from operator import itemgetter
from pandas import read_csv
import pandas
import math
import copy
import statistics
from sklearn import preprocessing
import csv

from copy import deepcopy

class Nueral_Net:

    def print_map(self):
        # For each layer
        print("Map\n")
        for i in range(len(self.net)):
            row = ""
            for j in range(len(self.net[i])):
                row += " O "
            row += " -1 "
            print(row)

    # Generate a blank map.
    # Layer Data -> A list of elements. Element 0 correspsonding to layer 0.
    def generate_map(self, layer_data, input_count, threshold):

        self.net = list(list())

        # Generate a layer for every single element in layer_Data, with given nueron count
        layer_count = len(layer_data)
        for i in range(layer_count): # For each layer we want to make
            nueron_count = layer_data[i]
            self.net.append(list())
            for j in range(nueron_count): # For each node in each layer
                # BIAS NODE
                weight_count = 0
                weights = list()
                # + 1 for bias node
                # Weight count is equal to the previous number of neurons in a given layer or the input count
                if i == 0:
                    weight_count = input_count + 1
                else:
                    weight_count = layer_data[i-1] + 1
                for k in range(weight_count): # For each weight in each node
                    weights.append(random.randint(0,10)/10)

                output_node = 0
                if i == layer_count - 1:
                    output_node = 1

                self.net[i].append(Nueron(threshold, weights, output_node))
        self.print_map()

    def process_data(self, features, targets, learning_rate, epoch_count):

        percent_correct = 0
        list_correct = list()

        for i in range(epoch_count):
            print("Starting Epoch: ", i)
            f_copy = deepcopy(features)
            targets_copy = deepcopy(targets)
            results = self.process_epoch( f_copy, targets_copy, learning_rate)

            correct = 0
            for j in range(len(results)):
                # Return first index with a 1
                for k in range(len(results[0])):
                    if results[j][k] == 1 and targets[j] == k:
                        correct += 1
            percent_correct = correct/len(targets)
            list_correct.append(percent_correct)
            print("Percent correct: ", percent_correct)
        return n.asarray(list_correct)


    def process_epoch(self, features, targets, learning_rate):
        results = list()
        layer_count = len(self.net)
        target_count = len(targets)
        correct = 0
        false = 0
        # Each layer output becomes the next input.

        # For every row of data
        for i in range(target_count):
            results.append(list())
            list_layer_inputs = list(list())
            list_layer_inputs.append(features[i])
            # For every single layer...
            for j in range(layer_count):
                # Row of inputs corresponds to j were on.
                row = list_layer_inputs[j]
                row.append(-1) # Append a bias node value
                # For every single Nueron
                list_layer_inputs.append(list())
                for k in range(len(self.net[j])):
                    fire = self.net[j][k].attempt_fire(row)
                    list_layer_inputs[j+1].append(fire)
                    if j == layer_count -1:
                      results[i].append(fire)

            # Back propagation time.
            # First update the error rates of every single output
            list_expected = list()
            out_layer_length = len(self.net[layer_count - 1])
            # Create a list of 0's and put a 1 for which node we expect to fire.
            # For each Nueron in the last layer
            for j in range(out_layer_length):
                if j == targets[i]:
                    list_expected.append(1)
                else:
                    list_expected.append(0)

            # For each node, it's Expected target value is found in list_expected
            for j in range(out_layer_length):
                nueron = self.net[layer_count - 1][j]
                error = (nueron.activation - list_expected[j]) * (nueron.activation)*(1 - nueron.activation)
                nueron.error = error

            # For each layer
            for j in range(len(self.net)):
                # Update the weights
                # Cycle through net in reverse order
                reverse_index = layer_count - 1 - j
                nueron_count = len(self.net[reverse_index])
                # For each Nueron in the layer
                for k in range(nueron_count):
                    # For each weight in the layer
                    weight_count = len(self.net[reverse_index][k].weights)
                    for l in range(weight_count):
                        activation = list_layer_inputs[reverse_index][l] # Weight activation is equal to the previous node's activation
                        self.net[reverse_index][k].weights[l] = self.net[reverse_index][k].weights[l] - learning_rate*activation*self.net[reverse_index][k].error
                # Update the Error rates of each Node
                # If we are at the first layer, end loop
                if reverse_index == 0:
                    break
                else:
                    layer_to_update_index = reverse_index - 1
                    # For each node in the layer to update
                    nodes_in_layer = len(self.net[layer_to_update_index])
                    for k in range(nodes_in_layer):
                        node = self.net[layer_to_update_index][k]
                        # For every single weight, in the k layer corresponding to this node, sum error * weight
                        weight_error_sum = 0
                        for l in range(nueron_count):
                            weight_error_sum += self.net[reverse_index][l].weights[k] * self.net[reverse_index][l].error
                        error_hidden_node = node.activation*(1-node.activation)*weight_error_sum
                        node.error = error_hidden_node

        return results




class Nueron:

    def __init__(self, threshold, weight_list, output=0):
        self.threshold = threshold
        self.weights = weight_list
        self.output = output
        self.activation = 0
        self.error = 0

    def set_weights(self, weights):
        self.weights = weights

    def attempt_fire(self, inputs):

        sum = 0
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]

        if self.output == 0:
            activation = 1 / (1 + math.e**(-sum))
            self.activation = activation
            return activation
        else:
            self.activation = 1 / (1 + math.e**(-sum))
            if sum < self.threshold:
                return 0
            else:
                return 1



# Class for running an option whenever a specified word is passed in.
class Tests:


    def __init__(self):
        self.option_dict = dict()
        self.initialize_program()

    def training_data_to_list(self):

        list_data = self.data_test.values.tolist()
        list_target = self.targets_test.values.tolist()
        return list_data, list_target

    # Initializes the program
    def initialize_program(self):
        print("Welcome to Matthew Brown's Machine Learning Project!")

        self.add_option("nueral", self.run_nueral_net_test)
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
        self.add_option("dt", self.decision_tree_tests)
        self.add_option("li", self.load_iris_almost_best)
        self.set_default_option(self.default_msg)


    def default_msg(self):
        print("Improper command! Type the command [help] for help or [quit] to exit the program!")
        return 0

    def help(self):
        print("[quit]: Quit the program.\n"
            "[nueral]: Runs the Nueral Net Test\n"
            "[dt]: Builds a Decision Tree and prints a text version of it\n"
            "[lc]: Load the car dataset. NOTE, YOU MUST LOAD A SET BEFORE RUNNING A TEST!!!\n"
            "[ld]: Load diabetes dataset. NOTE, YOU MUST LOAD A SET BEFORE RUNNING A TEST!!!\n"
            "[li]: Load Iris dataset."
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

    def decision_tree_tests(self):
        self.load_diabetes_data()
        test_obj = Decision_Tree_Classifier()
        test_data, test_target = self.training_data_to_list()
        tree_root = test_obj.build_tree_2(test_data, test_target)
        test_obj.print_tree(tree_root)



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


    def load_iris_data_best(self):
        print("Loading iris data!")
        iris = setup_iris_data()
        n_array = iris.data #numpy_array
        min_max_scaler = preprocessing.MinMaxScaler()
        array_scaled = min_max_scaler.fit_transform(n_array)
        d_data = pandas.DataFrame(array_scaled)

        self.data_train, self.data_test, self.targets_train, self.targets_test = \
            train_test_split(d_data, iris.target, test_size=0.33, random_state=46)

        self.targets_train = self.targets_train.tolist()
        self.targets_test = self.targets_test.tolist()

        print("Loaded the Iris Data Set!")
        print(d_data.head(5))
        return 0

    def load_iris_almost_best(self):

        headers = ["1", "2", "3", "4", "class"]
        df = read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data', names=headers, na_values="na")
        d_target = df['class']
        d_data = df.drop('class', axis=1)

        cleanup_nums = {'class': {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}}

        d_target = d_target.replace(cleanup_nums, inplace=True)


        self.data_train, self.data_test, self.targets_train, self.targets_test = \
            train_test_split(d_data, d_target, test_size=0.33, random_state=46)

        print(d_target.head(5))
        print("Loaded Iris data set!")

        return 0

    def load_diabetes_data(self):

        headers = ["Pregnant", "g-conc", "b-pressure", "fold-thickness", "insulin", "bmi", "pedigree", "age", "class"]
        df = read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
                      names=headers, na_values="0")

        d_target = df['class']

        d_data = df.drop('class', axis=1)
        d_data = d_data.fillna(df.mean())

        # normalize le data
        n_array = d_data.values #numpy_array
        min_max_scaler = preprocessing.MinMaxScaler()
        array_scaled = min_max_scaler.fit_transform(n_array)
        d_data = pandas.DataFrame(array_scaled)

        d_target = d_target.fillna(0)

        print("Loaded the Diabetes Data Set.")
        print(df.head(5))


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


    def run_nueral_net_test(self):
        print("Running Nueral Net Test!\n")
        net = Nueral_Net()

        data_list_train, data_list_target = self.training_data_to_list()
        training_rate = float(input("Training Rate: "))
        epoch = int(input("Epochs: "))
        hidden_layer = int(input("Hidden Nodes: "))
        file_name = input("Output file: ")
        input_count = len(data_list_train[0])
        net.generate_map([len(data_list_train[0]), hidden_layer, input_count], input_count, 1.2)
        result = net.process_data(data_list_train, data_list_target, training_rate, epoch)

        n.savetxt(file_name, result, delimiter=",")


        print("Percent correct: ", result)

        print("Test Finished")

        net2 = multilayer_perceptron()

        print("Running scilearn test!")


        self.load_iris_data_best()

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
        self.label = -1
        self.left = None
        self.right = None
        self.attribute = -1

    def __init__(self):
        self.parent = None
        self.label = -1
        self.left = None
        self.right = None
        self.attribute = -1

    def set_left(self, node):
        self.left = node
        node.parent = self

    def set_right(self, node):
        self.right = node
        node.parent = self

    def print_data(self):
        if self.label == -1:
           return "A: ", self.attribute
        else:
            return "L: ", self.label


class Decision_Tree:
    def __init__(self):
        pass


class Decision_Tree_Classifier:

    def __init__(self):
        self.counter = 0

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


    def remove_feature(self, data_list, feature_index):

        # Remove the feature just split
        for row in data_list:
            del row[feature_index]

        return data_list


        return data_list_left, target_list_left, data_list_right, target_list_right
    # Calculates the entropy of a given column
    def feature_entropy(self, target_data):

        if len(target_data) == 0:
            return 0

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
        attribute_list = list()
        # Find Population count
        pop_count = len(target_data)
        entropy_list = list()

        feature_count = len(data[0])
        # Calculate the information gain for each item
        for i in range(feature_count):
            # Split the data
            attribute = self.get_attritube(i, data)
            attribute_list.append(attribute)
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

        data_list_left = self.remove_feature(data_list_left, max_entropy_row_index)
        data_list_right = self.remove_feature(data_list_right, max_entropy_row_index)

        # Returns in the order of Entropy, data_list_left, target_list_left, data_list_right, target_list_right
        return attribute_list[max_entropy_row_index], chosen_row[0], chosen_row[1], chosen_row[2], chosen_row[3]


    def get_attritube(self, feature_index, data):

        row_count = len(data)
        column_list = list()

        for i in range(row_count):
            column_list.append(data[i][feature_index])

        #median = statistics.median(column_list)

        s_list = sorted(column_list)

        median = s_list[int(len(column_list)/2)]

        return median

    def most_common(self, list):
        return max(set(list), key=list.count)

    # Code taken from : https://stackoverflow.com/questions/1894846/printing-bfs-binary-tree-in-level-order-with-specific-formatting
    # built-in data structure we can use as a queue
    def print_tree(self, root):
        list = [root]
        while len(list) > 0:
            print ([e.print_data() for e in list])
            list = [e.left for e in list if e.left] + \
                   [e.right for e in list if e.right]

    def build_tree_2(self, data, target_data):
        node = DT_Node()
        #Cycle through every single node in target. if it is positive or all negative, return true.
        # BUG IF TARGET DATA IS EMPTY!
        if len(data) == 0:
            node.label = -2
            return node

        if self.is_same_value(target_data):
            node.label = target_data[0]
            return node

        if len(data[0]) == 0:
            node.label = self.most_common(target_data)
            return node

        # Now split the data
        at, d_left, t_left, d_right, t_right = self.find_best_gain(copy.deepcopy(data), copy.deepcopy(target_data))

        node.attribute = at

        node.set_left(self.build_tree_2(d_left, t_left))
        node.set_right(self.build_tree_2(d_right, t_right))

        return node

    def is_same_value(self, list):
        if len(list) == 0:
            return 0

        val = list[0]
        for i in range(len(list)):
            if list[i] == val:
                continue
            else:
                return False
        return True

    def build_tree(self, parent, data, target_data):

        if parent is None:
            parent = DT_Node()

        # If the number of features is 1, stop!
        if len(target_data[0]) == 1:
            return None
        else:
            combined_entropy, data_list_left, target_list_left, data_list_right, target_list_right = \
                self.find_best_gain(data, target_data)

            if combined_entropy == 0:
                parent.data = target_data[0]
                return parent

            # TODO: MAKE SURE REMOVING FEATURES WORKS 100%!!!
            left_child = DT_Node()
            left_child.parent = parent
            self.build_tree(left_child, data_list_left, target_list_left)
            parent.set_left(left_child)

            right_child = DT_Node()
            right_child.parent = parent
            self.build_tree(right_child, data_list_right, target_list_right)
            parent.set_right(right_child)

            return parent




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

