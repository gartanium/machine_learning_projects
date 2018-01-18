import random
from datetime import datetime
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split


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

    # Default option to be executed when an improper option is given
    def set_default_option(self, func):
        self.option_dict['default'] = func

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
          "to run the gussian test, and option [print] to print "
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

