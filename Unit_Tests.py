from Learning_Project import Decision_Tree_Classifier
import math

class unit_tests():
    def __init__(self):
        self.tests = dict()
        self.add_test(self.entropy_test, "Entropy")
        self.add_test(self.split_data_test, "Split Data ")
        self.add_test(self.column_tree_entropy_test, "Column_Tree_Entropy")
        self.add_test(self.attribute_test, "Attribute")
        self.add_test(self.best_feature_test, "Best Feature")

    def add_test(self, test, name):
        self.tests[name] = test

    def run_tests(self):
        print("Running all tests!")
        for key in self.tests:
            print("\tRunning test: ", key)

            if self.tests[key]() == 1:
                print("\t\tTest passed!")
            else:
                print("\t\tTest failed!")
        print("\tTest finished!")

    def is_equal(self, expected, actual):
        if math.isclose(expected, actual, rel_tol=0.005, abs_tol=0):
            return 1
        else:
            print("\t\t\tExpected: ", expected)
            print("\t\t\tActual: ", actual)
            return 0


    def entropy_test(self):
        testObj = Decision_Tree_Classifier()
        actual = testObj.entropy([5, 9])
        #Entropy Test P1= 5/14, P2 = 9/14, Expected = 0.9403, actual: ", actual)

        return self.is_equal(0.9403, actual)

    def split_data_test(self):
        testObj = Decision_Tree_Classifier()
        target_data = [0, 0, 0, 1, 1, 1]
        test_data = [[1], [2], [3], [4], [5], [6]]
        attribute = 4
        left_data, left_target, right_data, right_target = testObj.split_data(0, test_data, target_data, attribute)
        print("Left_Data: ", left_data)
        print("Left_Target: ", left_target)
        print("Right_Data: ", right_data)
        print("Right_Target: ", right_target)
        return 1

    def column_tree_entropy_test(self):
        testObj = Decision_Tree_Classifier()
        actual = testObj.feature_entropy([0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0])
        print("Entroy Test P1= 5/14, P2 = 9/14, Expected = 0.9403, actual: ", actual)
        return 1

    def attribute_test(self):
        testObj = Decision_Tree_Classifier()

        test_data = [[0, 1, 2], [1, 1, 2], [2, 1, 2], [3, 1, 2], [4, 1, 2], [5, 1, 2]]
        expected = 3

        print("Actual: ", testObj.get_attritube(0, test_data))
        print("Expected: ", expected)

        test_data = [[1, 1, 2], [0, 2, 3], [1, 2, 3]]
        print("Actual: ", testObj.get_attritube(0, test_data))
        print("Expected: ", 1)
        return 1

    def best_feature_test(self):
        testObj = Decision_Tree_Classifier()
        test_array = [[1, 3, 6], [0, 2, 5], [1, 4, 8], [0, 2, 5]]
        test_target = [0, 1, 0, 1]
        #entropy, d_l, t_l, d_r, t_r = testObj.find_best_gain(test_array, test_target)
        #print("Test Finished\n")
        #print("Data Left: ", d_l)
        #print("Test Left: ", t_l)
        #print("Data Right: ", d_r)
        #print("Test Right: ", t_r)
        #print("Entropy: ", entropy)
        return 0
        #return 1



def main():
    tests = unit_tests()
    tests.run_tests()


if __name__ == "__main__":
    main()

