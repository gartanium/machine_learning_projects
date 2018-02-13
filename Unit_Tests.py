from Learning_Project import Decision_Tree_Classifier
from Learning_Project import Tests
from Learning_Project import DT_Node
import math

class unit_tests():
    def __init__(self):
        self.tests = dict()
        self.add_test(self.entropy_test, "Entropy")
        self.add_test(self.split_data_test, "Split Data ")
        self.add_test(self.column_tree_entropy_test, "Column_Tree_Entropy")
        self.add_test(self.attribute_test, "Attribute")
        self.add_test(self.best_feature_test, "Best Feature")
        self.add_test(self.remove_feature_test, "Remove Feature Test")
        self.add_test(self.same_val_test, "Same Val test!")
        self.add_test(self.most_common_test, "Most Common Test")
        self.add_test(self.build_tree_test, "Build Tree Test")
        self.add_test(self.man_tree_test, "Man_Tree_Test")
        self.add_test(self.comprehensive_tree_test, "Comprehensive tree test")

    def man_tree_test(self):
        test_obj = DT_Node()
        test_obj.set_left(DT_Node())
        test_obj.left.set_left(DT_Node())
        test_obj.left.left.label = 2
        return self.is_equal(test_obj.left.left.label, 2)


    def comprehensive_tree_test(self):
        program = Tests()
        program.execute_option("ld")
        test_obj = Decision_Tree_Classifier()
        test_data, test_target = program.training_data_to_list()
        tree_root = test_obj.build_tree_2(test_data, test_target)
        test_obj.print_tree(tree_root)
        return 0

    def build_tree_test(self):
        test_obj = Decision_Tree_Classifier()
        test_data, test_target = self.setup_data()
        tree_root = test_obj.build_tree_2(test_data, test_target)
        return 0

    def most_common_test(self):
        test_obj = Decision_Tree_Classifier()
        test_data = [1, 2, 3, 3, 2, 3]
        actual = test_obj.most_common(test_data)
        return self.is_equal(3, actual)

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

    def is_true(self, statement):
        if statement is True:
            return 1
        else:
            print("\t\t\tStatement was false")
            return 0

    def same_val_test(self):
        test_obj = Decision_Tree_Classifier()
        test_data = [1, 1, 2]
        test_data2 = [1, 1, 1]

        return self.is_true(test_obj.is_same_value(test_data2)) and not test_obj.is_same_value(test_data)

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
        return self.is_true(left_target == [0, 0, 0]) and left_data == [[1], [2], [3]] and right_target == [1, 1, 1] and right_data == [[4], [5], [6]]

    def remove_feature_test(self):
        test_array, test_target = self.setup_data()
        testObj = Decision_Tree_Classifier()
        test_array = testObj.remove_feature(test_array, 0)
        return self.is_true(test_array == [[3, 6], [2, 5], [4, 8], [2, 5]])

    def column_tree_entropy_test(self):
        testObj = Decision_Tree_Classifier()
        actual = testObj.feature_entropy([0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0])
        return self.is_equal(0.9403, actual)

    def attribute_test(self):
        testObj = Decision_Tree_Classifier()

        test_data = [[0, 1, 2], [1, 1, 2], [2, 1, 2], [3, 1, 2], [4, 1, 2], [5, 1, 2]]
        expected_1 = 3


        actual_1 = testObj.get_attritube(0, test_data)


        test_data = [[1, 1, 2], [0, 2, 3], [1, 2, 3]]
        actual_2 = testObj.get_attritube(0, test_data)
        expected_2 = 1

        return self.is_equal(actual_1, expected_1) and self.is_equal(actual_2, expected_2)

    def setup_data(self):
        test_array = [[1, 3, 6], [0, 2, 5], [1, 4, 8], [0, 2, 5]]
        test_target = [0, 1, 0, 1]
        return test_array, test_target

    def best_feature_test(self):
        test_obj = Decision_Tree_Classifier()
        test_array, test_target = self.setup_data()
        at, d_l, t_l, d_r, t_r = test_obj.find_best_gain(test_array, test_target)

        return 0



def main():
    tests = unit_tests()
    tests.run_tests()


if __name__ == "__main__":
    main()

