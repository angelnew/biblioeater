import numpy as np
import random


def describe(numeric_list, description=""):
    np_array = np.array(numeric_list)

    print("\nSummary of {}".format(description))
    print("Average: {:.2f}".format(np.average(np_array)))

    print("Quartiles")
    print(np.percentile(np_array, np.arange(0, 100, 25)))


if __name__ == "__main__":
    f1 = [random.randint(10, 1000) for i in range(100)]
    describe(f1, "Random")


