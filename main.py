# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd

from DataSet import DataSet

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ds = DataSet()

    colmns = np.array([1, 2, 3])
    data = pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

    ds.init_by_df(data)

    print(ds.get_df())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
