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

    colmns = ['A', 'B', 'C']
    data = pd.DataFrame([[1, 2, 3]], columns=colmns)

    ds.init_by_df(data)
    ds.add_row([4, 5, 6])
    ds.add_column('D', [10, 11])
    ds.remove_columns([0, 2], True)
    #ds.load_csv('Z:/sign_dump.csv')
    #df = ds.get_rows('exclude', [0, 1, 2])
    print(ds.get_df())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
