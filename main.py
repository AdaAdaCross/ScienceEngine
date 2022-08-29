# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd

from DataSet import DataSet
from Signatures import Signatures

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ds = Signatures()
    ds.load_csv('Z:/ds_16.csv', num_of_harms=16)
    #ds.normalize('user_id', 'envelopes_2', 'column', r'\w+_diff\d_\w+_\d')
    #ds.generate_from_database('localhost', 'root', '192837465564738291yashka', 16, 'Fourier', True)
    print(ds.get_df())
    #ds.save_csv('Z:/ds_16.csv')
    ds.visualize('', '', None, None)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
def test_dataset():
    ds = DataSet()

    colmns = ['A', 'B', 'C', '0', '1', '2']
    str_colmns = " ".join(colmns)
    # print(str_colmns)
    data = pd.DataFrame([[1, 2, 3, 1, 2, 3]], columns=colmns)

    ds.init_by_df(data)
    ds.add_row([4, 5, 6, 4, 5, 6])
    # todo add few columns in row
    ds.add_column('D', [10, 11])
    # print(ds.get_df())

    data2 = pd.DataFrame([[10, 20], [30, np.NaN]], columns=['X', 'Y'])
    data3 = pd.DataFrame([[10, np.NaN], [30, np.NaN]], columns=['U', 'Z'])

    list_data = [data2, data3]
    # ds.set_value(3, 0, 100)
    # ds.load_csv('Z:/sign_dump.csv')
    # df = ds.get_rows('exclude', [0, 1, 2])
    ds.concatenate(list_data, 'right')
    # print(ds.get_df())
    print('_________')

    ds.extend_dataset('A', 3, 0.5, verbose=True)
    # print(ds.get_df())
    # ds.normalize('A', 'statistical', 'column')
    print(ds.get_df())
