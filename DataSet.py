import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split


class DataSet:
    """
    DataSet
    ----------
    Родительский класс для описания датасетов
    Содержит базовые атрибуты и функции для работы с датасетами
    Может использоваться как непосредственно для работы с датасетами,
    так и как база для решения узкоспециализированных задач

    Атрибуты
    ----------
    data_table : DataFrame from pandas
        контейнер данных датасета
    generated_indexes : list
        список индексов строк датафрема, которые были сгенерированы
        с помощью функции расширения датасета на основе случайной величины

    Методы
    ----------
    get_df()
        Возвращает текущий контейнер данных датасета из data_table
    load_csv()
        Загружает в data_table информацию из csv-файла по заданному пути или url
        см. pandas.read_csv()
    init_by_values(column_names, values=None)
        Инициализирует data_table датафреймом, созданным на основе входных данных
    init_by_df(df)
        Загружает в data_table данные из заданного датафрейма
    save_csv(path)
        Сохраняет текущий контейнер данных датасета data_table в виде csv по заданному пути
        см. pandas.to_csv()
    get_rows(type_of_select='include', indexes=None)
        Возвращает датасет данных, содержащий только заданные строки
        Строки выбираются по условиям параметров
    get_rows_by_value(column_name, value, type_of_select='include')
        Возвращает датафрейм, имеющий (не)соответствие в указанной колонке column_name указанному значению value
    remove_row(value, column_name='index')
        Удаляет строки из датафрейма по указанным параметрам
    add_row(list_of_values)
        Добавляет строку к датафрейму
    add_column(name, values=None)
        Добавляет колонку к датафрейму
    remove_columns(columns_names, is_index=False)
        Удаляет заданные колонки
    """

    def __init__(self):
        self._data_table = pd.DataFrame()
        self._generated_indexes = []

    def get_df(self):
        """
        Получить датафрейм
        :return: pandas.DataFrame
            Текущий контейнер данных датасета из data_table
        """
        return self._data_table

    def load_csv(self, path):
        """
        Загружает в data_table информацию из csv-файла по заданному пути или url
        см. pandas.read_csv()
        :param path: str
            Путь или URL до csv-файла данных
        """
        self._data_table = pd.read_csv(path)
        if self._data_table.columns[0] == 'Unnamed: 0':
            self._data_table = self._data_table.drop(self._data_table.columns[0], axis=1)

    def init_by_values(self, column_names, values=None):
        """
        Инициализирует data_table датафреймом, созданным на основе входных данных
        :param column_names: numpy.ndarray или list
            Список заголовков столбцов
        :param values: numpy.ndarray или list, optional
            Данные датасета
        """
        if type(column_names) == np.ndarray:
            columns = column_names.tolist()
        elif type(column_names) == list:
            columns = column_names
        else:
            raise TypeError('Invalid type of columns names')

        if values is not None:
            if type(values) == np.ndarray:
                values = values.tolist()
            elif type(values) != list:
                raise TypeError('Invalid type of values')

        self._data_table = pd.DataFrame(values, columns=columns)

    def init_by_df(self, df):
        """
        Загружает в data_table данные из заданного датафрейма
        :param df: pandas.DataFrame
            Датафрейм для инициализации
        """
        if type(df) == pd.DataFrame:
            self._data_table = df.copy()

    def save_csv(self, path):
        """
        Сохраняет текущий контейнер данных датасета data_table в виде csv по заданному пути
        см. pandas.to_csv()
        :param path: str
            Путь или URL до csv-файла данных
        """
        self._data_table.to_csv(path)

    def get_rows(self, type_of_select='include', indexes=None):
        """
        Возвращает датасет данных, содержащий только заданные строки
        Строки выбираются по условиям параметров

        :param indexes: list, numpy.ndarray, optional
            Индексы строк
        :param type_of_select: str
            Параметр выбора строк из датасета:
            'include' - вернуть только заданные строки,
            'exclude' - вернуть датасет без заданных строк,
            'random_\d\d' - вернуть заданный процент случайных строк (от 01 до 99)
        :return: pandas.DataFrame
            Сформированный контейнер данных датасета на основе data_table
        """
        if type(indexes) == np.ndarray:
            indexes = indexes.tolist()
        elif type(indexes) != list:
            raise TypeError('Invalid type of indexes')

        df_res = self._data_table.copy()
        if type_of_select == 'include':
            if indexes is None:
                raise AttributeError('Indexes required when include mode selected')
            df_res = df_res[df_res.index.isin(indexes)]
            return df_res
        if type_of_select == 'exclude':
            if indexes is None:
                raise AttributeError('Indexes required when exclude mode selected')
            df_res = df_res.drop(index=indexes, axis=0)
            return df_res
        if re.search(r'random_\d\d', type_of_select):
            percent = int(re.search(r'\d\d', type_of_select).group())
            rows = df_res.index.values
            row_drop, row_ret = train_test_split(rows, test_size=percent/100)
            df_res = df_res[df_res.index.isin(row_ret)]
            return df_res
        raise ValueError('Invalid type of select')

    def get_rows_by_value(self, column_name, value, type_of_select='include'):
        """
        Возвращает датафрейм, имеющий (не)соответствие в указанной колонке column_name указанному значению value
        :param column_name: str
            Название колонки, которй соотвествует значение value
        :param value:
            Искомое значение
        :param type_of_select: str
            Параметр выбора строк из датасета:
            'include' - вернуть только заданные строки,
            'exclude' - вернуть датасет без заданных строк,
            'random_%d%d' - вернуть заданный процент случайных строк (от 01 до 99)
                среди строк, значение которых соответсвует заданному value
        :return: pandas.DataFrame
            Сформированный контейнер данных датасета на основе data_table

        """
        # todo есть возможность расширить поиск колонок по нескольким значениям
        # todo см: https://overcoder.net/q/3301/выберите-строки-в-dataframe-на-основе-значений-в-столбце-в-пандах
        df_res = self._data_table.copy()
        if type_of_select == 'include':
            df_res = df_res.loc[df_res[column_name] == value]
            return df_res
        if type_of_select == 'exclude':
            df_res = df_res.loc[df_res[column_name] != value]
            return df_res
        if re.search(r'random_\d\d', type_of_select):
            percent = int(re.search(r'\d\d', type_of_select).group())
            df_res = df_res.loc[df_res[column_name] == value]
            rows = df_res.index.values
            row_drop, row_ret = train_test_split(rows, test_size=percent / 100)
            df_res = df_res[df_res.index.isin(row_ret)]
            return df_res
        raise ValueError('Invalid type of select')

    def remove_row(self, value, column_name='index'):
        """
        Удаляет строки из датафрейма по указанным параметрам
        :param column_name: str, optional
            Название колонки или слово 'index', если нужно удалить строки по индексам
        :param value:
            Значение ячейки, строку которой необходимо удалить.
            Если column_name=='index', то value хранит список индексов строк для удаления
        """
        if column_name == 'index':
            if type(value) != list:
                raise TypeError('Value MUST BE a list when removing rows by indexes')
            self._data_table = self._data_table.drop(index=value, axis=0)
        else:
            self._data_table = self._data_table.loc[self._data_table[column_name] != value]

    def add_row(self, list_of_values):
        """
        Добавляет строку к датафрейму
        :param list_of_values: list
            Список значений, добавляемых к датафрейму
        """
        new_row = pd.DataFrame([list_of_values], columns=self._data_table.columns)
        self._data_table = self._data_table.append(new_row, ignore_index=True)

    def add_column(self, name, values=None):
        """
        Добавляет колонку к датафрейму
        :param name: str
            Имя создаваемой колонки
        :param values: list, optional,
            Значения создаваемой колонки
        """
        self._data_table[name] = values

    def remove_columns(self, columns_names, is_index=False):
        """
        Удаляет заданные колонки
        :param columns_names: list
            Список имен или номеров колонок для удаления
        :param is_index: bool, optional
            Определяет интерпретировать columns_names как имена или как индексы колонок
        """
        if not is_index:
            self._data_table = self._data_table.drop(columns_names, axis=1)
        else:
            self._data_table = self._data_table.drop(self._data_table.columns[columns_names], axis=1)
