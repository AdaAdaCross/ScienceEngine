import pandas as pd
import numpy as np
import re
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


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
    normalization_list : list
        контейнер, хранящий тип нормализации для каждой колонки
        в формате [идентификатор колонки, тип нормализации, направление]

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
    get_columns(columns_names, type_of_select)
        Возвращает датафрейм, содержащий выбранные колонки
    get_value(column_name, row_index)
        Возвращает значение в указанной ячейке
    set_value(column_name, row_index, value)
        Устанавливает значение в указанной ячейке
    concatenate(dataframes, direction)
        Присоединяет к исходному датафрейму 1 или несколько датафреймов с заданной стороны
    get_train_data(label_column, data_columns=None, data_shape=None, test_size=0.3, is_onehotenc=True)
        Возвращает содержимое датасета в виде, готовым для пердачи классификатору на обучение
    extend_dataset(label_column, rate, eps, type_of_generator='uniform random', verbose=False)
        Расширяет датасет за счет генерации новых строк на основе заданных параметров и генератора
    get_generated_rows(indexes_only=True)
        Возращает индексы строк или датасет, которые были сгенерированые функцией extend_dataset
    replace_values(before, after)
        Заменяет все значения before на after
    normalize(label_column, type_of_normalize, direction, columns=None)
        Нормализует данные исходного датафрейма в соотвествии с параметрами
    get_info()
        Выводит информацию о датасете
    """

    def __init__(self):
        self._data_table = pd.DataFrame()
        self._generated_indexes = []
        self._normalization_list = []

    @staticmethod
    def __normalize_by_type(list_of_data, type_of_normalize):
        """
        Private. Нормализирует лист данных по выбранному методу
        :param list_of_data: list[]
            Данные для нормализации
        :param type_of_normalize: str, optional
            Выбор типа нормализации:
            'max' - нормализация относительно максимума
            'linear' - линейная нормализация
            'statistical' - статистическая нормализация
        :return: list, list
            Возвращает лист нормализованных значений и лист метаданных использованной нормализации
        """
        if type_of_normalize == 'max':
            norm = [float(i) / max(list_of_data) for i in list_of_data]
            return norm, max(list_of_data)
        if type_of_normalize == 'linear':
            norm = [(float(i) - min(list_of_data)) / (max(list_of_data) - min(list_of_data)) for i in list_of_data]
            return norm, [max(list_of_data), min(list_of_data)]
        if type_of_normalize == 'statistical':
            aver = np.mean(list_of_data, axis=0)
            sigma = np.std(list_of_data, axis=0)
            norm = [((float(i) - aver) / sigma) for i in list_of_data]
            return norm, [aver, sigma]
        raise ValueError('Invalid type of normalize')

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
        self._data_table.to_csv(path, index=False)

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
            'random_%d%d' - вернуть заданный процент случайных строк (от 01 до 99)
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

    def get_columns(self, columns_names, type_of_select):
        """
        Возвращает датафрейм, содержащий выбранные колонки
        :param columns_names: str, list[str], list[int]
            Контейнер, содержащий в себе описание требуемых колонок из датасета
            Если задана как строка - описание в виде регулярного выражения, которому должны соответствовать
                заголовки наименований колонок
            Если как список - содержит список необходимых колонок
        :param type_of_select: str
            Параметр выбора строк из датасета:
            'include' - вернуть только заданные столбцы,
            'exclude' - вернуть датасет без заданных столбцы
        :return: pandas.DataFrame
            Сформированный контейнер данных датасета на основе data_table
        """
        df_res = self._data_table.copy()
        selected_columns = []
        if type(columns_names) == str:
            list_columns = " ".join(df_res.columns)
            selected_columns = re.findall(columns_names, list_columns)
        elif type(columns_names) == list:
            if type(columns_names[0]) == int:
                selected_columns = res_data.columns[columns_names]
            elif type(columns_names[0]) == str:
                selected_columns = columns_names
            else:
                raise TypeError('Invalid type of data_columns list')
        else:
            raise TypeError('Invalid type of data_columns')

        if type_of_select == 'include':
            df_res = df_res[selected_columns]
            return df_res
        if type_of_select == 'exclude':
            df_res = df_res.drop(selected_columns, axis=1)
            return df_res
        raise ValueError('Invalid type of select')

    def get_value(self, column_name, row_index):
        """
        Возращает значение в указанной ячейке
        :param column_name: str, int
            Индекс или название колонки
        :param row_index: int
            Индекс строки
        :return:
            Значение, хранящееся в ячейке
        """
        if type(column_name) == str:
            return self._data_table.at[row_index, column_name]
        if type(column_name) == int:
            return self._data_table.iat[row_index, column_name]
        raise TypeError('Invalid type of column name')

    def set_value(self, column_name, row_index, value):
        """
        Устанавливает значение в указанной ячейке
        :param column_name: str, int
            Индекс или название колонки
        :param row_index: int
            Индекс строки
        :param value:
            Значение ячейки
        """
        if type(column_name) == str:
            self._data_table.at[row_index, column_name] = value
        elif type(column_name) == int:
            self._data_table.iat[row_index, column_name] = value
        else:
            raise TypeError('Invalid type of column name')

    def concatenate(self, dataframes, direction):
        """
        Присоединяет к исходному датафрейму 1 или несколько датафреймов с заданной стороны
        При несовпадении названий колонок пустые места будут заполнены Nan
        При пересечении имен колонок при добавлении слева или справа значения из присоединяемых
        датафреймов будут проигнорированы
        :param dataframes: list, pandas.DataFrame
            Датафрейм, который будет присоединен к основному
        :param direction: str
            Направление конкатинации относительно основного датафрейма:
            'left' - dataframes будет присоединен левее относительно основного
            'right' - dataframes будет присоединен правее относительно основного
            'up' - dataframes будет присоединен выше относительно основного
            'down' - dataframes будет присоединен снизу относительно основного
        """
        if type(dataframes) == pd.DataFrame:
            if direction == 'left':
                dataframes[self._data_table.columns] = self._data_table.values
                self._data_table = dataframes
            elif direction == 'right':
                self._data_table[dataframes.columns] = dataframes.values
            elif direction == 'up':
                dataframes = dataframes.append(self._data_table, ignore_index=True)
                self._data_table = dataframes
            elif direction == 'down':
                self._data_table = self._data_table.append(dataframes, ignore_index=True)
            else:
                raise ValueError('Invalid direction')
        elif type(dataframes) == list:
            if direction == 'left':
                for dataframe in dataframes:
                    dataframe[self._data_table.columns] = self._data_table.values
                    self._data_table = dataframe
            elif direction == 'right':
                for dataframe in dataframes:
                    self._data_table[dataframe.columns] = dataframe.values
            elif direction == 'up':
                for dataframe in dataframes:
                    dataframe = dataframe.append(self._data_table, ignore_index=True)
                    self._data_table = dataframe
            elif direction == 'down':
                for dataframe in dataframes:
                    self._data_table = self._data_table.append(dataframe, ignore_index=True)
            else:
                raise ValueError('Invalid direction')
        else:
            raise TypeError('Invalid type of dataframes')

    def get_train_data(self, label_column, data_columns=None, data_shape=None, test_size=0.3, is_onehotenc=True):
        """
        Возвращает содержимое датасета в виде, готовым для пердачи классификатору на обучение
        :param label_column: str
            Строка, содержащая наименование колонки с идентификаторами
        :param data_columns: str, list[str], list[int]
            Контейнер, содержащий в себе описание требуемых для обучения колонок из датасета
            Если задана как строка - описание в виде реуглярного выражения, которому должны соответствовать
                заголовки наименований колонок
            Если как список - содержит список необходимых колонок
        :param data_shape: list[int], optional
            Содержит описание кортежа измерений массива NxMx...xZ обучающей и тестовой выборок
            см. numpy.ndarray.shape
        :param test_size: float
            Соотношение X_test к общему размеру датасета
        :param is_onehotenc: bool, optional
            Определяет, необходимо ли приводить идентификаторы к OneHotEncoded виду
        :return: numpy.ndarray[4]
            Возвращает в указанном порядке:
            X_train - массив с выборкой для обучения
            X_test - массив с выборкой для валидации
            Y_train - массив с идентификаторами X_train
            Y_test - массив с идентификаторами X_test
        """
        res_data = self._data_table.copy()
        if data_columns is not None:
            if type(data_columns) == str:
                list_columns = " ".join(res_data.columns)
                need_columns = re.findall(data_columns, list_columns)
                if label_column not in need_columns:
                    need_columns.append(label_column)
                res_data = res_data[need_columns]
            elif type(data_columns) == list:
                if type(data_columns[0]) == int:
                    if res_data.columns.get_loc(label_column) not in data_columns:
                        data_columns.append(res_data.columns.get_loc(label_column))
                    res_data = res_data[res_data.columns[data_columns]]
                elif type(data_columns[0]) == str:
                    if label_column not in data_columns:
                        data_columns.append(label_column)
                    res_data = res_data[data_columns]
                else:
                    raise TypeError('Invalid type of data_columns list')
            else:
                raise TypeError('Invalid type of data_columns')
        X_train, X_test = train_test_split(res_data, test_size=test_size)
        Y_train = X_train[label_column]
        X_train = X_train.drop(columns=label_column)
        Y_test = X_test[label_column]
        X_test = X_test.drop(columns=label_column)
        if is_onehotenc:
            Y_train = to_categorical(Y_train)
            Y_test = to_categorical(Y_test)
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        Y_test = Y_test.to_numpy()
        Y_train = Y_train.to_numpy()
        if data_shape is not None:
            data_shape.insert(0, X_train.shape[0])
            X_train = X_train.reshape(data_shape)
            X_test = X_test.reshape(data_shape)

        return X_train, X_test, Y_train, Y_test

    def extend_dataset(self, label_column, rate, eps, type_of_generator='uniform random', verbose=False):
        """
        Расширяет датасет за счет генерации новых строк на основе заданных параметров и генератора
        :param label_column: str
            Строка, содержащая наименование колонки с идентификаторами
        :param rate: int
            Определяет во сколько раз расширить датасет
        :param eps: float
            Определяет размер отклонения влево и вправо сгенерированных значений от исходных
        :param type_of_generator: str
            Наименование типа генератора.
            Поддерживаемые генераторы:
            'uniform random' - случайное равномерное распределение
        :param verbose: bool, optional
            Вывод прогресса выполнения операции
        """

        if type_of_generator != 'uniform random':
            raise ValueError('Unsupported generator')

        # todo сделать больше генераторов

        res_data = self._data_table.copy()
        labels_for_data = res_data[label_column]
        res_data = res_data.drop(columns=label_column)
        noise_data = []
        offset = -1
        new_labels = []
        generated_index = res_data.shape[0]

        for i in range(res_data.shape[0]):
            for k in range(rate - 1):
                noise_data.append([])
                self._generated_indexes.append(generated_index)
                generated_index += 1
                offset += 1
                new_labels.append(labels_for_data.values[i])
                for j in range(res_data.shape[1]):
                    value = res_data.values[i][j]
                    value = value + (random.uniform(-eps, eps) * value)
                    noise_data[offset].append(value)
            if verbose:
                print("\rExtended", i+1, "out of", res_data.shape[0], "rows", end='')
        if verbose:
            print()

        gen_data = pd.DataFrame(data=noise_data, columns=res_data.columns)
        gen_data[label_column] = new_labels
        res_data[label_column] = labels_for_data
        self._data_table = res_data.append(gen_data, ignore_index=True)

    def get_generated_rows(self, indexes_only=True):
        """
        Возращает индексы строк или датасет, которые были сгенерированые функцией extend_dataset
        :param indexes_only: bool, optional
            Параметр, отвечающий за вид ответа функции:
            True - функция вернет лист индексов сгенерированных строк
            False - функция вернет датафрейм, состоящий из сгенерированных строк
        :return: list[int], pandas.DataFrame
            Возвращает индексы строк или датасет, которые были сгенерированы
        """
        if indexes_only:
            return self._generated_indexes
        else:
            return self._data_table[self._data_table.index.isin(self._generated_indexes)]

    def replace_values(self, before, after):
        """
        Заменяет все значения before на after
        :param before:
            Значение, которое необходимо заменить
        :param after:
            Значение, на которое необходимо заменить before
        :return: bool
            Возвращает True, если хотя бы одно значение было найдено и заменено
            False, если ни одного совпадения не найдено
        """
        if before not in self._data_table.values:
            return False
        self._data_table = self._data_table.replace(before, after)
        return True

    def normalize(self, label_column, type_of_normalize, direction, columns=None, metadata_save_path=None):
        """
        Нормализует данные исходного датафрейма в соотвествии с параметрами
        :param label_column: str
            Строка, содержащая наименование колонки с идентификаторами
        :param type_of_normalize: str, optional
            Выбор типа нормализации:
            'max' - нормализация относительно максимума
            'linear' - линейная нормализация
            'statistical' - статистическая Z-нормализация
        :param direction: str, optional
            Выбор направления нормализации:
            'row' - нормализовать по-строчно
            'column' - нормализовать по столбцам
        :param columns: str, list[str], list[int]
            Контейнер, содержащий в себе описание требуемых для обучения колонок из датасета
            Если задана как строка - описание в виде реуглярного выражения, которому должны соответствовать
                заголовки наименований колонок
            Если как список - содержит список необходимых колонок
        :param metadata_save_path: str, optional
            Строка, содержащая путь до места сохранения информации, уничтожаемой при нормализации, но позволяющей
            восстановить исходные данные подписи после нормализации
        """
        res_data = self._data_table.copy()
        res_data = res_data.drop(label_column, axis=1)
        if columns is not None:
            if type(columns) == str:
                list_columns = " ".join(res_data.columns)
                need_columns = re.findall(columns, list_columns)
                res_data = res_data[need_columns]
            elif type(columns) == list:
                if type(columns[0]) == int:
                    res_data = res_data[res_data.columns[columns]]
                elif type(columns[0]) == str:
                    res_data = res_data[columns]
                else:
                    raise TypeError('Invalid type of columns list')
            else:
                raise TypeError('Invalid type of columns')
        if direction == 'row':
            for index, row in res_data.iterrows():
                res_data.iloc[index], list_norm_param = DataSet.__normalize_by_type(row.values, type_of_normalize)
        elif direction == 'column':
            for column in res_data:
                res_data[column], list_norm_param = DataSet.__normalize_by_type(res_data[column].values, type_of_normalize)
        else:
            raise ValueError('Invalid direction')
        for column in res_data.columns:
            self._data_table[column] = res_data[column]
            self._normalization_list.append([column, type_of_normalize, direction]+list_norm_param)

        if metadata_save_path is not None:
            list_trans_norm = list(map(list, zip(*self._normalization_list)))
            print(list_trans_norm)
            labels_list = list_trans_norm[0]
            list_trans_norm.pop(0)
            data_list = list_trans_norm
            df_for_clean = pd.DataFrame(data = data_list, columns=labels_list)
            df_for_clean.to_csv(metadata_save_path, index=False)


    def get_info(self, column_info=False, normalization_info=False):
        """
        Выводит информацию о датасете
        :return: str
            Возвращает строку с информацией о данных
        """
        info_str = 'Dataset info:\n1) shape ' + str(self._data_table.shape) + '\n'
        info_str += '2) number of generated rows ' + str(len(self._generated_indexes)) + '\n'
        if column_info==True:
            info_str += '3) types of values in columns:\n'
            for column in self._data_table.columns:
                info_str += '\t' + column + ' : ' + str(type(self._data_table[column].values[0]))
                if np.isnan(self._data_table[column].values).any():
                    info_str += ' (contains NaN)'
                info_str += '\n'
        print(info_str)
        return info_str

        # todo сохранять матрицу с наименованиями нормализаций для каждой ячейки