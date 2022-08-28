import pandas as pd
import numpy as np
from DataSet import DataSet
from scipy.fft import *
import math
import cmath
import pywt
import mysql.connector as msc
import re


class Signatures(DataSet):
    """
    Signatures
    ----------
    Дочерний класс от DataSet для описания датасетов сигнатур подписи
    Содержит атрибуты и функции для работы с датасетами сигнатур подписи

    Атрибуты
    ----------
    data_table : DataFrame from pandas
        контейнер данных датасета
    generated_indexes : list
        список индексов строк датафрема, которые были сгенерированы
        с помощью функции расширения датасета на основе случайной величины
    p_type: str
        Содержит информацию о типе параметризации
    number_of_harms : int
        Содержит информацию о количестве гармоник

    Методы
    ----------
    load_csv(path, p_type='Furie', num_of_harms=None)
        Загружает в data_table информацию из csv-файла по заданному пути или url
        см. pandas.read_csv()
    generate_from_database(host, user, password, number_harms, par_type, trace=False)
        Генерирует датафрейм из базы данных sign_db с преобразованием данных по выбранному методу
    normalize(label_column, type_of_normalize, direction='column', columns=None)
        Нормализует данные исходного датафрейма в соотвествии с параметрами
    """
    def __init__(self):
        super().__init__()
        self._p_type = None
        self._number_of_harms = None
        self._envelopes = []

    @staticmethod
    def __calculate_harmonics(sequence, number):
        """
        Вычисляет гармоники заданной последовательности методом Фурье
        :param sequence: list[], np.ndarray
            Заданная последовательность
        :param number: int
            Количество генерируемых гармоник
        :return: {list, list, list}
            Возвращает кортеж списков:
            modulus - список модулей
            phase - список фаз
            frequencies - список частот
        """
        sampling_period = 0.005
        eps = 1e-10

        modulus = []
        phase = []
        frequencies = []

        avg = np.mean(sequence)
        sequence[:] = [x - avg for x in sequence]
        fft_res = rfft(sequence)
        fft_size = len(sequence)
        fft_freq = rfftfreq(fft_size, sampling_period)

        koefs = []
        for i in range(1, number):
            koefs.append(i)

        if avg > 0:
            modulus.append(avg)
            phase.append(0)
        else:
            modulus.append(abs(avg))
            phase.append(math.pi)
        frequencies.append(fft_freq[1])

        fft_phi = []
        spi = []

        for x in fft_res:
            (r, phi) = cmath.polar(x)
            spi.append(r)
            fft_phi.append(phi)

        for i in koefs:
            # todo добавить возможность сохранить модули без логарифмирования
            modulus.append(math.log10(spi[i] + eps))
            phase.append(fft_phi[i])
            frequencies.append(i)

        return modulus, phase, frequencies

    @staticmethod
    def __dwthaar(sequence, number):
        """
        Вычисляет гармоники заданной последовательности методом Вейвлета Хаара
        :param sequence:list[], np.ndarray
            Заданная последовательность
        :param number: int
            Количество генерируемых гармоник
        :return: list
            Возвращает преобразованные данные
        """
        # todo  добавить возможность сохранить вейвлет с логарифмированием
        while (len(sequence) > (2 * number)):
            (sequence, cD) = pywt.dwt(sequence, 'haar')
            sequence = sequence / math.sqrt(2)
        if (len(sequence) == number):
            return sequence
        delta = number - int(len(sequence) / 2)
        offset = int(delta / 2)
        result = []
        for i in range(offset):
            result.append(sequence[i])
        for i in range(int(len(sequence) / 2) - delta):
            value = (sequence[offset] + sequence[offset + 1]) / 2
            result.append(value)
            offset = offset + 2
        while ((offset < len(sequence)) & (len(result) < number)):
            result.append(sequence[offset])
            offset = offset + 1
        return result

    @staticmethod
    def __calculate_envelope(sequence, number_of_envelop):
        """
        Возвращает преобразованную конвертами исходную последовательность
        :param sequence: list[], np.ndarray
            Заданная последовательность
        :param number_of_envelop: int
            Разрядность генерируемых конвертов
            Результирующие значения будут представлены в виде [-%d, ..., -1, 0, 1, ..., %d]
        :return: {list, list}
            Возвращает кортеж из листов:
            result - список преобразованных данных
            envelope - список, состоящий из:
                envelope[0] - среднего значения
                envelope[1] - стандартное отклонение
        """
        envelope = []
        result = []
        envelope.append(np.mean(sequence))
        envelope.append(np.std(sequence))
        measure = [0]
        intervals = []
        for i in range(1, number_of_envelop + 1):
            measure = [-i] + measure + [i]
        for val in measure:
            if val!=0:
                intervals.append(envelope[0] + val * envelope[1])
        for value in sequence:
            is_inf = True
            for i in range(len(intervals)):
                if value<intervals[i]:
                    is_inf = False
                    break
            if is_inf:
                result.append(measure[-1])
            else:
                result.append(measure[i])
        return result, envelope



    def load_csv(self, path, p_type='Furie', num_of_harms=None):
        """
        Загружает в data_table информацию из csv-файла по заданному пути или url
        см. pandas.read_csv()
        :param path: str
            Путь или URL до csv-файла данных
        :param p_type: str
            Содержит информацию о типе параметризации
        :param num_of_harms:int
            Содержит информацию о количестве гармоник

        """
        super().load_csv(path)
        self._p_type = p_type
        self._number_of_harms = num_of_harms

    def generate_from_database(self, host, user, password, number_harms, par_type, trace=False):
        """
        Генерирует датафрейм из базы данных sign_db с преобразованием данных по выбранному методу
        База данных sign_db - стандартная база данных рукописных подписей, собранная
        на факультете безопасности ТУСУР. Данные представляют собой совокупность координат точек
        (X, Y и Z), силы нажатия на графический планшет и углов наклона пера относительно плоскости
        планшета каждые 5 миллисекунд в течение нанесения подписи. В результате снималось различное
        количество точек, от 160 до 2500 на одну подпись у различчных пользователей.
        :param host: str
            Адрес хоста, на котором развернута БД
        :param user: str
            Имя пользователя, имеющего доступ к БД
        :param password: str
            Пароль пользователя, имеющего доступ к БД
        :param number_harms: int
            Количество гармоник
        :param par_type: str
            Тип параметризации:
            'Fourier'- быстрое преобразрование Фурье
            'Wavelet' - перобразование Вейвлетом Хаара
        :param trace: bool, optional
            Параметр трассировки выполнения:
            True - выполнять трассировку
            False - не выполнять трассировку
        """
        # Читаем данные из sign_db - база данных подписей, собранная на ФБ
        conn = msc.connect(host=host, user=user, passwd=password, db='sign_db')
        cur = conn.cursor()
        cur.execute("SELECT id_sign FROM signatures")
        id_signs = []
        signs_data = []
        sign_owners = []
        for value in cur.fetchall():
            id_signs.append(value[0])
        for sign in id_signs:
            query = "SELECT * FROM packets_norm WHERE id_sign = " + str(sign)
            cur.execute(query)
            sign_data = []
            for itr in range(6):
                sign_data.append([])
            for record in cur.fetchall():
                for itr in range(6):
                    sign_data[itr].append(record[itr + 2])
            if len(sign_data[0]) == 0:
                print(str(sign) + ' no data')
            signs_data.append(sign_data)
            if trace:
                print('sign #' + str(sign) + ' - read')
            cur.execute("SELECT id_user FROM signatures WHERE id_sign = " + str(sign))
            user = cur.fetchall()
            sign_owners.append(user[0][0])
            if trace:
                print('sign owner - ' + str(user[0][0]))

        # параметризация
        if par_type == 'Fourier':
            params_data = [[]]
            signs_freq = []
            offset = 0
            if trace:
                print('processing...')
            for sign in signs_data:
                params_data[offset] = []
                for i in range(6):
                    (modulus, phase, frequencies) = self.__calculate_harmonics(sign[i], number_harms)
                    for mod in modulus:
                        params_data[offset].append(mod)
                    for pha in phase:
                        params_data[offset].append(pha)
                    if i == 0:
                        signs_freq.append(frequencies[0])
                    (modulus, phase, frequencies) = self.__calculate_harmonics(np.diff(sign[i], 1), number_harms)
                    for mod in modulus:
                        params_data[offset].append(mod)
                    for pha in phase:
                        params_data[offset].append(pha)
                    (modulus, phase, frequencies) = self.__calculate_harmonics(np.diff(sign[i], 2), number_harms)
                    for mod in modulus:
                        params_data[offset].append(mod)
                    for pha in phase:
                        params_data[offset].append(pha)
                params_data.append([])
                offset = offset + 1
            params_data = params_data[:-1]

            # формирование заголовков
            header = []
            for name in ['X', 'Y', 'Z', 'P', 'Al', 'Az']:
                for div in ['diff0', 'diff1', 'diff2']:
                    for value_type in ['module', 'phase']:
                        for i in range(number_harms):
                            header.append(name+'_'+div+'_'+value_type+'_'+str(i+1))

            # сохранение в DataFrame
            self._data_table = pd.DataFrame(data=params_data, columns=header)
            self._data_table['user_id'] = sign_owners
            self._data_table['frequency'] = signs_freq
            self._p_type = par_type
            self._number_of_harms = number_harms

        elif par_type == 'Wavelet':
            params_data = [[]]
            offset = 0
            for sign in signs_data:
                params_data[offset] = []
                for i in range(6):
                    sequence = sign[i]
                    cA = self.__dwthaar(sequence, number_harms)
                    for value in cA:
                        params_data[offset].append(value)
                    cA = self.__dwthaar(np.diff(sequence, 1), number_harms)
                    for value in cA:
                        params_data[offset].append(value)
                    cA = self.__dwthaar(np.diff(sequence, 2), number_harms)
                    for value in cA:
                        params_data[offset].append(value)
                params_data.append([])
                offset = offset + 1
            params_data = params_data[:-1]
            # формирование заголовков
            header = []
            for name in ['X', 'Y', 'Z', 'P', 'Al', 'Az']:
                for div in ['diff0', 'diff1', 'diff2']:
                    for i in range(number_harms):
                        header.append(name + '_' + div + '_' + str(i + 1))

            # сохранение в DataFrame
            self._data_table = pd.DataFrame(data=params_data, columns=header)
            self._data_table['user_id'] = sign_owners
            self._p_type = par_type
            self._number_of_harms = number_harms
        else:
            raise ValueError('Invalid type of parametrisation')

    def normalize(self, label_column, type_of_normalize, direction='column', columns=None):
        """
        Нормализует данные исходного датафрейма в соотвествии с параметрами
        Данный метод расширяет функионал базового метода класса DataSet
        Описание исходного функционала см. в DataSet.normalize

        :param type_of_normalize: str, optional
            Выбор типа нормализации:
            'envelopes_%d+' - нормализация "методом конвертов".
            Целое число после нижнего подчеркивания указывает разрядность генерируемых конвертов
            Результирующие значения будут представлены в виде [-%d, ..., -1, 0, 1, ..., %d]
            см. https://ieeexplore.ieee.org/abstract/document/7840688

        """
        if not re.search(r'envelopes_\d+', type_of_normalize):
            super().normalize(label_column, type_of_normalize, direction, columns)
        else:
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
            if re.search(r'envelopes_\d+', type_of_normalize):
                number_of_envelopes = int(re.search(r'\d+', type_of_normalize).group())
                self._envelopes = []
                if direction == 'row':
                    for index, row in res_data.iterrows():
                        res_data.iloc[index], current_envelope = Signatures.__calculate_envelope(row.values, number_of_envelopes)
                        self._envelopes.append([index, current_envelope])
                elif direction == 'column':
                    for column in res_data:
                        res_data[column], current_envelope = Signatures.__calculate_envelope(res_data[column].values, number_of_envelopes)
                        self._envelopes.append([column, current_envelope])
                else:
                    raise ValueError('Invalid direction')
                for column in res_data.columns:
                    self._data_table[column] = res_data[column]