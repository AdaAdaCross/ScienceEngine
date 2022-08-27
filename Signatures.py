import pandas as pd
import numpy as np
from DataSet import DataSet
from scipy.fft import *
import math
import cmath
import mysql.connector as msc


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

    """
    def __init__(self):
        super().__init__()
        self._p_type = None
        self._number_of_harms = None

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

        :param host:
        :param user:
        :param password:
        :param number_harms:
        :param par_type:
        :param trace:
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

                # todo сделать параметризацию производных

                for i in range(6):
                    (modulus, phase, frequencies) = self.__calculate_harmonics(sign[i], number_harms)
                    for mod in modulus:
                        params_data[offset].append(mod)
                    for pha in phase:
                        params_data[offset].append(pha)
                    if i == 0:
                        signs_freq.append(frequencies[0])
                params_data.append([])
                offset = offset + 1
            params_data = params_data[:-1]

            # формирование заголовков
            header = []
            for i in range(number_harms):
                header.append('X_module_' + str(i + 1))
            for i in range(number_harms):
                header.append('X_phase_' + str(i + 1))
            for i in range(number_harms):
                header.append('Y_module_' + str(i + 1))
            for i in range(number_harms):
                header.append('Y_phase_' + str(i + 1))
            for i in range(number_harms):
                header.append('Z_module_' + str(i + 1))
            for i in range(number_harms):
                header.append('Z_phase_' + str(i + 1))
            for i in range(number_harms):
                header.append('P_module_' + str(i + 1))
            for i in range(number_harms):
                header.append('P_phase_' + str(i + 1))
            for i in range(number_harms):
                header.append('Al_module_' + str(i + 1))
            for i in range(number_harms):
                header.append('Al_phase_' + str(i + 1))
            for i in range(number_harms):
                header.append('Az_module_' + str(i + 1))
            for i in range(number_harms):
                header.append('Az_phase_' + str(i + 1))

            # сохранение в DataFrame
            self._data_table = pd.DataFrame(data=params_data, columns=header)
            self._data_table['user_id'] = sign_owners
            self._data_table['frequency'] = signs_freq
