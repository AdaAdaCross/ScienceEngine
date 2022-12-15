import pandas as pd
import numpy as np
from DataSet import DataSet
from scipy.fft import *
import math
import matplotlib.pyplot as plt
import cmath
import pywt
import mysql.connector as msc
import re
from PIL import Image, ImageDraw


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
    p_type : str
        Содержит информацию о типе параметризации
    number_of_harms : int
        Содержит информацию о количестве гармоник
    envelopes : list
        Содержит список конвертов, рассчитанных для каждой колонки или строки
        в формате [идентификатор колонки или строки, [среднее значение, стандартное отклонение]]

    Методы
    ----------
    load_csv(path, p_type='Furie', num_of_harms=None)
        Загружает в data_table информацию из csv-файла по заданному пути или url
        см. pandas.read_csv()
    generate_from_database(host, user, password, number_harms, par_type, trace=False)
        Генерирует датафрейм из базы данных sign_db с преобразованием данных по выбранному методу
    normalize(label_column, type_of_normalize, direction='column', columns=None)
        Нормализует данные исходного датафрейма в соотвествии с параметрами
    visualize(file_name, row_index = None, user_id = None)
        Сохраняет отрисованные изображения подписи, кривые X,Y,Z и их гармоники в отдельные файлы
    set_numpy_data(numpy_data, indexes=None, columns=None)
        Заменить значения в заданных ячейках на значения из numpy.ndarray
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

    @staticmethod
    def __restore_sequence(modulus, phase, frequences, file_name):
        """
        Восстанавливает исходную последовательность из гармоник после быстрого преобразования Фурье
        :param modulus: list
            Список модулей
        :param phase: list
            Список фаз
        :param frequences: list
            Список частот
        :return: list
            Исходная последовательность

        """
        eps = 1e-10
        sampling_period = 0.005
        rev_size = math.ceil(100 / frequences[0])
        rev_fft = [0] * rev_size

        for i in range(1, len(modulus)):
            modulus[i] = 10 ** modulus[i] - eps
            rev_fft[int(frequences[i])] = cmath.rect(modulus[i], phase[i])

        fft_freq = rfftfreq(rev_size * 2 - 2, sampling_period)

        fig, ax = plt.subplots()
        ax.vlines(x=fft_freq[:17], ymin=0, ymax=np.abs(rev_fft)[:17], color=(0, 0, 0), linewidth=3)
        # ax.plot(fft_freq[:25], np.abs(rev_fft)[:25], color = (0, 0, 0), linewidth = 3)
        ax.set_xlabel('частота, Гц')
        ax.set_ylabel('модуль значения БФП')
        ax.grid()
        plt.savefig(file_name+'_harmonics.png')

        res_fft = irfft(rev_fft, math.ceil(200 / frequences[0]))
        res_fft[:] = [x + modulus[0] for x in res_fft]
        return res_fft

    def load_csv(self, path, p_type='Fourier', num_of_harms=None):
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
            orig_size = []
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
                orig_size.append(len(sign[0]))
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
            self._data_table['orig_size'] = orig_size
            self._p_type = par_type
            self._number_of_harms = number_harms
        else:
            raise ValueError('Invalid type of parametrisation')

    def normalize(self, label_column, type_of_normalize, direction='column', columns=None, metadata_save_path=None):
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
        :param metadata_save_path: str, optional
            Строка, содержащая путь до места сохранения информации, уничтожаемой при нормализации, но позволяющей
            восстановить исходные данные подписи после нормализации
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
            if metadata_save_path is not None:
                list_trans_envelopes = list(map(list, zip(*self._envelopes)))
                labels_list = list_trans_envelopes[0]
                data_list = list(map(list, zip(*list_trans_envelopes[1])))
                df_for_clean = pd.DataFrame(data=data_list, columns=labels_list)
                df_for_clean.to_csv(metadata_save_path, index=False)



    def visualize(self, file_name, row_index=None, user_id=None):
        """
        Сохраняет отрисованные изображения подписи, кривые X,Y,Z и их гармоники в отдельные файлы
        :param file_name: str
            Начало пути, по которому будут сохранены изображения
        :param row_index: int, optional
            Номер строки (от начала датасета или куска датасета заданного пользователя),
            подпись из которой нужно отрисовать
        :param user_id: int, optional
            Идентификатор пользователя, подпись которого нужно отрисовать
        """
        if user_id is not None:
            draw_sign = self._data_table.loc[self._data_table['user_id'] == user_id]
            if row_index is not None:
                draw_sign = draw_sign.iloc[row_index]
            else:
                draw_sign = draw_sign.iloc[0]
        else:
            draw_sign = self._data_table.iloc[0]

        if self._p_type == 'Fourier':
            names_modulus_X = []
            names_phases_X = []
            names_modulus_Y = []
            names_phases_Y = []
            names_modulus_Z = []
            names_phases_Z = []

            for i in range(self._number_of_harms):
                names_modulus_X.append('X_diff0_module_' + str(i + 1))
                names_phases_X.append('X_diff0_phase_' + str(i + 1))
                names_modulus_Y.append('Y_diff0_module_' + str(i + 1))
                names_phases_Y.append('Y_diff0_phase_' + str(i + 1))
                names_modulus_Z.append('Z_diff0_module_' + str(i + 1))
                names_phases_Z.append('Z_diff0_phase_' + str(i + 1))
            modulus_X = []
            phases_X = []
            modulus_Y = []
            phases_Y = []
            modulus_Z = []
            phases_Z = []
            frequences_X = [draw_sign['frequency']]
            frequences_Y = [draw_sign['frequency']]
            frequences_Z = [draw_sign['frequency']]
            for i in range( 1, self._number_of_harms):
                frequences_X.append(i)
                frequences_Y.append(i)
                frequences_Z.append(i)
            for i in names_modulus_X:
                modulus_X.append(draw_sign[i])
            for i in names_phases_X:
                phases_X.append(draw_sign[i])
            for i in names_modulus_Y:
                modulus_Y.append(draw_sign[i])
            for i in names_phases_Y:
                phases_Y.append(draw_sign[i])
            for i in names_modulus_Z:
                modulus_Z.append(draw_sign[i])
            for i in names_phases_Z:
                phases_Z.append(draw_sign[i])
            print(frequences_X)
            X = self.__restore_sequence(modulus_X, phases_X, frequences_X, file_name+'_X')
            Y = self.__restore_sequence(modulus_Y, phases_Y, frequences_Y, file_name+'_Y')
            Z = self.__restore_sequence(modulus_Z, phases_Z, frequences_Z, file_name+'_Z')

            fig, ax = plt.subplots()
            ax.plot(X, color=(0, 0, 0), linewidth=3)
            ax.set_xlabel('номер отсчета')
            ax.set_ylabel('значение X')
            ax.grid()
            plt.savefig(file_name+'_X_curve.png')

            fig, ax = plt.subplots()
            ax.plot(Y, color=(0, 0, 0), linewidth=3)
            ax.set_xlabel('номер отсчета')
            ax.set_ylabel('значение Y')
            ax.grid()
            plt.savefig(file_name + '_Y_curve.png')

            fig, ax = plt.subplots()
            ax.plot(Z, color=(0, 0, 0), linewidth=3)
            ax.set_xlabel('номер отсчета')
            ax.set_ylabel('значение Z')
            ax.grid()
            plt.savefig(file_name+'_Z_curve.png')

            image1 = Image.new("RGB", (800, 600), (255, 255, 255))
            draw = ImageDraw.Draw(image1)

            for i in range(len(X) - 1):
                if Z[i] < 50:
                    draw.line([(X[i] + 100), (Y[i] + 50), (X[i + 1] + 100), (Y[i + 1] + 50)], width=4, fill="black")
            image1.save(file_name+'_figure.png')
        elif self._p_type == 'Wavelet':
            restore = []
            sign_size = draw_sign['orig_size']
            for i in ['X', 'Y', 'Z']:
                mask = i + r'_diff0_\d+'
                list_columns = " ".join(draw_sign.index)
                need_columns = re.findall(mask, list_columns)
                sequence = draw_sign[need_columns]
                perc = (self._number_of_harms / sign_size) * 100
                x = list(range(int(sign_size)))
                xp = x[::int(100.0/perc)]
                index = xp[-1] + 1
                while len(xp) > self._number_of_harms:
                    xp.pop(int(len(xp) / 2))
                while len(xp) < self._number_of_harms:
                    xp.append(index)
                    index = index + 1
                sequence = np.interp(x, xp, sequence)
                restore.append(sequence)
            X = restore[0]
            Y = restore[1]
            Z = restore[2]
            image1 = Image.new("RGB", (800, 600), (255, 255, 255))
            draw = ImageDraw.Draw(image1)
            for i in range(len(X) - 1):
                if Z[i] < 50:
                    draw.line([(X[i] + 100), (Y[i] + 50), (X[i + 1] + 100), (Y[i + 1] + 50)], width=4, fill="black")
            image1.save(file_name + '_figure.png')
        else:
            ValueError('Unsupported parametrization type')

    def set_numpy_data(self, numpy_data, indexes=None, columns=None) :
        """
        Заменить значения в заданных ячейках на значения из numpy.ndarray
        :param numpy_data: numpy.ndarray
            Контейнер, содержащий в себе значения, помещаемые в датасет
        :param indexes: list[]
            Контейнер, содержащий в себе в себе список индексов строк
        :param columns: str, list[str], list[int]
            Контейнер, содержащий в себе описание требуемых для замены колонок из датасета
            Если задана как строка - описание в виде регулярного выражения, которому должны соответствовать
                заголовки наименований колонок
            Если как список - содержит список необходимых колонок
        """
        list_columns = []
        list_rows = []
        if columns is not None:
            if type(columns) == str:
                list_columns = " ".join(self._data_table.columns)
                list_columns = re.findall(columns, list_columns)
            elif type(columns) == list:
                if type(columns[0]) == int:
                    list_columns = self._data_table.columns[columns]
                elif type(columns[0]) == str:
                    list_columns = columns
                else:
                    raise TypeError('Invalid type of columns list')
            else:
                raise TypeError('Invalid type of columns')
        else:
            list_columns = self._data_table.columns
        if indexes is not None:
            list_rows = indexes
        else:
            list_rows = list(range(self._data_table.shape[0]))
        if numpy_data.shape[0] != len(list_rows) or numpy_data.shape[1] != len(list_columns):
            raise ValueError('Incorrect array size of insertion data')
        for i in range(len(list_columns)):
            for j in range(len(list_rows)):
                self.set_value(list_columns[i], list_rows[j], numpy_data[j][i])

    def get_info(self, envelopes_save_path=None):
        """
        Возвращает строку с информацией о подписи
        :return: str
            Строка с информацией о подписи
        """
        info_str1 = super().get_info()
        info_str2 = '4) type of parametrization: ' + self._p_type + '\n'
        info_str2 += '5) number of harmonics: ' + str(self._number_of_harms) + '\n'
        print(info_str2)

        if envelopes_save_path is not None:

            self._envelopes

        return info_str1 + info_str2

#todo параметры для вывода информации (например, все ли колонки выводить) P.S. сохранять конверты в файл
# todo сохранять нормаоизацию в файл