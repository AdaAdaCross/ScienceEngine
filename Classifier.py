import numpy as np
import yaml
from enum import Enum
from sklearn.model_selection import train_test_split

class ClassifierStates(Enum):
    """
    ClassifierStates
    ---------
    Класс-перечисление возможных состояний классификатора
    """

    NOT_INITIALIZED = 0
    """Модель классификатора еще не была определена или загружена"""
    MODEL_CREATED = 1
    """Модель классификатора была определена или загружена, но веса классификатора не определены"""
    MODEL_TRAINING = 2
    """Модель обучается, происходит вычисление весов"""
    MODEL_LOADING = 3
    """Загрузка весов из файла"""
    MODEL_READY = 10
    """Модель готова к классификации"""
    MODEL_EVALUATING = 11
    """Тестируются показатели эффективности работы модели"""
    MODEL_PREDICTING = 12
    """Модель в процессе классификации"""


class Classifier:
    """
    Classifier
    ----------
    Виртуальный базовый класс.
    Родительский класс для описания классификаторов на основе как нейронных сетей, так и классических алгоритмов
    классификации
    Содержит базовые атрибуты и функции для работы с моделями классификаторов
    Предполагается, что данный класс является виртуальным и не может использоваться самостоятельно

    Атрибуты
    ----------
    classifier_parameters_yaml : string
        содержит описание модели классификатора
        (используемые функции, описание слоев нейронной сети и т.п.)
         в формате YAML см. https://yaml.org/
    classifier_parameters : any_type
        разобранное описание модели классификатора
    current_state : ClassifierStates
        указывает текущее состояние классификатора
    learning_parameters_yaml : string
        содержит описание процесса обучения классификатора
        (например, количество эпох, используемый оптимизатор и т.п.)
        в формате YAML см. https://yaml.org/
    learning_parameters : any_type
        разобранное описание процесса обучения классификатора
    model : any_type
        содержит модель классификатора

    Методы
    ----------
    init_classifier(classifier_description, learning_description=None):
        Инициализирует модель классификатора и свойства для его обучения
    init_classifier_from_file(path_to_classifier_descr: str, path_to_learning_descr: str = None):
        Инициализирует модель классификатора и свойства для его обучения, считывая параметры из заданных файлов
    load_model(self, path: str):
        Загружает веса обученной модели в подготовленный классификатор
    learn(self, data, labels, train_test_ratio=0.7,save_learn_history_path=None, verbose=None, is_onehot_encoded=True):
        Запускает процесс обучения модели классификатора
    evaluate(self, data, labels, is_onehot_encoded=True):
        Тестирует обученную или загруженную модель классификатора на заданном наборе данных
    get_stats_error(self, data, labels, friend_users, alien_users, is_onehot_encoded=True):
        Вычисляет статистические ошибки, получаемые обученной моделью классификатора на заданном наборе данных
    get_prediction(self, data, is_onehot_encoded=True):
        Возвращает предсказание обученного классификатора для заданного набора данных
    get_current_state(self):
        Возвращает текущее внутреннее состояние классификатора
    get_summary(self, to_stdio=False):
        Возвращает текущую структуру классификатора (используемые функции, слои, их характеристики и т.п.)
    save_model(self, path: str):
        Сохраняет веса текущей модели классификатора в файл
    save_classifier_parameters(self, path: str):
        Сохраняет описание текущей модели классификатора в файл в формате YAML
    save_learning_parameters(self, path: str):
        Сохраняет параметры, с которыми текущая модель классификатора обучалась в файл в формате YAML
    """

    def __init__(self):
        self._classifier_parameters_yaml = ''
        self._classifier_parameters = None
        self._learning_parameters_yaml = ''
        self._learning_parameters = None
        self._current_state = ClassifierStates.NOT_INITIALIZED
        self._model = None

    def init_classifier(self, classifier_description, learning_description=None):
        """
        Виртуальный метод.
        Инициализирует модель классификатора и свойства для его обучения
        :param classifier_description: str
            Текстовое описание модели классификатора в формате YAML
        :param learning_description: str, optional
            Текстовое описание параметров обучения в формате YAML
        """
        self._classifier_parameters_yaml = classifier_description
        self._classifier_parameters = yaml.load(classifier_description, Loader=yaml.FullLoader)
        if learning_description is not None:
            self._learning_parameters_yaml = learning_description
            self._learning_parameters = yaml.load(learning_description, Loader=yaml.FullLoader)

    def init_classifier_from_file(self, path_to_classifier_descr: str, path_to_learning_descr: str = None):
        """
        Инициализирует модель классификатора и свойства для его обучения,
        считывая параметры из заданных файлов
        :param path_to_classifier_descr: str
            Путь к файлу с описанием модели классификатора в формате YAML
        :param path_to_learning_descr: str
            Путь к файлу с описанием параметров обучения в формате YAML
        """
        class_stream = open(path_to_classifier_descr, 'r')
        learn_stream = None
        if path_to_learning_descr is not None:
            learn_stream = open(path_to_learning_descr, 'r')

        # недокументированное использование функции init_classifier: вместо текста передаем открытые файлы,
        # реализация библиотеки yaml позволяет использовать ее таким образом
        self.init_classifier(class_stream, learn_stream)

    def load_model(self, path: str):
        """
        Виртуальный метод.
        Загружает веса обученной модели в подготовленный классификатор
        :param path: str
            Путь к сохраненным весам
        """
        if self._current_state == ClassifierStates.NOT_INITIALIZED:
            raise AttributeError('Model is not ready for loading weights')

    def learn(self, data, labels, train_test_ratio=0.7, save_learn_history_path=None, verbose=None, is_onehot_encoded=True):
        """
        Виртуальный метод.
        Запускает процесс обучения модели классификатора
        :param data: np.array
            данные для обучения
        :param labels: np.array or list
            метки данных для обучения
        :param train_test_ratio: float
            отношение размера тренировочного набора к размеру тестового набора,
            на которые будут поделены исходные данные перед обучением
        :param save_learn_history_path: str
            Строка, содержащая путь до файла с логами процесса обучения
        :param verbose: str
            Указывает будет ли выводиться информация о прогрессе обучения и если да, то в каком количестве
            Возможные варианты вывода:
            None - вывод не осуществляется
            'progress_only' - выводить только информацию о том, сколько процентов обучения уже было успешно пройдено
            'full' - выводить полную информацию
        :param is_onehot_encoded: bool
            Указывает приведены ли метки к OneHotEncoded виду
        """
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=train_test_ratio)
        # возврат разбитых обучающих и тестовых данных для использования в дочерних классах
        return train_data, test_data, train_labels, test_labels

    def evaluate(self, data, labels, is_onehot_encoded=True):
        """
        Виртуальный метод.
        Тестирует обученную или загруженную модель классификатора на заданном наборе данных
        :param data: np.array
            Набор данных для тестирования
        :param labels: np.array or list
            Метки набора данных для тестирования
        :param is_onehot_encoded: bool
            Указывает приведены ли метки к OneHotEncoded виду
        :return: float, float
            Полученные значения точности и функции потерь соответственно
        """
        if self._current_state != ClassifierStates.MODEL_READY:
            raise AttributeError('Model is not ready for data classification')
        return 0, np.Inf

    def get_stats_error(self, data, labels, friend_users, alien_users, is_onehot_encoded=True):
        """
        Виртуальный метод.
        Вычисляет статистические ошибки, получаемые обученной моделью классификатора на заданном наборе данных
        :param data: np.array
            Набор данных для тестирования
        :param labels: np.array or list
            Метки набора данных для тестирования
        :param friend_users: np.array or list
            Список легальных пользователей, которых система должна пропускать
        :param alien_users: np.array or list
            Список нелегальных пользователей, доступ в систему для которых закрыт
        :param is_onehot_encoded: bool
            Указывает приведены ли метки к OneHotEncoded виду
        :return: float, float
            Полученные значения ошибки первого и второго рода соответственно
        """
        if self._current_state != ClassifierStates.MODEL_READY:
            raise AttributeError('Model is not ready for data classification')
        # todo разобраться с ошибками 3 и 4 рода
        return np.Inf, np.Inf

    def get_prediction(self, data, is_onehot_encoded=True):
        """
        Виртуальный метод.
        Возвращает предсказание обученного классификатора для заданного набора данных
        :param data: np.array
            Данные для предсказания
        :param is_onehot_encoded: bool
            Указывает возвращать ли результаты в OneHotEncoded виде
        :return: np.array
            Возвращает предсказанные метки пользователей
        """
        if self._current_state != ClassifierStates.MODEL_READY:
            raise AttributeError('Model is not ready for data classification')
        return [0] * len(data)

    def get_current_state(self):
        """
        Возвращает текущее внутреннее состояние классификатора
        :return: ClassifierStates
            Состояние классификатора
        """
        return self._current_state

    def get_summary(self, to_stdio=False):
        """
        Виртуальный метод.
        Возвращает текущую структуру классификатора
        (используемые функции, слои, их характеристики и т.п.)
        :param to_stdio: bool
            Указывает, выводить ли результат работы функции в консоль
        :return: str
            Описание структуры классификатора
        """
        # todo возможно добавить графическую визуализацию классификатора
        if (to_stdio):
            print('Not implemented')
        raise NotImplementedError()

    def save_model(self, path: str):
        """
        Виртуальный метод.
        Сохраняет веса текущей модели классификатора в файл
        :param path: str, pathlike
            Путь к файлу для сохранения весов модели
        """
        if self._current_state != ClassifierStates.MODEL_READY:
            raise AttributeError('Models weights are not ready for saving')

    def save_classifier_parameters(self, path: str):
        """
        Виртуальный метод.
        Сохраняет описание текущей модели классификатора в файл в формате YAML
        :param path:
            Путь к файлу для сохранения описания модели
        """
        if self._current_state == ClassifierStates.NOT_INITIALIZED:
            raise AttributeError('Model is not ready for saving')

    def save_learning_parameters(self, path: str):
        """
        Виртуальный метод.
        Сохраняет параметры, с которыми текущая модель классификатора обучалась в файл в формате YAML
        :param path:
            Путь к файлу для сохранения параметров обучения модели
        """
        if self._current_state != ClassifierStates.MODEL_READY:
            raise AttributeError('Models params are not ready for saving')
