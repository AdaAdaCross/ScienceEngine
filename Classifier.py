import yaml
from enum import Enum

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
        :param path_to_classifier_descr:
            Путь к файлу с описанием модели классификатора в формате YAML
        :param path_to_learning_descr:
            Путь к файлу с описанием параметров обучения в формате YAML
        """
        class_stream = open(path_to_classifier_descr, 'r')
        learn_stream = None
        if path_to_learning_descr is not None:
            learn_stream = open(path_to_learning_descr, 'r')
        self.init_classifier(class_stream, learn_stream)
        print(self._classifier_parameters)

