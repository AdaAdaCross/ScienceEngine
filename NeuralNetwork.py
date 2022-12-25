from Classifier import *

class NeuralNetwork(Classifier):
    """
    NeuralNetwork
    ----------
    Виртуальный базовый класс.
    Дочерний класс от Classifier для определения классификаоров на базе нейронной сети.
    Содержит атрибуты и функции для работы с датасетами сигнатур подписи
    Предполагается, что данный класс является виртуальным и не может использоваться самостоятельно

    Атрибуты
    ----------
    layers_list : list
        Список, содержащий в себе описание слоев нейронной сети

    Методы
    ----------
    add_layer(type_of_layer, number_of_neurons, shape, extension = None)
        Добавляет слой в текущую модель нейронной сети
    clear_layers()
        Очищает список слоев нейронной сети

    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self._layers_list = []

    def add_layer(self, type_of_layer, **extension):
        """
        Виртуальный метод.
        Добавляет слой в текущую модель нейронной сети
        :param type_of_layer: str
            Строка, содержащая название класса, к которому принадлежит слой
        :param extension: dictionary
            Параметры, специфичные для заданного слоя
        """
        if self._current_state == ClassifierStates.NOT_INITIALIZED:
            raise AttributeError('Model is not ready for adding layers')

    def clear_layers(self):
        """
        Очищает список слоев нейронной сети
        """
        self._layers_list.clear()