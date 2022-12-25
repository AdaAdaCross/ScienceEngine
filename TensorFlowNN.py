from NeuralNetwork import *
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

class TensorFlowNN(NeuralNetwork):
    """
    TensorFlowNN
    ----------
    Дочерний класс от NeuralNetwork. Содержит атрибуты и реализацию методов, необходимых для работы
    с нейронными сетями на базе фреймворка TensorFlow

    Атрибуты
    ----------

    Методы
    ----------
    """
    
    def __init__(self):
        super(TensorFlowNN, self).__init__()

    def init_classifier(self, classifier_description, learning_description=None):
        """
        Расширение базового метода.
        Инициализирует модель нейронной сети и свойства для ее обучения
        :param classifier_description: str
            Текстовое описание модели классификатора в формате YAML
        :param learning_description: str, optional
            Текстовое описание параметров обучения в формате YAML
        """
        super(TensorFlowNN, self).init_classiifier(classifier_description, learning_description=learning_description)
        self._model = model_from_config(self._classifier_parameters)
        self._current_state = ClassifierStates.MODEL_CREATED
        self._layers_list = [self._model.input] + self._model.layers
        self._model.summary()

    def add_layer(self, type_of_layer, **extension):
        """
        Расширение базового метода.
        Добавляет слой в текущую модель нейронной сети
        Конфигурирование слоя осуществляется через extension согласно заданному классу слоя
        см. https://www.tensorflow.org/api_docs/python/tf/keras/layers
        :param type_of_layer: str
            Строка, содержащая название класса, к которому принадлежит слой
        :param extension: dictionary
            Параметры, специфичные для заданного слоя
        """
        super(TensorFlowNN, self).add_layer(type_of_layer, extension)
        layer_func = getattr(tf.keras.layers, type_of_layer)
        new_layer = layer_func(**extension)
        self._layers_list.append(new_layer)
        self._model.add(new_layer)
        self._model.summary()



