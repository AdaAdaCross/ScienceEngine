import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

from NeuralNetwork import *
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import json

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
    init_classifier(classifier_description, learning_description=None)
        Инициализирует модель нейронной сети и свойства для ее обучения
    add_layer(type_of_layer, **extension)
        Добавляет слой в текущую модель нейронной сети
    clear_layers()
        Очищает слои в моделе, оставляя только Input layer
    learn(data, labels, train_test_ratio=0.7, save_learn_history_path=None, verbose=None, is_onehot_encoded=True)
        Запускает процесс обучения модели классификатора
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
        super(TensorFlowNN, self).init_classifier(classifier_description, learning_description=learning_description)
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
        super(TensorFlowNN, self).add_layer(type_of_layer, **extension)
        layer_func = getattr(tf.keras.layers, type_of_layer)
        new_layer = layer_func(**extension)
        self._layers_list.append(new_layer)
        self._model.add(new_layer)
        self._model.summary()

    def clear_layers(self):
        """
        Расширение базового метода.
        Очищает слои в моделе, оставляя только Input layer.
        """
        self._layers_list.clear()
        model_func = getattr(tf.keras, self._classifier_parameters['class_name'])
        tmp_input_layer = self._model.input
        self._model = model_func()
        self._model.add(tmp_input_layer)
        self._model._name = self._classifier_parameters['config']['name']
        self._classifier_parameters = json.loads(self._model.to_json())
        self._classifier_parameters_yaml = yaml.dump(self._classifier_parameters)
        self._layers_list = [self._model.input]
        self._model.summary()

    def learn(self, data, labels, train_test_ratio=0.7, save_learn_history_path=None, verbose=None,
              is_onehot_encoded=True):
        """
        Расширение базового метода.
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
        if not is_onehot_encoded:
            labels = to_categorical(labels)

        train_data, test_data, train_labels, test_labels = \
            super(TensorFlowNN, self).learn(data, labels, train_test_ratio, save_learn_history_path, verbose,
                                            is_onehot_encoded)
        self._model.compile(**self._learning_parameters['compile_params'])
        verbose_level = 0
        if verbose is not None:
            if verbose == 'progress_only':
                verbose_level = 1
            if verbose == 'full':
                verbose_level = 2

        results = self._model.fit(train_data, train_labels, validation_data=(test_data, test_labels),
                                  verbose=verbose_level, **self._learning_parameters['fit_params'])
        if save_learn_history_path is not None:
            df_logs = pd.DataFrame(results.history)
            df_logs.to_csv(save_learn_history_path+'.csv', index=False)
            for metric in self._learning_parameters['compile_params']['metrics']:
                plt.plot(results.history[metric])
                plt.plot(results.history['val_' + metric])
                plt.title(metric)
                plt.ylabel(metric)
                plt.xlabel('epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                plt.savefig(save_learn_history_path+'_'+metric+'.png')
                plt.clf()

