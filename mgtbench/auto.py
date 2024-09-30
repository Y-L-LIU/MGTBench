from abc import ABC, abstractmethod
from .loading import load_pretrained
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
import numpy as np
import math

DETECTOR_MAPPING = {
    'gptzero' : 'mgtbench.methods.GPTZeroAPI',
    'll' : 'mgtbench.methods.LLDetector',
    'rank' : 'mgtbench.methods.RankDetector',
    'rank_GLTR' : 'mgtbench.methods.RankGLTRDetector',
    'entropy' : 'mgtbench.methods.EntropyDetector',
    'detectGPT' : 'mgtbench.methods.DetectGPTDetector',
    'NPR' : 'mgtbench.methods.NPRDetector',
    'LRR' : 'mgtbench.methods.LRRDetector',
    'GPTZero': 'mgtbench.methods.GPTZeroDetector',
    'OpenAI-D':'mgtbench.methods.SupervisedDetector',
    'ConDA':'mgtbench.methods.SupervisedDetector',
    'ChatGPT-D':'mgtbench.methods.SupervisedDetector',
    'LM-D':'mgtbench.methods.SupervisedDetector',
    'demasq' : 'mgtbench.methods.DemasqDetector',
}

EXPERIMENT_MAPPING = {
    'threshold' : 'mgtbench.experiment.ThresholdExperiment',
    'perturb' : 'mgtbench.experiment.PerturbExperiment',
    'supervised' : 'mgtbench.experiment.SupervisedExperiment',
    'demasq' : 'mgtbench.experiment.DemasqExperiment'
}

class BaseDetector(ABC):
    def __init__(self,name) -> None:
        self.name = name

    @abstractmethod
    def detect(self, **kargs):
        raise NotImplementedError('Invalid detector, implement detect first.')



class ModelBasedDetector(BaseDetector):
    def __init__(self,name,**kargs) -> None:
        super().__init__(name)


class BaseExperiment(ABC):
    def __init__(self, **kargs) -> None:
        self.loaded = False
        if "need_split_long_text" in kargs:
            self.need_split_long_text = kargs["need_split_long_text"]
        else:
            self.need_split_long_text = False


    @abstractmethod 
    def predict(self):
        raise NotImplementedError('Invalid Experiment, implement predict first.')

    def data_prepare(self, x, y, is_train_data=False):
        x, y = np.array(x), np.array(y)
        if self.need_split_long_text:
            x = np.array(reduce_text(x, self.train_text_mapping if is_train_data else self.test_text_mapping))
        select_index = ~np.isnan(x)
        x = x[select_index]
        y = y[select_index]
        x_train = np.expand_dims(x, axis=-1)
        return x_train, y
    
    def run_clf(self, clf, x, y):
        # print('x: ', x)
        # print('y: ', y)
        y_train_pred = clf.predict(x)
        y_train_pred_prob = clf.predict_proba(x)
        # print('y_pred: ', y_train_pred)
        # print('y_pred_prob: ', y_train_pred_prob)
        y_train_pred_prob = [_[1] for _ in y_train_pred_prob]
        return (y, y_train_pred, y_train_pred_prob)

    def cal_metrics(self, label, pred_label, pred_posteriors):
        if len(set(label)) < 3:
            acc = accuracy_score(label, pred_label)
            precision = precision_score(label, pred_label)
            recall = recall_score(label, pred_label)
            f1 = f1_score(label, pred_label)
            auc = roc_auc_score(label, pred_posteriors)
        else:
            acc = accuracy_score(label, pred_label)
            precision = precision_score(label, pred_label, average='weighted')
            recall = recall_score(label, pred_label, average='weighted')
            f1 = f1_score(label, pred_label, average='weighted')
            auc = -1.0
            conf_m = confusion_matrix(label, pred_label)
            print(conf_m)
        return Metric(acc, precision, recall, f1, auc)
    

    def load_data(self, data):
        self.loaded = True
        self.train_text = data['train']['text']
        self.train_label = data['train']['label']
        self.test_text = data['test']['text']
        self.test_label = data['test']['label']


    def launch(self, **config):
        if not self.loaded:
            raise RuntimeError('You should load the data first, call load_data.')
        print('Calculate result for each data point')
        predict_list = self.predict(**config)
        final_output = []
        for detector_predict in predict_list:
            train_metric = self.cal_metrics(*detector_predict['train_pred'])
            test_metric = self.cal_metrics(*detector_predict['test_pred'])
            final_output.append(DetectOutput(
                name = '',
                train = train_metric,
                test = test_metric
            ))
        return final_output 


@dataclass
class Metric:
    acc:float= None
    precision:float= None
    recall:float= None
    f1:float= None
    auc :float= None
    

@dataclass
class DetectOutput:
    name: str = None
    predictions=None
    train: Metric = None
    test: Metric = None
    clf  = None


def process_long_texts(texts):
    splited_texts = []
    mapping = []
    for i in range(len(texts)):
        splited_one_text = split_text(i, texts[i])
        mapping.append(len(expanding_texts))
        expanding_texts.extend(splited_one_text)
        expanding_labels.extend([labels[i]] * len(splited_one_text))
    mapping.append(len(expanding_texts))
    return expanding_texts, expanding_labels, mapping

MAX_WORD_NUM = 300

def split_text(i, text):
    words = text.split()
    # print(f"index {i}, len(words): {len(words)}")
    if len(words) > MAX_WORD_NUM:
        splited_text = []
        for j in range(math.floor(len(words)/MAX_WORD_NUM)):
        # for j in range(math.ceil(len(words)/MAX_WORD_NUM)):
            splited_text.append(' '.join(words[j*MAX_WORD_NUM:(j+1)*MAX_WORD_NUM]))
            # splited_text.append(' '.join(words[j*MAX_WORD_NUM:min((j+1)*MAX_WORD_NUM, len(words))]))
        return splited_text
    else:
        return [text]

def split_text_v2(i, text):
    pass

def reduce_text(text_features, mapping):
    # print(f'mapping: {mapping}, len(text_features): {len(text_features)}, text_features: {text_features}')
    reduced_text_features = []
    for i in range(len(mapping) - 1):
        reduced_text_feature = np.average(text_features[range(mapping[i], mapping[i + 1])])
        reduced_text_features.append(reduced_text_feature)
    # print(f'len(reduced_text_features): {len(reduced_text_features)}, reduced_text_features: {reduced_text_features}')
    return reduced_text_features
