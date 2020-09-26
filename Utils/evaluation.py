import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, \
    classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


class Evaluation(object):
    '''Evaluation class based on pathon list'''
    def __init__(self, predict, label):
        self.predict = predict
        self.label = label

        self.accuracy = self._accuracy()
        self.f1_measure = self._f1_measure()
        self.f1_macro = self._f1_macro()
        self.f1_macro_weighted = self._f1_macro_weighted()
        self.precision, self.recall = self._precision_recall(average='micro')
        self.precision_macro, self.recall_macro = self._precision_recall(average='macro')
        self.precision_weighted, self.recall_weighted = self._precision_recall(average='weighted')
        self.confusion_matrix = self._confusion_matrix()

    def _accuracy(self):
        assert len(self.predict) == len(self.label)
        correct = (np.array(self.predict) == np.array(self.label)).sum()
        return float(correct)/float(len(self.predict))

    def _f1_measure(self):
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='micro')

    def _f1_macro(self):
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='macro')

    def _f1_macro_weighted(self):
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='weighted')

    def _precision_recall(self, average=None):
        assert len(self.predict) == len(self.label)
        precision, recall, _, _ = precision_recall_fscore_support(label, predict, average=average)
        return precision, recall

    def auroc(self, prediction_scores: np.array = None, multi_class='ovo'):
        """
        Area Under Receiver Operating Characteristic Curve

        :param prediction_scores: array-like of shape (n_samples, n_classes). The multi-class ROC curve requires prediction scores for each class. If
        :param multi_class: {'ovo', 'ovr'}, default='ovo'
            'ovo' computes the average AUC of all possible pairwise combinations of classes.
            'ovr' Computes the AUC of each class against the rest.
        :return:
        """

        one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        #one_hot_encoder.fit(np.array(label+predict).reshape(-1, 1))
        one_hot_encoder.fit(np.array(label).reshape(-1, 1))
        true_scores = one_hot_encoder.transform(np.array(label).reshape(-1, 1))
        if prediction_scores is None:
            prediction_scores = one_hot_encoder.transform(np.array(predict).reshape(-1, 1))
        assert prediction_scores.shape == true_scores.shape
        return roc_auc_score(true_scores, prediction_scores, multi_class=multi_class)


    def _confusion_matrix(self, normalize=None):
        assert len(self.predict) == len(self.label)
        return confusion_matrix(self.label, self.predict, normalize=normalize)

    def plot_confusion_matrix(self, title='', labels: [str] = None, normalize=None, ax=None):
        conf_matrix = self._confusion_matrix(normalize)  # Evaluate the confusion matrix
        display = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)  # Generate the confusion matrix display

        # Formatting for the plot
        if labels:
            xticks_rotation = 'vertical'
        else:
            xticks_rotation = 'horizontal'

        display.plot(include_values=True, cmap=plt.cm.get_cmap('Blues'), xticks_rotation=xticks_rotation, ax=ax)
        plt.show()


    # TODO: F1_MACRO, area under ROC curve (AUROC), Percision, Recall

if __name__ == '__main__':
    predict = [1, 2, 3, 4, 5, 3, 3, 2, 2, 5, 6, 6, 4, 3, 2, 4, 5, 6, 6, 3, 2]
    label =   [2, 5, 3, 4, 5, 3, 2, 2, 4, 6, 6, 6, 3, 3, 2, 5, 5, 6, 6, 3, 3]


    eval = Evaluation(predict, label)
    print('Accuracy:', f"%.3f"%(eval.accuracy))
    print('F1-measure:', f'{eval.f1_measure:.3f}')
    print('F1-macro:', f'{eval.f1_macro:.3f}')
    print('F1-macro (weighted):', f'{eval.f1_macro_weighted:.3f}')
    print('precision:', f'{eval.precision:.3f}')
    print('precision (macro):', f'{eval.precision_macro:.3f}')
    print('precision (weighted):', f'{eval.precision_weighted:.3f}')
    print('recall:', f'{eval.recall:.3f}')
    print('recall (macro):', f'{eval.recall_macro:.3f}')
    print('recall (weighted):', f'{eval.recall_weighted:.3f}')

    # Generate "random prediction score" to test feeding in prediction score from NN
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    one_hot_encoder.fit(np.array(label).reshape(-1, 1))
    rand_prediction_scores = 2*one_hot_encoder.transform(np.array(predict).reshape(-1, 1))  # One hot
    rand_prediction_scores += np.random.rand(*rand_prediction_scores.shape)
    rand_prediction_scores /= rand_prediction_scores.sum(axis=1)[:, None]
    print('Area under ROC curve (with 100% confidence in prediction):', f'{eval.auroc():.3f}')
    print('Area under ROC curve (variable probability across classes):', f'{eval.auroc(prediction_scores=rand_prediction_scores):.3f}')
    print(eval.confusion_matrix)
    eval.plot_confusion_matrix()

    print(classification_report(label, predict, digits=3))