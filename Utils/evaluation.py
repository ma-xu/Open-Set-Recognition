import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, \
    classification_report, precision_recall_fscore_support, roc_auc_score

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

    def _auroc(self):
        """Area Under Receiver Operating Curve"""
        assert len(self.predict) == len(self.label)
        return roc_auc_score(label, )


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
    print(eval.confusion_matrix)
    eval.plot_confusion_matrix()

    print(classification_report(label, predict, digits=3))