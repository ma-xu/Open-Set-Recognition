import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay

class Evaluation(object):
    '''Evaluation class based on pathon list'''
    def __init__(self, predict, label):
        self.predict = predict
        self.label = label

        self.accuracy = self._accuracy()
        self.confusion_matrix = self._confusion_matrix()

    def _accuracy(self):
        assert len(self.predict) == len(self.label)
        correct = (np.array(self.predict) == np.array(self.label)).sum()
        return float(correct)/float(len(self.predict))

    def _confusion_matrix(self, normalize=None):
        assert len(self.predict) == len(self.label)
        return confusion_matrix(label, predict, normalize=normalize)

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


    # TODO: F1_MACRO, area under ROC curve (AUROC), Percision, Recall, Confuse_Matrix, ,

if __name__ == '__main__':
    predict = [1, 2, 3, 4, 5, 3, 3, 2, 2, 5, 6, 6, 4, 3, 2, 4, 5, 6, 6, 3, 2]
    label =   [2, 5, 3, 4, 5, 3, 2, 2, 4, 6, 6, 6, 3, 3, 2, 5, 5, 6, 6, 3, 3]
    eval = Evaluation(predict, label)
    print(f"%.3f"%(eval.accuracy))
    print(eval.confusion_matrix)
    eval.plot_confusion_matrix()
