import numpy as np
class Evaluation(object):
    '''Evaluation class based on pathon list'''
    def __init__(self, predict, label):
        self.predict = predict
        self.label = label

        self.accuracy = self._accuracy()

    def _accuracy(self):
        assert len(self.predict) == len(self.label)
        correct = (np.array(self.predict) == np.array(self.label)).sum()
        return float(correct)/float(len(self.predict))


if __name__ == '__main__':
    predict = [1, 2, 3, 4, 5, 3, 3, 2, 2, 5, 6, 6, 4, 3, 2, 4, 5, 6, 6, 3, 2]
    label =   [2, 5, 3, 4, 5, 3, 2, 2, 4, 6, 6, 6, 3, 3, 2, 5, 5, 6, 6, 3, 3]
    eval = Evaluation(predict,label)
    print(f"%.3f"%(eval.accuracy))
