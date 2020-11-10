from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import pprint, pickle
import argparse
import scipy.stats as stats
numpy2ri.activate()
mvt = importr('mvtnorm')

parser = argparse.ArgumentParser(description='PyTorch OSR Example')
parser.add_argument('--lamda', type=int, default=100, help='lamda in loss function')
parser.add_argument('--num_class', type=int, default=10, help='number of class')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold of gaussian model')
args = parser.parse_args()

def revise(epoch):
    train_rec = np.loadtxt('lvae%d/train_rec.txt' %args.lamda)
    rec_mean = np.mean(train_rec)
    rec_std = np.std(train_rec)
    rec_thres = rec_mean + 2 * rec_std #95%

    omn_rec = np.loadtxt('lvae%d/omn_rec.txt' %args.lamda)
    omn_pre = np.loadtxt('lvae%d/omn_pre.txt' %args.lamda)
    omn_pre[(omn_rec > rec_thres)] = args.num_class
    open('lvae%d/omn_pre.txt' %args.lamda , 'w').close()  # clear
    np.savetxt('lvae%d/omn_pre.txt' %args.lamda , omn_pre, delimiter=' ', fmt='%d')

    mnist_noise_rec = np.loadtxt('lvae%d/mnist_noise_rec.txt' %args.lamda )
    mnist_noise_pre = np.loadtxt('lvae%d/mnist_noise_pre.txt' %args.lamda)
    mnist_noise_pre[(mnist_noise_rec > rec_thres)] = args.num_class
    open('lvae%d/mnist_noise_pre.txt' %args.lamda , 'w').close()  # clear
    np.savetxt('lvae%d/mnist_noise_pre.txt' %args.lamda , mnist_noise_pre, delimiter=' ', fmt='%d')

    noise_rec = np.loadtxt('lvae%d/noise_rec.txt' %args.lamda)
    noise_pre = np.loadtxt('lvae%d/noise_pre.txt' %args.lamda)
    noise_pre[(noise_rec > rec_thres)] = args.num_class
    open('lvae%d/noise_pre.txt' %args.lamda , 'w').close()  # clear
    np.savetxt('lvae%d/noise_pre.txt' %args.lamda , noise_pre, delimiter=' ', fmt='%d')

class GAU(object):
    def __init__(self, epoch):
        self.trainfea = np.loadtxt('lvae%d/train_fea.txt' %args.lamda )
        self.traintar = np.loadtxt('lvae%d/train_tar.txt' %args.lamda )
        self.labelset = set(self.traintar)
        self.labelnum = len(self.labelset)
        self.num,self.dim = np.shape(self.trainfea)
        self.gau = self.train()

    def train(self):
        trainfea = self.trainfea
        traintar = self.traintar
        labelnum = self.labelnum
        trainsize = self.trainfea.shape[0]
        for i in range(labelnum):
            locals()['matrix' + str(i)] = np.empty(shape=[0,self.dim])

        gau = []
        muandsigma = []
        for j in range(trainsize):
            for i in range(labelnum):
                if traintar[j] == i:
                    locals()['matrix' + str(i)] = np.append((locals()['matrix' + str(i)]), np.array([np.array(trainfea[j])]),
                                                            axis=0)

        for i in range(labelnum):
            locals()['mu' + str(i)] = np.mean(np.array(locals()['matrix' + str(i)]),axis=0)
            locals()['sigma' + str(i)] = np.cov(np.array((locals()['matrix' + str(i)] - locals()['mu' + str(i)])).T)
            locals()['gau' + str(i)] = [locals()['mu' + str(i)],locals()['sigma' + str(i)]]
            print(i)
            print(locals()['mu' + str(i)])
            print(np.diag(locals()['sigma' + str(i)])**0.5)
            gau.append(locals()['gau' + str(i)])

        return gau

    def test(self,testsetlist,threshold = args.threshold):

        testfea = np.loadtxt(testsetlist[0])
        testtar = np.loadtxt(testsetlist[1])
        testpre = np.loadtxt(testsetlist[2])

        labelnum = self.labelnum
        gau = self.gau
        dim = self.dim
        performance = np.zeros([labelnum + 1, 5])
        testsize = testfea.shape[0]
        result = []
        if threshold != 0:
            print('threshold is', threshold)


        def multivariateGaussian(vector, mu, sigma):
            vector = np.array(vector)
            d = (np.mat(vector - mu)) * np.mat(np.linalg.pinv(sigma)) * (np.mat(vector - mu).T)
            p = np.exp(d * (-0.5)) / (((2 * np.pi) ** int(dim/2)) * (np.linalg.det(sigma)) ** (0.5))
            p = float(p)
            return p

        def multivariateGaussianNsigma(sigma,threshold):
            q = np.array(mvt.qmvnorm(threshold, sigma = sigma, tail = "both")[0])
            n = q[0]
            m = (np.diag(sigma) ** 0.5) * n
            d = (np.mat(m) * np.mat(np.linalg.pinv(sigma)) * (np.mat(m).T))
            p = np.exp(d * (-0.5)) / (((2 * np.pi) ** int(dim/2)) * (np.linalg.det(sigma)) ** (0.5))
            return p

        pNsigma = np.zeros(labelnum)
        p = np.zeros(labelnum)
        mu = []
        sigma = []

        for j in range(labelnum):
            mu.append(gau[j][0])
            sigma.append(gau[j][1])
            pNsigma[j] = multivariateGaussianNsigma(sigma[j],threshold)


        for i in range(testsize):
            for j in range(labelnum):
                p[j] = multivariateGaussian(testfea[i],mu[j],sigma[j])

            delta = p-pNsigma
            # print(delta)
            if len(delta[delta > 0]) == 0:
                #Unseen
                prediction = labelnum
            else:
                #Seen
                prediction = testpre[i]

            result.append(prediction)

        #result
        result = np.array(result)
        np.savetxt('lvae%d/Result.txt' %args.lamda,result)

        for i in range(labelnum+1):
            locals()['resultIndex' + str(i)] = np.argwhere(result == i)
            locals()['targetIndex' + str(i)] = np.argwhere(testtar == i)

        for i in range(labelnum+1):
            locals()['tp' + str(i)] = np.sum((testtar[(locals()['resultIndex' + str(i)])]) == i)
            locals()['fp' + str(i)] = np.sum((testtar[(locals()['resultIndex' + str(i)])]) != i)
            locals()['fn' + str(i)] = np.sum((result[locals()['targetIndex' + str(i)]]) != i)
            # print(locals()['tp' + str(i)],locals()['fp' + str(i)],locals()['fn' + str(i)])

            performance[i, 0] = locals()['tp' + str(i)]
            performance[i, 1] = locals()['fp' + str(i)]
            performance[i, 2] = locals()['fn' + str(i)]

        for i in range(labelnum+1):
            locals()['precision' + str(i)] = locals()['tp' + str(i)]/(locals()['tp' + str(i)]+locals()['fp' + str(i)] + 1)
            locals()['recall' + str(i)] = locals()['tp' + str(i)]/(locals()['tp' + str(i)]+locals()['fn' + str(i)] + 1)
            locals()['f-measure' + str(i)] = 2* locals()['precision' + str(i)]*locals()['recall' + str(i)]/(locals()['precision' + str(i)] + locals()['recall' + str(i)])

            performance[i, 3] = locals()['precision' + str(i)]
            performance[i, 4] = locals()['recall' + str(i)]

        performancesum = performance.sum(axis = 0)
        mafmeasure = 2*performancesum[3]*performancesum[4]/(performancesum[3]+performancesum[4])
        maperformance = np.append((performancesum)[3:],mafmeasure)/(labelnum+1)

        print(performance)
        np.savetxt('lvae%d/performance.txt' %args.lamda , performance)
        return maperformance

if __name__ == '__main__':

    for epoch in range(1):

        revise(epoch)
        gau = GAU(epoch)
        omn = ['lvae%d/omn_fea.txt' %args.lamda, 'lvae%d/omn_tar.txt' %args.lamda ,
                   'lvae%d/omn_pre.txt' %args.lamda ]
        mnist_noise = ['lvae%d/mnist_noise_fea.txt' %args.lamda, 'lvae%d/mnist_noise_tar.txt' %args.lamda ,
                 'lvae%d/mnist_noise_pre.txt' %args.lamda ]
        noise = ['lvae%d/noise_fea.txt' %args.lamda , 'lvae%d/noise_tar.txt' %args.lamda ,
                     'lvae%d/noise_pre.txt' %args.lamda ]

        perf_omn = gau.test(omn, args.threshold)
        perf_mnist_noise = gau.test(mnist_noise, args.threshold)
        perf_noise = gau.test(noise, args.threshold)
        #
        ma = [perf_omn, perf_mnist_noise, perf_noise]
        print(ma)
        np.savetxt('lvae%d/ma.txt' %args.lamda, ma)
