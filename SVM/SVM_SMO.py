import torch
import numpy as np
import time
from datetime import datetime

stat_all = time.time()


class SVM:
    def __init__(self, trainData, trainLabel, testData, testLabel, sigma, max_iter=2000, C=100.0, epsilon=0.01):
        self.trainData, self.trainLabel = trainData, trainLabel
        self.n, self.d = self.trainData.shape[0], self.trainData.shape[1]
        self.testData, self.testLabel = testData, testLabel
        self.max_iter = max_iter
        self.C = torch.tensor([C])
        self.c = C
        self.epsilon = epsilon
        self.alpha = torch.zeros(self.n)
        self.b = torch.tensor([1])
        self.w = torch.zeros(self.n)
        self.sigma = sigma
        self.E = torch.zeros(self.n)
        self.accuracy = 0
        self.start = time.time()
        self.count = 0
        self.result = torch.zeros(self.testData.shape[0])

    def SMO(self):
        n, d = self.n, self.d
        self.calE()
        while self.count <= self.max_iter:
            self.count += 1
            # Select the first variable
            idx1 = -1  # index of the first variable
            valid = 0
            for i in range(0, n):
                if 0 < self.alpha[i] < self.C:
                    obey, d = self.obeyKKT(self.trainData[i], self.trainLabel[i], self.alpha[i])
                    if obey == 0:
                        idx1 = i
                        # Select the second variable
                        if self.E[idx1] > 0:
                            idx2 = torch.min(self.E, 0)[1].item()
                        else:
                            idx2 = torch.max(self.E, 0)[1].item()
                        valid += self.updateAlpha(idx1, idx2)
                        if valid == 1:
                            break
            if idx1 == -1 or valid == 0:
                for k in range(0, n):
                    if self.alpha[k] == 0 or self.alpha[k] == self.C:
                        obey, d = self.obeyKKT(self.trainData[k], self.trainLabel[k], self.alpha[k])
                        if obey == 0:
                            idx1 = k
                            if self.E[idx1] > 0:
                                idx2 = torch.min(self.E, 0)[1].item()
                            else:
                                idx2 = torch.max(self.E, 0)[1].item()
                            valid += self.updateAlpha(idx1, idx2)
                            if valid == 1:
                                break
            if valid == 0:
                print("Optimization terminated!")
                return
            if idx1 == -1:
                print("Optimization completed!")
                return

    def updateAlpha(self, idx1, idx2):
        valid = 1
        # update alpha1 and alpha2
        alpha1Old = self.alpha[idx1]
        alpha2Old = self.alpha[idx2]
        x1, x2, y1, y2 = self.trainData[idx1], self.trainData[idx2], self.trainLabel[idx1], self.trainLabel[idx2]
        eta = self.kernel(x1, x1) + self.kernel(x2, x2) - 2 * self.kernel(x1, x2)
        alpha2Unc = alpha2Old + y2 * (self.E[idx1] - self.E[idx2]) / eta
        if y1 == y2:
            L = torch.max(torch.zeros(1), alpha1Old + alpha2Old - self.C)
            H = torch.min(self.C, alpha1Old + alpha2Old)
        else:
            L = torch.max(torch.zeros(1), alpha2Old - alpha1Old)
            H = torch.min(self.C, self.C + alpha2Old - alpha1Old)

        if alpha2Unc > H:
            alpha2New = H
        elif L <= alpha2Unc <= H:
            alpha2New = alpha2Unc
        else:
            alpha2New = L
        if alpha2New < 0.0000001:
            alpha2New = torch.zeros(1)
        alpha1New = alpha1Old + y1 * y2 * (alpha2Old - alpha2New)
        if alpha1New < 0.0000001:
            alpha1New = torch.zeros(1)
        # calculate E and B
        self.calB(alpha1New, alpha1Old, alpha2New, alpha2Old, idx1, idx2, self.b)
        self.calE(idx=idx1)
        self.calE(idx=idx2)
        if alpha1New == alpha1Old and alpha2New == alpha2Old:
            valid = 0
        # update alpha
        self.alpha[idx1], self.alpha[idx2] = alpha1New, alpha2New
        return valid

    def obeyKKT(self, x, label, alpha):
        result = -1
        a = label * self.f(x)
        distance = 0
        if alpha == 0:
            if a >= 1 - self.epsilon:
                result = 1
            elif a < 1 - self.epsilon:
                result = 0
                distance = 1 - a
        elif alpha == self.C:
            if a <= 1 + self.epsilon:
                result = 1
            elif a > 1 + self.epsilon:
                result = 0
                distance = a - 1
        elif self.C > alpha > 0:
            if 1 - self.epsilon < a < 1 + self.epsilon:
                result = 1
            else:
                result = 0
                distance = torch.abs(a - 1)
        return result, distance

    def f(self, x):
        result = 0
        for i in range(len(self.trainData)):
            result = result + self.alpha[i] * self.trainLabel[i] * self.kernel(x, self.trainData[i])
        result = result + self.b
        return result

    def kernel(self, x1, x2):
        return torch.exp(-((x1 - x2) * (x1 - x2)).sum() / (2 * (self.sigma ** 2)))

    def calE(self, idx=-1):
        if idx == -1:
            for i in range(self.n):
                self.E[i] = self.f(self.trainData[i]) - self.trainLabel[i]
        else:
            self.E[idx] = self.f(self.trainData[idx]) - self.trainLabel[idx]

    def calB(self, alpha1New, alpha1Old, alpha2New, alpha2Old, idx1, idx2, bOld):
        bNew_1 = -self.E[idx1] - self.trainLabel[idx1] * self.kernel(alpha1New, alpha1New) * (alpha1New - alpha1Old) - \
                 self.trainLabel[idx2] * self.kernel(alpha2New, alpha1New) * (alpha2New - alpha2Old) + bOld
        bNew_2 = -self.E[idx2] - self.trainLabel[idx1] * self.kernel(alpha1New, alpha2New) * (alpha1New - alpha1Old) - \
                 self.trainLabel[idx2] * self.kernel(alpha2New, alpha2New) * (alpha2New - alpha2Old) + bOld
        self.b = 0.5 * (bNew_1 + bNew_2)

    def predict(self):
        n = len(self.testLabel)
        for i in range(n):
            self.result[i] = torch.sign(self.f(self.testData[i]))
            if self.result[i] == self.testLabel[i]:
                self.accuracy += 1
        self.accuracy /= n
        print(datetime.now())
        print("Accuracy:", self.accuracy, ", C=", self.C, ", Sigma=", self.sigma, ", step=", self.count)

    def timeSpent(self, start):
        t = time.time() - start
        h = int(t / 3600)
        t = t - h * 3600
        m = int(t / 60)
        s = t - m * 60
        print(datetime.now())
        print("Time spent: %d : %d : %.2f, Step: %d. C=%d, sigma=%f" % (h, m, s, self.count, self.c, self.sigma))
        print()


def CrossValidation(trainData, trainLabel, C, sigma):
    n = 100
    accuracy = 0
    for k in range(5):
        start_all_cross = time.time()
        testData_ = trainData[k * n:(k + 1) * n, ...]
        testLabel_ = trainLabel[k * n:(k + 1) * n]
        trainData_ = torch.zeros(400, 784)
        trainLabel_ = torch.zeros(400)
        idx = 0
        for jj in range(500):
            if k * n <= jj < (k + 1) * n:
                continue
            trainData_[idx, ...] = trainData[jj, ...]
            trainLabel_[idx] = trainLabel[jj]
            idx += 1
        model = SVM(trainData=trainData_, trainLabel=trainLabel_, testData=testData_, testLabel=testLabel_,
                    sigma=sigma, C=C)
        model.SMO()
        model.predict()
        timeSpent(start_all_cross)
        print()
        accuracy += model.accuracy
    return accuracy / 5


# Timing
def timeSpent(start):
    t = time.time() - start
    h = int(t / 3600)
    t = t - h * 3600
    m = int(t / 60)
    s = t - m * 60
    print("Time spent: %d : %d : %.3f" % (h, m, s))


def load_data(file_name, N, num):
    with open(file_name) as fr:
        lines = fr.readlines()
    x = np.ones((N, 28 * 28), dtype=float)
    y = np.empty(N, dtype=int)
    for i in range(N):
        line = lines[i].strip().split(',')
        x[i] = line[1:]
        y[i] = line[0]
    # 压缩到0-1之间
    x /= 255
    # 如果是0,y就是1,否则就是-1
    y[y != num] = -1
    y[y == num] = 1
    return x, y


print("Training begins...")
for num in range(10):
    X, Y = load_data('./mnist_train.csv', num=num, N=1000)
    test_X, test_Y = load_data('./mnist_test.csv', num=num, N=100)
    trainingData = torch.tensor(X)
    trainingLabel = torch.tensor(Y)
    testData, testLabel = torch.tensor(test_X), torch.tensor(test_Y)
    C = 5000
    sigma = 50
    model = SVM(trainData=trainingData, trainLabel=trainingLabel, testData=testData, testLabel=testLabel,
                sigma=sigma, C=C)
    model.SMO()
    model.predict()
    timeSpent(stat_all)
    print("num=", num)
    print("Accuracy = ", model.accuracy)
    print(model.result)
    print("----------------------------------------------------------------------------------")
