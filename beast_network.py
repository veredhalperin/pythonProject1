import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import torch
from torch.nn import functional as F
from datetime import datetime
import logging
from CPC18PsychForestPython.CPC15_BEASTpred import CPC15_BEASTpred
from functions import logistic

torch.autograd.set_detect_anomaly(True)
logging.basicConfig(filename='out.log', level=logging.DEBUG)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
eps = torch.tensor([1e-20], dtype=torch.float32, requires_grad=True)


def lot_shape_convert2(lot_shape):
    if (lot_shape == [1, 0, 0, 0]).all(): return '-'
    if (lot_shape == [0, 1, 0, 0]).all(): return 'Symm'
    if (lot_shape == [0, 0, 1, 0]).all(): return 'L-skew'
    if (lot_shape == [0, 0, 0, 1]).all(): return 'R-skew'


def distSample(numbers, probabilities, rnd_num):
    tmp = F.softmax(10000 / (torch.cumsum(probabilities, 0) - rnd_num))
    return torch.reshape(
        torch.matmul(tmp, numbers),
        (1,))


def Uniform(dist):
    dist = dist + eps
    dist = dist / dist
    return dist / torch.reshape(torch.sum(dist), (1,))


class Outcome(torch.nn.Module):
    def __init__(self):
        super(Outcome, self).__init__()
        self.Linear = torch.nn.Linear(6, 6)
        self.LogSoftMax = torch.nn.LogSoftmax()

    def forward(self, pBias, trial, SignMax, RatioMin, gama,
                Range, values, dist, ObsPay, rndNum1, rndNum2, Min):
        Unbiased1 = distSample(values, dist, rndNum2)
        Unbiased2 = distSample(ObsPay, Uniform(ObsPay), rndNum2)
        Pessimism1 = Min
        Pessimism2 = distSample(values, Uniform(values), rndNum2)
        Sign1 = Range * distSample(F.softsign(values), dist, rndNum2)
        Sign2 = Range * distSample(F.softsign(ObsPay), Uniform(ObsPay), rndNum2)
        if rndNum1 > pBias and trial < torch.tensor([5], dtype=torch.float32, requires_grad=True):
            return Unbiased1
        elif rndNum1 > pBias:
            return Unbiased2
        elif rndNum1 > (2 / 3) * pBias:
            return Pessimism2
        elif rndNum1 > (1 / 3) * pBias and SignMax > 0 and RatioMin < gama:
            return Pessimism1
        elif rndNum1 > (1 / 3) * pBias:
            return Pessimism2
        elif trial < torch.tensor([5], dtype=torch.float32, requires_grad=True):
            return Sign1
        else:
            return Sign2
        """
        output=torch.cat((Unbiased1,Unbiased2,Pessimism1,Pessimism2,Sign1,Sign2))
        input=torch.cat((pBias, trial, SignMax, RatioMin, gama,rndNum1))
        oneHot=F.gumbel_softmax(self.LogSoftmax(self.Linear(input)),hard=True,tau=0.01)
        return torch.reshape(torch.matmul(oneHot,output),(1,))
        """


def Decision(BEVb, BEVa, kapa, pBias, trial, SignMax, RatioMin, gama, Range, valuesA, distA, ObsPayA, MinA, valuesB,
             pEstB, ObsPayB, MinB, Outcome, Dom, sigma):
    STa = torch.tensor([0], dtype=torch.float32, requires_grad=True)
    STb = torch.tensor([0], dtype=torch.float32, requires_grad=True)
    for s in range(kapa):
        rndNum1 = torch.rand(size=(1,), dtype=torch.float32, requires_grad=True)
        rndNum2 = torch.rand(size=(1,), dtype=torch.float32, requires_grad=True)
        STa = STa + Outcome(pBias, trial, SignMax, RatioMin, gama,
                            Range, valuesA, distA, ObsPayA, rndNum1, rndNum2, MinA)
        STb = STb + Outcome(pBias, trial, SignMax, RatioMin, gama,
                            Range, valuesB, pEstB, ObsPayB, rndNum1, rndNum2, MinB)
    kapaTensor = torch.tensor([kapa], dtype=torch.float32, requires_grad=True)
    STa = STa / kapaTensor
    STb = STb / kapaTensor
    return F.sigmoid(BEVb - BEVa + STb-STa), trial + torch.tensor([1],
                                                                                                    dtype=torch.float32,
                                                                                                    requires_grad=True)


def Trial(trial, BEVb, Amb, beta, theta, valuesA, distA, valuesB, distB, ObsPayA, ObsPayB, Corr):
    rndNumObs = torch.rand(size=(1,), dtype=torch.float32, requires_grad=True)
    prob = Corr * rndNumObs + torch.max(torch.cat((torch.tensor([-1], dtype=torch.float32, requires_grad=True) * Corr,
                                                   torch.tensor([0], dtype=torch.float32, requires_grad=True)))) + (
                   torch.tensor([1], dtype=torch.float32, requires_grad=True) - Corr * Corr) * torch.rand(size=(1,),
                                                                                                          dtype=torch.float32,
                                                                                                          requires_grad=True)
    lastB = distSample(valuesB, distB, prob)
    BEVb = Amb * (torch.tensor([0.95], dtype=torch.float32, requires_grad=True) * BEVb + torch.tensor([0.05],
                                                                                                      dtype=torch.float32,
                                                                                                      requires_grad=True) * lastB) + (
                   torch.tensor([1], dtype=torch.float32, requires_grad=True) - Amb) * BEVb
    pBias = beta / (beta + torch.tensor([1], dtype=torch.float32, requires_grad=True) + torch.pow(
        trial - torch.tensor([4], dtype=torch.float32, requires_grad=True), theta))
    return BEVb, pBias, torch.cat((ObsPayA, distSample(valuesA, distA, rndNumObs))), torch.cat(
        (ObsPayB, lastB))


class CPC15_BEASTpred(torch.nn.Module):
    def __init__(self):
        super(CPC15_BEASTpred, self).__init__()
        self.W = torch.nn.parameter.Parameter(
            torch.tensor([0.92, 0.97, 1, 1, 1], dtype=torch.float32))
        # self.Outcome = Outcome()

    def forward(self, Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr):
        Prediction = CPC15_BEASTpred(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr)
        Prediction.shape = (5,)
        return self.W * torch.tensor(Prediction, dtype=torch.float32)

        """
        BEVa = torch.reshape(torch.matmul(valuesA, distA), (1,))
        Prediction = torch.tensor([[0], [0], [0], [0], [0]], dtype=torch.float32, requires_grad=True)
        MaxB = torch.reshape(torch.max(valuesB), (1,))
        MaxOutcome = torch.reshape(torch.max(torch.cat((torch.reshape(torch.max(valuesA), (1,)), MaxB))), (1,))
        SignMax = F.softsign(MaxOutcome)
        MinA = torch.reshape(torch.min(valuesA), (1,))
        MinB = torch.reshape(torch.min(valuesB), (1,))
        absMinA = torch.abs(MinA)
        absMinB = torch.abs(MinB)
        Range = MaxOutcome - torch.reshape(torch.min(torch.cat((MinA, MinB))), (1,))
        RatioMin = torch.reshape(torch.max(torch.cat((torch.tensor([0], dtype=torch.float32, requires_grad=True),
                                                      torch.reshape(torch.min(torch.cat((absMinA, absMinB))), (1,)) / (
                                                              torch.reshape(
                                                                  torch.max(torch.cat((absMinA, absMinB))),
                                                                  (1,)) + eps)))), (1,))
        valuesBEps = valuesB + eps
        Len = torch.reshape(torch.sum(valuesBEps / valuesBEps), (1,)) - torch.tensor([1], dtype=torch.float32,
                                                                                     requires_grad=True)
        noMax = valuesB - MaxB
        Ones = noMax / (noMax + eps)
        Mean = torch.reshape(torch.sum(valuesB - MinB), (1,)) / Len + MinB
        distB = torch.cat((distB, torch.tensor([0], dtype=torch.float32, requires_grad=True)))
        valuesBOld = valuesB
        valuesB = torch.cat((valuesB, MaxB))
        nSims = 1
        nsimsTensor = torch.tensor([nSims], dtype=torch.float32, requires_grad=True)
        for sim in range(nSims):
            start = datetime.now()
            kapa = int(np.random.choice(range(1, 3), 1))
            beta = (torch.tensor([2.6], dtype=torch.float32, requires_grad=True)) * torch.rand(size=(1,),
                                                                                               dtype=torch.float32,
                                                                                               requires_grad=True)
            gama = (torch.tensor([0.5], dtype=torch.float32, requires_grad=True)) * torch.rand(size=(1,),
                                                                                               dtype=torch.float32,
                                                                                               requires_grad=True)
            psi = (torch.tensor([0.07], dtype=torch.float32, requires_grad=True)) * torch.rand(size=(1,),
                                                                                               dtype=torch.float32,
                                                                                               requires_grad=True)
            theta = (torch.tensor([1], dtype=torch.float32, requires_grad=True)) * torch.rand(size=(1,),
                                                                                              dtype=torch.float32,
                                                                                              requires_grad=True)
            sigma = (torch.tensor([7], dtype=torch.float32, requires_grad=True)) * torch.rand(size=(1,),
                                                                                              dtype=torch.float32,
                                                                                              requires_grad=True)
            ObsPayA = torch.tensor([], dtype=torch.float32, requires_grad=True)
            ObsPayB = torch.tensor([], dtype=torch.float32, requires_grad=True)
            pBias = beta / (beta + torch.tensor([1], dtype=torch.float32, requires_grad=True))
            BEVb = Amb * ((torch.tensor([1], dtype=torch.float32, requires_grad=True) - psi) * (
                    torch.reshape(torch.matmul(valuesBOld, Uniform(valuesBOld)), (1,)) + BEVa) / torch.tensor([2],
                                                                                                              dtype=torch.float32,
                                                                                                              requires_grad=True) + psi * MinB) + (
                           torch.tensor([1], dtype=torch.float32, requires_grad=True) - Amb) * torch.reshape(
                torch.matmul(valuesB, distB), (1,))
            t_SPminb = torch.reshape(torch.min(torch.cat((torch.tensor([1], dtype=torch.float32, requires_grad=True),
                                                          torch.reshape(torch.max(torch.cat((torch.tensor([0],
                                                                                                          dtype=torch.float32,
                                                                                                          requires_grad=True),
                                                                                             (BEVb - Mean) / (
                                                                                                     MinB - Mean + eps)))),
                                                                        (1,))))), (1,))
            pEstB = Amb * torch.cat(
                (t_SPminb, Ones * (torch.tensor([1], dtype=torch.float32, requires_grad=True) - t_SPminb) / Len)) + (
                            torch.tensor([1], dtype=torch.float32, requires_grad=True) - Amb) * distB
            trial = torch.tensor([0], dtype=torch.float32, requires_grad=True)
            Decision1, trial = Decision(BEVb, BEVa, kapa, pBias, trial, SignMax, RatioMin, gama, Range, valuesA, distA,
                                        ObsPayA, MinA, valuesB, pEstB, ObsPayB, MinB, self.Outcome,Dom,sigma)
            Decision2, trial = Decision(BEVb, BEVa, kapa, pBias, trial, SignMax, RatioMin, gama, Range, valuesA, distA,
                                        ObsPayA, MinA, valuesB, pEstB, ObsPayB, MinB, self.Outcome,Dom,sigma)
            Decision3, trial = Decision(BEVb, BEVa, kapa, pBias, trial, SignMax, RatioMin, gama, Range, valuesA, distA,
                                        ObsPayA, MinA, valuesB, pEstB, ObsPayB, MinB, self.Outcome,Dom,sigma)
            Decision4, trial = Decision(BEVb, BEVa, kapa, pBias, trial, SignMax, RatioMin, gama, Range, valuesA, distA,
                                        ObsPayA, MinA, valuesB, pEstB, ObsPayB, MinB, self.Outcome,Dom,sigma)
            Decision5, trial = Decision(BEVb, BEVa, kapa, pBias, trial, SignMax, RatioMin, gama, Range, valuesA, distA,
                                        ObsPayA, MinA, valuesB, pEstB, ObsPayB, MinB, self.Outcome,Dom,sigma)
            simPred = torch.reshape(torch.mean(torch.cat((Decision1, Decision2, Decision3, Decision4, Decision5))),
                                    (1,))
            for block in range(5, 25, 5):
                BEVb, pBias, ObsPayA, ObsPayB = Trial(trial, BEVb, Amb, beta, theta, valuesA, distA, valuesB, distB,
                                                      ObsPayA, ObsPayB, Corr)
                Decision1, trial = Decision(BEVb, BEVa, kapa, pBias, trial, SignMax, RatioMin, gama, Range, valuesA,
                                            distA, ObsPayA, MinA, valuesB, pEstB, ObsPayB, MinB, self.Outcome,Dom,sigma)
                BEVb, pBias, ObsPayA, ObsPayB = Trial(trial, BEVb, Amb, beta, theta, valuesA, distA, valuesB, distB,
                                                      ObsPayA, ObsPayB, Corr)
                Decision2, trial = Decision(BEVb, BEVa, kapa, pBias, trial, SignMax, RatioMin, gama, Range, valuesA,
                                            distA, ObsPayA, MinA, valuesB, pEstB, ObsPayB, MinB, self.Outcome,Dom,sigma)
                BEVb, pBias, ObsPayA, ObsPayB = Trial(trial, BEVb, Amb, beta, theta, valuesA, distA, valuesB, distB,
                                                      ObsPayA, ObsPayB, Corr)
                Decision3, trial = Decision(BEVb, BEVa, kapa, pBias, trial, SignMax, RatioMin, gama, Range, valuesA,
                                            distA, ObsPayA, MinA, valuesB, pEstB, ObsPayB, MinB, self.Outcome,Dom,sigma)
                BEVb, pBias, ObsPayA, ObsPayB = Trial(trial, BEVb, Amb, beta, theta, valuesA, distA, valuesB, distB,
                                                      ObsPayA, ObsPayB, Corr)
                Decision4, trial = Decision(BEVb, BEVa, kapa, pBias, trial, SignMax, RatioMin, gama, Range, valuesA,
                                            distA, ObsPayA, MinA, valuesB, pEstB, ObsPayB, MinB, self.Outcome,Dom,sigma)
                BEVb, pBias, ObsPayA, ObsPayB = Trial(trial, BEVb, Amb, beta, theta, valuesA, distA, valuesB, distB,
                                                      ObsPayA, ObsPayB, Corr)
                Decision5, trial = Decision(BEVb, BEVa, kapa, pBias, trial, SignMax, RatioMin, gama, Range, valuesA,
                                            distA, ObsPayA, MinA, valuesB, pEstB, ObsPayB, MinB, self.Outcome,Dom,sigma)
                simPred = torch.cat((simPred, torch.reshape(
                    torch.mean(torch.cat((Decision1, Decision2, Decision3, Decision4, Decision5))), (1,))))
            Prediction = Prediction + torch.reshape(self.w1 * simPred, (5, 1)) / nsimsTensor
            print(datetime.now() - start)
        return Prediction
        """


if __name__ == '__main__':
    Data = pd.read_csv('real.csv')
    model = CPC15_BEASTpred()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    nProblems = Data.shape[0]
    for i in range(4):
        for prob in range(0, nProblems):
            print(i * prob + prob)
            start = datetime.now()
            """
            DistA = CPC18_getDist(Data['Ha'][prob], Data['pHa'][prob], Data['La'][prob], LotShapeConvert(
                Data[['lot_shape__A', 'lot_shape_symm_A', 'lot_shape_L_A', 'lot_shape_R_A']].values[prob]),
                                  int(Data['LotNumA'][prob]))
            DistB = CPC18_getDist(Data['Hb'][prob], Data['pHb'][prob], Data['Lb'][prob], LotShapeConvert(
                Data[['lot_shape__B', 'lot_shape_symm_B', 'lot_shape_L_B', 'lot_shape_R_B']].values[prob]),
                                  int(Data['LotNumB'][prob]))
            trivial=CPC15_isStochasticDom(DistA,DistB)
            if trivial:
                dom=0
            else:
                dom=1
            loss = criterion(model(
                torch.tensor(DistA[:, 0], dtype=torch.float32, requires_grad=True),
                torch.tensor(DistA[:, 1], dtype=torch.float32, requires_grad=True),
                torch.tensor(DistB[:, 0], dtype=torch.float32, requires_grad=True),
                torch.tensor(DistB[:, 1], dtype=torch.float32, requires_grad=True),
                torch.tensor([Data['Amb'][prob]], dtype=torch.float32, requires_grad=True),
                torch.tensor([Data['Corr'][prob]], dtype=torch.float32, requires_grad=True),
                torch.tensor([dom], dtype=torch.float32, requires_grad=True)),
                torch.tensor([[Data['B.' + str(i)][prob]] for i in range(1, 6)], dtype=torch.float32, requires_grad=True))
            """
            # read problem's parameters
            Ha = Data['Ha'][prob]
            pHa = Data['pHa'][prob]
            La = Data['La'][prob]
            LotShapeA = lot_shape_convert2(
                Data[['lot_shape__A', 'lot_shape_symm_A', 'lot_shape_L_A', 'lot_shape_R_A']].values[prob])
            LotNumA = int(Data['LotNumA'][prob])
            Hb = Data['Hb'][prob]
            pHb = Data['pHb'][prob]
            Lb = Data['Lb'][prob]
            LotShapeB = lot_shape_convert2(
                Data[['lot_shape__B', 'lot_shape_symm_B', 'lot_shape_L_B', 'lot_shape_R_B']].values[prob])
            LotNumB = int(Data['LotNumB'][prob])
            Amb = Data['Amb'][prob]
            Corr = Data['Corr'][prob]
            true = torch.tensor([Data['B.' + str(i)][prob] for i in range(1, 6)], dtype=torch.float32,
                                requires_grad=True)
            pred = model(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr)
            loss = criterion(pred, true)
            logging.info(prob)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            for name, param in model.named_parameters():
                print(name, param)
            print(datetime.now() - start)
