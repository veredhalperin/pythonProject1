import numpy as np
import pandas as pd


def CPC15_isStochasticDom (DistA, DistB):
    # Check if one distribution dominates stochastically the other
    # Input: 2 discrete distributions which are set as matrices of 1st column
    # as outcome and 2nd its probability (DistA and DistB are numpy matrices)
    # Output: pandas data framed with 2 columns:
    # 'is' a logical output, 'which' a char output ('A', 'B', NaN)

    na = DistA.shape[0]
    nb = DistB.shape[0]
    if np.array_equal(DistA, DistB):
        dom = False
        which = None
    else:
        tempa = np.ones(shape=(na, 1))
        tempb = np.ones(shape=(nb, 1))
        for i in range(0, nb):
            sumpa = 0  # DistA(i,2)
            j = 0
            sumpb = np.sum(DistB[0:i + 1, 1])

            while (sumpa != 1) and (j < na) and (sumpa + DistA[j, 1] <= sumpb):
                sumpa += DistA[j, 1]
                if sumpa == sumpb:
                    break
                j += 1

            if j == na:
                j = na - 1
            if i == nb:
                i = nb - 1

            if DistB[i, 0] < DistA[j, 0]:
                tempb[i] = 0
                break

        if np.all(tempb != 0):
            dom = True
            which = 'B'
        else:
            for i in range(0, na):
                sumpb = 0  # DistA(i,2)
                j = 0
                sumpa = np.sum(DistA[0: i+1, 1])

                while (sumpb != 1) and (j < nb) and (sumpb + DistB[j, 1] <= sumpa):
                    sumpb += DistB[j, 1]
                    if sumpa == sumpb:
                        break
                    j += 1

                if j == nb:
                    j = nb - 1
                if i == na:
                    i = na - 1

                if DistA[i, 0] < DistB[j, 0]:
                    tempa[i] = 0
                    break

            if np.all(tempa != 0):
                dom = True
                which = 'A'
            else:
                dom = False
                which = None

    return pd.DataFrame([{'dom': dom, 'which': which}])
