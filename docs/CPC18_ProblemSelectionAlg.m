%% constants
MinEVa = -10;
MaxEVa = 30;
MinPay = -50;
MaxPay = 120;
MaxOutcomePay = 256;
ProbsSet =  [0.01 0.05 0.1 0.2 0.25 0.4 0.5 0.6 0.75 0.8 0.9 0.95 0.99 1];
MaxDEV = 20;
nUniDEV = 5;
pNa1 = 0.4;
pNoLotA = 0.6;
pSkewLotA = (1-pNoLotA)/2;
pNoLotB = 0.5;
pSkewLotB = 0.25;
MinSkew = -7;
MaxSkew = 8;
MaxSymmLot = 9;
pPosCorr = 0.1;
pNegCorr = 0.1;
pAmb = 0.2;
%% pool of games size
nGames = 5000;
%% prepare old problem dataset
load('DXcomp_DataEst.mat');
load('DXcomp_DataTest.mat');
oldProbs = vertcat(DataEst,DataTest);
LotShapeA = repmat({'-'},150,1);
LotNumA = ones(150,1);
oldProbs2cmp = [oldProbs(:,2:4), table(LotShapeA), table(LotNumA), oldProbs(:,5:7), oldProbs(:,10:12), oldProbs(:,9)];
oldProbs2cmp.Properties.VariableNames([9,10]) = {'LotShapeB' 'LotNumB'};
%% create pool of problems
nProbs = 0;
Games = table;
RDist = cell(nGames,1);
SDist = cell(nGames,1);
while nProbs < nGames
    [Param] = CPC18_setProblemParam(MinEVa, MaxEVa, MinPay, MaxPay, ...
        ProbsSet, MaxDEV, nUniDEV, pNa1, pNoLotA, pSkewLotA, pNoLotB, ...
        pSkewLotB, MinSkew, MaxSkew, MaxSymmLot, pPosCorr, pNegCorr, pAmb);
    distA = CPC18_getDist(Param.Ha,Param.pHa,Param.La,Param.LotShapeA,Param.LotNumA);    
    tmp = size(distA);
    nA = tmp(1);
    distB = CPC18_getDist(Param.Hb,Param.pHb,Param.Lb,Param.LotShapeB,Param.LotNumB);
    tmp = size(distB);
    nB = tmp(1);
    if distA(1,1) >= MinPay && distA(nA,1) <= MaxOutcomePay && distB(1,1) >= MinPay && distB(nB,1) <= MaxOutcomePay %if does not exceed max or min payoffs
        if Param.Ha ~= Param.Hb || Param.La ~= Param.Lb || not(strcmp(Param.LotShapeA,Param.LotShapeB)) || Param.LotNumA ~= Param.LotNumB || Param.Amb ~= 0 % if not the exact same distribution or ambigious
            if (Param.Hb ~= Param.Lb) || Param.Amb == 0 || not(strcmp(Param.LotShapeB,'-')) % if there is more than 1 outcome in R or there is no ambiguity 
                if (Param.Ha ~= Param.La && Param.Hb ~= Param.Lb) || Param.Corr == 0 % if no correlation where it doesn't make sense (no variance)
                     if not(ismember(Param,oldProbs2cmp)) && (nProbs == 0 || not(ismember(Param, Games)))
                         nProbs = nProbs + 1;
                         Games = vertcat(Games,Param);
                         nProbs
                     end
                end
            end
        end
    end
end
