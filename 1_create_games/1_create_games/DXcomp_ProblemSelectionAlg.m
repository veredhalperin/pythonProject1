%% constants
MinEVa = -10;
MaxEVa = 30;
MinPay = -50;
MaxPay = 120;
ProbsSet =  [0.01 0.05 0.1 0.2 0.25 0.4 0.5 0.6 0.75 0.8 0.9 0.95 0.99 1];
MaxDEV = 20;
nUniDEV = 5;
pNa1 = 0.5;
pNoNoise = 0.5;
pSkewNoise = 0.25;
MaxSkew = 8;
MaxSymmNoise = 8;
pPosCorr = 0.1;
pNegCorr = 0.1;
pUncertainty = 0.2;
nGames = 15000; 		%%% Meghan %%%
synth = 18; 			%%% Meghan %%%

%%
nProbs = 0;
Games = zeros(nGames,12);
% RDist = cell(nGames,1);
% SDist = cell(nGames,1);
while nProbs < nGames
    [Param] = DXcomp_setParamDesXper(MinEVa, MaxEVa, MinPay, MaxPay, ...
        ProbsSet, MaxDEV, nUniDEV, pNa1, pNoNoise, pSkewNoise, ...
        MaxSkew, MaxSymmNoise, pPosCorr, pNegCorr, pUncertainty, synth);
    Ha = Param(1);
    pHa = Param(2);
    La = Param(3);
	LotNumA = Param(4);
	LotShapeA = Param(5); 
    Hb = Param(6);
    pHb = Param(7);
    Lb = Param(8);
    LotNumB = Param(9);
    LotShapeB = Param(10);
    corr = Param(11);
    Amb = Param(12);
    if Ha < MaxPay && Hb < MaxPay && Lb >= MinPay && La >= MinPay %if does not exceed max or min payoffs
        if Ha ~= Hb || La ~= Lb || LotShapeA ~=  0 || LotShapeB ~=  0 || Amb ~= 0 % if not the exact same distribution or ambigious
            if (Ha ~= La && Hb ~= Lb) || corr == 0 % if no correlation where it doesn't make sense (no variance)
			%%% Meghan %%%
                if LotShapeA ~=  1  || LotShapeA ~=  -1  || (LotNumA ~= 1 && LotNumA ~=0 && LotNumA ~= -1) || LotShapeB ~=  1  || LotShapeB ~=  -1   || (LotNumB ~= 1 && LotNumB ~=0 && LotNumB ~= -1)  % if Skew noise with irrelevant parameter
                    if LotShapeA ~=  1  || LotShapeA ~=  -1 || (LotNumA > -6 && LotNumA < 7) || (LotNumA == -6 && Ha >= 7) || (LotNumA == -7 && Ha >= 70) || (LotNumA == 7 && Ha <= 136) || (LotNumA == 8 && Ha <= 9) || LotShapeB ~=  1  || LotShapeB ~=  -1  || (LotNumB > -6 && LotNumB < 7) || (LotNumB == -6 && Hb >= 7) || (LotNumB == -7 && Hb >= 70) || (LotNumB == 7 && Hb <= 136) || (LotNumB == 8 && Hb <= 9)
                        if (Hb ~= Lb) || (Ha ~= La) || Amb == 0 || LotShapeA ~= 0 || LotShapeB ~= 0 % if there is more than 1 outcome in R or there is no ambiguity 
                            nProbs = nProbs + 1;
                            Games(nProbs,:) = Param;
%                             RDist{nProbs} = getDesXperDist(Hb, pHb, Lb, LotShape, LotNum,max(MaxSymmNoise+2,MaxSkew+1));
%                             SDist{nProbs} = getDesXperDist(Ha, pHa, La, 0, 0, 2);
                        end
                    end
                end            
            end
        end
    end
    
end


