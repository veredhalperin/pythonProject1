function [Param] = CPC18_setProblemParam(MinEVa, MaxEVa, MinPay, MaxPay, ...
    ProbsSet, MaxDEV, nUniDEV, pNa1, pNoLotA, pSkewLotA, pNoLotB, ...
    pSkewLotB, MinSkew, MaxSkew, MaxSymmLot, pPosCorr, pNegCorr, pAmb)

% compute A outcomes and probs
tempEVa = randi([MinEVa,MaxEVa],1,1); % step 1
if rand(1) < pNa1 %Na = 1; step 2.1
    La = round(tempEVa);
    pHa = 1;
    Ha = La;
    LotShapeA = {'-'};
    LotNumA = 1;    
else % Na>1; % step 2.2
    pHa = ProbsSet(randi(length(ProbsSet),1));
    if pHa == 1 || pHa == 0 % step 2.2.1
        La = round(tempEVa);
        Ha = La;
    else % step 2.2.2
        a = MinPay;
        b = MaxPay;
        c = tempEVa;
        temp = c+sqrt(rand(1)).*(a-c+rand(1)*(b-a)); %Triangular dist

        if round(temp) > c % step 2.2.2.2
            Ha = round(temp);
            La = round((tempEVa - Ha*pHa)/(1-pHa));
        elseif round(temp) < c % step 2.2.2.1
            La = round(temp);
            Ha = round((tempEVa - La*(1-pHa))/pHa);
        else
            Ha = round(tempEVa);
            La = round(tempEVa);
        end
    end
    % compute A Option lottery distribution; step 2.2.3
    rndNum = rand(1);
    if rndNum < pNoLotA % no lot; step 2.2.3.1
        LotShapeA = {'-'};
        LotNumA = 1;
    elseif rndNum <  pNoLotA + pSkewLotA % step 2.2.3.2
        tmpParam = randsample([MinSkew:1:-2,2:1:MaxSkew],1);
        if tmpParam > 0 
            LotShapeA = {'R-skew'};
            LotNumA = tmpParam;
        else
            LotShapeA = {'L-skew'};
            LotNumA = -tmpParam;         
        end
        %         noiseParamA = randi([MinSkew,MaxSkew],1,1);
    else % step 2.2.3.3
        LotShapeA = {'Symm'};
        possibleN = 3:2:MaxSymmLot;
        LotNumA = possibleN(randi(length(possibleN),1));
    end
end
EVa = La * (1-pHa) + Ha * pHa; % for step 4

% compute B outcomes and probs
DEV = mean(2*MaxDEV*rand(nUniDEV,1)-MaxDEV); %step 3
tempEVb = EVa + DEV; % step 4
if tempEVb > MinPay % for step 4.1
    pHb = ProbsSet(randi(length(ProbsSet),1)); % step 5
    if pHb == 1 || pHb == 0 % step 5.1
        Lb = round(tempEVa);
        Hb = Lb;
    else % step 5.2
        a = MinPay;
        b = MaxPay;
        c = tempEVb;
        temp = c+sqrt(rand(1)).*(a-c+rand(1)*(b-a)); 

         if round(temp) > c %step 5.2.2
            Hb = round(temp);
            Lb = round((tempEVb - Hb*pHb)/(1-pHb));
         elseif round(temp) < c % step 5.2.1
            Lb = round(temp);
            Hb = round((tempEVb - Lb*(1-pHb))/pHb);
         else
             Hb = round(tempEVb);
             Lb = round(tempEVb);
         end
    end
else % EV(Risk) is lower than minimal possible pay
    Lb = -999;
    pHb = -999;
    Hb = -999;
end

% compute B Option noise distribution; step 6
rndNum = rand(1);
if rndNum < pNoLotB % no noise; step 6.1
    LotShapeB = {'-'};
    LotNumB = 1;
elseif rndNum <  pNoLotB + pSkewLotB % step 6.2
   tmpParam = randsample([MinSkew:1:-2,2:1:MaxSkew],1);
   if tmpParam > 0 
       LotShapeB = {'R-skew'};
       LotNumB = tmpParam;
   else
       LotShapeB = {'L-skew'};
       LotNumB = -tmpParam;         
   end
%     noiseParamB = randi([-1*MaxSkew,MaxSkew],1,1);
else % step 6.3
    LotShapeB = {'Symm'};
    possibleN = 3:2:MaxSymmLot;
    LotNumB = possibleN(randi(length(possibleN),1));
end

% compute correlation; step 7
rndNum = rand(1);
if rndNum < pPosCorr
    Corr = 1;
elseif rndNum < pPosCorr + pNegCorr
    Corr = -1;
else
    Corr = 0;
end

% compute ambiguity; step 8
if rand(1) < pAmb
    Amb = 1;
else
    Amb = 0;
end

% sum up evreything by order
Param = table(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Corr, Amb); 

end