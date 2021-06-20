function [Param] = DXcomp_setParamDesXper(MinEVa, MaxEVa, MinPay, MaxPay, ...
    ProbsSet, MaxDEV, nUniDEV, pNa1, pNoNoise, pSkewNoise, ...
    MaxSkew, MaxSymmNoise, pPosCorr, pNegCorr, pUncertainty, synth)
		% compute safe outcomes and probs
		%1. 	Draw randomly EVA’ ~ Uni(-10, 30) (a continuous uniform distribution) 
		
	tempEVa = randi([MinEVa,MaxEVa],1,1); 		
	%2. 	Draw number of outcomes for Option A, NA: 1 with probability .5; 2 otherwise. 
	% 2.1. 	If NA = 1 then set LA = HA = Round(EVA’); pHA = 1 
	if rand(1) < pNa1 %Ns = 1
		if synth == 15						%%%% Meghan %%%%
			LotNumA = 1 ;					%%%% Meghan %%%%
		end
		La = round(tempEVa);
		pHa = 1;
		Ha = La;
		%2.2. 	If NA = 2 then 
	else % Na=2
		if synth == 15						%%%% Meghan %%%%
			LotNumA = 2 ; 					%%%% Meghan %%%%
		end
		% draw pHA uniformly from the set {.01, .05, .1, .2, .25, .4, .5, .6, .75, .8, .9, .95, .99, 1} 
		pHa = ProbsSet(randi(length(ProbsSet),1));
		% 2.2.1. 	If pHA = 1 then set LA = HA = Round(EVA’) 
		if pHa == 1 || pHa == 0
			La = round(tempEVa);
			Ha = La;
			%  2.2.2. 	If pHA < 1 then draw an outcome temp ~ Triangular[-50, EVA’, 120] 
		else
			a = MinPay;
			b = MaxPay;
			c = tempEVa;
			temp = c+sqrt(rand(1)).*(a-c+rand(1)*(b-a)); %Triangular dist
			% 2.2.2.2. 	If Round(temp) > EVA’ then set HA = Round(temp); LA = Round[EVA’ – HA ∙ pHA/(1 – pHA)] 
			if round(temp) > c %&& pHa < 1
				Ha = round(temp);
				La = round((tempEVa - Ha*pHa)/(1-pHa));
				% 2.2.2.1. 	If Round(temp) < EVA’ then set LA = Round(temp); HA = Round{[EVA’ – LA(1 – pHA)]/pHA} 
			elseif round(temp) < c %&& pHa > 0
				La = round(temp);
				Ha = round((tempEVa - La*(1-pHa))/pHa);
			else
				Ha = round(tempEVa);
				La = round(tempEVa);
			end
		end
	end
	EVa = La * (1-pHa) + Ha * pHa;

	% compute B outcomes and probs
	% 3. 	Draw difference in expected values between options, DEV: DEV = (1/5)∙∑Ui, where Ui ~ Uni[-20, 20] 
	DEV = mean(2*MaxDEV*rand(nUniDEV,1)-MaxDEV);
	% 4. 	Set EVB’ = EVA + DEV , where EVA is the real expected value of Option A. 
	tempEVb = EVa + DEV;

	if tempEVb > MinPay 
		% 5. 	Draw pHB uniformly from the set {.01, .05, .1, .2, .25, .4, .5, .6, .75, .8, .9, .95, .99, 1}
		pHb = ProbsSet(randi(length(ProbsSet),1));
		%       5.1. 	If pHB = 1 then set LB = HB = Round(EVB’)
		if pHb == 1 || pHb == 0
			Lb = round(tempEVa);
			Hb = Lb;
			% 5.2. 	If pHB < 1 then draw an outcome temp ~ Triangular[-50, EVB’, 120]
		else
			a = MinPay;
			b = MaxPay;
			c = tempEVb;
			temp = c+sqrt(rand(1)).*(a-c+rand(1)*(b-a)); 
			% 5.2.2. 	If Round(temp) > EVB’ then set HB = Round(temp); LB = Round[(EVB’ – HB ∙ pHB)/(1 – pHB)] 
			 if round(temp) > c
				Hb = round(temp);
				Lb = round((tempEVb - Hb*pHb)/(1-pHb));
				%  5.2.1. 	If Round(temp) < EVB’ then set LB = Round(temp); HB = Round{[EVB’ – LB(1 – pHB)]/pHB} 
			 elseif round(temp) < c
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
	% 6. 	Set lottery 
	% compute B Option noise distribution
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Option B %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% %6.1.  With probability .5,
	rndNum = rand(1);
	%  the lottery is degenerate. Set LotNum = 1 and LotShpae = "-" 
	
	if rndNum < pNoNoise % no noise
		noiseType = 0 ;
		noiseParam = 0 ;
		LotNumB = 1 ;		%%%% Meghan %%%%
		LotShapeB = 0 ;
	% 6.2.  With probability .25, the lottery is skewed. Draw temp uniformly from the set {-7, -6, … , 3, 2, 2, 3, … , 7, 8} 
	elseif rndNum <  pNoNoise + pSkewNoise
		noiseType = 1;
		noiseParam = randi([-1*MaxSkew,MaxSkew],1,1);
		while abs(noiseParam) < 2 
			noiseParam = randi([-1*MaxSkew,MaxSkew],1,1);
		end
		if noiseParam > 0 							%%%% Meghan %%%%
			LotNumB = noiseParam;					%%%% Meghan %%%%
			LotShapeB = 1 ;					%%%% Meghan %%%%
		elseif noiseParam < 0 
			LotNumB = - noiseParam; 				%%%% Meghan %%%%
			LotShapeB = 3 ;					%%%% Meghan %%%%
		end
	else
		%      6.3.  With probability .25, the lottery is symmetric. Set LotShape = "Symm" and draw LotNum uniformly from the set {3, 5, 7, 9}
		noiseType = 2;
		possibleN = 2:2:MaxSymmNoise;
		noiseParam = possibleN(randi(length(possibleN),1));
		LotNumB = noiseParam; 					%%%% Meghan %%%%
		LotShapeB = 2 ; 						%%%% Meghan %%%%
	end


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Option A %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% %6.1.  With probability .5,
	rndNum = rand(1);
	%  the lottery is degenerate. Set LotNum = 1 and LotShpae = "-" 
	if rndNum < pNoNoise % no noise
		noiseType = 0;
		noiseParam = 0;
		if synth == 18
			LotNumA = 1 ;				%%%% Meghan %%%%
		end
		LotShapeA = 0;			%%%% Meghan %%%%
		% 6.2.  With probability .25, the lottery is skewed. Draw temp uniformly from the set {-7, -6, … , 3, 2, 2, 3, … , 7, 8} 
	elseif rndNum <  pNoNoise + pSkewNoise
		noiseType = 1;
		noiseParam = randi([-1*MaxSkew,MaxSkew],1,1);
		while abs(noiseParam) < 2 
			noiseParam = randi([-1*MaxSkew,MaxSkew],1,1);
		end
		if noiseParam > 0 							%%%% Meghan %%%%
			LotShapeA = 1 ;					%%%% Meghan %%%%
			if synth == 18
				LotNumA = noiseParam;					%%%% Meghan %%%%
			end
		elseif noiseParam < 0
			LotShapeA = 3 ;					%%%% Meghan %%%%
			if synth == 18
				LotNumA = - noiseParam; 				%%%% Meghan %%%%
			end
		end
	else
		%6.3.  With probability .25, the lottery is symmetric. Set LotShape = "Symm" and draw LotNum uniformly from the set {3, 5, 7, 9}
		noiseType = 2;
		possibleN = 2:2:MaxSymmNoise;
		noiseParam = possibleN(randi(length(possibleN),1));
		if synth == 18
			LotNumA = noiseParam; 					%%%% Meghan %%%%
		end
		LotShapeA = 2 ; 						%%%% Meghan %%%%

	end 

	% compute correlation
	rndNum = rand(1);
	if rndNum < pPosCorr
		Corr = 1;
	elseif rndNum < pPosCorr + pNegCorr
		Corr = -1;
	else
		Corr = 0;
	end

	% compute uncertainty
	if rand(1) < pUncertainty
		Uncertainty = 1;
	else
		Uncertainty = 0;
	end

	% sum up evreything by order
	Param = [Ha, pHa, La, LotNumA, LotShapeA, Hb, pHb, Lb, LotNumB, LotShapeB, Corr, Uncertainty]; 
	% Param = [Ha, pHa, La, Hb, pHb, Lb, Corr, Uncertainty, noiseType, noiseParam]; 

	end