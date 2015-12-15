%% Test notes
%% % *Weak classifiers: Tree,Knn,Discriminant
    %* Tweak Nlearners
    %* other params (
    %* for Bag: try 'regression' too
    %* for Boost: try different boosts
    %* for Stacking ?:
   %For table: use start=tic; time=toc(start); and Evaluate fcn (needs modifiyng for multi class)
        % So we have same data to put on table
   %plots: dimention vs. class , 2-dimension vs. class rtc.
        % http://web.cs.wpi.edu/~ruiz/KDDRG/Resources/Clustering/viewClusters.png
        % have the plot template on github if you have one please.
        %rsLoss = resubLoss(ClassTreeEns,'Mode','Cumulative');
   %
%%
rng(12312,'twister')
%%
load poker-hand-testing.data
load poker-hand-training-true.data
%%
pokerTest.Data  = poker_hand_testing(:,1:end-1);
pokerTest.Labels = poker_hand_testing(:,end);

pokerTrain.Data = poker_hand_training_true(:,1:end-1);
pokerTrain.Labels = poker_hand_training_true(:,end);
%% Creating claffication noise to test
% http://www.phillong.info/publications/LS10_potential.pdf
T = length(pokerTrain.Data);
LabelNoise = round(normrnd(0, 0.32, T,1));
pokerTrain.NoisyLabels = pokerTrain.Labels + LabelNoise;
%error ratio
er_rat = sum(pokerTrain.NoisyLabels ~= pokerTrain.Labels)/T
disp(er_rat>0.1)
%% Bagging
pokerBag.mdlTree = fitensemble(pokerTrain.Data,pokerTrain.Labels, ...
  'Bag', 20, 'Tree',  'Type', 'classification');
%%
%[~, score]=oobPredict(poker.bag)
pokerBag.predict=predict(pokerBag.mdlTree,pokerTest.Data);
%%
sum(pokerBag.predict==pokerTest.Labels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% B. Boosting
finalBoost 



%% Stacking


%% Results Table and Plots