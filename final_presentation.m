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
%% Bagging
pokerBag.mdlTree = fitensemble(pokerTrain.Data,pokerTrain.Labels, ...
  'Bag', 20, 'Tree',  'Type', 'classification');
%%
%[~, score]=oobPredict(poker.bag)
pokerBag.predict=predict(pokerBag.mdl,pokerTest.Data);
%%
sum(pokerBag.predict==pokerTest.Labels)

%% Boosting
pokerAdaBoost.mdlTree = fitensemble(pokerTrain.Data,pokerTrain.Labels, ...
  'AdaBoostM2', 20, 'Tree');
%%
%[~, score]=oobPredict(poker.bag)
pokerAdaBoost.predict=predict(pokerAdaBoost.mdlTree,pokerTest.Data);
%%
sum(pokerAdaBoost.predict==pokerTest.Labels)



%% Stacking


%% Results Table and Plots