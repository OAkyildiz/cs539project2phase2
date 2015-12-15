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
%% Poker Data set
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
pokerTrain.er_rat = sum(pokerTrain.NoisyLabels ~= pokerTrain.Labels)/T;

%%  Wine Data set
load wine.data
%% stratify
windex =cvpartition(wine(:,1),'Holdout',0.25);
wineTrain.Data=wine(windex.training(),2:end);
wineTrain.Labels=wine(windex.training(),1);

wineTest.Data=wine(windex.test(),2:end);
wineTest.Labels=wine(windex.test(),1);
%% Classification noise
T = length(wineTrain.Data);
LabelNoise = round(normrnd(0, 0.5, T,1));
wineTrain.NoisyLabels = wineTrain.Labels + LabelNoise;
wineTrain.er_rat = sum(wineTrain.NoisyLabels ~= wineTrain.Labels)/T;
%% Page_block Data set

load page-blocks.data
%% stratify
pagedex =cvpartition(page_blocks(:,1),'Holdout',0.25);
pageTrain.Data=page_blocks(pagedex.training(),1:end-1);
pageTrain.Labels=page_blocks(pagedex.training(),end);

pageTest.Data=page_blocks(pagedex.test(),1:end-1);
pageTest.Labels=page_blocks(pagedex.test(),end);
%% Classification noise
T = length(pageTrain.Data);
LabelNoise = round(normrnd(0, 0.32, T,1));
pageTrain.NoisyLabels = pageTrain.Labels + LabelNoise;
pageTrain.er_rat = sum(pageTrain.NoisyLabels ~= pageTrain.Labels)/T;
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