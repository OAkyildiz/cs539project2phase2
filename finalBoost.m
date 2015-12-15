%  finalBoost.m             %% Ozan Akyýldýz
   % 1. AdaBoost
    % a. Ada Tree
boostTree=templateTree('MergeLeaves','On');
%%
pokerAdaBoost.tree.mdl = fitensemble(pokerTrain.Data,pokerTrain.NoisyLabels, ...
  'AdaBoostM2', 20, boostTree);
%%
%[~, score]=oobPredict(poker.bag)
pokerAdaBoost.tree.predict=predict(pokerAdaBoost.tree.mdl,pokerTest.Data);
%%
sum(pokerAdaBoost.tree.predict==pokerTest.Labels)
%%%%%
%%
    %  b. Ada Tree 2
pokerAdaBoost.tree100.mdl = fitensemble(pokerTrain.Data,pokerTrain.Labels, ...
  'AdaBoostM2', 200, 'Tree');
%%
%[~, score]=oobPredict(poker.bag)
pokerAdaBoost.tree100.predict=predict(pokerAdaBoost.tree100.mdl,pokerTest.Data);
%%
sum(pokerAdaBoost.tree100.predict==pokerTest.Labels)
%%%%%
%%
    % c. Ada Discriminant
pokerAdaBoost.discr.mdl = fitensemble(pokerTrain.Data,pokerTrain.Labels, ...
  'AdaBoostM2', 20, 'Discriminant');
%%
%[~, score]=oobPredict(pokerc.bag)
pokerAdaBoost.discr.predict=predict(pokerAdaBoost.discr.mdl,pokerTest.Data);
%%
sum(pokerAdaBoost.discr.predict==pokerTest.Labels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

tree.mdl = fitensemble(train, trainLab, ...
  'AdaBoostM2', 20, boostTree);
%%
%[~, score]=oobPredict(poker.bag)
tree.predict=predict(tree.mdl,test);
%%
sum(tree.predict==testLab)