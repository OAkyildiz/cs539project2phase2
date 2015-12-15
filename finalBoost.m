%  finalBoost.m             %% Ozan Akyýldýz
   % 1. Ada Tree
pokerAdaBoost.tree.mdl = fitensemble(pokerTrain.Data,pokerTrain.Labels, ...
  'AdaBoostM2', 20, 'Tree');
%%
%[~, score]=oobPredict(poker.bag)
pokerAdaBoost.tree.predict=predict(pokerAdaBoost.tree.mdl,pokerTest.Data);
%%
sum(pokerAdaBoost.tree.predict==pokerTest.Labels)
%%%%%
%%
    %  2.Ada KNN
pokerAdaBoost.knn.mdl = fitensemble(pokerTrain.Data,pokerTrain.Labels, ...
  'AdaBoostM2', 20, 'Knn');
%%
%[~, score]=oobPredict(poker.bag)
pokerAdaBoost.knn.predict=predict(pokerAdaBoost.knn.mdl,pokerTest.Data);
%%
sum(pokerAdaBoost.knn.predict==pokerTest.Labels)
%%%%%
    % Ada Discriminant
pokerAdaBoost.discr.mdl = fitensemble(pokerTrain.Data,pokerTrain.Labels, ...
  'AdaBoostM2', 20, 'Discriminant');
%%
%[~, score]=oobPredict(poker.bag)
pokerAdaBoost.discr.predict=predict(pokerAdaBoost.discr.mdl,pokerTest.Data);
%%
sum(pokerAdaBoost.discr.predict==pokerTest.Labels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%