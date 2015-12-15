%  finalBoost.m             %% Ozan Aky�ld�z
 
boostTree=templateTree();
pageClasses = [1 2 3 4 5];
wineClasses = [1 2 3];
pokerClasses = [1 2 3 4 5 6 7 8 9];
%% Adaboost Tree 20 - page_blocks
disp(' Adaboost Tree 20 - page_block')
pageAdaBoost.tree.mdl = fitensemble(pageTrain.Data, ...
    pageTrain.Labels,'AdaBoostM2', 20, boostTree);
pageAdaBoost.tree.predict=predict(pageAdaBoost.tree.mdl,pageTest.Data);
sum(pageAdaBoost.tree.predict==pageTest.Labels)
eval1 = Evaluate(pageTest.Labels,pageAdaBoost.tree.predict, pageClasses)

%% Adaboost Tree 1 - page_blocks
disp(' Adaboost Tree 1 - page_block')
pageAdaBoost.tree1.mdl = fitensemble(pageTrain.Data, ...
    pageTrain.Labels,'AdaBoostM2', 1, boostTree);
pageAdaBoost.tree1.predict=predict(pageAdaBoost.tree1.mdl,pageTest.Data);
sum(pageAdaBoost.tree1.predict==pageTest.Labels)
eval2 = Evaluate(pageTest.Labels,pageAdaBoost.tree1.predict, pageClasses)

%% Adaboost Tree 20 - poker
disp('Adaboost Tree 20 - poker')
pokerAdaBoost.tree.mdl = fitensemble(pokerTrain.Data, ...
    pokerTrain.Labels,'AdaBoostM2', 30, boostTree);
pokerAdaBoost.tree.predict=predict(pokerAdaBoost.tree.mdl,pokerTest.Data);
sum(pokerAdaBoost.tree.predict==pokerTest.Labels)
eval3 = Evaluate(pokerTest.Labels,pokerAdaBoost.tree.predict, pokerClasses)

%% Adaboost Wine 
disp('Adaboost Wine ')
wineAdaBoost.tree.mdl = fitensemble(wineTrain.Data, ...
    wineTrain.Labels,'AdaBoostM2', 10, boostTree);
wineAdaBoost.tree.predict=predict(wineAdaBoost.tree.mdl,wineTest.Data);
sum(wineAdaBoost.tree.predict==wineTest.Labels)
eval4 = Evaluate(wineTest.Labels,wineAdaBoost.tree.predict, wineClasses)

%% Adaboost Noisy Wine
disp('Adaboost Noisy Wine')
wineAdaBoost.treeNoisy.mdl = fitensemble(wineTrain.Data, ...
    wineTrain.NoisyLabels,'AdaBoostM2', 10, boostTree);
wineAdaBoost.treeNoisy.predict=predict(wineAdaBoost.treeNoisy.mdl,wineTest.Data);
sum(wineAdaBoost.treeNoisy.predict==wineTest.Labels)
eval5 = Evaluate(wineTest.Labels,wineAdaBoost.treeNoisy.predict, wineClasses)

%% RUSBoost 20 - poker
disp('RUSBoost 20 - poker')
pokerRUSBoost.tree.mdl = fitensemble(pokerTrain.Data, ...
    pokerTrain.Labels,'RUSBoost', 30, boostTree);
pokerRUSBoost.tree.predict=predict(pokerRUSBoost.tree.mdl,pokerTest.Data);
sum(pokerRUSBoost.tree.predict==pokerTest.Labels)
eval6 = Evaluate(pokerTest.Labels,pokerRUSBoost.tree.predict, pokerClasses)

%% TotalBoost Tree 20 - page_blocks
disp(' TotalBoost Tree 20 - page_block')
pageTotalBoost.tree.mdl = fitensemble(pageTrain.Data, ...
    pageTrain.Labels,'TotalBoost', 30, boostTree);
pageTotalBoost.tree.predict=predict(pageTotalBoost.tree.mdl,pageTest.Data);
sum(pageTotalBoost.tree.predict==pageTest.Labels)
eval7 = Evaluate(pageTest.Labels,pageTotalBoost.tree.predict, pageClasses)

%% Totalboost Tree 1 - page_blocks
disp(' TotalBoost Tree 1 - page_block')
pageTotalBoost.tree1.mdl = fitensemble(pageTrain.Data, ...
    pageTrain.Labels,'TotalBoost', 1, boostTree);
pageTotalBoost.tree1.predict=predict(pageTotalBoost.tree1.mdl,pageTest.Data);
sum(pageTotalBoost.tree1.predict==pageTest.Labels)
eval8 = Evaluate(pageTest.Labels,pageTotalBoost.tree1.predict, pageClasses)
