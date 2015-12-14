 function EVAL = Evaluate(ACTUAL,PREDICTED,labels)
% 
% This fucntion evaluates the performance of a classification model by 
% calculating :Accuracy, Sensitivity, 
% Specificity, Precision, Recall, F-Measure, G-mean.
% Input: ACTUAL = Column matrix with actual class labels of the training
%                 examples
%        PREDICTED = Column matrix with predicted class labels by the
%                    classification model
% Output: EVAL = Row matrix with all the performance measures
% modified by oakyildiz.

idx = strcmp(ACTUAL(),labels{1});

p = sum(strcmp(ACTUAL(),labels{1}));
n = sum(strcmp(ACTUAL(),labels{2}));
N = p+n;

tp = sum(strcmp(ACTUAL,PREDICTED));   %True positives
tn = sum(~strcmp(ACTUAL,PREDICTED)); %True negatives
fp = n-tn;  %false positives
fn = p-tp;  %false negatives

tp_rate = tp/p;
%tn_rate = tn/n;

accuracy = (tp+tn)/N;
%sensitivity = tp_rate;
%specificity = tn_rate;
precision = tp/(tp+fp);
recall = tp_rate;
%f_measure = 2*((precision*recall)/(precision + recall));
%gmean = sqrt(tp_rate*tn_rate);

%EVAL = [accuracy sensitivity specificity precision recall f_measure gmean];
EVAL = [accuracy precision recall];
