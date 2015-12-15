 function EVAL = Evaluate(ACTUAL,PREDICTED,labels)
% Evaluate This fucntion evaluates the performance of a classification model by 
% calculating :Accuracy, Sensitivity, Specificity, Precision, Recall, F-Measure, G-mean.
%
%   The function assumes there are no unclassified instances.
%   Output is tabel ready (cells) 
%    Input: ACTUAL = Column matrix with actual class labels of the training
%                 examples
%        PREDICTED = Column matrix with predicted class labels by the
%                    classification model
% Output: EVAL = Row matrix with all the performance measures
% modified by oakyildiz.


C = length(labels);
N = length (ACTUAL);


accuracy = zeros(C,1);
sensitivity = zeros(C,1);
specificity = zeros(C,1);
precision = zeros(C,1);
recall = zeros(C,1);
f_measure = zeros(C,1);
gmean = zeros(C,1);
%% New
c_mat = confusionmat(ACTUAL,PREDICTED)
%%

for c=1:C
%    p = sum(PREDICTED==c); %positives
%    n = N-p;            %negatives (relative)
    tp =  c_mat(c,c); %True positives
    fp = sum(c_mat(:,1))-tp;  %false positives
    fn = sum(c_mat(1,:))-tp;  %false negatives
    tn = N-tp-fp-fn;  %True negatives

    tp_rate = tp/(tp+fn); 
    tn_rate = tn/(fp+tn);
    
    accuracy(c) = (tp+tn)/N;
    sensitivity(c) = tp_rate;
    specificity(c) = tn_rate;
    precision(c) = tp/(tp+fp);
    recall(c) = tp_rate;
    f_measure(c) = 2*((precision(c)*recall(c))/(precision(c) + recall(c)));
    gmean(c) = sqrt(precision(c)*recall(c));
    
end
EVAL = [accuracy sensitivity specificity precision recall f_measure gmean];