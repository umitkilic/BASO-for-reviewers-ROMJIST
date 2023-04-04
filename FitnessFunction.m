%-------------------------------------------------------------------------%
%  Binary Anarchich Society Algorithm (BASO) source codes                 %
%  for Feature Selection                                                  %
%                                                                         %
%  Umit Kilic                                                             %
%                                                                         %
%  email: ukilic@atu.edu.tr & umitkilic21@gmail.com                       %
%-------------------------------------------------------------------------%

function [errorrate,fmeasure,accuracy]=jFitnessFunction(feat,label,X)
% Parameter setting for k-value of KNN
k=3; 
% Parameter setting for number of cross-validation
kfold=10;
% Error rate
%disp(['X pos: ' num2str(X)]);
[errorrate,fmeasure]=jwrapperKNN(feat(:,X==1),label,k,kfold);
%disp(['pos: ' num2str(X)]);
accuracy=1-errorrate;
%accuracy
%disp(['fitness:',num2str(fitness)])
end

% Perform KNN with k-folds cross-validation
function [ER,fmeasure]=jwrapperKNN(feat,label,k,kfold)
%disp(['size feat: ' num2str(size(feat)) ' size label: ' num2str(size(label))]);
Model=fitcknn(feat,label,'NumNeighbors',k,'Distance','euclidean'); 
C=crossval(Model,'KFold',kfold);
% Error rate 
ER=kfoldLoss(C);

%A=classperf(Model);

pred=kfoldPredict(C);
%ccc=classperf(C);
%disp(['sized of label:' num2str(size(label)) 'size of pred:' num2str(size(pred))]);
confmat=confusionmat(label,pred);
conf_matrix=confmat;
[fmeasure,p,r]=micro_averaged_f1(conf_matrix);
%eval=Evaluate(label,pred);
%disp(['Our results:fm: ' num2str(fmeasure) ' precision: ' num2str(p) ' recall: ' num2str(r)...
%    ' Their res: fm: ' num2str(eval(6)) ' prec: ' num2str(eval(4)) ' recall: ' num2str(eval(5))...
%    'accuracy: ' num2str(eval(1))]);
end

function [fmeasure,precision,recall]=micro_averaged_f1(cm)
total_tp=0; total_fp=0; total_fn=0; total_tn=0;
size(cm);
    if size(cm,1)==2
        total_tp=cm(1,1);
        total_fp=cm(1,2);
        total_fn=cm(2,1);
        total_tn=cm(2,2);
        
        
        precision=(total_tp/(total_tp+total_fp));
        recall=(total_tp/(total_tp+total_fn));
        fmeasure=2*((precision*recall)/(precision+recall));
        
        %disp(['TP:' num2str(total_tp) 'FP: ' num2str(total_fp) 'FN: ' num2str(total_fn) 'TN: ' num2str(total_tn)]);
    else

        for i=1:size(cm,1)

            for k=1:size(cm,2)
                if(i==k)
                    total_tp = total_tp + cm(i,k);
                else
                    total_fp = total_fp + cm(i,k);
                end

            end
        end
           
        total_fn=total_fp;
        precision=(total_tp/(total_tp+total_fp));
        recall=(total_tp/(total_tp+total_fn));
        fmeasure=2*((precision*recall)/(precision+recall));
        %disp(['TP:' num2str(total_tp) 'FP: ' num2str(total_fp) 'FN: ' num2str(total_fn) 'TN: ' num2str(total_tn)]);
    end
    
    
end


% ! NOT USED !
function EVAL = Evaluate(ACTUAL,PREDICTED)
% This fucntion evaluates the performance of a classification model by 
% calculating the common performance measures: Accuracy, Sensitivity, 
% Specificity, Precision, Recall, F-Measure, G-mean.
% Input: ACTUAL = Column matrix with actual class labels of the training
%                 examples
%        PREDICTED = Column matrix with predicted class labels by the
%                    classification model
% Output: EVAL = Row matrix with all the performance measures
idx = (ACTUAL()==1);
p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;
tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp;
tp_rate = tp/p;
tn_rate = tn/n;
accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);
EVAL = [accuracy sensitivity specificity precision recall f_measure gmean];
end

