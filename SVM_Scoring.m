% Object classification into one of twenty classes
% Loads SVM models
% Predicts label of the test images

clc;
clear all;
close all;

K= 20;
tic;

load('man_feature.mat');
load('man_names.mat');

Xval= cnn_features;
yval= names';

yv= 100*ones(length(yval),1); % This one later gets 1/-1
y_pred= yv;

for i= 1:K
    
   disp(i);    
      
   yv(yval==(i-1))= 1;
   yv(yval~=(i-1))= -1;
   

    %% Testing Phase
    SVMModel = loadCompactModel(['SVMModel_' num2str(i-1)]);
    
    y_svm= predict(SVMModel, Xval);
    ind= find(y_svm==1);
    y_pred(ind)= i-1;    
        
    disp('==========Validation Data============');
    disp('True classification rate of all images');
    Tr= sum(y_pred==yval)/length(yval)
    
    
end

y_pred(y_pred==100)= datasample(0:K-1,sum(y_pred==100));

disp('==========Validation Data============');

disp('True classification rate of all images');
Tr= sum(y_pred==yval)/length(yval) 


% Class based accuracy
for i= 1:K

    ind= find(yval==i-1);
    Tr= sum(y_pred(ind)==yval(ind))/length(ind);
    disp(['Class ' num2str(i-1) ' accuracy: ' num2str(Tr*100)]);
    
end

toc;




