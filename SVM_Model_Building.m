% Object classification into one of twenty classes
% Basic SVM and main
% The future prediction overwrites the previous prediction
% Polynomial Kernel

clc;
clear all;
close all;

error= [ ];

K= 20;
train_ratio= 0.99;
tic;

for j= 1:1

[X, y, Xval, yval]= VOC_dataset(train_ratio);
yt= 100*ones(length(y),1); % This one later gets 1/-1
y_pred_train= yt;
yv= 100*ones(length(yval),1); % This one later gets 1/-1
y_pred= yv;

for i= 1:K
    
   disp(i);    
      
   yt(y==(i-1))= 1;
   yt(y~=(i-1))= -1;
   
   yv(yval==(i-1))= 1;
   yv(yval~=(i-1))= -1;
   

    %% Learning Phase
    SVMModel = fitcsvm(X,yt,'KernelFunction','polynomial','KernelScale','auto','Standardize',true);
    saveCompactModel(SVMModel,['SVMModel_' num2str(i-1)]);

    
    %% Training data classification
    y_svm= predict(SVMModel, X);
    
    ind= find(y==i-1);
    y_pred_train(ind)= (i-1)*(y_svm(ind)==1)+100*(y_svm(ind)~=1);
%     y_pred_train(y_svm(ind)==-1)= datasample(setdiff((0:K-1),i-1),sum(y_svm(ind)==-1));


    %% Validation data classification
    y_svm= predict(SVMModel, Xval);
    
    ind= find(y_svm==1);
    y_pred(ind)= i-1;    
    
    disp('==========Training Data============');
    disp('True classification rate of all images');
    Tr= sum(y_pred_train==y)/length(y) 

    
    disp('==========Validation Data============');
    disp('True classification rate of all images');
    Tr= sum(y_pred==yval)/length(yval)
    
    
end

y_pred_train(y_pred_train==100)= datasample(0:K-1,sum(y_pred_train==100));
y_pred(y_pred==100)= datasample(0:K-1,sum(y_pred==100));

% Overall accuracy
disp('==========Training Data============');

disp('True classification rate of all images');
Tr= sum(y_pred_train==y)/length(y) 

disp('==========Validation Data============');

disp('True classification rate of all images');
Tr= sum(y_pred==yval)/length(yval) 


end

% Class based accuracy
for i= 1:K

    ind= find(yval==i-1);
    Tr= sum(y_pred(ind)==yval(ind))/length(ind);
    disp(['Class ' num2str(i-1) ' accuracy: ' num2str(Tr*100)]);
    
end

toc




