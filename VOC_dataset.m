%%%%% Creation of train and test data

function [Train_feature, Train_label, Test_feature, Test_label]= VOC_dataset(train_ratio)

Train_feature= [ ];
Train_label= [ ];
Test_feature= [ ];
Test_label= [ ];

load('vgg19_feature.mat');
load('im_names.mat');


for i= 1:20
    
    obj_ind= find(names==i-1);
    L= length(obj_ind);
    ind1= datasample(obj_ind,round(train_ratio*L),'Replace',false);
    ind2= setdiff(obj_ind, ind1);    
    
    Train_feature= [Train_feature; cnn_features(ind1,: )];
    Train_label= [Train_label; names(ind1)'];

    Test_feature= [Test_feature; cnn_features(ind2,: )];
    Test_label= [Test_label; names(ind2)'];    
    
    
end


    
end


