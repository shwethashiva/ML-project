%SVM implementation for SMS classification
%Name: Prachi Shirode, Red Id : 819718205

%% load data 
load('sms.mat')
%% Split training data into validation 
n=1000; 

%split data 
%S=randperm(n); % comment this line after 1st run to use same data 
%save('randOnetime.mat', 'S'); % comment this line after 1st run to use same data 
load('randOnetime.mat', 'S'); % use this data for different models

%500 training samples
X_train1=X_train(S(1:500),:);
y_train1=y_train(S(1:500),:);

%saving the training data in file
save('trainData.mat', 'X_train1', 'y_train1')

%500 validation samples
X_valid=X_train(S(501:end),:);
y_valid=y_train(S(501:end),:);

%saving the validation data in file
save('validData.mat', 'X_valid', 'y_valid')
%% Applying svm on training data
C = 0.05;
kernelType = 'linear';
MDL = fitcsvm(X_train1, y_train1, 'KernelFunction',kernelType, 'BoxConstraint',C);
MDL

%% Applying svm model on validation data
[labelvalid, scoretest] = predict(MDL, X_valid);

%% Calculate errors  for validation data i.e. number of mismatch labels
errorsvalid = sum(~(y_valid == labelvalid));
errorsvalid

%% Applying svm model on testing data
[labeltest, scoretest] = predict(MDL, X_test);

%% Calculate errors  for testing data i.e. number of mismatch labels
errorstest = sum(~(y_test == labeltest));
errorstest

%% Calculate accuracy,precision and recall rate
stats = confusionmatStats(y_test,labeltest);

%% plotting C versus errors
a= [0.01,0.05,0.1,0.4,0.5,0.8,1]; % C value
b= [34,22,19,21,23,27,27];  % error values calculated for C values mentioned in previous stmt.
figure;
plot(a,b), xlabel('C value'), ylabel('error');
title('Comparison between C values');


%% plotting Kernel types versus errors
e= [21,230]; % error value
figure;
bar(e,0.05);
set(gca,'XTickLabel',{'linear','RBF'})
xlabel('kernel type'), ylabel('error');
title('Comparison between kernel type');
