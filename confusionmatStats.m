% Name: Shwetha and Prachi Shirode
% Function to compute confusion matrix, accuracy, precision and recall

function stats = confusionmatStats(group,grouphat)
field1 = 'confusionMat';
if nargin < 2
    value1 = group;
else
    value1 = confusionmat(group,grouphat);
end
value1

numOfClasses = size(value1,1);
totalSamples = sum(sum(value1));

field2 = 'accuracy';  value2 = trace(value1)/(totalSamples);

value2

[TP,TN,FP,FN,recall,precision] = deal(zeros(numOfClasses,1));
for class = 1:numOfClasses
    TP(class) = value1(class,class);
    tempMat = value1;
    tempMat(:,class) = []; % remove column
    tempMat(class,:) = []; % remove row
    
    TN(class) = sum(sum(tempMat));
    FP(class) = sum(value1(:,class))-TP(class);
    FN(class) = sum(value1(class,:))-TP(class);
end

for class = 1:numOfClasses
    recall(class) = TP(class) / (TP(class) + FN(class));
    precision(class) = TP(class) / (TP(class) + FP(class));
end

field3 = 'precision';  value3 = precision;
field4 = 'recall';  value4 = recall;

precision
recall

stats = struct(field1,value1,field2,value2,field3,value3,field4,value4);
