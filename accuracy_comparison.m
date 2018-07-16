% Comparison of accuracy of all three algorithms implemented

acc= [94.50,93,97.50];
bar(acc,0.05);
set(gca,'XTickLabel',{'SVM', 'Neural network', 'Logistic'})
ylim([90 100]);
xlabel('Algorithm name');
ylabel('Accuarcy');
title('Comparison of Accuracy');