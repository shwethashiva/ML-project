%Name: Shwetha Shivaraju  and Prachi Shirode

%% Load
load sms.mat;  % loads X_train,X_test,y_train,y_test
[m, n] = size(X_train);
%X_train([1],:)
%X_fst = X_train([1],:);
%X_normalize = normalize_features(X_train([2:3],:));
X_train = normalize_features(X_train);

% Initialize theta
theta_init = zeros(n,1);

%% Train
% theta
alpha = 0.01;
iters = 100;
[theta, J_history] = gradient_descent(X_train, y_train, theta_init, alpha, iters);

%% plot the cost history
costsz = size(J_history, 2);
complexity = 1:costsz;

subplot (1, 1, 1);
plot(complexity, J_history);
xlabel('Complexity');
ylabel('Cost');
title('Cost vs Complexity');

%% Testing

% theta to make predictions for test set
z = theta' * X_test';
y = logistic(z);

% accuracy on the test set
total_msgs = size(X_test,1);

% define: if predicted to be spam by logistic regression, y = 1
predicted_spam = (logistic(z) >= 0.5)'
% how many msgs were correctly predicted as spam?
crrct_spam = sum(y_test==1 & predicted_spam==1)
% how many msgs were correctly predicted as ham?
crrct_ham = sum(y_test == 0 & predicted_spam ==0 )

percent_crrct = ((crrct_spam +crrct_ham)/total_msgs)*100

%% Model check for spam and non spam words

word_weights = theta(2:end);
[~,I] = sort(word_weights, 'descend');
[~,J] = sort(word_weights, 'ascend');

k = 10;

fprintf('Top %d spam words\n', k);
fprintf('  %s\n', dict{I(1:k)});
fprintf('\n');
fprintf('Top %d nonspam words\n', k);
fprintf('  %s\n', dict{J(1:k)});

%% Predict for a new message input

text = 'Machine learning is the latest technology. There will be seminar to discuss about it at conference hall at 10 pm tommorrow';
text2 = 'Sale for Black Friday. Contact us on website for Free coupons and Discounts';
x_my_ham = sms_extract_features(text, dict);
x_my_spam = sms_extract_features(text2, dict);
y_my_ham = logistic(theta'*x_my_ham');
y_my_spam = logistic(theta'*x_my_spam');

if y_my_ham >=0.5
    fprintf( 'Your ham message is spam\n')
else
    fprintf('Your ham message is ham\n')
end

if y_my_spam >=0.5
    fprintf('Your spam message is spam\n')
else
    fprintf('Your spam message is ham\n')
end

%% Calculate accuracy,precision and recall rate
stats = confusionmatStats(y_test,predicted_spam);

