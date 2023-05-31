clear all

nesto1 = csvread('fashion-mnist_train.csv', 1, 0);
nesto2 = csvread('fashion-mnist_test.csv', 1, 0);

X_train = double(nesto1(:, 2:end));
X_train = X_train'/255;
y_train = double(nesto1(:, 1));
y_train = y_train';

X_test = double(nesto2(:, 2:end));
X_test = X_test'/255;
y_test = double(nesto2(:, 1));
y_test = y_test';

clear nesto1;
clear nesto2;

% Reshape X_train
X_train_reshaped = reshape(X_train, 28, 28, size(X_train, 2));
X_train_new = zeros(14, 14, size(X_train, 2));

% Grouping and averaging pixels
for i = 1:size(X_train_reshaped, 3)
    image = X_train_reshaped(:,:,i);
    new_image = imresize(image, [14 14]);
    X_train_new(:,:,i) = new_image;
end

X_train_new = reshape(X_train_new, [], size(X_train, 2));
X_train = X_train_new;
clear X_train_new

% Reshape X_test
X_test_reshaped = reshape(X_test, 28, 28, size(X_test, 2));
X_test_new = zeros(14, 14, size(X_test, 2));

% Grouping and averaging pixels
for i = 1:size(X_test_reshaped, 3)
    image = X_test_reshaped(:,:,i);
    new_image = imresize(image, [14 14]);
    X_test_new(:,:,i) = new_image;
end

X_test_new = reshape(X_test_new, [], size(X_test, 2));
X_test = X_test_new;
clear X_test_new


m = 25000; % size(X_train,2);

% Set the parameters
totalIndices = 60000;  % Total number of indices
numIndices = m;    % Number of random indices to select

% Generate random indices
randomIndices = randperm(totalIndices);
randomIndices = randomIndices(1:numIndices);

X_train_new = X_train(:, randomIndices);
y_train_new = y_train(:, randomIndices);

G = zeros(m,m);
alpha = 0.8;

for i = 1:m
    for j = 1:m
        G(i,j) = kernel_f(X_train_new(:,i),X_train_new(:,j),alpha);
    end
end

d = 10;

Y = one_hot_encode(y_train_new, d, m);

Z = Y/G;
% Z = Y * pinv(G);

figure(2)
imagesc(reshape(X_test(:,1),[14,14])');colormap(gray);

x = X_test(:,1);
y = f(X_train_new,Z,x,alpha);
y
y_test(1)

figure(3)
imagesc(reshape(X_test(:,4001),[14,14])');colormap(gray);

x = X_test(:,4001);
y = f(X_train_new,Z,x,alpha);
y
y_test(4001)

figure(4)
imagesc(reshape(X_test(:,9001),[14,14])');colormap(gray);

x = X_test(:,9001);
y = f(X_train_new,Z,x,alpha);
y
y_test(9001)



% Set the parameters
totalIndices = 10000;  % Total number of indices
numIndices = 10000;    % Number of random indices to select

randomIndices = randperm(totalIndices);
randomIndices = randomIndices(1:numIndices);

X_test_new = X_test(:, randomIndices);
y_test_new = y_test(:, randomIndices);

y_preds = zeros(1, numIndices);

for i = 1:numIndices
    x = X_test_new(:,i);
    rates = f(X_train_new, Z, x, alpha);
    [~, pred] = max(abs(rates));
    y_preds(1,i) = pred-1;
end

% Assuming y_preds and y_test are row vectors of shape (1, numIndices)

% Count the number of correct predictions
numCorrect = sum(y_preds == y_test_new);

% Calculate the accuracy
accuracy = (numCorrect / numIndices) * 100;

% Display the accuracy
fprintf('Accuracy: %.2f%%\n',accuracy);