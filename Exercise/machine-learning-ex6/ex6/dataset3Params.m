function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_temp = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_temp = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
m = size(C_temp, 2);
n = size(sigma_temp, 2);
cv_error = ones(m,n)

% find the optical C and sigma by using the CV_error. 
for i = 1:m
   for j = 1:n 
       C = C_temp(i);
       sigma = sigma_temp(j);
       model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
       predictions = svmPredict(model, Xval);
       cv_error(i, j) = mean(double(predictions ~= yval));
    endfor
endfor

[cVal, cIndex] = min(cv_error);
[eMin ,cmin]= min(cVal);
r_min = cIndex(cmin); 

C = C_temp(r_min);
sigma = sigma_temp(cmin);

% =========================================================================

end
