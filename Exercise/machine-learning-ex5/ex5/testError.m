%the function is to compute the test error, to modify whether the modle is precise.

function test_error = testError(Xtest, ytest, theata, lambda)

test_error = 0;

test_error = linearRegCostFunction(Xtest, ytest, theata, lambda);

end
