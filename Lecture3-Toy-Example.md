# Forward Propagation in Neural Networks.

Solution:

```
function predictions = predict(W1,W2, X)
  a1 = X; % input layer
  z2 = a1 * W1;
  a2 = sigmoid(z2); % hidden layer
  predictions = a2 * W2; % output layer
endfunction

function g = sigmoid(z)
  g = 1.0 ./ (1.0 + exp(-z));
end
```
