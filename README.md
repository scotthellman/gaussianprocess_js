gaussianprocess_js
==================

Gaussian Process Regression in Javascript

Requires sylvester.js, and example code requires flot for plotting.

To use, create a kernel using the kernel builder. The following code will create a kernel with 
an even weighting between a constant term, a linear term, a gaussian noise term, and a squared
exponential term with a width of -1.

```javascript
var K = Kernels.kernelBuilder(Kernels.constant(1),
Kernels.linear(1),
Kernels.gaussianNoise(1),
Kernels.squaredExponential(1,-1));
```

Construct a Gaussian Process given a kernel
```javascript
var gpr = GaussianProcess(K);
```

You can use gradient descent to find parameters that maximize the marginal likelihood of the Gaussian
Process using the gradient descent function:

```javascript
gpr.gradientDescent(labels,values,cutoff,gamma,iterations);
```

And evaluate a new set of points given training data
```javascript
gpr.evaluate(training_data,training_labels,testing_data)
```
where training_data and testing_data are arrays of sylvester.js vectors and training labels is an array of scalars.

License
=======
The MIT License (MIT)
Copyright (c) <year> <copyright holders>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
