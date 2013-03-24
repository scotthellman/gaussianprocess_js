(function(){
	var tolerance;
	module('Gaussian Process Regression Tests', {
		setup: function() {
			tolerance = 0.000001;
		},
		teardown: function() {
		}
	});

	test( "constant test input invariance ", function(){
		var constant = Kernels.constant(1);
		ok(constant.kernel($V([1]),$V([1])) == constant.kernel($V([-5,3]),$V([2,10])));
	});

	test( "constant test output depends on theta ", function(){
		var constant = Kernels.constant(10);
		equal(constant.kernel($V([3]),$V([1])),10);
	});

	test( "constant test output handles 0 theta ", function(){
		var constant = Kernels.constant(0);
		equal(constant.kernel($V([3]),$V([1])),0);
	});

	test( "contant gradient input invariance", function(){
		var constant = Kernels.constant(1);
		ok(constant.gradients[0]($V([1]),$V([1])) == constant.gradients[0]($V([-5,2]),$V([10,20])));
	});

	test( "contant gradient is 1", function(){
		var constant = Kernels.constant(1);
		equal(constant.gradients[0]($V([0]),$V([-21])),1);
	});

	test( "linear test output is linear ", function(){
		var linear = Kernels.linear(1);
		equal(linear.kernel($V([2]),$V([3])),6);
	});

	test( "linear test output correct in higher dimensions ", function(){
		var linear = Kernels.linear(1);
		var expected = 2*3 + 4*-3;
		ok(Math.abs(linear.kernel($V([2,4]),$V([3,-3])) - expected) < tolerance);
	});

	test( "linear test output scales with theta", function(){
		var linear = Kernels.linear(-2);
		var expected = 2*3 + 4*-3;
		expected *= -2
		ok(Math.abs(linear.kernel($V([2,4]),$V([3,-3])) - expected) < tolerance);
	});

	test( "linear test output handles 0 theta", function(){
		var linear = Kernels.linear(0);
		equal(linear.kernel($V([2,4]),$V([3,-3])), 0);
	});

	test( "linear gradient correctness ", function(){
		var linear = Kernels.linear(4);
		equal(linear.gradients[0]($V([2]),$V([3])),6);
	});

	test( "linear gradient multivariate correctness ", function(){
		var linear = Kernels.linear(-1);
		equal(linear.gradients[0]($V([-1,2]),$V([0,3])),6);
	});


})();
