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
		ok(constant.kernel($V([1]),$V([1])) === constant.kernel($V([-5,3]),$V([2,10])));
	});

	test( "constant test output depends on theta ", function(){
		var constant = Kernels.constant(10);
		deepEqual(constant.kernel($V([3]),$V([1])),10);
	});

	test( "constant test output handles 0 theta ", function(){
		var constant = Kernels.constant(0);
		deepEqual(constant.kernel($V([3]),$V([1])),0);
	});

	test( "contant gradient input invariance", function(){
		var constant = Kernels.constant(1);
		ok(constant.gradients[0]($V([1]),$V([1])) === constant.gradients[0]($V([-5,2]),$V([10,20])));
	});

	test( "contant gradient is 1", function(){
		var constant = Kernels.constant(1);
		deepEqual(constant.gradients[0]($V([0]),$V([-21])),1);
	});

	test( "linear test output is linear ", function(){
		var linear = Kernels.linear(1);
		deepEqual(linear.kernel($V([2]),$V([3])),6);
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
		deepEqual(linear.kernel($V([2,4]),$V([3,-3])), 0);
	});

	test( "linear gradient correctness ", function(){
		var linear = Kernels.linear(4);
		deepEqual(linear.gradients[0]($V([2]),$V([3])),6);
	});

	test( "linear gradient multivariate correctness ", function(){
		var linear = Kernels.linear(-1);
		deepEqual(linear.gradients[0]($V([-1,2]),$V([0,3])),6);
	});

	test( "gaussian is constant on diagonal" , function(){
		var gauss = Kernels.gaussianNoise(1);
		deepEqual(gauss.kernel($V([0]),$V([0])), gauss.kernel($V([4]),$V([4])));
	});

	test( "gaussian is constant on diagonal multivariate" , function(){
		var gauss = Kernels.gaussianNoise(1);
		deepEqual(gauss.kernel($V([0,4]),$V([0,4])), gauss.kernel($V([20,-2]),$V([20,-2])));
	});

	test( "gaussian is 0 on off-diagonal" , function(){
		var gauss = Kernels.gaussianNoise(1);
		deepEqual(gauss.kernel($V([0]),$V([1])), 0);
	});

	test( "gaussian is 0 on off-diagonal multivariate" , function(){
		var gauss = Kernels.gaussianNoise(1);
		deepEqual(gauss.kernel($V([0,20]),$V([1,21])), 0);
	});

	test( "gaussian scales with theta" , function(){
		var gauss = Kernels.gaussianNoise(5);
		deepEqual(gauss.kernel($V([4,4]),$V([4,4])), 5);
	});

	test( "gaussian has no gradient" , function(){
		var gauss = Kernels.gaussianNoise(2);
		deepEqual(gauss.gradients,[]);
	});

	test( "squared exponential correctness ", function(){
		var exp = Kernels.squaredExponential(1,-0.5);
		expected = Math.exp(-16);
		ok(Math.abs(exp.kernel($V([1]),$V([3])) - expected) < tolerance);
	});

	test( "squared exponential correctness multivariate ", function(){
		var exp = Kernels.squaredExponential(1,-0.5);
		expected = Math.exp(-52);
		ok(Math.abs(exp.kernel($V([1,2]),$V([3,-1])) - expected) < tolerance);
	});

	test( "squared exponential scales with theta ", function(){
		var exp = Kernels.squaredExponential(3,-0.5);
		expected = 3 * Math.exp(-52);
		ok(Math.abs(exp.kernel($V([1,2]),$V([3,-1])) - expected) < tolerance);
	});

	test( "squared exponential scales with width ", function(){
		var exp = Kernels.squaredExponential(1,-0.25);
		expected = Math.exp(-104);
		ok(Math.abs(exp.kernel($V([1,2]),$V([3,-1])) - expected) < tolerance);
	});

	test( "squared exponential d_theta correct", function(){
		var exp = Kernels.squaredExponential(3,-0.5);
		expected = Math.exp(-52);
		ok(Math.abs(exp.gradients[0]($V([1,2]),$V([3,-1])) - expected) < tolerance);
	});

	test( "squared exponential d_width correct", function(){
		var exp = Kernels.squaredExponential(3,-2);
		expected = 1.07439299;
		ok(Math.abs(exp.gradients[1]($V([1,2]),$V([3,1])) - expected) < tolerance);
	});

})();
