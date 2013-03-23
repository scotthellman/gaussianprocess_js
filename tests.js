(function(){
	module('Gaussian Process Regression Tests', {
		setup: function() {
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
})();
