(function(){
	module('Gaussian Process Regression Tests', {
		setup: function() {
		},
		teardown: function() {
		}
	});

	test( "constant test input invariance ", function(){
		var constant = Kernels.constant(1);
		ok(constant.kernel(1,1) == constant.kernel(-5,10));
	});

	test( "constant test output depends on theta ", function(){
		var constant = Kernels.constant(10);
		equal(constant.kernel(3,1),10);
	});

	test( "constant test output handles 0 theta ", function(){
		var constant = Kernels.constant(0);
		equal(constant.kernel(3,1),0);
	});

	test( "contant gradient input invariance", function(){
		var constant = Kernels.constant(1);
		ok(constant.gradients[0](1,1) == constant.gradients[0](-5,10));
	});

	test( "contant gradient is 1", function(){
		var constant = Kernels.constant(1);
		equal(constant.gradients[0](0,-21),1);
	});
})();
