function gradientDescent(y,X,kernel,cutoff,gamma,max_iterations){
	var delta = null;
	var max_delta = 10;
	var current = [];
	var function_indices = [];
	var parameter_indices = [];
	for(var i = 0; i < kernel.functions.length; i++){
		for(var j = 0; j < kernel.functions[i].gradients.length; j++){
			function_indices.push(i);
			parameter_indices.push(j);
			current.push(kernel.functions[i].parameters[j])
		}
	}
	var iterations = 0;
	while(max_delta > cutoff && iterations < max_iterations){
		max_delta = 0;
		iterations++;
		var K = applyKernel(X,X,kernel);
		var K_inv = K.inv();
		for(var i = 0; i < current.length; i++){
			delta = d_likelihood_slow(y,X,K,K_inv,kernel,function_indices[i],parameter_indices[i]);
			max_delta = Math.max(Math.abs(delta),max_delta);
			var old = current[i];
			current[i] -= delta * gamma;
		}
		for(var i = 0; i < current.length; i++){
			kernel.functions[function_indices[i]].parameters[parameter_indices[i]] = current[i];
		}
	}
}

//TODO: doesn't work
function d_likelihood(y,X,K,K_inv,kernel,function_index,param_index){
	var result = K_inv.x(y);
	result = result.x(result.transpose());
	result = result.subtract(K_inv);
	result = result.x(applyKernelGradient(X,X,kernel,function_index,param_index));
	return 0.5 * result.trace();
}

function d_likelihood_slow(y,X,K,K_inv,kernel,function_index,param_index){
	var d_K = applyKernelGradient(X,X,kernel,function_index,param_index);
	var result = y.transpose().x(K_inv).x(d_K).x(K_inv).x(y).e(0,0);
	var penalty = K_inv.x(d_K).trace();
	result = 0.5 * (result - penalty);
	return result
}

function applyKernel(X,Y,kernel){
	var result_array = []
	for(var i = 0; i < X.length; i++){
		result_array.push([]);
		for(var j = 0; j < Y.length; j++){
			result_array[i].push(kernel.kernel(X[i],Y[j]));
		}
	}
	return $M(result_array);
}

function applyKernelGradient(X,Y,kernel,function_index,param_index){
	var result_array = []
	for(var i = 0; i < X.length; i++){
		result_array.push([]);
		for(var j = 0; j < Y.length; j++){
			result_array[i].push(kernel.gradient(X[i],Y[j],function_index,param_index));
		}
	}
	return $M(result_array);
}

var Kernels = function(){

	function constant(x,y){
		return 1;
	}

	function linear(x,y){
		return x.dot(y);
	}

	function gaussianNoise(x,y){
		if(x.eql(y)){
			return 1;
		}
		return 0;
	}

	function squaredExponential(x,y,l){
		var diff = x.subtract(y);
		diff = diff.dot(diff);
		return Math.exp(-1 * (diff/(l*l)));
	}

	//matern with nu=3/2
	function matern(x,y,l){
		var diff = x.subtract(y);
		diff = diff.dot(diff);
		diff = Math.sqrt(diff);
		var result = (1 + Math.sqrt(3) * diff / l);
		result *= Math.exp(-1 * Math.sqrt(3) * diff/l);
		return result;
	}

	function kernelBuilder(){
		var functions = arguments;
		return{
			kernel : function(x,y){
				var result = 0;
				for(var i = 0; i < functions.length; i++){
					result += functions[i].kernel(x,y);
				}
				return result;
			},
			gradient : function(x,y,func,parameter){
				return functions[func].gradients[parameter](x,y);
			},

			functions : functions
		}
	}

	return {
		constant : function(theta) { 
			var parameters = [theta];
			return {
				kernel : function(x,y){return parameters[0] * constant(x,y);},
				gradients : [function(x,y){return constant(x,y);}],
				parameters : parameters
			}
		},

		linear : function(theta) {
			var parameters = [theta];
			return{
				kernel : function(x,y){return parameters[0] * linear(x,y);},
				gradients : [function(x,y){return linear(x,y);}],
				parameters : parameters
			}
		},
		gaussianNoise : function(theta) {
			var parameters = [theta];
			return{
				kernel : function(x,y){return parameters[0] * gaussianNoise(x,y);},
				gradients : [], //not differentiable
				parameters : parameters
			}
		},
		squaredExponential : function(theta,l) {
			var parameters = [theta,l];
			return{
				kernel : function(x,y){return parameters[0] * squaredExponential(x,y,parameters[1]);},
				gradients : [function(x,y){return squaredExponential(x,y,parameters[1])},
				             function(x,y){
				             	var result = -1 * Math.pow(x-y,2) * parameters[0] * squaredExponential(x,y,parameters[1])/Math.pow(parameters[1],3)
				             	return result;
				             }],
				parameters : parameters
			}
		},
		matern : function(theta,l) {
			var parameters = [theta,l];
			return{
				kernel : function(x,y){return parameters[0] * matern(x,y,parameters[1]);},
				gradients : [function(x,y){return matern(x,y,parameters[1])},
				             function(x,y){
				             	var diff = Math.abs(x - y);
				             	var result = 3 * parameters[0] * diff * diff;
				             	result *= Math.exp(-Math.sqrt(3) * diff / parameters[1]);
				             	result /= Math.pow(parameters[1],3);
				             	return result;
				             }],
				parameters : parameters
			}
		},
		kernelBuilder : kernelBuilder 
	}
}();

function GaussianProcess(kernel){
	function train(training_data,training_labels,testing_data){
		//build covariance matrix components
		var C = applyKernel(training_data,training_data,kernel);
		var k = applyKernel(training_data,testing_data,kernel);
		var Cinv = C.inv(); 
		var c = applyKernel(testing_data,testing_data,kernel);

		//condition
		var mu = k.transpose().x(Cinv.x(training_labels));
		var sigma = c.subtract(k.transpose().x(Cinv.x(k)));

		return{
			mu:mu,
			sigma:sigma
		}
	}

	function eval(point){
	}
	

	return{
		train:train,
		eval:eval,
	}
}

function wrapScalarsAsVectors(xs){
	result = [];
	for(var i = 0; i < xs.length; i++){
		result.push(Vector.create([xs[i]]));
	}
	return result;
}

