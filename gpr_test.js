var constant_theta = 1;
var linear_theta = 1;
var gaussian_theta = 1;
var exp_theta = 1;
var exp_width = -0.5;
var matern_theta = 1;
var matern_width = 1;

$(function(){
	$('#constant').keyup(function(){
		constant_theta = parseFloat($(this).val());
		if(isNaN(constant_theta)){
			constant_theta = 1;
		}
		K.functions[0].parameters[0] = constant_theta;
		replot();
	});
	$('#linear').keyup(function(){
		linear_theta = parseFloat($(this).val());
		if(isNaN(linear_theta)){
			linear_theta = 1;
		}
		K.functions[1].parameters[0] = linear_theta;
		replot();
	});
	$('#gaussian').keyup(function(){
		gaussian_theta = parseFloat($(this).val());
		if(isNaN(gaussian_theta)){
			gaussian_theta = 1;
		}
		K.functions[2].parameters[0] = gaussian_theta;
		replot();
	});
	$('#exp_theta').keyup(function(){
		exp_theta = parseFloat($(this).val());
		if(isNaN(exp_theta)){
			exp_theta = 1;
		}
		K.functions[3].parameters[0] = exp_theta;
		replot();
	});
	$('#exp_width').keyup(function(){
		exp_width = parseFloat($(this).val());
		if(isNaN(exp_width)){
			exp_width = -0.5;
		}
		K.functions[3].parameters[1] = exp_width;
		replot();
	});
	$('#matern_theta').keyup(function(){
		matern_theta = parseFloat($(this).val());
		if(isNaN(matern_theta)){
			matern_theta = 1;
		}
		K.functions[3].parameters[0] = matern_theta;
		replot();
	});
	$('#matern_width').keyup(function(){
		matern_width = parseFloat($(this).val());
		if(isNaN(matern_width)){
			matern_width = 1;
		}
		K.functions[3].parameters[1] = matern_width;
		replot();
	});
	$('#train').click(function(){
		gradientDescent(f1,m1,K,0.1,0.01);
		$('#constant').val(K.functions[0].parameters[0]);
		$('#linear').val(K.functions[1].parameters[0]);
		$('#gaussian').val(K.functions[2].parameters[0]);
		$('#exp_theta').val(K.functions[3].parameters[0]);
		$('#exp_width').val(K.functions[3].parameters[1]);
		$('#matern_theta').val(K.functions[4].parameters[0]);
		$('#matern_width').val(K.functions[4].parameters[1]);
		replot();
	})
	replot();
})

m1 = [];
plotting = [];
for(var i = -50; i < 50; i++){
	if(i > 10 && i < 30){
		continue;
	}
    m1.push(i/10);
}

var m2 = [];
for(var i = -70; i < 150; i++){
    m2.push(i/10 + Math.random()*0.1);
}

var f = [];
for(var i = 0; i < m1.length; i++){
	if(i < 60){
	    f.push((m1[i]) * m1[i]  + Math.random()*5 + 4);
	}
	else{
		f.push(m1[i] * m1[i] * m1[i] + Math.random()*m1[i]*5);
	}
}
f1 = $M(f);
// var K = Kernels.kernelBuilder(Kernels.constant(constant_theta),
// 	                          Kernels.linear(linear_theta),
// 	                          Kernels.gaussianNoise(gaussian_theta),
// 	                          Kernels.squaredExponential(exp_theta,exp_width));
var K = Kernels.kernelBuilder(Kernels.constant(constant_theta),
	                          Kernels.linear(linear_theta),
	                          Kernels.gaussianNoise(gaussian_theta),
 	                          Kernels.squaredExponential(exp_theta,exp_width),
	                          Kernels.matern(matern_theta,matern_width));
// var K = Kernels.kernelBuilder(Kernels.gaussianNoise(gaussian_theta),
// 	                          Kernels.squaredExponential(exp_theta,exp_width));
function replot(){
	// var K = Kernels.kernelBuilder(Kernels.constant(constant_theta),
		                          // Kernels.linear(linear_theta),
		                          // Kernels.gaussianNoise(gaussian_theta));
	var GPR = GaussianProcess(K); 
	var v_m1 = wrapScalarsAsVectors(m1);
	var v_m2 = wrapScalarsAsVectors(m2);

    var result = GPR.train(v_m1,f1,v_m2);

    var mu = result.mu;
    var sigma = result.sigma;

    var predicted = [];
    var upper = [];
    var lower = [];
    for(var i = 0; i < m2.length; i++){
        var mean = mu.e(i+1,1);
        predicted.push([m2[i],mean]);
        var error = sigma.e(i+1,i+1);
        upper.push([m2[i],mean+error]);
        lower.push([m2[i],mean-error]);
    }
    truth = [];
    for(var i = 0; i < m1.length; i++){
        truth.push([m1[i],f[i]]);
    }

    $.plot($("#placeholder"), [ predicted, upper, lower, truth]);
}