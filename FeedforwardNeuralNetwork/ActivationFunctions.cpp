#include "stdafx.h"
#include "ActivationFunctions.h"

auto relu = [](const double input)
{
	if (input < 0.0)
	{
		return 0.0;
	}
	return input;
};

Eigen::VectorXd Relu(Eigen::VectorXd inputs)
{
	return inputs.unaryExpr(relu);
}

auto tanhype = [](const double input)
{
	return tanh(input);
};

Eigen::VectorXd Tanh(Eigen::VectorXd inputs)
{
	return inputs.unaryExpr(tanhype);
}

auto sigm = [](const double input)
{
	return 1.0 / (1 + exp(-input));
};

Eigen::VectorXd sigmoid(Eigen::VectorXd inputs)
{
	return inputs.unaryExpr(sigm);
}

auto soft = [](const double x)
{
	return exp(x);
};

Eigen::VectorXd softmax(Eigen::VectorXd inputs)
{
	Eigen::VectorXd a = inputs.unaryExpr(soft);
	return a / a.sum();
}

Eigen::VectorXd differentialSigmoid(Eigen::VectorXd output)
{
	return output.array() * (Eigen::VectorXd::Ones(output.size()) - output).array();
}

Eigen::VectorXd differentialTanh(Eigen::VectorXd output)
{
	return Eigen::VectorXd::Ones(output.size()).array() - output.array() * output.array();
}

auto diffRelu = [](const double input)
{
	if (input <= 0.0)
	{
		return 0.0;
	}
	return 1.0;
};

Eigen::VectorXd differentialRelu(Eigen::VectorXd output)
{
	return output.unaryExpr(diffRelu);
}
