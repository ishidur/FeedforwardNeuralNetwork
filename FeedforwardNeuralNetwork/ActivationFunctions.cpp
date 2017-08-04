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
	double s = a.sum();
	Eigen::VectorXd b = a / s;
	return b;
}

Eigen::VectorXd differentialSigmoid(Eigen::VectorXd input)
{
	Eigen::VectorXd result = input.array() * (Eigen::VectorXd::Ones(input.size()) - input).array();
	return result;
}

Eigen::VectorXd differentialTanh(Eigen::VectorXd input)
{
	Eigen::VectorXd result = Eigen::VectorXd::Ones(input.size()).array() - input.array() * input.array();
	return result;
}

auto diffRelu = [](const double input)
{
	if (input < 0.0)
	{
		return 0.0;
	}
	return 1.0;
};

Eigen::VectorXd differentialRelu(Eigen::VectorXd input)
{
	Eigen::VectorXd result = input.unaryExpr(relu);
	return result;
}
