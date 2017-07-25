#pragma once
class XORDataSet
{
public:
	MatrixXd dataSet;
	MatrixXd teachSet;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	XORDataSet();
};

