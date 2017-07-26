#pragma once
class XORDataSet
{
public:
	MatrixXd dataSet;
	MatrixXd teachSet;
	MatrixXd testDataSet;
	MatrixXd testTeachSet;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	XORDataSet();
};

