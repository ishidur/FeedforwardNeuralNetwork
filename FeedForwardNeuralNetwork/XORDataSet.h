#pragma once
class XORDataSet
{
public:
	bool useSoftmax = false;
	Eigen::MatrixXd dataSet;
	Eigen::MatrixXd teachSet;
	Eigen::MatrixXd testDataSet;
	Eigen::MatrixXd testTeachSet;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	XORDataSet();
	void load();
};

