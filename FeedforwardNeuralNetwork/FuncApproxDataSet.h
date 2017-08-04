#pragma once

class FuncApproxDataSet
{
public:
	const bool useSoftmax = false;
	Eigen::MatrixXd dataSet;
	Eigen::MatrixXd teachSet;
	Eigen::MatrixXd testDataSet;
	Eigen::MatrixXd testTeachSet;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	FuncApproxDataSet();
	void load();
	void show();
	void update(Eigen::VectorXd outputs);
};
