#pragma once
class TwoSpiralDataSet
{
public:
	bool useSoftmax = false;
	Eigen::MatrixXd dataSet;
	Eigen::MatrixXd teachSet;
	Eigen::MatrixXd testDataSet;
	Eigen::MatrixXd testTeachSet;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	TwoSpiralDataSet();
	/**
	 * \brief create dataset
	 */
	void load();
};
