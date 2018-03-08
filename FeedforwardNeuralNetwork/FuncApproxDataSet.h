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
	/**
	 * \brief create dataset
	 */
	void load();
	/**
	 * \brief visualize
	 */
	void show();
	/**
	 * \brief update visual
	 * \param outputs 
	 */
	void update(Eigen::VectorXd outputs);
};
