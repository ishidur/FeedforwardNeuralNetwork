#pragma once
#include "Eigen/Core"

class DataSet
{
public:
    DataSet(){};
    virtual ~DataSet() = default;
	bool useSoftmax;
	Eigen::MatrixXd inputSet;
	Eigen::MatrixXd teachSet;
	Eigen::MatrixXd testInputSet;
	Eigen::MatrixXd testTeachSet;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	/**
	 * \brief create dataset
	 */
	virtual void load();
};

