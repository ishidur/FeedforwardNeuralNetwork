#pragma once
#include "DataSet.h"
class FuncApproxDataSet: public DataSet
{
public:
	const bool useSoftmax = false;
	/**
	 * \brief create dataset
	 */
	void load() override;
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
