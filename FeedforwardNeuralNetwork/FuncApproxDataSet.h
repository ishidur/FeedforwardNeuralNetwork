#pragma once
#include "DataSet.h"
class FuncApproxDataSet final : public DataSet
{
public:
    FuncApproxDataSet();
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
