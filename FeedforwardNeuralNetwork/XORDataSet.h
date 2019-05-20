#pragma once
#include "DataSet.h"

class XORDataSet: public DataSet
{
public:
	const bool useSoftmax = false;
	/**
	 * \brief create dataset
	 */
	void load() override;
};
