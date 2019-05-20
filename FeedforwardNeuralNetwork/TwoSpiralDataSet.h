#pragma once
#include "DataSet.h"

class TwoSpiralDataSet final : public DataSet
{
public:
    TwoSpiralDataSet();
	/**
	 * \brief create dataset
	 */
	void load() override;
};
