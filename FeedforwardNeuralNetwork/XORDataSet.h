#pragma once
#include "DataSet.h"

class XORDataSet final : public DataSet
{
public:
	XORDataSet();
	/**
	 * \brief create dataset
	 */
	void load() override;
};
