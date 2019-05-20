#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include "DataSet.h"

class MNISTDataSet: public DataSet
{
public:
	const bool useSoftmax = true;
	/**
	 * \brief create dataset
	 */
	void load() override;
};
