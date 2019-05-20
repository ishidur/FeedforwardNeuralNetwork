#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include "DataSet.h"

class MNISTDataSet final : public DataSet
{
public:
    MNISTDataSet();
	/**
	 * \brief create dataset
	 */
	void load() override;
};
