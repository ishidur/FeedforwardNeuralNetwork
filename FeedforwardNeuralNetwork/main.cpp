// FeedForwardNeuralNetwork.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include<windows.h>
#include <imagehlp.h>
#pragma comment(lib, "imagehlp.lib")
#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <ppl.h>
#include <tuple>
#include <vector>

#include "NeuralNetwork.h"
#include "config.h"
#include "DataSet.h"


int main()
{
    data_set->load();
	//	if (typeid(dataSet) == typeid(FuncApproxDataSet))
	//	{
	//		dataSet.show();
	//	}
	for (double init_val : initVals)
	{
        std::string dirName = "data\\";
        std::ostringstream sout;
		sout << std::fixed << init_val;
        std::string s = sout.str();
		dirName += s;
		dirName += "\\";
		if (!MakeSureDirectoryPathExists(dirName.c_str())) { break; }
        std::string filename = dirName;
		filename += "static.csv";
        std::ofstream ofs2(filename);
		for (std::vector<int> structure : structures)
		{
            std::string layers = "";
			for (int i = 0; i < structure.size(); ++i)
			{
				layers += std::to_string(structure[i]);
				if (i < structure.size() - 1) { layers += "X"; }
			}
			ofs2 << "structures," << layers << std::endl;
            std::cout << "structures; " << layers << std::endl;
			int correct = 0;
			for (int i = 0; i < TRIALS_PER_STRUCTURE; ++i)
			{
				time_t epoch_time;
				epoch_time = time(nullptr);
				ofs2 << "try," << i << std::endl;
                std::cout << "try: " << i << std::endl;
                std::string fileName = dirName;
				fileName += "result-";
				fileName += std::to_string(structure.size()) + "-layers-";
				fileName += layers;
				fileName += "-" + std::to_string(epoch_time);
				ofs2 << "file," << fileName << std::endl;
                std::cout << fileName << std::endl;
				double err;
				int n;
                NeuralNetwork network = NeuralNetwork(structure, init_val, data_set);
                
	            if (needPretrain)
	            {
		            //pretraining process
                    std::ofstream preofs(filename + "-ae" + ".csv");
		            network.pretrain(preofs);
		            preofs.close();
	            }
                std::tie(err, n) = network.learn(fileName);
				ofs2 << "error," << err << ",,learning time," << n << std::endl;
                std::cout << "error; " << err << std::endl;
				if (err < ERROR_BOTTOM) { correct++; }
			}
			ofs2 << correct << ", /, " << TRIALS_PER_STRUCTURE << ", success" << std::endl;
            std::cout << correct << " / " << TRIALS_PER_STRUCTURE << " success" << std::endl;
		}
		ofs2.close();
	}
	return 0;
}