#include "stdafx.h"
#include "LoadCSVDataSet.h"
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

#define SET_PATERNS 9999
#define INPUT_DIM 4
#define OUTPUT_DIM 1
#define importFileName "normalizedTrainingSet10000Eps10.0.csv"


LoadCSVDataSet::LoadCSVDataSet()
{
}

void LoadCSVDataSet::load()
{
	dataSet.resize(SET_PATERNS, INPUT_DIM);
	teachSet.resize(SET_PATERNS, OUTPUT_DIM);

	//ファイルの読み込み
	ifstream ifs(importFileName);
	if (!ifs) {
		cout << "入力エラー";
	}

	//csvファイルを1行ずつ読み込む
	string str;
	int lineCount = 0;

	while (getline(ifs, str)) {
		int rowCount = 0;
		string token;
		istringstream stream(str);

		//1行のうち、文字列とコンマを分割する
		while (getline(stream, token, ',')) {
			//すべて文字列として読み込まれるため
			//数値は変換が必要
			double temp = stof(token); //stof(string str) : stringをfloatに変換

			if(rowCount<INPUT_DIM)
			{
				//cout << "I:" << temp;
				//cout << rowCount << endl;
				dataSet(lineCount,rowCount) = temp;
			}

			else
			{
				//cout << "O:" << temp;
				teachSet(lineCount, rowCount-INPUT_DIM) = temp;

			}
			rowCount++;
		}
		//cout << endl;
		//countColum++;
		lineCount++;
	}

	//cout << dataSet << endl;
	//cout << teachSet << endl;
	testDataSet = dataSet;
	testTeachSet = teachSet;
}