#include "stdafx.h"
#include "MNISTDataSet.h"

#define PIXEL 784
#define TRAIN_IMAGE_PATH "./MNISTData/train-images-idx3-ubyte/train-images.idx3-ubyte"
#define TRAIN_LABEL_PATH "./MNISTData/train-labels-idx1-ubyte/train-labels.idx1-ubyte"
#define TEST_IMAGE_PATH "./MNISTData/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"
#define TEST_LABEL_PATH "./MNISTData/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"

int reverseInt(int i)
{
	unsigned char c1 = i & 255;
	unsigned char c2 = (i >> 8) & 255;
	unsigned char c3 = (i >> 16) & 255;
	unsigned char c4 = (i >> 24) & 255;

	return (int(c1) << 24) + (int(c2) << 16) + (int(c3) << 8) + c4;
}

Eigen::MatrixXd readImageFile(std::string filename)
{
	std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
	int magic_number = 0;
	int number_of_images = 0;
	int rows = 0;
	int cols = 0;
	ifs.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	ifs.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);
	ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
	rows = reverseInt(rows);
	ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));
	cols = reverseInt(cols);

	Eigen::MatrixXd images = Eigen::MatrixXd::Zero(number_of_images, PIXEL);
	std::cout << magic_number << " " << number_of_images << " " << rows << " " << cols << std::endl;
	std::cout << "fetching image data..." << std::endl;
	std::string progress = "";

	for (int i = 0; i < number_of_images; i++)
	{
		double status = double(i * 100.0 / (number_of_images - 1));
		if (progress.size() < int(status) / 5)
		{
			progress += "#";
		}
		std::cout << "progress: " << std::setw(4) << std::right << std::fixed << std::setprecision(1) << (status) << "% " << progress << "\r" << std::flush;
		for (int row = 0; row < rows; row++)
		{
			for (int col = 0; col < cols; col++)
			{
				unsigned char temp = 0;
				ifs.read(reinterpret_cast<char*>(&temp), sizeof(temp));
				double a = double(temp / 255.0);
				images(i, rows * row + col) = double(a);
			}
		}
	}
	std::cout << std::endl << "Done." << std::endl;
	return images;
}

Eigen::MatrixXd readLabelFile(std::string filename)
{
	std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
	int magic_number = 0;
	int number_of_images = 0;
	ifs.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	ifs.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);

	Eigen::MatrixXd _label = Eigen::MatrixXd::Zero(number_of_images, 10);

	std::cout << number_of_images << std::endl;

	std::cout << "fetching label data..." << std::endl;
	std::string progress = "";

	for (int i = 0; i < number_of_images; i++)
	{
		double status = double(i * 100.0 / (number_of_images - 1));
		if (progress.size() < int(status) / 5)
		{
			progress += "#";
		}
		std::cout << "progress: " << std::setw(4) << std::right << std::fixed << std::setprecision(1) << (status) << "% " << progress << "\r" << std::flush;
		unsigned char temp = 0;
		ifs.read(reinterpret_cast<char*>(&temp), sizeof(temp));
		_label(i, int(temp)) = 1.0;
	}
	std::cout << std::endl << "Done." << std::endl;
	return _label;
}

MNISTDataSet::MNISTDataSet()
{
}

void MNISTDataSet::load()
{
	//	dataSet = readImageFile(TEST_IMAGE_PATH);
	//	teachSet = readLabelFile(TEST_LABEL_PATH);
	dataSet = readImageFile(TRAIN_IMAGE_PATH);
	teachSet = readLabelFile(TRAIN_LABEL_PATH);
	testDataSet = readImageFile(TEST_IMAGE_PATH);
	testTeachSet = readLabelFile(TEST_LABEL_PATH);
}
