#pragma once

#include <vector>
#include <format>
#include <fstream>
#include <sstream>
#include "../headers/Dataset.hpp"

namespace Neuranet
{
	void Dataset::parse(std::string& inputsFileName, std::string& expectedOutputsFileName, std::vector<Matrix2D>& inputs, std::vector<Matrix2D>& expectedOutputs)
	{
		inputs = std::vector<Matrix2D>();
		expectedOutputs = std::vector<Matrix2D>();
		
		std::fstream fin;
		std::string line, entry;
		std::vector<double> entries;

		fin.open(inputsFileName, std::ios::in);

		if (fin.is_open())
		{
			while (std::getline(fin, line))
			{
				entries.clear();
				std::stringstream linestream(line);

				while (std::getline(linestream, entry, ','))
				{
					entries.push_back(std::stod(entry));
				}

				Matrix2D input(entries.size(), 1, &entries[0]);
				inputs.push_back(input);
			}
		}

		fin.close();
		fin.open(expectedOutputsFileName, std::ios::in);

		if (fin.is_open())
		{
			while (std::getline(fin, line))
			{
				entries.clear();
				std::stringstream linestream(line);

				while (std::getline(linestream, entry, ','))
				{
					entries.push_back(std::stod(entry));
				}

				Matrix2D expectedOutput(entries.size(), 1, &entries[0]);
				expectedOutputs.push_back(expectedOutput);
			}
		}
	}

	void Dataset::parse(std::string& inputsFileName, std::string& expectedOutputsFileName, std::vector<Matrix3D>& inputs, std::vector<Matrix2D>& expectedOutputs)
	{
		inputs = std::vector<Matrix3D>();
		expectedOutputs = std::vector<Matrix2D>();

		std::fstream fin;
		std::string line, entry;
		std::vector<double> entries;

		fin.open(inputsFileName, std::ios::in);

		if (fin.is_open())
		{
			int lineNum = 0;
			uint16_t rows, cols, lays;

			while (std::getline(fin, line))
			{
				if (lineNum == 0)
				{
					entries.clear();
					std::stringstream linestream(line);

					while (std::getline(linestream, entry, ','))
					{
						entries.push_back(std::stoi(entry));
					}

					rows = entries[0];
					cols = entries[1];
					lays = entries[2];
				}
				else
				{
					entries.clear();
					std::stringstream linestream(line);

					while (std::getline(linestream, entry, ','))
					{
						entries.push_back(std::stod(entry));
					}

					Matrix3D input(rows, cols, lays, &entries[0]);
					inputs.push_back(input);
				}
				
				lineNum += 1;
			}
		}

		fin.close();
		fin.open(expectedOutputsFileName, std::ios::in);

		if (fin.is_open())
		{
			while (std::getline(fin, line))
			{
				entries.clear();
				std::stringstream linestream(line);

				while (std::getline(linestream, entry, ','))
				{
					entries.push_back(std::stod(entry));
				}

				Matrix2D expectedOutput(entries.size(), 1, &entries[0]);
				expectedOutputs.push_back(expectedOutput);
			}
		}
	}
	
	static int reverseInt(int i)
	{
		unsigned char c1, c2, c3, c4;

		c1 = i & 255;
		c2 = (i >> 8) & 255;
		c3 = (i >> 16) & 255;
		c4 = (i >> 24) & 255;

		return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
	}

	void Dataset::parseMNISTimages(std::string& imagesFilePath, std::string& labelsFilePath, std::vector<Matrix3D>& images, std::vector<Matrix2D>& labels)
	{
		// Read images.

		std::ifstream file(imagesFilePath, std::ios::binary);
		
		if (file.is_open())
		{
			int magicNumber, numImages, rowCount, colCount = 0;
			
			file.read((char*) &magicNumber, sizeof(magicNumber));
			magicNumber = reverseInt(magicNumber);
			
			file.read((char*) &numImages, sizeof(numImages));
			numImages = reverseInt(numImages);
			
			file.read((char*) &rowCount, sizeof(rowCount));
			rowCount = reverseInt(rowCount);
			
			file.read((char*) &colCount, sizeof(colCount));
			colCount = reverseInt(colCount);

			std::cout << "Reading MNIST Database files..." << std::endl;
			std::cout << "\tNumber of images: " << numImages << std::endl;
			std::cout << "\tRows per image: " << rowCount << std::endl;
			std::cout << "\tColumns per image: " << colCount << "\n" << std::endl;
			
			images = std::vector<Matrix3D>(numImages);
			
			for (int index = 0; index < numImages; index += 1)
			{
				images[index] = Matrix3D(rowCount, colCount, 1);
				for (int row = 0; row < rowCount; row += 1)
				{
					for (int col = 0; col < colCount; col += 1)
					{
						unsigned char temp = 0;
						file.read((char*) &temp, sizeof(temp));

						images[index].set(row, col, 0, (double)temp / 255.0);
					}
				}

				if ((index + 1) % 50 == 0)
				{
					std::cout << std::format("\r\tReading image data: ({}/{})  {}% ", index + 1, numImages, (int) (10000.0 * (1.0 + index) / numImages) / 100.0);
				}
			}

			std::cout << "\n\n";

			file.close();
		}
		else
		{
			std::cout << "FAILED TO PARSE MNIST DATASET" << std::endl;
		}

		// Read labels.

		file.close();
		file = std::ifstream(labelsFilePath, std::ios::binary);

		if (file.is_open())
		{
			int magicNumber, numLabels = 0;
			
			file.read((char*) &magicNumber, sizeof(magicNumber));
			magicNumber = reverseInt(magicNumber);

			file.read((char*)&numLabels, sizeof(numLabels));
			numLabels = reverseInt(numLabels);

			labels = std::vector<Matrix2D>(numLabels);

			for (int index = 0; index < numLabels; index += 1)
			{
				unsigned char temp = 0;
				file.read((char*) &temp, 1);
				labels[index] = Matrix2D(10, 1);
				labels[index].set(temp, 0, 1.0);

				if ((index + 1) % 50 == 0)
				{
					std::cout << std::format("\r\tReading label data: ({}/{})  {}% ", index + 1, numLabels, (int)(10000.0 * (1.0 + index) / numLabels) / 100.0);
				}
			}

			std::cout << "\n\n";
		}
		else
		{
			std::cout << "FAILED TO PARSE MNIST DATASET" << std::endl;
		}
	}
}