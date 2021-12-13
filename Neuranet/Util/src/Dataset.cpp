#pragma once

#include <vector>
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
}