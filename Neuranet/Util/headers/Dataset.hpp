#pragma once

#include <string>
#include "../../Math/headers/Matrix2D.hpp"
#include "../../Math/headers/Matrix3D.hpp"

namespace Neuranet
{
	/**
	 * @brief Static class that returns parsed data from files.
     * @author Nolan Bridges
	 * @version 1.0.0
	 */
	class Dataset
	{
	public:
		/**
		 * @brief Constructs a dataset based on two file paths of input and expected output data. This function is for 2D data primarily.
		 *
		 * @param inputsFileName The path to the .csv file with the set of input data. Each row
		 *						 is an input vector, and each column is a value of each input vector.
		 * @param expectedOutputsFileName The path to the .csv file with the set of expected output data. Each row is an
		 *							      expected output vector, and each column is a value of each expected output vector.
		 * @param inputs The variable to which to return the inputs.
		 * @param expectedOutputs The variable to which to return the inputs.
		 */
		static void parse(std::string& inputsFileName, std::string& expectedOutputsFileName, std::vector<Matrix2D>& inputs, std::vector<Matrix2D>& expectedOutputs);
		
		/**
		 * @brief Constructs a dataset based on two file paths of input and expected output data. This function is for 3D data, such as images.
		 *
		 * @param inputsFileName The path to the .csv file with the set of input data. Each row
		 *						 is an input vector, and each column is a value of each input vector.
		 * @param expectedOutputsFileName The path to the .csv file with the set of expected output data. Each row is an
		 *							      expected output vector, and each column is a value of each expected output vector.
		 * @param inputs The variable to which to return the inputs.
		 * @param expectedOutputs The variable to which to return the inputs.
		 */
		//static void parse(std::string& inputsFileName, std::string& expectedOutputsFileName, std::vector<Matrix3D>& inputs, std::vector<Matrix2D>& expectedOutputs);

	};
}