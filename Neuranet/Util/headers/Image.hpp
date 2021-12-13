#pragma once

#include "../../Math/headers/Matrix2D.hpp"
#include "../../Math/headers/Matrix3D.hpp"

namespace Neuranet
{
	/**
	 * @brief Class used for reading images to and writing images from file.
	 * @author Nolan Bridges
	 * @version 1.0.0
	 */
	class Image
	{
	public:
		/**
		 * @brief Writes an RGB image to file, given its pixels' colors.
		 * @param fileName The file path to write the image to.
		 * @param pixels The pixel colors in RGB format, represented as an mxnx3 matrix.
		 */
		static void write(std::string& fileName, const Matrix3D& pixels);

		/**
		 * @brief Writes a grayscale image to file, given its pixels' grayscale values.
		 * @param fileName The file path to write the image to.
		 * @param pixels The pixel grayscale values, represented as an mxn matrix.
		 */
		static void write(std::string& fileName, const Matrix2D& pixels);

		/**
		 * @brief Reads an RGB image from file.
		 * @param fileName The file path from which to read the image.
		 * @param pixels The variable to return the pixels' RGB values to.
		 */
		static void read(std::string& fileName, Matrix3D& pixels);

		/**
		 * @brief Reads a grayscale image from file.
		 * @param fileName The file path from which to read the image.
		 * @param pixels The variable to return the pixels' grayscale values to.
		 */
		static void read(std::string& fileName, Matrix2D& pixels);
	};
}