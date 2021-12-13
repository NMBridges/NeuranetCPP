#pragma once

#include "../../../Math/headers/Matrix.hpp"
#include "../../../Math/headers/Matrix2D.hpp"
#include "../../../Math/headers/Matrix3D.hpp"
#include "../headers/Activation.hpp"

namespace Neuranet
{
	/**
	 * @brief An abstract class representing a type of neural network.
	 * 
	 * @author Nolan Bridges
	 * @version 1.0.0
	 */
	class Network
	{
	protected:
		/**
		 * @brief Calculates the loss of the network, given an expected output and output.
		 *
		 * @param expectedOutput The expected output of the network for a certain input.
		 * @param output The actual output of the network for that certain input.
		 * @return The loss of the network based on the inputted results.
		 */
		double loss(const Matrix2D& expectedOutput, const Matrix2D& output);

		/**
		 * @brief Puts a number through a sigmoid function.
		 *
		 * @param input The number to pass through the function.
		 * @return The sigmoid output.
		 */
		double sigmoid(double input);

		/**
		 * @brief Puts a number through a ReLU function.
		 * 
		 * @param input The number to pass through the function.
		 * @return The ReLU output.
		 */
		double reLU(double input);

		/**
		 * @brief Activates an inputted matrix.
		 *
		 * @param input The matrix to modify by activating it.
		 * @param activationType The activation function to use.
		 */
		void activate(Matrix2D& input, Activation activationType);

		/**
		 * @brief Activates an inputted matrix.
		 *
		 * @param input The matrix to modify by activating it.
		 * @param activationType The activation function to use.
		 */
		void activate(Matrix3D& input, Activation activationType);

		/**
		 * @brief Returns the derivative of an inputted matrix along an activation function.
		 *
		 * @param input The matrix to take the derivative of along the activation function.
		 * @param activationType The activation function to use.
		 * 
		 * @return The derivative of the activation function at the inputted matrix.
		 */
		Matrix2D activateDerivative(Matrix2D& input, Activation activationType);

		/**
		 * @brief Returns the derivative of an inputted matrix along an activation function.
		 *
		 * @param input The matrix to take the derivative of along the activation function.
		 * @param activationType The activation function to use.
		 *
		 * @return The derivative of the activation function at the inputted matrix.
		 */
		Matrix3D activateDerivative(Matrix3D& input, Activation activationType);
	};
}