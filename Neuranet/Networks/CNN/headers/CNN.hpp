#pragma once

#include "../../Common/headers/Network.hpp"

namespace Neuranet
{
	/**
	 * @brief Class representing a convolutional neural network.
	 * @author Nolan Bridges
	 * @version 1.0.0
	 */
	class CNN : public Network
	{
	private:
		/**
		 * @brief The convolutional layers of the convolutional stage of the network.
		 */
		std::vector<Convolution*> convolutions;
		
		/**
		 * @brief The fully-connected layers.
		 */
		std::vector<FCL> fcls;

		/**
		 * @brief The number of nodes in the fully-connected layers.
		 */
		std::vector<uint16_t> nodeCounts;

	public:
		/**
		 * @brief No-arg constructor that creates a blank CNN.
		 */
		CNN(uint16_t numOutputs, Activation finalActivationType);

		/**
		 * @brief Destructor.
		 */
		~CNN();

		/**
		 * @brief Adds a convolution to the end of the list of convolutions.
		 *
		 * @param conv The convolution to add.
		 */
		void addConvolution(Convolution* conv);

		/**
		 * @brief Adds a hidden layer before the output layer of the fully-connected section of the network.
		 *
		 * @param nodeCount The number of nodes in the hidden layer.
		 * @param activationType The activation type used by the hidden layer.
		 */
		void addHiddenLayer(uint16_t nodeCount, Activation activationType);

		/**
		 * @brief Gets the average loss of the model for the given input / output pairs. All inputs must be of the correct size.
		 *
		 * @param inputs The inputs to test.
		 * @param expectedOutputs The corresponding expected outputs.
		 *
		 * @return The average loss of the model for the given input / output pairs.
		 */
		double getAverageLoss(const std::vector<Matrix3D>& inputs, const std::vector<Matrix2D>& expectedOutputs);

		/**
		 * @brief Computes the output vector of the network.
		 * 
		 * @param input The input for which to compute the output.
		 * @return The output of the network.
		 */
		Matrix2D compute(const Matrix3D& input);

		/**
		 * @brief Provides the CNN with datapoints to test and adapt the model to. Defaults the learning
		 *		  rate to 1.0.
		 *
		 * @param inputs The list of input matrices.
		 * @param expectedOutputs The list of expected output matrices.
		 * @param epochs The number of times for the model to learn from the dataset.
		 * @param batchSize The number of input/outputs to get gradients from before modifying the weights and biases of the model with the averaged gradients.
		 * @param learningRate The learning rate.
		 */
		void learn(const std::vector<Matrix3D>& inputs, const std::vector<Matrix2D>& expectedOutputs, uint16_t epochs, uint16_t batchSize, double learningRate);

	//protected:
		/**
		 * @brief Gets the gradients for the weights and biases with respect to the loss function,
		 *		  given an input and expected output.
		 *
		 * @param input The input values.
		 * @param expectedOutput The expected output values.
		 * @param weightGradient The return variable for the filter weight gradients.
		 * @param biasGradient The return variable for the filter bias gradients.
		 */
		void getGradients(const Matrix3D& input, const Matrix2D& expectedOutput, std::vector<std::vector<Matrix3D>>& filterWeightGradients, std::vector<std::vector<Matrix2D>>& filterBiasGradients);

	public:
		/**
		 * @brief Overrides the bitshifting operator such that the state
		 *        of the object as a string is printed.
		 */
		friend std::ostream& operator<<(std::ostream& os, const CNN& a);
	};
}