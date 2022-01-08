#pragma once

#include <vector>
#include "../../Common/headers/Network.hpp"

namespace Neuranet
{
	/**
	 * @brief Class representing an artificial neural network.
	 * @author Nolan Bridges
	 * @version 1.0.0
	 */
	class NeuralNetwork : public Network
	{
	private:
		/**
		 * @brief The weights of the neural network.
		 */
		std::vector<Matrix2D> weights;
		
		/**
		 * @brief The biases of the neural network.
		 */
		std::vector<Matrix2D> biases;
		
		/**
		 * @brief The type of activation function that the neural network uses.
		 */
		Activation activationType;

	public:
		/**
		 * @brief Full-args initializer for the neural network. The nodeCounts pointer
		 *		  is deleted automatically.
		 * 
		 * @param nodeCounts The number of nodes in each layer of the neural network, including those in the input and output.
		 * @param activationType The type of activation function that the neural network uses.
		 */
		NeuralNetwork(std::vector<uint16_t> nodeCounts, Activation activationType);

		/**
		 * @brief One-arg initializer for the neural network. Defaults the activation type to SIGMOID. The nodeCounts pointer
		 *		  is deleted automatically.
		 * 
		 * @param nodeCounts The number of nodes in each layer of the neural network, including those in the input and output.
		 */
		NeuralNetwork(std::vector<uint16_t> nodeCounts) : NeuralNetwork(nodeCounts, Activation::SIGMOID) {};

		/**
		 * @brief [DO NOT USE] No-arg initializer for the neural network.
		 */
		NeuralNetwork() : NeuralNetwork(std::vector<uint16_t>(0), Activation::SIGMOID) {};
		
		/**
		 * @brief Destructor for the neural network class.
		 */
		~NeuralNetwork();

		/**
		 * @brief Gets the average loss of the model for the given input / output pairs.
		 * 
		 * @param inputs The inputs to test.
		 * @param expectedOutputs The corresponding expected outputs.
		 * 
		 * @return The average loss of the model for the given input / output pairs.
		 */
		double getAverageLoss(const std::vector<Matrix2D>& inputs, const std::vector<Matrix2D>& expectedOutputs);

		/**
		 * @brief Gets the average loss of the model for the given input / output pairs.
		 *
		 * @param inputs The inputs to test.
		 * @param expectedOutputs The corresponding expected outputs.
		 *
		 * @return The average loss of the model for the given input / output pairs.
		 */
		double getAverageLoss(std::vector<Matrix3D>& inputs, const std::vector<Matrix2D>& expectedOutputs);

		/**
		 * @brief Computes the output for the neural network with the given input.
		 * 
		 * @param input The mx1 matrix input.
		 * @return The output of the neural network at the given input.
		 */
		Matrix2D compute(const Matrix2D& input);

		/**
		 * @brief Provides the neural network with datapoints to test and adapt the model to.
		 *
		 * @param inputs The list of input matrices.
		 * @param expectedOutputs The list of expected output matrices.
		 * @param epochs The number of times for the model to learn from the dataset.
		 * @param batchSize The number of input/outputs to get gradients from before modifying the weights and biases of the model with the averaged gradients.
		 * @param learningRate The learning rate of the model. Standard is 1.0.
		 */
		void learn(const std::vector<Matrix2D>& inputs, const std::vector<Matrix2D>& expectedOutputs, uint16_t epochs, uint16_t batchSize, double learningRate);

		/**
		 * @brief Provides the neural network with datapoints to test and adapt the model to.
		 *
		 * @param inputs The list of input matrices.
		 * @param expectedOutputs The list of expected output matrices.
		 * @param epochs The number of times for the model to learn from the dataset.
		 * @param batchSize The number of input/outputs to get gradients from before modifying the weights and biases of the model with the averaged gradients.
		 * @param learningRate The learning rate of the model. Standard is 1.0.
		 */
		void learn(std::vector<Matrix3D>& inputs, const std::vector<Matrix2D>& expectedOutputs, uint16_t epochs, uint16_t batchSize, double learningRate);

		/**
		 * @brief Provides the neural network with datapoints to test and adapt the model to.
		 *
		 * @param inputs The file name of the list of input matrices (.csv).
		 * @param expectedOutputs The file name of the list of expected output matrices (.csv).
		 * @param epochs The number of times for the model to learn from the dataset.
		 * @param batchSize The number of input/outputs to get gradients from before modifying the weights and biases of the model with the averaged gradients.
		 * @param learningRate The learning rate of the model. Standard is 1.0.
		 */
		void learn(std::string& inputsFilePath, std::string& expectedOutputsFilePath, uint16_t epochs, uint16_t batchSize, double learningRate);

		/**
		 * @brief Provides the neural network with datapoints to test and adapt the model to. Defaults the learning
		 *		  rate to 1.0.
		 *
		 * @param inputs The list of input matrices.
		 * @param expectedOutputs The list of expected output matrices.
		 * @param epochs The number of times for the model to learn from the dataset.
		 * @param batchSize The number of input/outputs to get gradients from before modifying the weights and biases of the model with the averaged gradients.
		 */
		void learn(const std::vector<Matrix2D>& inputs, const std::vector<Matrix2D>& expectedOutputs, uint16_t epochs, uint16_t batchSize);

		/**
		 * @brief Provides the neural network with datapoints to test and adapt the model to. Defaults the learning
		 *		  rate to 1.0.
		 *
		 * @param inputs The file name of the list of input matrices (.csv).
		 * @param expectedOutputs The file name of the list of expected output matrices (.csv).
		 * @param epochs The number of times for the model to learn from the dataset.
		 * @param batchSize The number of input/outputs to get gradients from before modifying the weights and biases of the model with the averaged gradients.
		 */
		void learn(std::string& inputsFilePath, std::string& expectedOutputsFilePath, uint16_t epochs, uint16_t batchSize);

		/**
		 * @brief Provides the neural network with datapoints to test and adapt the model to. Defaults the learning
		 *		  rate to 1.0 and the batchSize to 1.
		 *
		 * @param inputs The list of input matrices.
		 * @param expectedOutputs The list of expected output matrices.
		 * @param epochs The number of times for the model to learn from the dataset.
		 */
		void learn(const std::vector<Matrix2D>& inputs, const std::vector<Matrix2D>& expectedOutputs, uint16_t epochs);

		/**
		 * @brief Provides the neural network with datapoints to test and adapt the model to. Defaults the learning
		 *		  rate to 1.0 and batch size to 1.
		 *
		 * @param inputs The file name of the list of input matrices (.csv).
		 * @param expectedOutputs The file name of the list of expected output matrices (.csv).
		 * @param epochs The number of times for the model to learn from the dataset.
		 */
		void learn(std::string& inputsFilePath, std::string& expectedOutputsFilePath, uint16_t epochs);

		/**
		 * @brief Provides the neural network with datapoints to test and adapt the model to. Defaults the learning
		 *		  rate to 1.0, the batchSize to 1, and the epochs to 1.
		 *
		 * @param inputs The list of input matrices.
		 * @param expectedOutputs The list of expected output matrices.
		 */
		void learn(const std::vector<Matrix2D>& inputs, const std::vector<Matrix2D>& expectedOutputs);

		/**
		 * @brief Provides the neural network with datapoints to test and adapt the model to. Defaults the learning
		 *		  rate to 1.0, batch size to 1, and epochs to 1.
		 *
		 * @param inputs The file name of the list of input matrices (.csv).
		 * @param expectedOutputs The file name of the list of expected output matrices (.csv).
		 */
		void learn(std::string& inputsFilePath, std::string& expectedOutputsFilePath);
		
		/**
		 * @brief Gets the gradients for the weights and biases with respect to the loss function,
		 *		  given an input and expected output.
		 * 
		 * @param input The input values.
		 * @param expectedOutput The expected output values.
		 * @param weightGradient The return variable for the weight gradients.
		 * @param biasGradient The return variable for the bias gradients.
		 */
		void getGradients(const Matrix2D& input, const Matrix2D& expectedOutput, std::vector<Matrix2D>& weightGradients, std::vector<Matrix2D>& biasGradients);

		/**
		 * @brief Overrides the equals operator such that a neural network can be assigned to another.
		 *
		 * @param a The neural network to set the current one to.
		 * @return The new neural network.
		 */
		NeuralNetwork& operator=(const NeuralNetwork& a);

		/**
		 * @brief Writes the weights and biases as images in the specified folder.
		 * 
		 * @param folderPath The path of the folder to which to write the images.
		 */
		void writeToPath(std::string& folderPath);

		/**
		 * @brief Returns the state of the neural network as a string.
		 *
		 * @return the state of the neural network as a string.
		 */
		std::string toString();

		/**
		 * @brief Overrides the bitshifting operator such that the state
		 *        of the object as a string is printed.
		 */
		friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork& a);
	};
}