#pragma once

#include <vector>
#include "../../../Math/headers/Matrix2D.hpp"
#include "../../../Math/headers/Matrix3D.hpp"

namespace Neuranet
{
	/**
	 * @brief Class representation a type of activation function.
	 * @author Nolan Bridges
	 * @version 1.0.0
	 */
	enum class Activation
	{
		SIGMOID, RELU, RELU_NORMALIZED, SOFTMAX
	};

	/**
	 * @brief Class representation a type of loss function.
	 * @author Nolan Bridges
	 * @version 1.0.0
	 */
	enum class Loss
	{
		EUCLIDEAN, NEGATIVE_LOG
	};

	/**
	 * @brief Class representation a type of pooling function.
	 * @author Nolan Bridges
	 * @version 1.0.0
	 */
	enum class Pooling
	{
		MAX, AVERAGE
	};

	/**
	 * @brief A structure representing a fully-connected layer transformation.
	 */
	struct FCL
	{
		/**
		 * @brief The weights of the FCL.
		 */
		Matrix2D weights;

		/**
		 * @brief The biases of the FCL.
		 */
		Matrix2D biases;

		/**
		 * @brief The type of activation function used by the FCL.
		 */
		Activation activationType;

		FCL(Matrix2D weights, Matrix2D biases, Activation activationType) : weights(weights), biases(biases), activationType(activationType) {};

		FCL(const FCL& other) : weights(other.weights), biases(other.biases), activationType(other.activationType) {};

		FCL() : weights(Matrix2D(0, 0)), biases(Matrix2D(0, 0)), activationType(Activation::SIGMOID) {};
	};

	/**
	 * @brief The type of convolution.
	 */
	enum class ConvolutionType
	{
		CONVOLUTIONAL_LAYER, POOLING_LAYER, NO_TYPE
	};

	/**
	 * @brief A structure representing a layer in a CNN.
	 */
	struct Convolution
	{
		/**
		 * @brief The type of convolution.
		 */
		ConvolutionType convType;

		Convolution(ConvolutionType convType) : convType(convType) {};

		Convolution() : convType(ConvolutionType::NO_TYPE) {};

		virtual ~Convolution() {};
	};

	/**
	 * @brief A structure representing a convolution.
	 */
	struct ConvolutionalLayer : public Convolution
	{
		/**
		 * @brief The filter weights of the convolution.
		 */
		std::vector<Matrix3D> filterWeights;
		
		/**
		 * @brief The filter biases of the convolution.
		 */
		std::vector<double> filterBiases;

		/**
		 * @brief The stride used by the filters of the convolution.
		 */
		std::tuple<uint8_t, uint8_t> filterStride;

		/**
		 * @brief The padding used by the convolution.
		 */
		uint8_t padding;

		/**
		 * @brief The type of activation function used by the convolution.
		 */
		Activation activationType;

		ConvolutionalLayer(const ConvolutionalLayer& other) :
			filterWeights(other.filterWeights), filterBiases(other.filterBiases), filterStride(other.filterStride), padding(other.padding),
				activationType(other.activationType), Convolution(ConvolutionType::CONVOLUTIONAL_LAYER) {};

		ConvolutionalLayer(uint16_t numFilters, uint8_t filterWeightRowCount, uint8_t filterWeightColCount, uint8_t filterWeightLayCount, std::tuple<uint8_t, uint8_t> filterStride, uint8_t padding, Activation activationType) :
			filterStride(filterStride), padding(padding), activationType(activationType), Convolution(ConvolutionType::CONVOLUTIONAL_LAYER)
		{
			filterWeights = std::vector<Matrix3D>(numFilters);
			filterBiases = std::vector<double>(numFilters);

			for (uint16_t index = 0; index < numFilters; index += 1)
			{
				double lowerBound;
				double upperBound;

				switch (activationType)
				{
				case Activation::SIGMOID:
					lowerBound = -1.0;
					upperBound = 1.0;
					break;
				case Activation::RELU:
					lowerBound = 0.001;
					upperBound = 1.0;
					break;
				case Activation::RELU_NORMALIZED:
					lowerBound = 0.001;
					upperBound = 1.0;
					break;
				default:
					lowerBound = -1.0;
					upperBound = 1.0;
				}

				filterWeights[index] = Matrix3D::random(filterWeightRowCount, filterWeightColCount, filterWeightLayCount, lowerBound, upperBound);
				filterBiases[index] = 0.0;
			}
		};

		ConvolutionalLayer() : padding(0), filterWeights(std::vector<Matrix3D>(0)), filterBiases(std::vector<double>(0)), filterStride(std::make_tuple<uint8_t, uint8_t>(1, 1)), activationType(Activation::SIGMOID), Convolution(ConvolutionType::CONVOLUTIONAL_LAYER) {};
		
		~ConvolutionalLayer() {};
	};

	struct PoolingLayer : public Convolution
	{
		/**
		 * @brief The type of pooling function used by the convolution.
		 */
		Pooling poolingType;

		/**
		 * @brief The size of the pool used by the convolution.
		 */
		std::tuple<uint8_t, uint8_t> poolSize;

		/**
		 * @brief The stride used by the pooling method of the convolution.
		 */
		std::tuple<uint8_t, uint8_t> poolingStride;

		PoolingLayer(std::tuple<uint8_t, uint8_t> poolSize, std::tuple<uint8_t, uint8_t> poolingStride, Pooling poolingType) :
			poolingType(poolingType), poolSize(poolSize), poolingStride(poolingStride), Convolution(ConvolutionType::POOLING_LAYER) {};

		PoolingLayer() : poolingType(Pooling::MAX), poolSize(std::make_tuple<uint8_t, uint8_t>(1, 1)), poolingStride(std::make_tuple<uint8_t, uint8_t>(1, 1)), Convolution(ConvolutionType::POOLING_LAYER) {};
	
		~PoolingLayer() {};
	};

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
		 * @param lossType The loss function type to use.
		 * @return The loss of the network based on the inputted results.
		 */
		double loss(const Matrix2D& expectedOutput, const Matrix2D& output, Loss lossType);

		/**
		 * @brief Gets the bounds for the random weights of a given activation type.
		 * 
		 * @param activationType The activation type for which to find the bounds for weights.
		 * @return The bounds of the weights.
		 */
		std::tuple<double, double> getWeightBounds(uint16_t numInputs, uint16_t numOutputs, Activation activationType);

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
		 * @brief Puts a 2D matrix or vector through a softmax function.
		 *
		 * @param input The 2D matrix or vector to pass through the function.
		 * @return The softmax vector.
		 */
		Matrix2D softmax(Matrix2D& input);

		/**
		 * @brief Puts a 3D matrix through a softmax function.
		 *
		 * @param input The 3D matrix to pass through the function.
		 * @return The softmax vector.
		 */
		Matrix3D softmax(Matrix3D& input);

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