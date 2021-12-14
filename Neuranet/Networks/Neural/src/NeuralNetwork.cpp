#pragma once

#include <format>
#include <direct.h>
#include "../headers/NeuralNetwork.hpp"
#include "../../../Util/headers/Image.hpp"
#include "../../../Util/headers/Dataset.hpp"

namespace Neuranet
{
	NeuralNetwork::NeuralNetwork(std::vector<uint16_t> nodeCounts, Activation activationType)
	{
		this->activationType = activationType;

		if (nodeCounts.size() > 0)
		{
			this->weights = std::vector<Matrix2D>(nodeCounts.size() - 1);
			this->biases = std::vector<Matrix2D>(nodeCounts.size() - 1);
		}

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
			upperBound =   1.0;
			break;
		case Activation::RELU_NORMALIZED:
			lowerBound = 0.001;
			upperBound =   1.0;
			break;
		default:
			lowerBound = -1.0;
			upperBound =  1.0;
		}

		for (uint16_t layer = 0; layer < nodeCounts.size() - 1; layer += 1)
		{
			this->weights[layer] = Matrix2D(nodeCounts[layer + 1], nodeCounts[layer]);
			this->weights[layer].randomize(lowerBound, upperBound);
			this->biases[layer] = Matrix2D(nodeCounts[layer + 1], 1);
		}
	}

	NeuralNetwork::~NeuralNetwork()
	{

	}

	double NeuralNetwork::getAverageLoss(const std::vector<Matrix2D>& inputs, const std::vector<Matrix2D>& expectedOutputs)
	{
		if (inputs.size() == 0 || expectedOutputs.size() == 0)
		{
			return 0.0;
		}
		double totalLoss = 0.0;
		for (int index = 0; index < inputs.size(); index += 1)
		{
			totalLoss += loss(expectedOutputs[index], compute(inputs[index]));
		}
		return totalLoss / inputs.size();
	}

	Matrix2D NeuralNetwork::compute(const Matrix2D& input)
	{
		Matrix2D current = input;

		for (uint16_t layer = 0; layer < this->weights.size(); layer += 1)
		{
			current = weights[layer] * current + biases[layer];
			activate(current, this->activationType);
		}

		return current;
	}

	void NeuralNetwork::learn(const std::vector<Matrix2D>& inputs, const std::vector<Matrix2D>& expectedOutputs, uint16_t epochs, uint16_t batchSize, double learningRate)
	{
		for (uint16_t epoch = 0; epoch < epochs; epoch += 1)
		{
			for (int batchStartIndex = 0; batchStartIndex < inputs.size(); batchStartIndex += batchSize)
			{
				// Sums the gradients of the weights and biases for all datasets.

				uint16_t thisBatchSize = std::min((uint16_t)(inputs.size() - batchStartIndex), batchSize);

				std::vector<Matrix2D> totalWeightGradients = std::vector<Matrix2D>(this->weights.size());
				std::vector<Matrix2D> totalBiasGradients = std::vector<Matrix2D>(this->biases.size());

				for (int batchIndex = 0; batchIndex < thisBatchSize; batchIndex += 1)
				{
					std::vector<Matrix2D> weightGradients;
					std::vector<Matrix2D> biasGradients;
					getGradients(inputs[batchStartIndex + batchIndex], expectedOutputs[batchStartIndex + batchIndex], weightGradients, biasGradients);
					
					for (uint16_t gradientIndex = 0; gradientIndex < this->weights.size(); gradientIndex += 1)
					{
						if (totalWeightGradients[gradientIndex].getRowCount() == 0 && totalWeightGradients[gradientIndex].getColumnCount() == 0)
						{
							totalWeightGradients[gradientIndex] = weightGradients[gradientIndex];
							totalBiasGradients[gradientIndex] = biasGradients[gradientIndex];
						}
						else
						{
							totalWeightGradients[gradientIndex] += weightGradients[gradientIndex];
							totalBiasGradients[gradientIndex] += biasGradients[gradientIndex];
						}
					}
				}

				// Modifies the weights and biases by the average gradient of the batch.

				for (uint16_t gradientIndex = 0; gradientIndex < this->weights.size(); gradientIndex += 1)
				{
					this->weights[gradientIndex] -= totalWeightGradients[gradientIndex] / thisBatchSize * learningRate;
					this->biases[gradientIndex] -= totalBiasGradients[gradientIndex] / thisBatchSize * learningRate;
				}
			}
		}
	}

	void NeuralNetwork::learn(std::string& inputsFilePath, std::string& expectedOutputsFilePath, uint16_t epochs, uint16_t batchSize, double learningRate)
	{
		std::vector<Matrix2D> ins;
		std::vector<Matrix2D> expouts;
		Dataset::parse(inputsFilePath, expectedOutputsFilePath, ins, expouts);
		learn(ins, expouts, epochs, batchSize, learningRate);
	}

	void NeuralNetwork::learn(const std::vector<Matrix2D>& inputs, const std::vector<Matrix2D>& expectedOutputs, uint16_t epochs, uint16_t batchSize)
	{
		learn(inputs, expectedOutputs, epochs, batchSize, 1.0);
	}

	void NeuralNetwork::learn(std::string& inputsFilePath, std::string& expectedOutputsFilePath, uint16_t epochs, uint16_t batchSize)
	{
		learn(inputsFilePath, expectedOutputsFilePath, epochs, batchSize, 1.0);
	}

	void NeuralNetwork::learn(const std::vector<Matrix2D>& inputs, const std::vector<Matrix2D>& expectedOutputs, uint16_t epochs)
	{
		learn(inputs, expectedOutputs, epochs, 1, 1.0);
	}

	void NeuralNetwork::learn(std::string& inputsFilePath, std::string& expectedOutputsFilePath, uint16_t epochs)
	{
		learn(inputsFilePath, expectedOutputsFilePath, epochs, 1, 1.0);
	}

	void NeuralNetwork::learn(const std::vector<Matrix2D>& inputs, const std::vector<Matrix2D>& expectedOutputs)
	{
		learn(inputs, expectedOutputs, 1, 1, 1.0);
	}

	void NeuralNetwork::learn(std::string& inputsFilePath, std::string& expectedOutputsFilePath)
	{
		learn(inputsFilePath, expectedOutputsFilePath, 1, 1, 1.0);
	}

	void NeuralNetwork::getGradients(const Matrix2D& input, const Matrix2D& expectedOutput, std::vector<Matrix2D>& weightGradients, std::vector<Matrix2D>& biasGradients)
	{
		// Forward propagate, storing the z-values
		std::vector<Matrix2D> zValues = std::vector<Matrix2D>(this->weights.size() + 1);
		zValues[0] = input;

		Matrix2D output = input;

		for (uint16_t layer = 0; layer < this->weights.size(); layer += 1)
		{
			output = weights[layer] * output + biases[layer];
			zValues[layer + 1] = output;
			activate(output, this->activationType);
		}

		// Back propagate, getting the gradients.

		weightGradients = std::vector<Matrix2D>(this->weights.size());
		biasGradients = std::vector<Matrix2D>(this->biases.size());

		Matrix2D dCda_l = output - expectedOutput;
		Matrix2D delta_l = Matrix2D();

		for (int layer = this->weights.size() - 1; layer >= 0; layer -= 1)
		{
			/** Unactivated node values (z) at layer l. */
			Matrix2D z_l = zValues[layer + 1];
			
			/** Unactivated node values (z) at layer l-1. */
			Matrix2D z_lminusOne = zValues[layer];
			
			/** Activated node values (a) at layer l-1. */
			Matrix2D a_lminusOne = z_lminusOne;
			activate(a_lminusOne, this->activationType);
			
			/** The derivative of the activation function at layer l. */
			Matrix2D sigma_lprime = activateDerivative(z_l, this->activationType);

			/** Recalculates the cost at the current layer. */
			if (layer == this->weights.size() - 1) {
				delta_l = Matrix2D::hadamardMultiply(dCda_l, sigma_lprime);
			}
			else {
				delta_l = Matrix2D::hadamardMultiply(weights[layer + 1].getTranspose() * delta_l, sigma_lprime);
			}

			/**
			 * Adjusts the weight and bias gradients based on the error
			 * at the current layer.
			 */
			weightGradients[layer] = delta_l * a_lminusOne.getTranspose();
			biasGradients[layer] = delta_l;
		}

	}

	NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& a)
	{
		this->activationType = a.activationType;
		this->weights = a.weights;
		this->biases = a.biases;

		return *this;
	}

	void NeuralNetwork::writeToPath(std::string& folderPath)
	{
		_mkdir(folderPath.c_str());
		for (uint16_t layer = 0; layer < this->weights.size(); layer += 1)
		{
			Matrix3D layerWeights3D = Matrix3D(this->weights[layer].getRowCount(), this->weights[layer].getColumnCount(), 3);
			Matrix2D layerWeightsRescaled = this->weights[layer].getRescaled(-255.0, 255.0);

			for (uint16_t row = 0; row < layerWeights3D.getRowCount(); row += 1)
			{
				for (uint16_t col = 0; col < layerWeights3D.getColumnCount(); col += 1)
				{
					if (layerWeightsRescaled.get(row, col) >= 0)
					{
						layerWeights3D.set(row, col, 1, layerWeightsRescaled.get(row, col));
					}
					else
					{
						layerWeights3D.set(row, col, 0, -layerWeightsRescaled.get(row, col));
					}
				}
			}

			std::string weightFilePath = std::format("{}/{}.jpg", folderPath, layer);
			Image::write(weightFilePath, layerWeights3D);
		}
	}

	std::string NeuralNetwork::toString()
	{
		std::string out = "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Neural Network  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";

		if (this->weights.size() > 0) {
			out += std::format("\n\n\tLAYER 1 ({} nodes)\n", weights[0].getColumnCount());
		}

		for (int index = 0; index < this->weights.size(); index += 1) {
			out += std::format("\nWeights:{}\nBiases:{}", weights[index].toString(), biases[index].toString());
			out += std::format("\n\tLAYER {} ({} nodes)\n", index + 2, weights[index].getRowCount());
		}
		out += "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
		return out;
	}

	std::ostream& operator<<(std::ostream& os, const NeuralNetwork& a)
	{
		std::string out = "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Neural Network  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";

		if (a.weights.size() > 0) {
			out += std::format("\n\n\tLAYER 1 ({} nodes)\n", a.weights[0].getColumnCount());
		}

		for (int index = 0; index < a.weights.size(); index += 1) {
			out += std::format("\nWeights:{}\nBiases:{}", Matrix2D(a.weights[index]).toString(), Matrix2D(a.biases[index]).toString());
			out += std::format("\n\tLAYER {} ({} nodes)\n", index + 2, a.weights[index].getRowCount());
		}
		out += "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

		os << out;
		return os;
	}
}