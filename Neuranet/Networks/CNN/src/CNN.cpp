#pragma once

#include <format>
#include "../headers/CNN.hpp"

namespace Neuranet
{
	CNN::CNN(uint16_t numOutputs, Activation finalActivationType)
	{
		this->convolutions = std::vector<Convolution*>(0);
		this->nodeCounts = { numOutputs };
		this->fcls = { FCL(Matrix2D(0, 0), Matrix2D(numOutputs, 1), finalActivationType) };

		Matrix2D::initializeOpenCL();
	}

	CNN::~CNN()
	{
		//delete Matrix2D::cl_context;
		//delete Matrix2D::cl_device;
		//delete Matrix2D::cl_platform;
		//delete Matrix2D::cl_program;
		//delete Matrix2D::cl_plus1Kernel;
	}

	void CNN::addConvolution(Convolution* conv)
	{
		this->convolutions.push_back(conv);
	}

	void CNN::addHiddenLayer(uint16_t nodeCount, Activation activationType)
	{
		this->nodeCounts.push_back(this->nodeCounts[this->nodeCounts.size() - 1]);
		this->nodeCounts[this->nodeCounts.size() - 2] = nodeCount;

		// Reconstructs weights.

		this->fcls.push_back(FCL(this->fcls[this->fcls.size() - 1]));

		auto [lowerBound, upperBound] = getWeightBounds(this->nodeCounts[this->fcls.size() - 2], this->nodeCounts[this->fcls.size() - 1], this->fcls[this->fcls.size() - 1].activationType);
		this->fcls[this->fcls.size() - 1].weights = Matrix2D::random(this->nodeCounts[this->fcls.size() - 1], this->nodeCounts[this->fcls.size() - 2], lowerBound, upperBound);
		
		if (this->fcls.size() > 2)
		{
			auto [lowerBound2, upperBound2] = getWeightBounds(this->nodeCounts[this->fcls.size() - 3], this->nodeCounts[this->fcls.size() - 2], activationType);
			this->fcls[this->fcls.size() - 2].weights = Matrix2D::random(this->nodeCounts[this->fcls.size() - 2], this->nodeCounts[this->fcls.size() - 3], lowerBound2, upperBound2);
		}
		
		this->fcls[this->fcls.size() - 2].biases = Matrix2D(this->nodeCounts[this->fcls.size() - 2], 1);
		this->fcls[this->fcls.size() - 2].activationType = activationType;
	}

	double CNN::getAverageLoss(const std::vector<Matrix3D>& inputs, const std::vector<Matrix2D>& expectedOutputs)
	{
		std::cout << "Calculating average loss..." << std::endl;

		if (inputs.size() == 0 || expectedOutputs.size() == 0)
		{
			return 0.0;
		}
		
		double totalLoss = 0.0;
		
		Loss lossType;
		switch (this->fcls[this->fcls.size() - 1].activationType)
		{
		case Activation::SOFTMAX:
			lossType = Loss::NEGATIVE_LOG;
			break;
		case Activation::SIGMOID:
			lossType = Loss::EUCLIDEAN;
			break;
		default:
			lossType = Loss::EUCLIDEAN;
			break;
		}
		
		for (int index = 0; index < inputs.size(); index += 1)
		{
			totalLoss += loss(expectedOutputs[index], compute(inputs[index]), lossType);
			if ((index + 1) % 50 == 0)
			{
				std::cout << std::format("\r\tComputing: ({}/{})  {}% ", index + 1, inputs.size(), (int)(10000.0 * (1.0 + index) / inputs.size()) / 100.0);
			}
		}

		std::cout << "\n\n";
		
		return totalLoss / inputs.size();
	}

	Matrix2D CNN::compute(const Matrix3D& input)
	{
		Matrix3D current = input;

		for (uint16_t conv = 0; conv < this->convolutions.size(); conv += 1)
		{
			switch (this->convolutions[conv]->convType)
			{
			case ConvolutionType::CONVOLUTIONAL_LAYER:
				{
					ConvolutionalLayer* convLay = dynamic_cast<ConvolutionalLayer*>(this->convolutions[conv]);
					uint8_t padding = convLay->padding;
					uint8_t filterRowCount = convLay->filterWeights[0].getRowCount();
					uint8_t filterColCount = convLay->filterWeights[0].getColumnCount();
					uint8_t filterRowStride = std::get<0>(convLay->filterStride);
					uint8_t filterColStride = std::get<1>(convLay->filterStride);

					Matrix3D filtered((current.getRowCount() - filterRowCount + 2 * padding) / filterRowStride + 1, (current.getColumnCount() - filterColCount + 2 * padding) / filterColStride + 1, convLay->filterWeights.size());

					for (uint16_t filter = 0; filter < filtered.getLayerCount(); filter += 1)
					{
						// Applying filters.

						for (int row = -padding; row < current.getRowCount() + padding - filterRowCount + 1; row += filterRowStride)
						{
							for (int col = -padding; col < current.getColumnCount() + padding - filterColCount + 1; col += filterColStride)
							{
								double dotProduct = Matrix3D::hadamardProduct(current.getSubmatrix(row, col, 0, row + filterRowCount, col + filterColCount, current.getLayerCount()), convLay->filterWeights[filter]).getSumOfEntries();
								filtered.set((row + padding) / filterRowStride, (col + padding) / filterColStride, filter, dotProduct + convLay->filterBiases[filter]);
							}
						}
					}

					activate(filtered, convLay->activationType);

					if (convLay->activationType == Activation::RELU_NORMALIZED)
					{
						filtered /= filtered.getValues()[filtered.getIndexOfMax()];
					}

					current = filtered;
					break;
				}
			case ConvolutionType::POOLING_LAYER:
				{
					PoolingLayer* poolLay = dynamic_cast<PoolingLayer*>(this->convolutions[conv]);
					uint8_t poolingRowCount = std::get<0>(poolLay->poolSize);
					uint8_t poolingColCount = std::get<1>(poolLay->poolSize);
					uint8_t poolingRowStride = std::get<0>(poolLay->poolingStride);
					uint8_t poolingColStride = std::get<1>(poolLay->poolingStride);

					Matrix3D pooled((current.getRowCount() - poolingRowCount) / poolingRowStride + 1, (current.getColumnCount() - poolingColCount) / poolingColStride + 1, current.getLayerCount());

					for (uint16_t filter = 0; filter < current.getLayerCount(); filter += 1)
					{
						// Applying pooling.

						for (uint16_t row = 0; row < current.getRowCount() - poolingRowCount + 1; row += poolingRowStride)
						{
							for (uint16_t col = 0; col < current.getColumnCount() - poolingColCount + 1; col += poolingColStride)
							{
								Matrix3D subMatrix = current.getSubmatrix(row, col, filter, row + poolingRowCount, col + poolingColCount, filter + 1);

								switch (poolLay->poolingType)
								{
								case Pooling::MAX:
									pooled.set(row / poolingRowStride, col / poolingColStride, filter, subMatrix.getValues()[subMatrix.getIndexOfMax()]);
									break;
								case Pooling::AVERAGE:
									pooled.set(row / poolingRowStride, col / poolingColStride, filter, subMatrix.getSumOfEntries() / (subMatrix.getRowCount() * subMatrix.getColumnCount()));
									break;
								default:
									pooled.set(row / poolingRowStride, col / poolingColStride, filter, subMatrix.getValues()[subMatrix.getIndexOfMax()]);
									break;
								}
							}
						}
					}

					current = pooled;
					break;
				}
			}
		}

		// Computes fully-connected layers.
		
		Matrix2D flattened = current.getVectorized();

		if (this->fcls[0].weights.getRowCount() == 0 || this->fcls[0].weights.getColumnCount() == 0)
		{
			const auto [lowerBound, upperBound] = getWeightBounds(flattened.getRowCount(), this->nodeCounts[0], this->fcls[0].activationType);

			this->fcls[0].weights = Matrix2D::random(this->nodeCounts[0], flattened.getRowCount(), lowerBound, upperBound);
			this->fcls[0].biases = Matrix2D(this->nodeCounts[0], 1);
		}

		for (uint8_t layer = 0; layer < this->fcls.size(); layer += 1)
		{
			flattened = this->fcls[layer].weights * flattened + this->fcls[layer].biases;
			activate(flattened, this->fcls[layer].activationType);
		}

		return flattened;
	}

	void CNN::learn(const std::vector<Matrix3D>& inputs, const std::vector<Matrix2D>& expectedOutputs, uint16_t epochs, uint16_t batchSize, double learningRate)
	{
		std::cout << "\nLearning...\n" << std::endl;
		for (uint16_t epoch = 0; epoch < epochs; epoch += 1)
		{
			std::cout << std::format("Epoch {} of {}:\n", (epoch + 1), epochs) << std::endl;
			for (int batchStartIndex = 0; batchStartIndex < inputs.size(); batchStartIndex += batchSize)
			{
				// Sums the gradients of the weights and biases for all datasets.

				uint16_t thisBatchSize = std::min((uint16_t)(inputs.size() - batchStartIndex), batchSize);

				std::vector<std::vector<Matrix3D>> totalWeightGradients = std::vector<std::vector<Matrix3D>>(this->convolutions.size() + this->fcls.size());
				std::vector<std::vector<Matrix2D>> totalBiasGradients = std::vector<std::vector<Matrix2D>>(this->convolutions.size() + this->fcls.size());

				for (int batchIndex = 0; batchIndex < thisBatchSize; batchIndex += 1)
				{
					std::vector<std::vector<Matrix3D>> weightGradients;
					std::vector<std::vector<Matrix2D>> biasGradients;
					getGradients(inputs[batchStartIndex + batchIndex], expectedOutputs[batchStartIndex + batchIndex], weightGradients, biasGradients);
					
					for (uint16_t gradientIndex = 0; gradientIndex < this->convolutions.size() + this->fcls.size(); gradientIndex += 1)
					{
						for (uint16_t gradientLayer = 0; gradientLayer < weightGradients[gradientIndex].size(); gradientLayer += 1)
						{
							if (batchIndex == 0 && gradientLayer == 0)
							{
								totalWeightGradients[gradientIndex] = std::vector<Matrix3D>(weightGradients[gradientIndex].size());
								totalBiasGradients[gradientIndex] = std::vector<Matrix2D>(biasGradients[gradientIndex].size());
							}

							if (batchIndex == 0)
							{
								totalWeightGradients[gradientIndex][gradientLayer] = weightGradients[gradientIndex][gradientLayer];
								totalBiasGradients[gradientIndex][gradientLayer] = biasGradients[gradientIndex][gradientLayer];
							}
							else
							{
								totalWeightGradients[gradientIndex][gradientLayer] += weightGradients[gradientIndex][gradientLayer];
								totalBiasGradients[gradientIndex][gradientLayer] += biasGradients[gradientIndex][gradientLayer];
							}
						}
					}
				}

				// Modifies the weights and biases by the average gradient of the batch.

				for (uint16_t gradientIndex = 0; gradientIndex < totalWeightGradients.size(); gradientIndex += 1)
				{
					for (uint16_t gradientLayer = 0; gradientLayer < totalWeightGradients[gradientIndex].size(); gradientLayer += 1)
					{
						if (gradientIndex < this->convolutions.size())
						{
							switch (this->convolutions[gradientIndex]->convType)
							{
							case ConvolutionType::CONVOLUTIONAL_LAYER:
							{
								ConvolutionalLayer* convLay = dynamic_cast<ConvolutionalLayer*>(this->convolutions[gradientIndex]);
								convLay->filterWeights[gradientLayer] -= totalWeightGradients[gradientIndex][gradientLayer] / batchSize * learningRate;
								convLay->filterBiases[gradientLayer] -= totalBiasGradients[gradientIndex][gradientLayer].get(0,0) / batchSize * learningRate;
								break;
							}
							case ConvolutionType::POOLING_LAYER:
							{
								break;
							}
							}
						}
						else
						{
							Matrix3D weight3D = totalWeightGradients[gradientIndex][gradientLayer];
							Matrix2D weight2D = Matrix2D(weight3D.getRowCount(), weight3D.getColumnCount(), weight3D.getValues());
							this->fcls[gradientIndex - this->convolutions.size()].weights -= weight2D / batchSize * learningRate;
							this->fcls[gradientIndex - this->convolutions.size()].biases -= totalBiasGradients[gradientIndex][gradientLayer] / batchSize * learningRate;
						}
						
					}
				}
				
				std::cout << std::format("\r\tBatch {} of {}  {}%", (batchStartIndex / batchSize + 1), (inputs.size() / batchSize), (int)(10000.0 * (batchStartIndex / batchSize + 1) / (100.0 * inputs.size() / batchSize)));
			}

			std::cout << "\n" << std::endl;
		}
	}

	void CNN::getGradients(const Matrix3D& input, const Matrix2D& expectedOutput, std::vector<std::vector<Matrix3D>>& filterWeightGradients, std::vector<std::vector<Matrix2D>>& filterBiasGradients)
	{
		std::vector<Matrix3D> z3d(this->convolutions.size() + 1);
		std::vector<Matrix2D> z2d(this->fcls.size() + 1);

		Matrix3D current = input;
		z3d[0] = current;

		for (uint16_t conv = 0; conv < this->convolutions.size(); conv += 1)
		{
			switch (this->convolutions[conv]->convType)
			{
			case ConvolutionType::CONVOLUTIONAL_LAYER:
			{
				ConvolutionalLayer* convLay = dynamic_cast<ConvolutionalLayer*>(this->convolutions[conv]);
				uint8_t padding = convLay->padding;
				uint8_t filterRowCount = convLay->filterWeights[0].getRowCount();
				uint8_t filterColCount = convLay->filterWeights[0].getColumnCount();
				uint8_t filterRowStride = std::get<0>(convLay->filterStride);
				uint8_t filterColStride = std::get<1>(convLay->filterStride);

				Matrix3D filtered((current.getRowCount() - filterRowCount + 2 * padding) / filterRowStride + 1, (current.getColumnCount() - filterColCount + 2 * padding) / filterColStride + 1, convLay->filterWeights.size());

				for (uint16_t filter = 0; filter < filtered.getLayerCount(); filter += 1)
				{
					// Applying filters.

					for (int row = -padding; row < current.getRowCount() + padding - filterRowCount + 1; row += filterRowStride)
					{
						for (int col = -padding; col < current.getColumnCount() + padding - filterColCount + 1; col += filterColStride)
						{
							double dotProduct = Matrix3D::hadamardProduct(current.getSubmatrix(row, col, 0, row + filterRowCount, col + filterColCount, current.getLayerCount()), convLay->filterWeights[filter]).getSumOfEntries();
							filtered.set((row + padding) / filterRowStride, (col + padding) / filterColStride, filter, dotProduct + convLay->filterBiases[filter]);
						}
					}
				}

				z3d[conv + 1] = filtered;

				activate(filtered, convLay->activationType);

				if (convLay->activationType == Activation::RELU_NORMALIZED)
				{
					filtered /= filtered.getValues()[filtered.getIndexOfMax()];
				}

				current = filtered;
				break;
			}
			case ConvolutionType::POOLING_LAYER:
			{
				PoolingLayer* poolLay = dynamic_cast<PoolingLayer*>(this->convolutions[conv]);
				uint8_t poolingRowCount = std::get<0>(poolLay->poolSize);
				uint8_t poolingColCount = std::get<1>(poolLay->poolSize);
				uint8_t poolingRowStride = std::get<0>(poolLay->poolingStride);
				uint8_t poolingColStride = std::get<1>(poolLay->poolingStride);

				Matrix3D pooled((current.getRowCount() - poolingRowCount) / poolingRowStride + 1, (current.getColumnCount() - poolingColCount) / poolingColStride + 1, current.getLayerCount());

				for (uint16_t filter = 0; filter < current.getLayerCount(); filter += 1)
				{
					// Applying pooling.

					for (uint16_t row = 0; row < current.getRowCount() - poolingRowCount + 1; row += poolingRowStride)
					{
						for (uint16_t col = 0; col < current.getColumnCount() - poolingColCount + 1; col += poolingColStride)
						{
							Matrix3D subMatrix = current.getSubmatrix(row, col, filter, row + poolingRowCount, col + poolingColCount, filter + 1);

							switch (poolLay->poolingType)
							{
							case Pooling::MAX:
								pooled.set(row / poolingRowStride, col / poolingColStride, filter, subMatrix.getValues()[subMatrix.getIndexOfMax()]);
								break;
							case Pooling::AVERAGE:
								pooled.set(row / poolingRowStride, col / poolingColStride, filter, subMatrix.getSumOfEntries() / (subMatrix.getRowCount() * subMatrix.getColumnCount()));
								break;
							default:
								pooled.set(row / poolingRowStride, col / poolingColStride, filter, subMatrix.getValues()[subMatrix.getIndexOfMax()]);
								break;
							}
						}
					}
				}

				z3d[conv + 1] = pooled;

				current = pooled;
				break;
			}
			}
		}

		// Computes fully-connected layers.

		Matrix2D flattened = current.getVectorized();

		if (this->fcls[0].weights.getRowCount() == 0 || this->fcls[0].weights.getColumnCount() == 0)
		{
			const auto [lowerBound, upperBound] = getWeightBounds(flattened.getRowCount(), this->nodeCounts[0], this->fcls[0].activationType);

			this->fcls[0].weights = Matrix2D::random(this->nodeCounts[0], flattened.getRowCount(), lowerBound, upperBound);
			this->fcls[0].biases = Matrix2D(this->nodeCounts[0], 1);
		}
		
		z2d[0] = flattened;

		for (uint8_t layer = 0; layer < this->fcls.size(); layer += 1)
		{
			flattened = this->fcls[layer].weights * flattened + this->fcls[layer].biases;
			z2d[layer + 1] = flattened;
			activate(flattened, this->fcls[layer].activationType);
		}

		// The z values from the convolution and fully-connected stages are now cached.
		// At this time we can begin backpropagation.

		uint16_t totalLayers = this->convolutions.size() + this->fcls.size();
		filterWeightGradients = std::vector<std::vector<Matrix3D>>(totalLayers);
		filterBiasGradients = std::vector<std::vector<Matrix2D>>(totalLayers);

		// Backpropagates the fully-connected stage.
		Matrix2D lastFCLdelta = Matrix2D();
		{
			Matrix2D dCda_l = flattened - expectedOutput;
			Matrix2D delta_l = Matrix2D();

			for (int layer = this->fcls.size() - 1; layer >= 0; layer -= 1)
			{
				/** Unactivated node values (z) at layer l. */
				Matrix2D z_l = z2d[layer + 1];

				/** Unactivated node values (z) at layer l-1. */
				Matrix2D z_lminusOne = z2d[layer];

				/** Activated node values (a) at layer l-1. */
				Matrix2D a_lminusOne = z_lminusOne;
				activate(a_lminusOne, this->fcls[layer].activationType);

				/** The derivative of the activation function at layer l. */
				Matrix2D sigma_lprime = activateDerivative(z_l, this->fcls[layer].activationType);

				/** Recalculates the cost at the current layer. */
				if (layer == this->fcls.size() - 1)
				{
					if (this->fcls[layer].activationType != Activation::SOFTMAX)
					{
						delta_l = Matrix2D::hadamardProduct(dCda_l, sigma_lprime);
					}
					else
					{
						//delta_l = Matrix2D::hadamardProduct(dCda_l, sigma_lprime);
						delta_l = dCda_l;
					}
					//std::cout << dCda_l << std::endl;
					//std::cout << sigma_lprime << std::endl;
					//std::cout << delta_l << std::endl;
				}
				else
				{
					delta_l = Matrix2D::hadamardProduct(this->fcls[layer + 1].weights.getTranspose() * delta_l, sigma_lprime);
				}

				/**
				 * Adjusts the weight and bias gradients based on the error
				 * at the current layer.
				 */
				Matrix2D prod = delta_l * a_lminusOne.getTranspose();
				filterWeightGradients[layer + this->convolutions.size()] = { Matrix3D(prod.getRowCount(), prod.getColumnCount(), 1, prod.getValues()) };
				filterBiasGradients[layer + this->convolutions.size()] = { delta_l };

				if (layer == 0)
				{
					lastFCLdelta = this->fcls[0].weights.getTranspose() * delta_l;
				}
			}
		}

		// Backpropagates the convolutional stage.
		{
			// Error at layer L
			//std::cout << "last fcl delta " << lastFCLdelta << std::endl;
			Matrix3D delta_l = Matrix3D(z3d[z3d.size() - 1].getRowCount(), z3d[z3d.size() - 1].getColumnCount(), z3d[z3d.size() - 1].getLayerCount(), lastFCLdelta.getValues());

			for (int conv = this->convolutions.size() - 1; conv >= 0; conv -= 1)
			{
				switch (this->convolutions[conv]->convType)
				{
				case ConvolutionType::CONVOLUTIONAL_LAYER:
				{
					ConvolutionalLayer* convLay_l = dynamic_cast<ConvolutionalLayer*>(this->convolutions[conv]);

					/** Unactivated node values (z) at layer l. */
					Matrix3D z_l = z3d[conv + 1];

					/** Unactivated node values (z) at layer l-1. */
					Matrix3D z_lminusOne = z3d[conv];

					/** Activated node values (a) at layer l-1. */
					Matrix3D a_lminusOne = z_lminusOne;
					activate(a_lminusOne, convLay_l->activationType);

					/** The derivative of the activation function at layer l. */
					Matrix3D sigma_lprime = activateDerivative(z_l, convLay_l->activationType);

					/** Recalculates the cost at the current layer. */
					if (conv == this->convolutions.size() - 1)
					{
						delta_l = Matrix3D::hadamardProduct(delta_l, sigma_lprime);

						//std::cout << "delta1: " << delta_l << std::endl;
					}
					else
					{
						/** Gets the delta/error at the previous layer. */
						ConvolutionalLayer* convLay_lplusOne = dynamic_cast<ConvolutionalLayer*>(this->convolutions[conv + 1]);
						uint8_t filterRowStride = std::get<0>(convLay_lplusOne->filterStride);
						uint8_t filterColStride = std::get<1>(convLay_lplusOne->filterStride);
						int lMinusOneRowCount = z_l.getRowCount();
						int lMinusOneColCount = z_l.getColumnCount();
						int lMinusOneLayCount = z_l.getLayerCount();
						Matrix3D delta_lminusOne(lMinusOneRowCount, lMinusOneColCount, lMinusOneLayCount);

						for (uint16_t lay = 0; lay < delta_l.getLayerCount(); lay += 1)
						{
							for (uint16_t row = 0; row < delta_l.getRowCount(); row += 1)
							{
								for (uint16_t col = 0; col < delta_l.getColumnCount(); col += 1)
								{
									delta_lminusOne.addSubmatrix(convLay_lplusOne->filterWeights[lay] * delta_l.get(row, col, lay), row * filterRowStride - convLay_lplusOne->padding, col * filterColStride - convLay_lplusOne->padding, 0);
								}
							}
						}

						//std::cout << "delta: " << delta_lminusOne << std::endl;

						delta_l = Matrix3D::hadamardProduct(delta_lminusOne, sigma_lprime);
					}

					/**
					 * Adjusts the weight and bias gradients based on the error
					 * at the current layer.
					 */
					filterWeightGradients[conv] = std::vector<Matrix3D>(convLay_l->filterWeights.size());
					filterBiasGradients[conv] = std::vector<Matrix2D>(convLay_l->filterBiases.size());

					for (uint16_t lay = 0; lay < delta_l.getLayerCount(); lay += 1)
					{
						double* bias = new double(delta_l.getSumOfEntries());
						filterBiasGradients[conv][lay] = Matrix2D(1, 1, bias);
						delete bias;

						Matrix3D totalWeightedError = convLay_l->filterWeights[lay];
						totalWeightedError.zero();

						for (uint16_t row = 0; row < delta_l.getRowCount(); row += 1)
						{
							for (uint16_t col = 0; col < delta_l.getColumnCount(); col += 1)
							{
								uint8_t filterRowStride = std::get<0>(convLay_l->filterStride);
								uint8_t filterColStride = std::get<1>(convLay_l->filterStride);
								int r = row * filterRowStride - convLay_l->padding;
								int c = col * filterColStride - convLay_l->padding;
								totalWeightedError += a_lminusOne.getSubmatrix(r, c, 0, r + totalWeightedError.getRowCount(), c + totalWeightedError.getColumnCount(), totalWeightedError.getLayerCount()) * delta_l.get(row, col, lay);
								//totalWeightedError += convLay_l->filterWeights[lay] * delta_l.get(row, col, lay);
							}
						}

						//totalWeightedError /= (delta_l.getRowCount() * delta_l.getColumnCount());
						filterWeightGradients[conv][lay] = totalWeightedError;

						//std::cout << "WEIGHTddddS: " << totalWeightedError << std::endl;
						//std::cout << "BIASEddddS: " << filterBiasGradients[conv][lay] << std::endl;

					}
					
					break;
				}
				case ConvolutionType::POOLING_LAYER:
				{
					PoolingLayer* poolLay = dynamic_cast<PoolingLayer*>(this->convolutions[conv]);

					break;
				}
				}
			}
		}
	}

	std::ostream& operator<<(std::ostream& os, const CNN& a)
	{
		std::string out = "\n~~~~~~~~~~~~~~~~~~~~~~~~~  Convolutional Neural Network  ~~~~~~~~~~~~~~~~~~~~~~~~~~";

		for (uint16_t convLayer = 0; convLayer < a.convolutions.size(); convLayer += 1)
		{
			std::string convType;
			
			switch (a.convolutions[convLayer]->convType)
			{
			case ConvolutionType::CONVOLUTIONAL_LAYER:
				convType = "CONVOLUTIONAL LAYER";
				break;
			case ConvolutionType::POOLING_LAYER:
				convType = "POOLING LAYER";
				break;
			case ConvolutionType::NO_TYPE:
				convType = "ERROR_NO_CONVOLUTION_TYPE";
				break;
			}
			
			out += std::format("\n\n\tLAYER {} ({})\n", convLayer + 1, convType);

			if (a.convolutions[convLayer]->convType == ConvolutionType::CONVOLUTIONAL_LAYER)
			{
				out += "\n\tFILTERS:\n";

				ConvolutionalLayer* convLay = dynamic_cast<ConvolutionalLayer*>(a.convolutions[convLayer]);
				
				for (uint16_t layer = 0; layer < convLay->filterWeights.size(); layer += 1)
				{
					out += "\nWEIGHT " + std::to_string(layer + 1) + ":\n" + convLay->filterWeights[layer].toString();
					out += "\nBIAS " + std::to_string(layer + 1) + ": " + std::to_string(convLay->filterBiases[layer]);
				}
			}
			else
			{
				out += "\n\[n/a]";

				if (convLayer + 1 == a.convolutions.size())
				{
					out += "\n";
				}
			}
		}

		out += "\n\tFLATTENED - FULLY-CONNECTED LAYER 1\n";

		for (int fclLayer = 0; fclLayer < a.fcls.size(); fclLayer += 1)
		{
			out += std::format("\nWEIGHTS ({}):{}\nBIASES:{}", a.fcls[fclLayer].weights.getDimensions(), Matrix2D(a.fcls[fclLayer].weights).toString(), Matrix2D(a.fcls[fclLayer].biases).toString());

			if (fclLayer != a.fcls.size() - 1)
			{
				out += std::format("\n\tFULLY-CONNECTED LAYER {} ({} nodes)\n", fclLayer + 3, a.nodeCounts[fclLayer]);
			}
			else
			{
				out += std::format("\n\tOUTPUT LAYER ({} nodes)\n", a.nodeCounts[fclLayer]);
			}
		}
		out += "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

		os << out;
		return os;
	}
}