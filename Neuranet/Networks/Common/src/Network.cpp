#pragma once

#include "../headers/Network.hpp"
#include <float.h>

namespace Neuranet
{
	double Network::loss(const Matrix2D& expectedOutput, const Matrix2D& output, Loss lossType)
	{
		switch (lossType)
		{
		case Loss::EUCLIDEAN:
			return Matrix2D::power(Matrix2D(expectedOutput) - output, 2.0).getSumOfEntries() / (output.getRowCount() * output.getColumnCount());
		case Loss::NEGATIVE_LOG:
			return -Matrix2D::hadamardProduct(Matrix2D::logarithmic(output), expectedOutput).getSumOfEntries();
		default:
			return 0;
		}
	}

	std::tuple<double, double> Network::getWeightBounds(uint16_t numInputs, uint16_t numOutputs, Activation activationType)
	{
		std::tuple<double, double> out;

		switch (activationType)
		{
		case Activation::SIGMOID:
			// Normalized Xavier Weight Initialization
			std::get<0>(out) = -sqrt(6) / sqrt(numInputs + numOutputs);
			std::get<1>(out) = sqrt(6) / sqrt(numInputs + numOutputs);
			break;
		case Activation::RELU:
			// He Weight Initialization
			std::get<0>(out) = 0.001;
			std::get<1>(out) = sqrt(2 / numInputs);
			break;
		case Activation::RELU_NORMALIZED:
			// He Weight Initialization
			std::get<0>(out) = 0.001;
			std::get<1>(out) = sqrt(2 / numInputs);
			break;
		default:
			std::get<0>(out) = -1.0;
			std::get<1>(out) = 1.0;
		}

		return out;
	}

	double Network::sigmoid(double input)
	{
		return 1 / (1 + exp(-input));
	}

	double Network::reLU(double input)
	{
		return std::max(0.0, input);
	}

	Matrix2D Network::softmax(Matrix2D& input)
	{
		double max = input.getValues()[input.getIndexOfMax()];
		Matrix2D raised = Matrix2D::exponential(input - Matrix2D::random(input.getRowCount(), input.getColumnCount(), max, max));
		return (raised / raised.getSumOfEntries());
	}

	Matrix3D Network::softmax(Matrix3D& input)
	{
		double max = input.getValues()[input.getIndexOfMax()];
		Matrix3D raised = Matrix3D::exponential(input - Matrix3D::random(input.getRowCount(), input.getColumnCount(), input.getLayerCount(), max, max));
		return (raised / raised.getSumOfEntries());
	}

	void Network::activate(Matrix2D& input, Activation activationType)
	{
		if (activationType == Activation::SOFTMAX)
		{
			input = softmax(input);
			return;
		}

		double maxValue = DBL_MIN;

		for (uint16_t row = 0; row < input.getRowCount(); row += 1)
		{
			for (uint16_t col = 0; col < input.getColumnCount(); col += 1)
			{
				switch (activationType)
				{
				case Activation::SIGMOID:
					input.set(row, col, sigmoid(input.get(row, col)));
					break;
				case Activation::RELU:
					input.set(row, col, reLU(input.get(row, col)));
					break;
				case Activation::RELU_NORMALIZED:
					input.set(row, col, reLU(input.get(row, col)));
					if (input.get(row, col) > maxValue)
					{
						maxValue = input.get(row, col);
					}
					break;
				}
			}
		}

		if (activationType == Activation::RELU_NORMALIZED && maxValue > 0.0)
		{
			input /= maxValue;
		}
	}

	void Network::activate(Matrix3D& input, Activation activationType)
	{
		if (activationType == Activation::SOFTMAX)
		{
			input = softmax(input);
			return;
		}

		double maxValue = DBL_MIN;

		for (uint16_t row = 0; row < input.getRowCount(); row += 1)
		{
			for (uint16_t col = 0; col < input.getColumnCount(); col += 1)
			{
				for (uint16_t lay = 0; lay < input.getLayerCount(); lay += 1)
				{
					switch (activationType)
					{
					case Activation::SIGMOID:
						input.set(row, col, lay, sigmoid(input.get(row, col, lay)));
						break;
					case Activation::RELU:
						input.set(row, col, lay, reLU(input.get(row, col, lay)));
						break;
					case Activation::RELU_NORMALIZED:
						input.set(row, col, lay, reLU(input.get(row, col, lay)));
						if (input.get(row, col, lay) > maxValue)
						{
							maxValue = input.get(row, col, lay);
						}
						break;
					}
				}
			}
		}

		if (activationType == Activation::RELU_NORMALIZED && maxValue > 0.0)
		{
			input /= maxValue;
		}
	}

	Matrix2D Network::activateDerivative(Matrix2D& input, Activation activationType)
	{
		if (activationType == Activation::SOFTMAX)
		{
			Matrix2D s = softmax(input);
			double sum = s.getSumOfEntries();
			Matrix2D deriv = s - (input * sum);
			return deriv;
		}

		Matrix2D derivative(input.getRowCount(), input.getColumnCount());

		for (uint16_t row = 0; row < input.getRowCount(); row += 1)
		{
			for (uint16_t col = 0; col < input.getColumnCount(); col += 1)
			{
				switch (activationType)
				{
				case Activation::SIGMOID:
					derivative.set(row, col, sigmoid(input.get(row, col)) * (1 - sigmoid(input.get(row, col))));
					break;
				case Activation::RELU:
					derivative.set(row, col, input.get(row, col) > 0.0 ? 1.0 : 0.0);
					break;
				case Activation::RELU_NORMALIZED:
					derivative.set(row, col, input.get(row, col) > 0.0 ? 1.0 : 0.0);
					break;
				default:
					derivative.set(row, col, 0.0);
				}
			}
		}

		return derivative;
	}

	Matrix3D Network::activateDerivative(Matrix3D& input, Activation activationType)
	{
		Matrix3D derivative(input.getRowCount(), input.getColumnCount(), input.getLayerCount());

		for (uint16_t row = 0; row < input.getRowCount(); row += 1)
		{
			for (uint16_t col = 0; col < input.getColumnCount(); col += 1)
			{
				for (uint16_t lay = 0; lay < input.getLayerCount(); lay += 1)
				{
					switch (activationType)
					{
					case Activation::SIGMOID:
						derivative.set(row, col, lay, sigmoid(input.get(row, col, lay)) * (1 - sigmoid(input.get(row, col, lay))));
						break;
					case Activation::RELU:
						derivative.set(row, col, lay, input.get(row, col, lay) > 0.0 ? 1.0 : 0.0);
						break;
					case Activation::RELU_NORMALIZED:
						derivative.set(row, col, lay, input.get(row, col, lay) > 0.0 ? 1.0 : 0.0);
						break;
					default:
						derivative.set(row, col, lay, 0.0);
					}
				}
			}
		}

		return derivative;
	}
}