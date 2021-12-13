#pragma once

#include "../headers/Network.hpp"
#include <float.h>

namespace Neuranet
{
	double Network::loss(const Matrix2D& expectedOutput, const Matrix2D& output)
	{
		return Matrix2D::power(Matrix2D(expectedOutput) - output, 2.0).getSumOfEntries() / (output.getRowCount() * output.getColumnCount());
	}

	double Network::sigmoid(double input)
	{
		return 1 / (1 + exp(-input));
	}

	double Network::reLU(double input)
	{
		return std::max(0.0, input);
	}

	void Network::activate(Matrix2D& input, Activation activationType)
	{
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