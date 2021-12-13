#pragma once

#include <stdint.h>
#include "../headers/Image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../../External/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../External/stb/stb_image_write.h"

namespace Neuranet
{
	void Image::write(std::string& fileName, const Matrix3D& pixels)
	{
		int w = pixels.getColumnCount();
		int h = pixels.getRowCount();

		uint8_t* rgb_image;
		rgb_image = (uint8_t*)malloc(w * h * 3);

		for (uint16_t row = 0; row < h; row += 1)
		{
			for (uint16_t col = 0; col < w; col += 1)
			{
				rgb_image[(row * w + col) * 3] = (uint8_t) pixels.get(row, col, 0);
				rgb_image[(row * w + col) * 3 + 1] = (uint8_t) pixels.get(row, col, 1);
				rgb_image[(row * w + col) * 3 + 2] = (uint8_t) pixels.get(row, col, 2);
			}
		}

		stbi_write_png(fileName.c_str(), w, h, 3, rgb_image, w * 3);
	}

	void Image::write(std::string& fileName, const Matrix2D& pixels)
	{
		int w = pixels.getColumnCount();
		int h = pixels.getRowCount();

		uint8_t* grayscale_image;
		grayscale_image = (uint8_t*)malloc(w * h);

		for (uint16_t row = 0; row < h; row += 1)
		{
			for (uint16_t col = 0; col < w; col += 1)
			{
				grayscale_image[row * w + col] = (uint8_t)pixels.get(row, col);
			}
		}

		stbi_write_png(fileName.c_str(), w, h, 1, grayscale_image, w);
	}

	void Image::read(std::string& fileName, Matrix3D& pixels)
	{
		int w, h, comp;

		uint8_t* rgb_image = stbi_load(fileName.c_str(), &w, &h, &comp, 3);

		pixels = Matrix3D(h, w, 3);

		for (uint16_t row = 0; row < h; row += 1)
		{
			for (uint16_t col = 0; col < w; col += 1)
			{
				pixels.set(row, col, 0, rgb_image[(row * w + col) * 3]);
				pixels.set(row, col, 1, rgb_image[(row * w + col) * 3 + 1]);
				pixels.set(row, col, 2, rgb_image[(row * w + col) * 3 + 2]);
			}
		}

		stbi_image_free(rgb_image);
	}

	void Image::read(std::string& fileName, Matrix2D& pixels)
	{
		int w, h, comp;

		uint8_t* grayscale_image = stbi_load(fileName.c_str(), &w, &h, &comp, 1);

		pixels = Matrix2D(h, w);

		for (uint16_t row = 0; row < h; row += 1)
		{
			for (uint16_t col = 0; col < w; col += 1)
			{
				pixels.set(row, col, grayscale_image[row * w + col]);
			}
		}

		stbi_image_free(grayscale_image);
	}
}