#pragma once

#include <iostream>
#include <string>

namespace Neuranet
{
    /**
     * @brief Abstract class representing a matrix
     *        of any size.
     * @author Nolan Bridges
     * @version 1.0.0
     */
    class Matrix
    {
    protected:
        /**
         * @brief The number of rows in the matrix.
         */
        uint16_t rowCount;

        /**
         * @brief The number of columns in the matrix.
         */
        uint16_t colCount;
        
        /**
         * @brief The values of the matrix as a 1D array. Values 'snake' along
         *        the matrix as one would read a book, starting from the top left.
         */
        double* values;

    public:
        /**
         * @brief Get the dimensions of the matrix.
         * 
         * @return the dimensions of the matrix, as a string.
         */
        virtual constexpr std::string getDimensions() const noexcept = 0;

        /**
         * @brief The number of dimensions of the matrix, as an int.
         */
        uint16_t dimensionCount;
    };
}