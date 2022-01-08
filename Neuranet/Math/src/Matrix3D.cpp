#pragma once

#include <ctime>
#include <chrono>
#include <string>
#include <format>
#include "../headers/Matrix3D.hpp"
#include <Windows.h>

namespace Neuranet
{
    int Matrix3D::randomSeed;

    Matrix3D::Matrix3D(uint16_t rows, uint16_t columns, uint16_t layers, double* values)
    {
        this->dimensionCount = 3;

        this->rowCount = rows;
        this->colCount = columns;
        this->layCount = layers;

        this->values = new double[rows * columns * layers];

        if (values != nullptr)
        {
            for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
            {
                this->values[index] = values[index];
            }
        }
        else
        {
            for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
            {
                this->values[index] = 0.0;
            }
        }
    }

    Matrix3D::~Matrix3D()
    {
        delete[] this->values;
    }

    void Matrix3D::zero()
    {
        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            this->values[index] = 0.0;
        }
    }

    Matrix3D Matrix3D::operator+(const Matrix3D& a)
    {
        if (this->rowCount != a.rowCount || this->colCount != a.colCount || this->layCount != a.layCount)
        {
            std::string header = "Matrix3D Matrix3D::operator+(const Matrix3D& a)";
            std::string excep = std::format("Exception at: {}\n\tMatrices of dimensions {} and {}x{}x{} cannot be added.",
                header, this->getDimensions(), a.rowCount, a.colCount, a.layCount);
            throw excep;
        }

        Matrix3D summedMatrix(a.rowCount, a.colCount, a.layCount);
        
        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            summedMatrix.values[index] = this->values[index] + a.values[index];
        }

        return summedMatrix;
    }

    Matrix3D& Matrix3D::operator+=(const Matrix3D& a)
    {
        if (this->rowCount != a.rowCount || this->colCount != a.colCount || this->layCount != a.layCount)
        {
            std::string header = "Matrix3D& Matrix3D::operator+=(const Matrix3D& a)";
            std::string excep = std::format("Exception at: {}\n\tMatrices of dimensions {} and {}x{}x{} cannot be added.",
                header, this->getDimensions(), a.rowCount, a.colCount, a.layCount);
            throw excep;
        }

        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            this->values[index] += a.values[index];
        }
        
        return *this;
    }

    void Matrix3D::addSubmatrix(const Matrix3D& a, uint16_t rowStart, uint16_t colStart, uint16_t layStart)
    {
        uint16_t rowEnd = (this->rowCount < rowStart + a.rowCount) ? this->rowCount : rowStart + a.rowCount;
        uint16_t colEnd = (this->colCount < colStart + a.colCount) ? this->colCount : colStart + a.colCount;
        uint16_t layEnd = (this->layCount < layStart + a.layCount) ? this->layCount : layStart + a.layCount;
        
        for (uint16_t lay = (layStart > 0 ? layStart : 0); lay < layEnd; lay += 1)
        {
            for (uint16_t row = (rowStart > 0 ? rowStart : 0); row < rowEnd; row += 1)
            {
                for (uint16_t col = (colStart > 0 ? colStart : 0); col < colEnd; col += 1)
                {
                    set(row, col, lay, a.values[(col - colStart) + (row - rowStart) * a.colCount + (lay - layStart) * a.colCount * a.rowCount]);
                }
            }
        }
    }

    Matrix3D Matrix3D::operator-(const Matrix3D& a)
    {
        if (this->rowCount != a.rowCount || this->colCount != a.colCount || this->layCount != a.layCount)
        {
            std::string header = "Matrix3D Matrix3D::operator-(const Matrix3D& a)";
            std::string excep = std::format("Exception at: {}\n\tMatrices of dimensions {} and {}x{}x{} cannot be subtracted.",
                header, this->getDimensions(), a.rowCount, a.colCount, a.layCount);
            throw excep;
        }

        Matrix3D differenceMatrix(a.rowCount, a.colCount, a.layCount);
        
        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            differenceMatrix.values[index] = this->values[index] - a.values[index];
        }

        return differenceMatrix;
    }

    Matrix3D& Matrix3D::operator-=(const Matrix3D& a)
    {
        if (this->rowCount != a.rowCount || this->colCount != a.colCount || this->layCount != a.layCount)
        {
            std::string header = "Matrix3D& Matrix3D::operator-=(const Matrix3D& a)";
            std::string excep = std::format("Exception at: {}\n\tMatrices of dimensions {} and {}x{}x{} cannot be subtracted.",
                header, this->getDimensions(), a.rowCount, a.colCount, a.layCount);
            throw excep;
        }

        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            this->values[index] -= a.values[index];
        }

        return *this;
    }

    Matrix3D Matrix3D::operator*(const Matrix3D& a)
    {
        if (this->colCount != a.rowCount || this->layCount != a.layCount)
        {
            std::string header = "Matrix3D Matrix3D::operator*(const Matrix3D& a)";
            std::string excep = std::format("Exception at: {}\n\tMatrices of dimensions {} and {}x{}x{} cannot be multiplied.",
                header, this->getDimensions(), a.rowCount, a.colCount, a.layCount);
            throw excep;
        }

        Matrix3D productMatrix(this->rowCount, a.colCount, this->layCount);
        
        for (uint16_t lay = 0; lay < this->layCount; lay += 1)
        {
            for (uint16_t row = 0; row < this->rowCount; row += 1)
            {
                for (uint16_t col = 0; col < a.colCount; col += 1)
                {
                    // The dot product of the 'row'th row of the original matrix and the 'col'th col of matrix a.
                    double dotProduct = 0.0;

                    for (uint16_t i = 0; i < this->colCount; i += 1)
                    {
                        // FIX
                        dotProduct += this->values[i + row * this->colCount + lay * this->rowCount * this->colCount]
                                        * a.values[i * a.colCount + col + lay * a.rowCount * a.colCount];
                    }

                    productMatrix.set(row, col, lay, dotProduct);
                }
            }
        }
        
        return productMatrix;
    }

    Matrix3D& Matrix3D::operator*=(const Matrix3D& a)
    {
        if (this->colCount != a.rowCount || this->layCount != a.layCount)
        {
            std::string header = "Matrix3D& Matrix3D::operator*=(const Matrix3D& a)";
            std::string excep = std::format("Exception at: {}\n\tMatrices of dimensions {} and {}x{}x{} cannot be multiplied.",
                header, this->getDimensions(), a.rowCount, a.colCount, a.layCount);
            throw excep;
        }

        Matrix3D previousMatrix(this->rowCount, this->colCount, this->layCount, this->values);
        
        this->colCount = a.colCount;
        
        delete[] this->values;
        this->values = nullptr;
        this->values = new double[this->rowCount * this->colCount * this->layCount];

        for (uint16_t lay = 0; lay < this->layCount; lay += 1)
        {
            for (uint16_t row = 0; row < this->rowCount; row += 1)
            {
                for (uint16_t col = 0; col < a.colCount; col += 1)
                {
                    // The dot product of the 'row'th row of the original matrix and the 'col'th col of matrix a.
                    double dotProduct = 0.0;

                    for (uint16_t i = 0; i < this->colCount; i += 1)
                    {
                        dotProduct += previousMatrix.values[i + row * previousMatrix.colCount + lay * this->rowCount * this->colCount]
                                                 * a.values[i * a.colCount + col + lay * a.rowCount * a.colCount];
                    }

                    this->set(row, col, lay, dotProduct);
                }
            }
        }

        return *this;
    }

    Matrix3D Matrix3D::operator*(double factor)
    {
        Matrix3D scaledMatrix(this->rowCount, this->colCount, this->layCount);
        
        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            scaledMatrix.values[index] = this->values[index] * factor;
        }
        
        return scaledMatrix;
    }

    Matrix3D& Matrix3D::operator*=(double factor)
    {
        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            this->values[index] *= factor;
        }
        
        return *this;
    }

    Matrix3D Matrix3D::hadamardProduct(const Matrix3D& a, const Matrix3D& b)
    {
        if (a.rowCount != b.rowCount || a.colCount != b.colCount || a.layCount != b.layCount)
        {
            std::string header = "Matrix3D Matrix3D::hadamardMultiply(const Matrix3D& a, const Matrix3D& b)";
            std::string excep = std::format("Exception at: {}\n\tMatrices of dimensions {}x{}x{} and {}x{}x{} cannot be multiplied element-wise.",
                header, a.rowCount, a.colCount, a.layCount, b.rowCount, b.colCount, b.layCount);
            throw excep;
        }

        Matrix3D productMatrix(a.rowCount, a.colCount, a.layCount);
        
        for (int index = 0; index < a.rowCount * a.colCount * a.layCount; index += 1)
        {
            productMatrix.values[index] = a.values[index] * b.values[index];
        }
        
        return productMatrix;
    }

    Matrix3D Matrix3D::operator/(double factor)
    {
        Matrix3D quotientMatrix(this->rowCount, this->colCount, this->layCount);
        
        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            quotientMatrix.values[index] = this->values[index] / factor;
        }
        
        return quotientMatrix;
    }

    Matrix3D& Matrix3D::operator/=(double factor)
    {
        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            this->values[index] /= factor;
        }
        
        return *this;
    }

    Matrix3D Matrix3D::random(uint16_t rows, uint16_t columns, uint16_t layers, double minValue, double maxValue)
    {
        Matrix3D::randomSeed += 1;
        std::srand(static_cast<std::time_t>(Matrix3D::randomSeed) * std::time(NULL) * GetTickCount64());
        
        Matrix3D randomMatrix(rows, columns, layers);
        
        for (int index = 0; index < rows * columns * layers; index += 1)
        {
            double percent = (double) std::rand() / RAND_MAX;
            randomMatrix.values[index] = minValue + percent * (maxValue - minValue);
        }
        
        return randomMatrix;
    }

    void Matrix3D::randomize(double minValue, double maxValue)
    {
        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            double percent = (double) std::rand() / RAND_MAX;
            this->values[index] = minValue + percent * (maxValue - minValue);
        }
    }

    /**
     * @brief Returns the tranpose of the matrix.
     *
     * @return The transposed matrix.
     */
    Matrix3D Matrix3D::getTranspose()
    {
        Matrix3D transposeMatrix(this->colCount, this->rowCount, this->layCount);

        for (uint16_t lay = 0; lay < this->layCount; lay += 1)
        {
            for (uint16_t row = 0; row < this->rowCount; row += 1)
            {
                for (uint16_t col = 0; col < this->colCount; col += 1)
                {
                    transposeMatrix.set(col, row, lay, this->values[row * this->colCount + col]);
                }
            }
        }
        
        return transposeMatrix;
    }

    Matrix3D Matrix3D::getFlippedHori()
    {
        Matrix3D flippedMatrix = Matrix3D(this->rowCount, this->colCount, this->layCount);

        for (uint16_t lay = 0; lay < this->layCount; lay += 1)
        {
            for (uint16_t row = 0; row < this->rowCount; row += 1)
            {
                for (uint16_t col = 0; col < this->colCount; col += 1)
                {
                    flippedMatrix.set(row, col, lay, get(this->rowCount - row - 1, col, lay));
                }
            }
        }

        return flippedMatrix;
    }

    Matrix3D Matrix3D::getFlippedVert()
    {
        Matrix3D flippedMatrix = Matrix3D(this->rowCount, this->colCount, this->layCount);

        for (uint16_t lay = 0; lay < this->layCount; lay += 1)
        {
            for (uint16_t row = 0; row < this->rowCount; row += 1)
            {
                for (uint16_t col = 0; col < this->colCount; col += 1)
                {
                    flippedMatrix.set(row, col, lay, get(row, this->colCount - col - 1, lay));
                }
            }
        }

        return flippedMatrix;
    }

    Matrix3D Matrix3D::getFlippedHoriAndVert()
    {
        Matrix3D flippedMatrix = Matrix3D(this->rowCount, this->colCount, this->layCount);

        for (uint16_t lay = 0; lay < this->layCount; lay += 1)
        {
            for (uint16_t row = 0; row < this->rowCount; row += 1)
            {
                for (uint16_t col = 0; col < this->colCount; col += 1)
                {
                    flippedMatrix.set(row, col, lay, get(this->rowCount - row - 1, this->colCount - col - 1, lay));
                }
            }
        }

        return flippedMatrix;
    }

    Matrix3D Matrix3D::power(const Matrix3D& a, double factor)
    {
        Matrix3D powerMatrix(a.rowCount, a.colCount, a.layCount);

        for (int index = 0; index < a.rowCount * a.colCount * a.layCount; index += 1)
        {
            powerMatrix.values[index] = pow(a.values[index], factor);
        }

        return powerMatrix;
    }

    Matrix3D Matrix3D::exponential(const Matrix3D& a)
    {
        Matrix3D powerMatrix(a.rowCount, a.colCount, a.layCount);

        for (int index = 0; index < a.rowCount * a.colCount * a.layCount; index += 1)
        {
            powerMatrix.values[index] = exp(a.values[index]);
        }

        return powerMatrix;
    }

    Matrix3D Matrix3D::absoluteValue(const Matrix3D& a)
    {
        Matrix3D absMatrix(a.rowCount, a.colCount, a.layCount);

        for (int index = 0; index < a.rowCount * a.colCount * a.layCount; index += 1)
        {
            absMatrix.values[index] = abs(a.values[index]);
        }
        
        return absMatrix;
    }

    Matrix3D* Matrix3D::getRows(uint16_t* length)
    {
        Matrix3D* rows = new Matrix3D[this->rowCount];
        
        for (uint16_t row = 0; row < this->rowCount; row += 1)
        {
            Matrix3D tempRow(1, this->colCount, this->layCount);
            
            for (uint16_t lay = 0; lay < this->layCount; lay += 1)
            {
                for (uint16_t col = 0; col < this->colCount; col += 1)
                {
                    tempRow.set(0, col, lay, this->get(row, col, lay));
                }
            }

            rows[row] = Matrix3D(tempRow);
        }

        if (length != nullptr)
        {
            *length = this->rowCount;
        }
        
        return rows;
    }

    Matrix3D* Matrix3D::getColumns(uint16_t* length)
    {
        Matrix3D* columns = new Matrix3D[this->colCount];
        
        for (uint16_t col = 0; col < this->colCount; col += 1)
        {
            Matrix3D tempCol(this->rowCount, 1, this->layCount);
            
            for (uint16_t lay = 0; lay < this->layCount; lay += 1)
            {
                for (uint16_t row = 0; row < this->rowCount; row += 1)
                {
                    tempCol.set(row, 0, lay, this->get(row, col, lay));
                }
            }

            columns[col] = Matrix3D(tempCol);
        }

        if (length != nullptr)
        {
            *length = this->colCount;
        }

        return columns;
    }

    Matrix3D* Matrix3D::getLayers(uint16_t* length)
    {
        Matrix3D* layers = new Matrix3D[this->layCount];
        
        for (uint16_t lay = 0; lay < this->layCount; lay += 1)
        {
            Matrix3D tempLay(this->rowCount, this->colCount, 1);
            
            for (uint16_t col = 0; col < this->colCount; col += 1)
            {
                for (uint16_t row = 0; row < this->rowCount; row += 1)
                {
                    tempLay.set(row, col, 0, this->get(row, col, lay));
                }
            }

            layers[lay] = Matrix3D(tempLay);
        }

        if (length != nullptr)
        {
            *length = this->layCount;
        }

        return layers;
    }

    Matrix2D Matrix3D::getVectorized()
    {
        Matrix2D flattened(this->rowCount * this->colCount * this->layCount, 1);

        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            flattened.set(index, 0, this->values[index]);
        }

        return flattened;
    }

    void Matrix3D::unflatten(uint16_t rows, uint16_t cols, uint16_t lays)
    {
        this->rowCount = rows;
        this->colCount = cols;
        this->layCount = lays;
    }

    double* Matrix3D::getValues()
    {
        return this->values;
    }

    Matrix3D Matrix3D::getSubmatrix(int rowStart, int colStart, int layStart, uint16_t rowEnd, uint16_t colEnd, uint16_t layEnd)
    {
        if (rowStart > rowEnd || colStart > colEnd || layStart > layEnd)
        {
            std::string header = "Matrix3D Matrix3D::getSubmatrix(int rowStart, int colStart, int layStart, uint16_t rowEnd, uint16_t colEnd, uint16_t layEnd)";
            std::string excep = std::format("Exception at: {}\n\tUpper bound ({}) greater than lower bound ({}), left bound ({}) greater than right bound ({}),"
                    " or front bound({}) greater than back bound({}).", header, rowStart, rowEnd, colStart, colEnd, layStart, layEnd);
            throw excep;
        }

        Matrix3D subMatrix(rowEnd - rowStart, colEnd - colStart, layEnd - layStart);
        
        for (int lay = layStart; lay < layEnd; lay += 1)
        {
            for (int row = rowStart; row < rowEnd; row += 1)
            {
                for (int col = colStart; col < colEnd; col += 1)
                {
                    if (row > -1 && row < this->rowCount && col > -1 && col < this->colCount && lay > -1 && lay < this->layCount)
                    {
                        subMatrix.set(row - rowStart, col - colStart, lay - layStart, this->values[col + row * this->colCount + lay * this->colCount * this->rowCount]);
                    }
                }
            }
        }
        return subMatrix;
    }

    double Matrix3D::getSumOfEntries()
    {
        double sum = 0.0;

        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            sum += this->values[index];
        }
        
        return sum;
    }

    int Matrix3D::getIndexOfMax()
    {
        double maxValue = std::numeric_limits<double>::lowest();
        uint16_t maxIndex = 0;

        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            if (this->values[index] > maxValue)
            {
                maxValue = this->values[index];
                maxIndex = index;
            }
        }
        return maxIndex;
    }

    int Matrix3D::getIndexOfMin()
    {
        double minValue = DBL_MAX;
        uint16_t minIndex = 0;

        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            if (this->values[index] < minValue)
            {
                minValue = this->values[index];
                minIndex = index;
            }
        }
        return minIndex;
    }

    Matrix3D Matrix3D::getRescaled(double lowValue, double highValue)
    {
        Matrix3D rescaledMatrix(this->rowCount, this->colCount, this->layCount);

        double minValue = this->values[getIndexOfMin()];
        double maxValue = this->values[getIndexOfMax()];

        for (uint16_t lay = 0; lay < this->layCount; lay += 1)
        {
            for (uint16_t row = 0; row < this->rowCount; row += 1)
            {
                for (uint16_t col = 0; col < this->colCount; col += 1)
                {
                    if (maxValue != minValue)
                    {
                        rescaledMatrix.set(row, col, lay, lowValue + (highValue - lowValue) * (get(row, col, lay) - minValue) / (maxValue - minValue));
                    }
                    else
                    {
                        rescaledMatrix.set(row, col, lay, lowValue + (highValue - lowValue) * 0.5);
                    }
                }
            }
        }

        return rescaledMatrix;
    }

    void Matrix3D::rescale(double lowValue, double highValue)
    {
        double minValue = this->values[getIndexOfMin()];
        double maxValue = this->values[getIndexOfMax()];

        for (uint16_t lay = 0; lay < this->layCount; lay += 1)
        {
            for (uint16_t row = 0; row < this->rowCount; row += 1)
            {
                for (uint16_t col = 0; col < this->colCount; col += 1)
                {
                    if (maxValue != minValue)
                    {
                        set(row, col, lay, lowValue + (highValue - lowValue) * (get(row, col, lay) - minValue) / (maxValue - minValue));
                    }
                    else
                    {
                        set(row, col, lay, lowValue + (highValue - lowValue) * 0.5);
                    }
                }
            }
        }
    }

    constexpr double Matrix3D::get(uint16_t row, uint16_t column, uint16_t layer) const noexcept
    {
        if (row >= this->rowCount || column >= this->colCount || layer >= this->layCount)
        {
            std::string header = "double Matrix3D::get(uint16_t row, uint16_t column, uint16_t layer)";
            std::string excep = std::format("Exception at: {}\n\t({}, {}, {}) is an invalid index for matrix of dimensions {}.", header, row, column, layer, getDimensions());
            throw excep;
        }

        return this->values[column + row * this->colCount + layer * this->colCount * this->rowCount];
    }

    void Matrix3D::set(uint16_t row, uint16_t column, uint16_t layer, double value)
    {
        if (row >= this->rowCount || column >= this->colCount || layer >= this->layCount)
        {
            std::string header = "void Matrix3D::set(uint16_t row, uint16_t column, uint16_t layer, double value)";
            std::string excep = std::format("Exception at: {}\n\t({}, {}, {}) is an invalid index for matrix of dimensions {}.", header, row, column, layer, getDimensions());
            throw excep;
        }

        this->values[column + row * this->colCount + layer * this->colCount * this->rowCount] = value;
    }

    void Matrix3D::setRow(uint16_t row, const Matrix3D& a)
    {
        if (row >= this->rowCount)
        {
            std::string header = "void Matrix3D::setRow(uint16_t row, const Matrix3D& a)";
            std::string excep = std::format("Exception at: {}\n\tCannot set row of index {} for matrix of dimensions {}.", header, row, getDimensions());
            throw excep;
        }

        for (uint16_t lay = 0; lay < this->layCount; lay += 1)
        {
            for (uint16_t col = 0; col < this->colCount; col += 1)
            {
                if (col < a.colCount && lay < a.layCount)
                {
                    this->values[col + row * this->colCount + lay * this->colCount * this->rowCount] = a.values[col + lay * a.colCount * a.rowCount];
                }
                else
                {
                    this->values[col + row * this->colCount + lay * this->colCount * this->rowCount] = 0.0;
                }
            }
        }
    }

    void Matrix3D::setColumn(uint16_t col, const Matrix3D& a)
    {
        if (col >= this->colCount)
        {
            std::string header = "void Matrix3D::setColumn(uint16_t col, const Matrix3D& a)";
            std::string excep = std::format("Exception at: {}\n\tCannot set column of index {} for matrix of dimensions {}.", header, col, getDimensions());
            throw excep;
        }

        for (uint16_t lay = 0; lay < this->layCount; lay += 1)
        {
            for (uint16_t row = 0; row < this->rowCount; row += 1)
            {
                if (row < a.rowCount && lay < a.layCount)
                {
                    this->values[col + row * this->colCount + lay * this->colCount * this->rowCount] = a.values[row * a.colCount + lay * a.colCount * a.rowCount];
                }
                else
                {
                    this->values[col + row * this->colCount + lay * this->colCount * this->rowCount] = 0.0;
                }
            }
        }
    }

    void Matrix3D::setLayer(uint16_t lay, const Matrix3D& a)
    {
        if (lay >= this->layCount)
        {
            std::string header = "void Matrix3D::setLayer(uint16_t lay, const Matrix3D& a)";
            std::string excep = std::format("Exception at: {}\n\tCannot set layer of index {} for matrix of dimensions {}.", header, lay, getDimensions());
            throw excep;
        }

        int layerOffset = lay * this->colCount * this->rowCount;

        for (uint16_t row = 0; row < this->rowCount; row += 1)
        {
            for (uint16_t col = 0; col < this->colCount; col += 1)
            {
                if (row < a.rowCount && col < a.colCount)
                {
                    this->values[col + row * this->colCount + layerOffset] = a.values[col + row * a.colCount];
                }
                else
                {
                    this->values[col + row * this->colCount + layerOffset] = 0.0;
                }
            }
        }
    }

    constexpr int Matrix3D::getRowCount() const noexcept
    {
        return this->rowCount;
    }

    constexpr int Matrix3D::getColumnCount() const noexcept
    {
        return this->colCount;
    }

    constexpr int Matrix3D::getLayerCount() const noexcept
    {
        return this->layCount;
    }

    constexpr std::string Matrix3D::getDimensions() const noexcept
    {
        std::string out = std::to_string(getRowCount()) + "x" + std::to_string(getColumnCount()) + "x" + std::to_string(getLayerCount());
        return out;
    }

    bool Matrix3D::operator==(const Matrix3D& a)
    {
        if (this->colCount != a.colCount || this->rowCount != a.rowCount || this->layCount != a.layCount)
        {
            return false;
        }
        
        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            if (this->values[index] != a.values[index])
            {
                return false;
            }
        }
        
        return true;
    }

    Matrix3D& Matrix3D::operator=(const Matrix3D& a)
    {
        if (this == &a)
        {
            return *this;
        }

        this->rowCount = a.rowCount;
        this->colCount = a.colCount;
        this->layCount = a.layCount;

        delete[] this->values;
        this->values = nullptr;
        this->values = new double[a.rowCount * a.colCount * a.layCount];

        for (int index = 0; index < this->rowCount * this->colCount * this->layCount; index += 1)
        {
            this->values[index] = a.values[index];
        }

        return *this;
    }

    std::string Matrix3D::toString()
    {
        if (this->rowCount == 0 || this->colCount == 0 || this->layCount == 0)
        {
            return "\nEMPTY MATRIX; DIMENSIONS " + getDimensions() + "\n";
        }
        
        std::string out = "";
        
        Matrix3D* layers = getLayers(nullptr);

        out += "\nMatrix3D: [\n";
        for (uint16_t layer = 0; layer < this->layCount; layer += 1)
        {
            Matrix2D mat2d(this->rowCount, this->colCount, layers[layer].values);
            out += mat2d.toString();
        }
        out += "\n]\n";

        delete[] layers;

        return out;
    }

    std::ostream& operator<<(std::ostream& os, const Matrix3D& a)
    {
        std::string out = "";

        if (a.rowCount == 0 || a.colCount == 0 || a.layCount == 0)
        {
            out = "\nEMPTY MATRIX; DIMENSIONS " + std::to_string(a.rowCount) + "x" + std::to_string(a.colCount)
                + "x" + std::to_string(a.layCount) + "\n";
            os << out;
            return os;
        }

        out += "\nMatrix3D: [\n";

        Matrix3D* layers = new Matrix3D[a.layCount];
        
        for (uint16_t lay = 0; lay < a.layCount; lay += 1)
        {
            Matrix2D mat2d(a.rowCount, a.colCount);

            for (uint16_t col = 0; col < a.colCount; col += 1)
            {
                for (uint16_t row = 0; row < a.rowCount; row += 1)
                {
                    mat2d.set(row, col, a.values[col + row * a.colCount + lay * a.colCount * a.rowCount]);
                }
            }

            out += mat2d.toString();
        }

        out += "\n]\n";

        delete[] layers;

        os << out;
        return os;
    }
}