#pragma once

#include <ctime>
#include <chrono>
#include <format>
#include "../headers/Matrix2D.hpp"
#include <Windows.h>

namespace Neuranet
{
    int Matrix2D::randomSeed;

    Matrix2D::Matrix2D(uint16_t rows, uint16_t columns, double* values)
    {
        this->dimensionCount = 2;

        this->rowCount = rows;
        this->colCount = columns;

        this->values = new double[rows * columns];

        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            this->values[index] = values[index];
        }
    }

    Matrix2D::~Matrix2D()
    {
        delete[] this->values;
    }

    void Matrix2D::zero()
    {
        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            this->values[index] = 0.0;
        }
    }

    Matrix2D Matrix2D::operator+(const Matrix2D& a)
    {
        if (this->rowCount != a.rowCount || this->colCount != a.colCount)
        {
            std::string header = "Matrix2D Matrix2D::operator+(const Matrix2D& a)";
            std::string excep = std::format("Exception at: {}\n\tMatrices of dimensions {} and {}x{} cannot be added.",
                 header, this->getDimensions(), a.rowCount, a.colCount);
            throw excep;
        }

        Matrix2D summedMatrix(a.rowCount, a.colCount);
        
        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            summedMatrix.values[index] = this->values[index] + a.values[index];
        }

        return summedMatrix;
    }

    Matrix2D& Matrix2D::operator+=(const Matrix2D& a)
    {
        if (this->rowCount != a.rowCount || this->colCount != a.colCount)
        {
            std::string header = "Matrix2D& Matrix2D::operator+=(const Matrix2D& a)";
            std::string excep = std::format("Exception at: {}\n\tMatrices of dimensions {} and {}x{} cannot be added.",
                header, this->getDimensions(), a.rowCount, a.colCount);
            throw excep;
        }

        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            this->values[index] += a.values[index];
        }
        
        return *this;
    }

    Matrix2D Matrix2D::operator-(const Matrix2D& a)
    {
        if (this->rowCount != a.rowCount || this->colCount != a.colCount)
        {
            std::string header = "Matrix2D Matrix2D::operator-(const Matrix2D& a)";
            std::string excep = std::format("Exception at: {}\n\tMatrices of dimensions {} and {}x{} cannot be subtracted.",
                header, this->getDimensions(), a.rowCount, a.colCount);
            throw excep;
        }

        Matrix2D differenceMatrix(a.rowCount, a.colCount);
        
        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            differenceMatrix.values[index] = this->values[index] - a.values[index];
        }

        return differenceMatrix;
    }

    Matrix2D& Matrix2D::operator-=(const Matrix2D& a)
    {
        if (this->rowCount != a.rowCount || this->colCount != a.colCount)
        {
            std::string header = "Matrix2D& Matrix2D::operator-=(const Matrix2D& a)";
            std::string excep = std::format("Exception at: {}\n\tMatrices of dimensions {} and {}x{} cannot be subtracted.",
                header, this->getDimensions(), a.rowCount, a.colCount);
            throw excep;
        }

        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            this->values[index] -= a.values[index];
        }

        return *this;
    }

    Matrix2D Matrix2D::operator*(const Matrix2D& a)
    {
        if (this->colCount != a.rowCount)
        {
            std::string header = "Matrix2D Matrix2D::operator*(const Matrix2D & a)";
            std::string excep = std::format("Exception at: {}\n\tMatrices of dimensions {} and {}x{} cannot be multiplied.",
                header, this->getDimensions(), a.rowCount, a.colCount);
            throw excep;
        }

        Matrix2D productMatrix = Matrix2D(this->rowCount, a.colCount);
        
        for (uint16_t row = 0; row < this->rowCount; row += 1)
        {
            for (uint16_t col = 0; col < a.colCount; col += 1)
            {
                // The dot product of the 'row'th row of the original matrix and the 'col'th col of matrix a.
                double dotProduct = 0.0;

                for (uint16_t i = 0; i < this->colCount; i += 1)
                {
                    dotProduct += this->values[row * this->colCount + i] * a.values[i * a.colCount + col];
                }

                productMatrix.set(row, col, dotProduct);
            }
        }

        return productMatrix;
    }

    Matrix2D& Matrix2D::operator*=(const Matrix2D& a)
    {
        if (this->colCount != a.rowCount)
        {
            std::string header = "Matrix2D& Matrix2D::operator*=(const Matrix2D& a)";
            std::string excep = std::format("Exception at: {}\n\tMatrices of dimensions {} and {}x{} cannot be multiplied.",
                header, this->getDimensions(), a.rowCount, a.colCount);
            throw excep;
        }

        Matrix2D previousMatrix(this->rowCount, this->colCount, this->values);
        
        this->colCount = a.colCount;

        delete[] this->values;
        this->values = nullptr;
        this->values = new double[this->rowCount* this->colCount];

        for (uint16_t row = 0; row < this->rowCount; row += 1)
        {
            for (uint16_t col = 0; col < a.colCount; col += 1)
            {
                // The dot product of the 'row'th row of the original matrix and the 'col'th col of matrix a.
                double dotProduct = 0.0;

                for (uint16_t i = 0; i < this->colCount; i += 1)
                {
                    dotProduct += previousMatrix.values[row * previousMatrix.colCount + i] * a.values[i * a.colCount + col];
                }

                this->set(row, col, dotProduct);
            }
        }

        return *this;
    }

    Matrix2D Matrix2D::operator*(double factor)
    {
        Matrix2D scaledMatrix(this->rowCount, this->colCount);
        
        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            scaledMatrix.values[index] = this->values[index] * factor;
        }
        
        return scaledMatrix;
    }

    Matrix2D& Matrix2D::operator*=(double factor)
    {
        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            this->values[index] *= factor;
        }
        
        return *this;
    }

    Matrix2D Matrix2D::hadamardMultiply(const Matrix2D& a, const Matrix2D& b)
    {
        if (a.rowCount != b.rowCount || a.colCount != b.colCount)
        {
            std::string header = "Matrix2D Matrix2D::hadamardMultiply(const Matrix2D& a, const Matrix2D& b)";
            std::string excep = std::format("Exception at: {}\n\tMatrices of dimensions {}x{} and {}x{} cannot be multiplied element-wise.",
                header, std::to_string(a.rowCount), a.colCount, b.rowCount, b.colCount);
            throw excep;
        }

        Matrix2D productMatrix(a.rowCount, a.colCount);
        
        for (int index = 0; index < a.rowCount * a.colCount; index += 1)
        {
            productMatrix.values[index] = a.values[index] * b.values[index];
        }
        
        return productMatrix;
    }

    Matrix2D Matrix2D::operator/(double factor)
    {
        Matrix2D quotientMatrix(this->rowCount, this->colCount);
        
        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            quotientMatrix.values[index] = this->values[index] / factor;
        }
        
        return quotientMatrix;
    }

    Matrix2D& Matrix2D::operator/=(double factor)
    {
        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            this->values[index] /= factor;
        }
        
        return *this;
    }

    Matrix2D Matrix2D::random(uint16_t rows, uint16_t columns, double minValue, double maxValue)
    {
        Matrix2D::randomSeed += 1;
        std::srand(static_cast<std::time_t>(Matrix2D::randomSeed) * std::time(NULL) * GetTickCount64());
        
        Matrix2D randomMatrix = Matrix2D(rows, columns);
        
        for (int index = 0; index < rows * columns; index += 1)
        {
            double percent = (double) std::rand() / RAND_MAX;
            randomMatrix.values[index] = minValue + percent * (maxValue - minValue);
        }
        
        return randomMatrix;
    }

    void Matrix2D::randomize(double minValue, double maxValue)
    {
        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            double percent = (double) std::rand() / RAND_MAX;
            this->values[index] = minValue + percent * (maxValue - minValue);
        }
    }

    Matrix2D Matrix2D::power(const Matrix2D& a, double factor)
    {
        Matrix2D powerMatrix(a.rowCount, a.colCount);

        for (int index = 0; index < a.rowCount * a.colCount; index += 1)
        {
            powerMatrix.values[index] = pow(a.values[index], factor);
        }
        
        return powerMatrix;
    }

    Matrix2D Matrix2D::getTranspose()
    {
        Matrix2D transposeMatrix(this->colCount, this->rowCount);

        for (uint16_t row = 0; row < this->rowCount; row += 1)
        {
            for (uint16_t col = 0; col < this->colCount; col += 1)
            {
                transposeMatrix.set(col, row, this->values[row * this->colCount + col]);
            }
        }
        return transposeMatrix;
    }

    Matrix2D Matrix2D::absoluteValue(const Matrix2D& a)
    {
        Matrix2D absMatrix(a.rowCount, a.colCount);

        for (int index = 0; index < a.rowCount * a.colCount; index += 1)
        {
            absMatrix.values[index] = abs(a.values[index]);
        }
        
        return absMatrix;
    }

    Matrix2D* Matrix2D::getRows(uint16_t* length)
    {
        Matrix2D* rows = new Matrix2D[this->rowCount];
        
        for (uint16_t row = 0; row < this->rowCount; row += 1)
        {
            Matrix2D tempRow(1, this->colCount);

            for (uint16_t col = 0; col < this->colCount; col += 1)
            {
                tempRow.set(0, col, this->get(row, col));
            }

            rows[row] = Matrix2D(tempRow);
        }

        *length = this->rowCount;

        return rows;
    }

    Matrix2D* Matrix2D::getColumns(uint16_t* length)
    {
        Matrix2D* columns = new Matrix2D[this->colCount];
        
        for (uint16_t col = 0; col < this->colCount; col += 1)
        {
            Matrix2D tempCol(this->rowCount, 1);

            for (uint16_t row = 0; row < this->rowCount; row += 1)
            {
                tempCol.set(row, 0, this->get(row, col));
            }

            columns[col] = Matrix2D(tempCol);
        }

        *length = this->colCount;

        return columns;
    }

    Matrix2D Matrix2D::flatten()
    {
        Matrix2D flattened(this->rowCount * this->colCount, 1);

        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            flattened.set(index, 0, this->values[index]);
        }

        return flattened;
    }

    double* Matrix2D::getValues()
    {
        return this->values;
    }

    Matrix2D Matrix2D::getSubmatrix(int rowStart, int colStart, uint16_t rowEnd, uint16_t colEnd)
    {
        if (rowStart > rowEnd || colStart > colEnd)
        {
            std::string header = "Matrix2D Matrix2D::getSubmatrix(int rowStart, int colStart, uint16_t rowEnd, uint16_t colEnd)";
            std::string excep = std::format("Exception at: {}\n\tUpper bound ({}) greater than lower bound ({}) or left bound ({}) greater than right bound ({}).",
                header, rowStart, rowEnd, colStart, colEnd);
            throw excep;
        }

        Matrix2D subMatrix(rowEnd - rowStart, colEnd - colStart);
        
        for (int row = rowStart; row < rowEnd; row += 1)
        {
            for (int col = colStart; col < colEnd; col += 1)
            {
                if (row > -1 && row < this->rowCount && col > -1 && col < this->colCount)
                {
                    subMatrix.set(row - rowStart, col - colStart, this->values[row * this->colCount + col]);
                }
            }
        }
        return subMatrix;
    }

    Matrix2D Matrix2D::getMinor(uint16_t row, uint16_t col)
    {
        Matrix2D minorMatrix(this->rowCount - 1, this->colCount - 1);
        
        for (uint16_t r = 0; r < this->rowCount; r += 1)
        {
            if (r != row)
            {
                for (uint16_t c = 0; c < this->colCount; c += 1)
                {
                    if (c != col) {
                        minorMatrix.set((r > row ? r - 1 : r), (c > col ? c - 1 : c), this->values[r * this->colCount + c]);
                    }
                }
            }
        }
        return minorMatrix;
    }

    Matrix2D Matrix2D::getCofactors()
    {
        if (this->rowCount != this->colCount)
        {
            std::string header = "Matrix2D Matrix2D::getCofactors()";
            std::string excep = std::format("Exception at: {}\n\tCannot find cofactors for matrix of dimensions {}.",
                header, getDimensions());
            throw excep;
        }

        Matrix2D cofactorMatrix(this->rowCount, this->colCount);
        
        for (uint16_t row = 0; row < this->rowCount; row += 1)
        {
            for (uint16_t col = 0; col < this->colCount; col += 1)
            {
                Matrix2D minor = this->getMinor(row, col);
                double det = minor.getDeterminant();
                cofactorMatrix.set(row, col, det * (((row + col) % 2) * (-2) + 1));
            }
        }
        
        return cofactorMatrix;
    }

    Matrix2D Matrix2D::getAdjoint()
    {
        if (this->rowCount != this->colCount)
        {
            std::string header = "Matrix2D Matrix2D::getAdjoint()";
            std::string excep = std::format("Exception at: {}\n\tCannot find the adjoint for matrix of dimensions {}.",
                header, getDimensions());
            throw excep;
        }
        return (this->getCofactors()).getTranspose();
    }

    double Matrix2D::getDeterminant()
    {
        if (this->rowCount != this->colCount)
        {
            std::string header = "double Matrix2D::getDeterminant()";
            std::string excep = std::format("Exception at: {}\n\tCannot find the determinant for matrix of dimensions {}.",
                header, getDimensions());
            throw excep;
        }
        if (this->colCount == 1)
        {
            return this->values[0];
        }
        else
        {
            double sum = 0.0;
            for (uint16_t col = 0; col < this->colCount; col += 1)
            {
                Matrix2D minor = this->getMinor(0, col);
                sum += this->values[col] * minor.getDeterminant() * ((col % 2) * (-2) + 1);
            }
            return sum;
        }
    }

    Matrix2D Matrix2D::getInverse()
    {
        if (this->rowCount != this->colCount)
        {
            std::string header = "Matrix2D Matrix2D::getInverse()";
            std::string excep = std::format("Exception at: {}\n\tCannot find the inverse of matrix of dimensions {}.",
                header, getDimensions());
            throw excep;
        }
        double det = this->getDeterminant();
        if (abs(det) > 0.000001)
        {
            return this->getAdjoint() / det;
        }
        else
        {
            std::string header = "Matrix2D Matrix2D::getInverse()";
            std::string excep = std::format("Exception at: {}\n\tCannot find the inverse of matrix with determinant of 0.",
                header);
            throw excep;
        }
    }

    double Matrix2D::getSumOfEntries()
    {
        double sum = 0.0;

        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            sum += this->values[index];
        }
        
        return sum;
    }

    double Matrix2D::getTrace()
    {
        double trace = 0.0;

        for (int index = 0; index < this->rowCount * this->colCount; index += this->colCount + 1)
        {
            trace += this->values[index];
        }

        return trace;
    }

    int Matrix2D::getIndexOfMax()
    {
        double maxValue = std::numeric_limits<double>::lowest();
        uint16_t maxIndex = 0;

        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            if (this->values[index] > maxValue)
            {
                maxValue = this->values[index];
                maxIndex = index;
            }
        }

        return maxIndex;
    }

    int Matrix2D::getIndexOfMin()
    {
        double minValue = DBL_MAX;
        uint16_t minIndex = 0;

        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            if (this->values[index] < minValue)
            {
                minValue = this->values[index];
                minIndex = index;
            }
        }

        return minIndex;
    }

    Matrix2D Matrix2D::getRescaled(double lowValue, double highValue)
    {
        Matrix2D rescaledMatrix(this->rowCount, this->colCount);

        double minValue = this->values[getIndexOfMin()];
        double maxValue = this->values[getIndexOfMax()];

        for (uint16_t row = 0; row < this->rowCount; row += 1)
        {
            for (uint16_t col = 0; col < this->colCount; col += 1)
            {
                if (maxValue != minValue)
                {
                    rescaledMatrix.set(row, col, lowValue + (highValue - lowValue) * (get(row, col) - minValue) / (maxValue - minValue));
                }
                else
                {
                    rescaledMatrix.set(row, col, lowValue + (highValue - lowValue) * 0.5);
                }
            }
        }

        return rescaledMatrix;
    }

    void Matrix2D::rescale(double lowValue, double highValue)
    {
        double minValue = this->values[getIndexOfMin()];
        double maxValue = this->values[getIndexOfMax()];

        for (uint16_t row = 0; row < this->rowCount; row += 1)
        {
            for (uint16_t col = 0; col < this->colCount; col += 1)
            {
                if (maxValue != minValue)
                {
                    set(row, col, lowValue + (highValue - lowValue) * (get(row, col) - minValue) / (maxValue - minValue));
                }
                else
                {
                    set(row, col, lowValue + (highValue - lowValue) * 0.5);
                }
            }
        }
    }

    constexpr double Matrix2D::get(uint16_t row, uint16_t column) const noexcept
    {
        if (row >= this->rowCount || column >= this->colCount)
        {
            std::string header = "double Matrix2D::get(uint16_t row, uint16_t column)";
            std::string excep = std::format("Exception at: {}\n\t({}, {}) is an invalid index for matrix of dimensions {}.",
                header, row, column, getDimensions());
            throw excep;
        }

        return this->values[row * this->colCount + column];
    }

    void Matrix2D::set(uint16_t row, uint16_t column, double value)
    {
        if (row >= this->rowCount || column >= this->colCount)
        {
            std::string header = "void Matrix2D::set(uint16_t row, uint16_t column, double value)";
            std::string excep = std::format("Exception at: {}\n\t({}, {}) is an invalid index for matrix of dimensions {}.",
                header, row, column, getDimensions());
            throw excep;
        }

        this->values[row * this->colCount + column] = value;
    }

    void Matrix2D::setRow(uint16_t row, const Matrix2D& a)
    {
        if (row >= this->rowCount)
        {
            std::string header = "void Matrix2D::setRow(uint16_t row, const Matrix2D& a)";
            std::string excep = std::format("Exception at: {}\n\tCannot set row of index {} in matrix of dimensions {}.",
                header, row, getDimensions());
            throw excep;
        }

        for (uint16_t col = 0; col < this->colCount; col += 1)
        {
            if (col < a.colCount)
            {
                this->values[row * this->colCount + col] = a.values[col];
            }
            else
            {
                this->values[row * this->colCount + col] = 0.0;
            }
        }
    }

    void Matrix2D::setColumn(uint16_t col, const Matrix2D& a)
    {
        if (col >= this->colCount)
        {
            std::string header = "void Matrix2D::setColumn(uint16_t col, const Matrix2D& a)";
            std::string excep = std::format("Exception at: {}\n\tCannot set column of index {} in matrix of dimensions {}.",
                header, col, getDimensions());
            throw excep;
        }

        for (uint16_t row = 0; row < this->rowCount; row += 1)
        {
            if (row < a.rowCount)
            {
                this->values[row * this->colCount + col] = a.values[row * a.colCount];
            }
            else
            {
                this->values[row * this->colCount + col] = 0.0;
            }
        }
    }

    constexpr int Matrix2D::getRowCount() const noexcept
    {
        return this->rowCount;
    }

    constexpr int Matrix2D::getColumnCount() const noexcept
    {
        return this->colCount;
    }

    constexpr std::string Matrix2D::getDimensions() const noexcept
    {
        std::string out = std::to_string(getRowCount()) + "x" + std::to_string(getColumnCount());
        return out;
    }

    bool Matrix2D::operator==(const Matrix2D& a)
    {
        if (this->colCount != a.colCount || this->rowCount != a.rowCount)
        {
            return false;
        }
        
        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            if (this->values[index] != a.values[index])
            {
                return false;
            }
        }
        
        return true;
    }

    Matrix2D& Matrix2D::operator=(const Matrix2D& a)
    {
        if (this == &a)
        {
            return *this;
        }
        this->rowCount = a.rowCount;
        this->colCount = a.colCount;

        delete[] this->values;
        this->values = nullptr;
        this->values = new double[a.rowCount * a.colCount];

        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            this->values[index] = a.values[index];
        }

        return *this;
    }

    std::string Matrix2D::toString()
    {
        if (rowCount == 0 || colCount == 0)
        {
            return "\nEMPTY MATRIX; DIMENSIONS " + getDimensions() + "\n";
        }
        
        std::string out = "";

        for (uint16_t row = 0; row < this->rowCount; row += 1)
        {
            out += "\n[ ";
            for (uint16_t col = 0; col < this->colCount; col += 1)
            {
                out += std::to_string(this->values[row * this->colCount + col]) + " ";
            }
            out += "]";
        }

        return out + "\n";
    }

    std::ostream& operator<<(std::ostream& os, const Matrix2D& a)
    {
        std::string out = "";

        if (a.rowCount == 0 || a.colCount == 0)
        {
            out = "\nEMPTY MATRIX; DIMENSIONS " + std::to_string(a.rowCount) + "x" + std::to_string(a.colCount) + "\n";
            os << out;
            return os;
        }

        for (uint16_t row = 0; row < a.rowCount; row += 1)
        {
            out += "\n[ ";
            for (uint16_t col = 0; col < a.colCount; col += 1)
            {
                out += std::to_string(a.values[row * a.colCount + col]) + " ";
            }
            out += "]";
        }

        out += "\n";

        os << out;
        return os;
    }
}