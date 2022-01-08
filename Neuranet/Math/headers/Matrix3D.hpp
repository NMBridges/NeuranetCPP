#pragma once

#include "Matrix.hpp"
#include "Matrix2D.hpp"

namespace Neuranet
{
    /**
     * @brief Class representing a 3-dimensional matrix
     *        of any size.
     * @author Nolan Bridges
     * @version 1.0.0
     */
    class Matrix3D : public Matrix
    {
    private:
        /**
         * @brief The number of layers in the matrix.
         */
        uint16_t layCount;
        /**
         * @brief The seed to use for calling random functions.
         */
        static int randomSeed;

    public:
        /**
         * @brief Construct a new Matrix3D object with size 0.
         */
        Matrix3D() : Matrix3D(0, 0, 0, nullptr) {};
        
        /**
         * @brief Construct a new Matrix3D object with the specified dimensions and values.
         */
        Matrix3D(uint16_t rows, uint16_t columns, uint16_t layers, double* values);
        
        /**
         * @brief Construct a new Matrix3D object with the specified dimensions (default values = 0.0).
         */
        Matrix3D(uint16_t rows, uint16_t columns, uint16_t layers) : Matrix3D(rows, columns, layers, nullptr) {};
        
        /**
         * @brief Construct a new Matrix3D object from another Matrix3D object.
         */
        Matrix3D(const Matrix3D& a) : Matrix3D(a.rowCount, a.colCount, a.layCount, a.values) {};

        /**
         * @brief Destroys the Matrix3D object.
         */
        ~Matrix3D();

        /**
         * @brief Zeroes the entries of the matrix.
         */
        void zero();

        /**
         * @brief Overrides the addition operator such that two matrices can be added.
         * 
         * @param a The matrix to add to the current one.
         * @return The summed matrix. 
         */
        Matrix3D operator+(const Matrix3D& a);

        /**
         * @brief Overrides the addition operator such that two matrices can be added.
         * 
         * @param a The matrix to add to the current one.
         * @return The summed matrix. 
         */
        Matrix3D& operator+=(const Matrix3D& a);

        /**
         * @brief Adds a smaller matrix to the existing matrix starting at a given index.
         *
         * @param a The matrix to add to the current one.
         */
        void addSubmatrix(const Matrix3D& a, uint16_t rowStart, uint16_t colStart, uint16_t layStart);

        /**
         * @brief Overrides the subtraction operator such that two matrices can be subtracted.
         * 
         * @param a The matrix to subtract from the current one.
         * @return The difference matrix. 
         */
        Matrix3D operator-(const Matrix3D& a);

        /**
         * @brief Overrides the subtraction operator such that two matrices can be subtracted.
         * 
         * @param a The matrix to subtract from the current one.
         * @return The difference matrix. 
         */
        Matrix3D& operator-=(const Matrix3D& a);

        /**
         * @brief Overrides the multiplication operator such that two matrices can be multiplied.
         * 
         * @param a The matrix to multiply to the current one (from the right).
         * @return The product matrix. 
         */
        Matrix3D operator*(const Matrix3D& a);

        /**
         * @brief Overrides the multiplication operator such that two matrices can be multiplied.
         * 
         * @param a The matrix to multiply to the current one (from the right).
         * @return The product matrix. 
         */
        Matrix3D& operator*=(const Matrix3D& a);

        /**
         * @brief Overrides the multiplication operator such a matrix can be individually scaled.
         * 
         * @param factor The factor to scale the matrix entries by.
         * @return The scaled matrix. 
         */
        Matrix3D operator*(double factor);

        /**
         * @brief Overrides the multiplication operator such a matrix can be individually scaled.
         * 
         * @param factor The factor to scale the matrix entries by.
         * @return The scaled matrix. 
         */
        Matrix3D& operator*=(double factor);

        /**
         * @brief Multiply two matrices in an element-wise fashion.
         * 
         * @param a The first matrix.
         * @param b The second matrix.
         * @return The hadamard product matrix. 
         */
        static Matrix3D hadamardProduct(const Matrix3D& a, const Matrix3D& b);

        /**
         * @brief Overrides the division operator such that a matrix can be individually divided.
         * 
         * @param factor The factor to divide the matrix entries by.
         * @return The 'quotient' matrix. 
         */
        Matrix3D operator/(double factor);

        /**
         * @brief Overrides the division operator such that a matrix can be individually divided.
         * 
         * @param factor The factor to divide the matrix entries by.
         * @return The 'quotient' matrix. 
         */
        Matrix3D& operator/=(double factor);

        /**
         * @brief Creates an array of specified dimensions with random values between
         *        the two specified values.
         * 
         * @param rows The number of rows in the matrix.
         * @param columns The number of columns in the matrix.
         * @param layers The number of layers in the matrix.
         * @param minValue The minimum value of entries in the matrix.
         * @param maxValue The maximum value of entries in the matrix.
         * @return The random matrix.
         */
        static Matrix3D random(uint16_t rows, uint16_t columns, uint16_t layers, double minValue, double maxValue);

        /**
         * @brief Randomizes the entries of the matrix to values between the specified values.
         * 
         * @param minValue The lower bound of random numbers (inclusive).
         * @param maxValue The upper bound of random numbers (exclusive).
         */
        void randomize(double minValue, double maxValue);

        /**
         * @brief Returns the tranpose of the matrix.
         *
         * @return The transposed matrix.
         */
        Matrix3D getTranspose();

        /**
         * @brief Gets the flipped matrix along the horizontal axis.
         */
        Matrix3D getFlippedHori();

        /**
         * @brief Gets the flipped matrix along the vertical axis.
         */
        Matrix3D getFlippedVert();

        /**
         * @brief Gets the flipped matrix along the horizontal and vertical axis (180 degree rotation).
         */
        Matrix3D getFlippedHoriAndVert();

        /**
         * @brief Raises the entries of a matrix to a power.
         * 
         * @param a The matrix to raise.
         * @param factor The power to raise each entry to.
         * @return The raised matrix.
         */
        static Matrix3D power(const Matrix3D& a, double factor);

        /**
         * @brief Raises the entries of a matrix to e to the orignal entry-th power.
         *
         * @param a The matrix to raise.
         * @return The raised matrix.
         */
        static Matrix3D exponential(const Matrix3D& a);

        /**
         * @brief Returns the matrix with the absolute values of entries of the inputted matrix.
         * 
         * @param a The inputted matrix to find the absolute value of.
         * @return The absolute value matrix of the original matrix.
         */
        static Matrix3D absoluteValue(const Matrix3D& a);

        /**
         * @brief Get the rows of the matrix.
         * 
         * @param length the variable to return the length of the array to.
         * @return The rows, as an array of 1xmxn matrices.
         */
        Matrix3D* getRows(uint16_t* length);

        /**
         * @brief Get the columns of the matrix.
         * 
         * @param length the variable to return the length of the array to.
         * @return The columns, as an array of mx1xn matrices.
         */
        Matrix3D* getColumns(uint16_t* length);

        /**
         * @brief Get the layers of the matrix.
         * 
         * @param length the variable to return the length of the array to.
         * @return The layers, as an array of mxnx1 matrices.
         */
        Matrix3D* getLayers(uint16_t* length);

        /**
         * @brief Flattens the entries of the matrix into a (m*n*p)x1 matrix.
         *
         * @return The vectorized 2D matrix.
         */
        Matrix2D getVectorized();

        /**
         * @brief Unflattens the entries of a 1xb matrix into an mxnxp matrix.
         *
         * @param rows The number of rows in the unflattened matrix.
         * @param cols The number of columns in the unflattened matrix.
         * @param lays The number of layers in the unflattened matrix.
         */
        void unflatten(uint16_t rows, uint16_t cols, uint16_t lays);

        /**
         * @brief Get the values of the entries as a 1-d array.
         * 
         * @return The values of the matrix entries.
         */
        double* getValues();

        /**
         * @brief Returns a submatrix bounded by the specified values.
         *        e.g. getSubmatrix(1, 2, 4, 3) returns a 3x1 matrix
         *        that includes the second, third, and fourth rows
         *        (indices 1, 2, 3, respectively) and the third column (index 2).
         * 
         * @param rowStart The index of the top-most row. Values beyond bounds of
         *                 original matrix will be defaulted to 0.0.
         * @param colStart The index of the left-most column. Values beyond bounds
         *                 of original matrix will be defaulted to 0.0.
         * @param layStart The index of the front-most layer. Values beyond bounds
         *                 of original matrix will be defaulted to 0.0.
         * @param rowEnd The index of the row below the bottom-most row. Values beyond
         *               bounds of original matrix will be defaulted to 0.0.
         * @param colEnd The index of the column to the right of the right-most row.
         *               Values beyond the bounds of the original matrix will be defauled
         *               to 0.0.
         * @param layEnd The index of the layer one layer behind the back-most row.
         *               Values beyond the bounds of the original matrix will be defauled
         *               to 0.0.
         * @return The submatrix bounded by the specified indices.
         */
        Matrix3D getSubmatrix(int rowStart, int colStart, int layStart, uint16_t rowEnd, uint16_t colEnd, uint16_t layEnd);

        /**
         * @brief Computes the sum of the entries of the matrix.
         * 
         * @return The sum of the entries of the matrix, as a double. 
         */
        double getSumOfEntries();

        /**
         * @brief Gets the index of the max entry in the matrix.
         *        index = (row index * number of columns) + column index.
         *
         * @return The index of the max entry in the matrix.
         */
        int getIndexOfMax();

        /**
         * @brief Gets the index of the minimum entry in the matrix.
         *        index = (layer index * number of columns * number of rows) + (row index * number of columns) + column index.
         *
         * @return The index of the minimum entry in the matrix.
         */
        int getIndexOfMin();

        /**
         * @brief Rescales the entries of the matrix to be between the
         *        specified low value and high value. Preserves relational
         *        values between entries (i.e. >, <, ==). Matrices with all
         *        equivalent elements will be put halfway between the low
         *        and high value.
         * @param lowValue The new low bound of the matrix.
         * @param highValue The new high bound of the matrx.
         *
         * @return The rescaled matrix.
         */
        Matrix3D getRescaled(double lowValue, double highValue);

        /**
         * @brief Rescales the entries of the matrix to be between the
         *        specified low value and high value. Preserves relational
         *        values between entries (i.e. >, <, ==). Matrices with all
         *        equivalent elements will be put halfway between the low
         *        and high value.
         * @param lowValue The new low bound of the matrix.
         * @param highValue The new high bound of the matrx.
         */
        void rescale(double lowValue, double highValue);
        
        /**
         * @brief Gets the value at the specified index.
         * 
         * @param row The row index.
         * @param column The column index.
         * @param layer The layer index.
         * @return The value at the specified index.
         */
        constexpr double get(uint16_t row, uint16_t column, uint16_t layer) const noexcept;

        /**
         * @brief Sets the value at the specified index.
         * 
         * @param row The row index.
         * @param column The column index.
         * @param layer The layer index.
         * @param value The new value.
         */
        void set(uint16_t row, uint16_t column, uint16_t layer, double value);

        /**
         * @brief Sets the values in the row of a matrix to the values
         *        of a 1xmxn matrix. Extra values will be ignored, and
         *        if m < the column count of the original matrix or if
         *        n < the layer count of the original matrix, the
         *        remaining values will be set to 0.0.
         * 
         * @param row The row index to set.
         * @param a The values to set the row to, as a 1xmxn matrix.
         */
        void setRow(uint16_t row, const Matrix3D& a);

        /**
         * @brief Sets the values in the column of a matrix to the values
         *        of a mx1xn matrix. Extra values will be ignored, and
         *        if m < the row count of the original matrix or if
         *        n < the layer count of the original matrix, the
         *        remaining values will be set to 0.0.
         * 
         * @param col The column index to set.
         * @param a The values to set the column to, as a mx1xn matrix.
         */
        void setColumn(uint16_t col, const Matrix3D& a);

        /**
         * @brief Sets the values in the layer of a matrix to the values
         *        of a mxnx1 matrix. Extra values will be ignored, and
         *        if m < the row count of the original matrix or if
         *        n < the column count of the original matrix, the
         *        remaining values will be set to 0.0.
         * 
         * @param lay The layer index to set.
         * @param a The values to set the layer to, as a mxnx1 matrix.
         */
        void setLayer(uint16_t lay, const Matrix3D& a);
        
        /**
         * @brief Returns the row count of the matrix.
         * 
         * @return The number of rows in the matrix.
         */
        constexpr int getRowCount() const noexcept;
        
        /**
         * @brief Returns the column count of the matrix.
         * 
         * @return The number of columns in the matrix.
         */
        constexpr int getColumnCount() const noexcept;
        
        /**
         * @brief Returns the layer count of the matrix.
         * 
         * @return The number of layers in the matrix.
         */
        constexpr int getLayerCount() const noexcept;

        /**
         * @brief Gets the size of the 3D matrix.
         * 
         * @return The size of the 3D matrix, as a string.
         */
        constexpr std::string getDimensions() const noexcept;

        /**
         * @brief Overrides the equals operator such that two matrices can be compared.
         *
         * @param a The matrix to be compared to the current one.
         * @return The equivalence relation result.
         */
        bool operator==(const Matrix3D& a);

        /**
         * @brief Overrides the equals operator such that a matrix can be assigned to another.
         *
         * @param a The matrix to set the current one to.
         * @return The new matrix.
         */
        Matrix3D& operator=(const Matrix3D& a);

        /**
         * @brief Returns the state of the matrix as a string.
         * 
         * @return the state of the matrix as a string.
         */
        std::string toString();

        /**
         * @brief Overrides the bitshifting operator such that the state
         *        of the object as a string is printed.
         */
        friend std::ostream& operator<<(std::ostream& os, const Matrix3D& a);
    };
}