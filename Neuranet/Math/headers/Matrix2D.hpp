#pragma once

#include "CL/cl.hpp"
#include "Matrix.hpp"

namespace Neuranet
{
    /**
     * @brief Class representing a 2-dimensional matrix
     *        of any size.
     * @author Nolan Bridges
     * @version 1.0.0
     */
    class Matrix2D : public Matrix
    {
    private:
        /**
         * @brief The seed to use for calling random functions.
         */
        static int randomSeed;

    public:
        /**
         * @brief OpenCL / GPU acceleration variables.
         */
        static inline cl::Platform cl_platform;
        static inline cl::Device cl_device;
        static inline cl::Context cl_context;
        static inline cl::Program cl_program;
        static inline cl::Kernel cl_plus1Kernel;
        static inline cl::Kernel cl_plus2Kernel;
        static inline cl::Kernel cl_plus3Kernel;
        static inline cl::Kernel cl_plus4Kernel;
        static inline cl::Kernel cl_multiply1Kernel;

    public:
        /**
         * @brief Construct a new Matrix2D object with size 0.
         */
        Matrix2D() : Matrix2D(0, 0, nullptr) {};
        
        /**
         * @brief Construct a new Matrix2D object with the specified dimensions and values.
         */
        Matrix2D(uint16_t rows, uint16_t columns, double* values);

        /**
         * @brief Construct a new Matrix2D object with the specified dimensions, filled entirely by
         *        by the specified value.
         */
        Matrix2D(uint16_t rows, uint16_t columns, double value);
        
        /**
         * @brief Construct a new Matrix2D object with the specified dimensions (default values = 0.0).
         */
        Matrix2D(uint16_t rows, uint16_t columns) : Matrix2D(rows, columns, nullptr) {};
        
        /**
         * @brief Construct a new Matrix2D object from another Matrix2D object.
         */
        Matrix2D(const Matrix2D& a) : Matrix2D(a.rowCount, a.colCount, a.values) {};

        /**
         * @brief Destroys the Matrix2D object.
         */
        ~Matrix2D();
    
    protected:
        /**
         * @brief Creates a Matrix2D object with the specified dimensions and no ininitialized values.
         */
        static Matrix2D createUninitialized(uint16_t rows, uint16_t columns);

    public:

        /**
         * @brief Initializes variables used for GPU acceleration.
         */
        static void initializeOpenCL();

        /**
         * @brief Zeroes the entries of the matrix.
         */
        void zero();

        /**
         * @brief Sets the entries of the matrix to one.
         */
        void one();

        /**
         * @brief Overrides the addition operator such that two matrices can be added.
         * 
         * @param a The matrix to add to the current one.
         * @return The summed matrix. 
         */
        Matrix2D operator+(const Matrix2D& a);

        /**
         * @brief Overrides the addition operator such that a matrix can be incremented entry-wise by a scalar.
         *
         * @param a The value to add to the current matrix entries.
         * @return The summed matrix.
         */
        Matrix2D operator+(double a);

        /**
         * @brief Overrides the addition operator such that two matrices can be added.
         * 
         * @param a The matrix to add to the current one.
         * @return The summed matrix. 
         */
        Matrix2D& operator+=(const Matrix2D& a);

        /**
         * @brief Overrides the subtraction operator such that two matrices can be subtracted.
         * 
         * @param a The matrix to subtract from the current one.
         * @return The difference matrix. 
         */
        Matrix2D operator-(const Matrix2D& a);

        /**
         * @brief Overrides the addition operator such that a matrix can be incremented entry-wise by a scalar.
         *
         * @param a The value to add to the current matrix entries.
         * @return The summed matrix.
         */
        Matrix2D operator-(double a);

        /**
         * @brief Overrides the subtraction operator such that two matrices can be subtracted.
         * 
         * @param a The matrix to subtract from the current one.
         * @return The difference matrix. 
         */
        Matrix2D& operator-=(const Matrix2D& a);

        /**
         * @brief Overrides the multiplication operator such that two matrices can be multiplied.
         * 
         * @param a The matrix to multiply to the current one (from the right).
         * @return The product matrix. 
         */
        Matrix2D operator*(const Matrix2D& a);

        /**
         * @brief Overrides the multiplication operator such that two matrices can be multiplied.
         * 
         * @param a The matrix to multiply to the current one (from the right).
         * @return The product matrix. 
         */
        Matrix2D& operator*=(const Matrix2D& a);

        /**
         * @brief Overrides the multiplication operator such a matrix can be individually scaled.
         * 
         * @param factor The factor to scale the matrix entries by.
         * @return The scaled matrix. 
         */
        Matrix2D operator*(double factor);

        /**
         * @brief Overrides the multiplication operator such a matrix can be individually scaled.
         * 
         * @param factor The factor to scale the matrix entries by.
         * @return The scaled matrix. 
         */
        Matrix2D& operator*=(double factor);

        /**
         * @brief Multiply two matrices in an element-wise fashion.
         * 
         * @param a The first matrix.
         * @param b The second matrix.
         * @return The hadamard product matrix. 
         */
        static Matrix2D hadamardProduct(const Matrix2D& a, const Matrix2D& b);

        /**
         * @brief Overrides the division operator such that a matrix can be individually divided.
         * 
         * @param factor The factor to divide the matrix entries by.
         * @return The 'quotient' matrix. 
         */
        Matrix2D operator/(double factor);

        /**
         * @brief Overrides the division operator such that a matrix can be individually divided.
         * 
         * @param factor The factor to divide the matrix entries by.
         * @return The 'quotient' matrix. 
         */
        Matrix2D& operator/=(double factor);

        /**
         * @brief Divides two matrices in an element-wise fashion.
         *
         * @param a The first matrix.
         * @param b The second matrix.
         * @return The hadamard quotient matrix.
         */
        static Matrix2D hadamardQuotient(const Matrix2D& a, const Matrix2D& b);

        /**
         * @brief Creates an array of specified dimensions with random values between
         *        the two specified values.
         * 
         * @param rows The number of rows in the matrix.
         * @param columns The number of columns in the matrix.
         * @param minValue The minimum value of entries in the matrix.
         * @param maxValue The maximum value of entries in the matrix.
         * @return The random matrix.
         */
        static Matrix2D random(uint16_t rows, uint16_t columns, double minValue, double maxValue);

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
        Matrix2D getTranspose();

        /**
         * @brief Raises the entries of a matrix to a power.
         *
         * @param a The matrix to raise.
         * @param factor The power to raise each entry to.
         * @return The raised matrix.
         */
        static Matrix2D power(const Matrix2D& a, double factor);

        /**
         * @brief Raises the entries of a matrix to e to the orignal entry-th power.
         *
         * @param a The matrix to raise.
         * @return The raised matrix.
         */
        static Matrix2D exponential(const Matrix2D& a);

        /**
         * @brief Finds the entry-wise log of the matrix.
         *
         * @param a The matrix of which to find the log of each entry.
         * @return The logged matrix.
         */
        static Matrix2D logarithmic(const Matrix2D& a);

        /**
         * @brief Returns the matrix with the absolute values of entries of the inputted matrix.
         * 
         * @param a The inputted matrix to find the absolute value of.
         * @return The absolute value matrix of the original matrix.
         */
        static Matrix2D absoluteValue(const Matrix2D& a);

        /**
         * @brief Get the rows of the matrix.
         * 
         * @param length the variable to return the length of the array to.
         * @return The rows, as an array of 1xm matrices.
         */
        Matrix2D* getRows(uint16_t* length);

        /**
         * @brief Get the columns of the matrix.
         * 
         * @param length the variable to return the length of the array to.
         * @return The columns, as an array of mx1 matrices.
         */
        Matrix2D* getColumns(uint16_t* length);

        /**
         * @brief Flattens the entries of the matrix into a (m*n*p)x1 matrix.
         *
         * @return The vectorized 2D matrix.
         */
        Matrix2D getVectorized();

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
         * @param rowEnd The index of the row below the bottom-most row. Values beyond
         *               bounds of original matrix will be defaulted to 0.0.
         * @param colEnd The index of the column to the right of the right-most row.
         *               Values beyond the bounds of the original matrix will be defauled
         *               to 0.0.
         * @return The submatrix bounded by the specified indices.
         */
        Matrix2D getSubmatrix(int rowStart, int colStart, uint16_t rowEnd, uint16_t colEnd);

        /**
         * @brief Gets the minor of a matrix at a given index.
         * 
         * @param row The index of the row.
         * @param col The index of the column.
         * @return The minor of the matrix at the given index. 
         */
        Matrix2D getMinor(uint16_t row, uint16_t col);

        /**
         * @brief Gets the matrix of cofactors of the matrix.
         * 
         * @return The matrix of cofactors.
         */
        Matrix2D getCofactors();

        /**
         * @brief Gets the adjoint of the matrix.
         * 
         * @return The adjoint of the matrix. 
         */
        Matrix2D getAdjoint();

        /**
         * @brief Computes the determinant of the matrix.
         * 
         * @return The determinant of the matrix, as a double. 
         */
        double getDeterminant();

        /**
         * @brief Gets the inverse of the matrix.
         * 
         * @return The inverse of the matrix.
         */
        Matrix2D getInverse();

        /**
         * @brief Computes the sum of the entries of the matrix.
         * 
         * @return The sum of the entries of the matrix, as a double. 
         */
        double getSumOfEntries();

        /**
         * @brief Computes the trace of the matrix.
         * 
         * @return The trace of the matrix, as a double. 
         */
        double getTrace();

        /**
         * @brief Gets the index of the max entry in the matrix.
         *        index = (row index * number of columns) + column index.
         *
         * @return The index of the max entry in the matrix.
         */
        int getIndexOfMax();

        /**
         * @brief Gets the index of the minimum entry in the matrix.
         *        index = (row index * number of columns) + column index.
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
        Matrix2D getRescaled(double lowValue, double highValue);

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
         * @return The value at the specified index.
         */
        constexpr double get(uint16_t row, uint16_t column) const noexcept;

        /**
         * @brief Sets the value at the specified index.
         * 
         * @param row The row index.
         * @param column The column index.
         * @param value The new value.
         */
        void set(uint16_t row, uint16_t column, double value);

        /**
         * @brief Sets the values in the row of a matrix to the values
         *        of a 1xm matrix. Extra values will be ignored, and
         *        if m < the column count of the original matrix, the
         *        remaining values will be set to 0.0.
         * 
         * @param row The row index to set.
         * @param a The values to set the row to, as a 1xm matrix.
         */
        void setRow(uint16_t row, const Matrix2D& a);

        /**
         * @brief Sets the values in the column of a matrix to the values
         *        of a mx1 matrix. Extra values will be ignored, and
         *        if m < the row count of the original matrix, the
         *        remaining values will be set to 0.0.
         * 
         * @param col The column index to set.
         * @param a The values to set the column to, as a mx1 matrix.
         */
        void setColumn(uint16_t col, const Matrix2D& a);
        
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
         * @brief Gets the size of the 2D matrix.
         * 
         * @return The size of the 2D matrix, as a string.
         */
        constexpr std::string getDimensions() const noexcept;

        /**
         * @brief Overrides the equals operator such that two matrices can be compared.
         * 
         * @param a The matrix to be compared to the current one.
         * @return The equivalence relation result. 
         */
        bool operator==(const Matrix2D& a);

        /**
         * @brief Overrides the equals operator such that a matrix can be assigned to another.
         *
         * @param a The matrix to set the current one to.
         * @return The new matrix.
         */
        Matrix2D& operator=(const Matrix2D& a);

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
        friend std::ostream& operator<<(std::ostream& os, const Matrix2D& a);
    };
}