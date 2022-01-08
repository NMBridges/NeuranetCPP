#pragma once

#include <ctime>
#include <chrono>
#include <format>
#include <fstream>
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

        if (values != nullptr)
        {
            for (int index = 0; index < this->rowCount * this->colCount; index += 1)
            {
                this->values[index] = values[index];
            }
        }
        else
        {
            for (int index = 0; index < this->rowCount * this->colCount; index += 1)
            {
                this->values[index] = 0.0;
            }
        }
    }

    Matrix2D::Matrix2D(uint16_t rows, uint16_t columns, double value)
    {
        this->dimensionCount = 2;

        this->rowCount = rows;
        this->colCount = columns;

        this->values = new double[rows * columns];

        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            this->values[index] = value;
        }
    }

    Matrix2D::~Matrix2D()
    {
        delete[] this->values;
    }

    Matrix2D Matrix2D::createUninitialized(uint16_t rows, uint16_t columns)
    {
        Matrix2D uninitialized = Matrix2D();

        uninitialized.rowCount = rows;
        uninitialized.colCount = columns;

        delete[] uninitialized.values;
        uninitialized.values = nullptr;
        uninitialized.values = new double[rows * columns];

        return uninitialized;
    }

    void Matrix2D::initializeOpenCL()
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        Matrix2D::cl_platform = platforms.front();

        std::vector<cl::Device> devices;
        Matrix2D::cl_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        Matrix2D::cl_device = devices.front();

        Matrix2D::cl_context = cl::Context(devices.front());

        std::string m2dstring = "__kernel void plus1(__global const double* a, __global const double* b, __global const int* rows, __global const int* columns, __global double* c)"
                                "{"
                                    "int arrLength = *rows * *columns;"
                                    "for (int gid = get_global_id(0); gid < arrLength; gid += 1024)"
                                    "{"
                                        "c[gid] = a[gid] + b[gid];"
                                    "}"
                                "}"
                                "__kernel void plus2(__global const double* a, __global const double* b, __global const int* rows, __global const int* columns, __global double* c)"
                                "{"
                                    "int arrLength = *rows * *columns;"
                                    "for (int gid = get_global_id(0); gid < arrLength; gid += 1024)"
                                    "{"
                                        "c[gid] = a[gid] + b[0];"
                                    "}"
                                "}"
                                "__kernel void plus3(__global double* a, __global const double* b, __global const int* rows, __global const int* columns)"
                                "{"
                                    "int arrLength = *rows * *columns;"
                                    "for (int gid = get_global_id(0); gid < arrLength; gid += 1024)"
                                    "{"
                                        "a[gid] += b[gid];"
                                    "}"
                                "}"
                                "__kernel void multiply1(__global const double* a, __global const int* aRows, __global const int* aCols, __global const double* b, __global const int* bRows, __global const int* bCols, __global double* c)"
                                "{"
                                    "int arrLength = *aRows * *bCols;"
                                    "for (int gid = get_global_id(0); gid < arrLength; gid += 128)"
                                    "{"
                                        "int rowOffset = gid / *bCols * *aCols;"
                                        "int col = gid % *bCols;"
                                        "double dotProduct = 0.0;"
                                        "for (int i = 0; i < *aCols; i += 1)"
                                        "{"
                                            "dotProduct += a[rowOffset + i] * b[i * *bCols + col];"
                                        "}"
                                        "c[gid] = dotProduct;"
                                    "}"
                                "}";

        cl::Program::Sources sources = cl::Program::Sources(1, std::make_pair(m2dstring.c_str(), m2dstring.length() + 1));

        Matrix2D::cl_program = cl::Program(Matrix2D::cl_context, sources);

        cl_int err = Matrix2D::cl_program.build("-cl-std=CL1.2");

        std::cout << Matrix2D::cl_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(Matrix2D::cl_device, &err);

        bool hadError = false;
        cl_int lastErr;

        Matrix2D::cl_plus1Kernel = cl::Kernel(Matrix2D::cl_program, "plus1", &err);
        if (err != 0)
        {
            hadError = true;
            lastErr = err;
        }
        Matrix2D::cl_plus2Kernel = cl::Kernel(Matrix2D::cl_program, "plus2", &err);
        if (err != 0)
        {
            hadError = true;
            lastErr = err;
        }
        Matrix2D::cl_plus3Kernel = cl::Kernel(Matrix2D::cl_program, "plus3", &err);
        if (err != 0)
        {
            hadError = true;
            lastErr = err;
        }
        Matrix2D::cl_multiply1Kernel = cl::Kernel(Matrix2D::cl_program, "multiply1", &err);
        if (err != 0)
        {
            hadError = true;
            lastErr = err;
        }

        if (!hadError)
        {
            std::cout << "OpenCL initialized successfully." << std::endl;
        }
        else
        {
            std::cout << "Error " << lastErr << std::endl;
        }
    }

    void Matrix2D::zero()
    {
        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            this->values[index] = 0.0;
        }
    }

    void Matrix2D::one()
    {
        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            this->values[index] = 1.0;
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

        Matrix2D summedMatrix = createUninitialized(a.rowCount, a.colCount);

        /** I have found that this process is faster on the CPU for some reason. */
        //cl_int err;
        //cl::Buffer memBuf1(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(double) * this->rowCount * this->colCount, this->values, &err);
        ////std::cout << "1 " << err << std::endl;
        //cl::Buffer memBuf2(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(double) * this->rowCount * this->colCount, a.values, &err);
        ////std::cout << "2 " << err << std::endl;
        //
        //cl_int rC = a.rowCount;
        //cl::Buffer memBuf3(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &rC, &err);
        ////std::cout << "3 " << err << std::endl;
        //
        //cl_int cC = a.colCount;
        //cl::Buffer memBuf4(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &cC, &err);
        ////std::cout << "4 " << err << std::endl;
        //
        //cl::Buffer memBuf5(Matrix2D::cl_context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * this->rowCount * this->colCount, summedMatrix.values, &err);
        ////std::cout << "5 " << err << std::endl;
        //err = Matrix2D::cl_plus1Kernel.setArg(0, memBuf1());
        ////std::cout << "6 " << err << std::endl;
        //err = Matrix2D::cl_plus1Kernel.setArg(1, memBuf2());
        ////std::cout << "7 " << err << std::endl;
        //err = Matrix2D::cl_plus1Kernel.setArg(2, memBuf3());
        ////std::cout << "8 " << err << std::endl;
        //err = Matrix2D::cl_plus1Kernel.setArg(3, memBuf4());
        ////std::cout << "9 " << err << std::endl;
        //err = Matrix2D::cl_plus1Kernel.setArg(4, memBuf5());
        ////std::cout << "10 " << err << std::endl;
        //
        //cl::CommandQueue queue(Matrix2D::cl_context, Matrix2D::cl_device);
        //
        //constexpr int magicInt = 4096;
        //int local = min(magicInt, this->rowCount * this->colCount);
        //int global = (int)ceil((this->rowCount * this->colCount) * 1.0 / local) * local;
        //
        //queue.enqueueNDRangeKernel(Matrix2D::cl_plus1Kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(local));
        //queue.finish();
        //queue.enqueueReadBuffer(memBuf5, CL_FALSE, 0, sizeof(double) * this->rowCount * this->colCount, summedMatrix.values);
        //queue.finish();
        
        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            summedMatrix.values[index] = this->values[index] + a.values[index];
        }

        return summedMatrix;
    }

    Matrix2D Matrix2D::operator+(double a)
    {
        Matrix2D summedMatrix = createUninitialized(this->rowCount, this->colCount);

        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            summedMatrix.values[index] = this->values[index] + a;
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

    Matrix2D Matrix2D::operator-(double a)
    {
        Matrix2D differenceMatrix(this->rowCount, this->colCount);

        for (int index = 0; index < this->rowCount * this->colCount; index += 1)
        {
            differenceMatrix.values[index] = this->values[index] - a;
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

        Matrix2D productMatrix = Matrix2D::createUninitialized(this->rowCount, a.colCount);

        //cl_int err;
        //cl::Buffer memBuf1(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(double) * this->rowCount * this->colCount, this->values, &err);
        ////std::cout << "1 " << err << std::endl;
        //
        //cl_int rC1 = this->rowCount;
        //cl::Buffer memBuf2(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &rC1, &err);
        ////std::cout << "2 " << err << std::endl;
        //
        //cl_int cC1 = this->colCount;
        //cl::Buffer memBuf3(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &cC1, &err);
        ////std::cout << "3 " << err << std::endl;
        //
        //cl::Buffer memBuf4(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(double) * a.rowCount * a.colCount, a.values, &err);
        ////std::cout << "4 " << err << std::endl;
        //
        //cl_int rC2 = a.rowCount;
        //cl::Buffer memBuf5(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &rC2, &err);
        ////std::cout << "5 " << err << std::endl;
        //
        //cl_int cC2 = a.colCount;
        //cl::Buffer memBuf6(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &cC2, &err);
        ////std::cout << "6 " << err << std::endl;
        //
        //cl::Buffer memBuf7(Matrix2D::cl_context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(double) * this->rowCount * a.colCount, productMatrix.values, &err);
        ////std::cout << "7 " << err << std::endl;
        //
        //err = Matrix2D::cl_multiply1Kernel.setArg(0, memBuf1());
        ////std::cout << "8 " << err << std::endl;
        //err = Matrix2D::cl_multiply1Kernel.setArg(1, memBuf2());
        ////std::cout << "9 " << err << std::endl;
        //err = Matrix2D::cl_multiply1Kernel.setArg(2, memBuf3());
        ////std::cout << "10 " << err << std::endl;
        //err = Matrix2D::cl_multiply1Kernel.setArg(3, memBuf4());
        ////std::cout << "11 " << err << std::endl;
        //err = Matrix2D::cl_multiply1Kernel.setArg(4, memBuf5());
        ////std::cout << "12 " << err << std::endl;
        //err = Matrix2D::cl_multiply1Kernel.setArg(5, memBuf6());
        ////std::cout << "13 " << err << std::endl;
        //err = Matrix2D::cl_multiply1Kernel.setArg(6, memBuf7());
        ////std::cout << "14 " << err << std::endl;
        //
        //cl::CommandQueue queue(Matrix2D::cl_context, Matrix2D::cl_device);
        //
        //constexpr int magicInt = 4096;
        //int local = min(magicInt, this->rowCount * a.colCount);
        //int global = (int) ceil((this->rowCount * a.colCount) * 1.0 / local) * local;
        //
        //queue.enqueueNDRangeKernel(Matrix2D::cl_multiply1Kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(local));
        //queue.finish();
        //queue.enqueueReadBuffer(memBuf7, CL_FALSE, 0, sizeof(double) * this->rowCount * a.colCount, productMatrix.values);
        //queue.finish();

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
        this->values = new double[this->rowCount * this->colCount];
        
        //cl_int err;
        //cl::Buffer memBuf1(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(double) * previousMatrix.rowCount * previousMatrix.colCount, previousMatrix.values, &err);
        ////std::cout << "1 " << err << std::endl;
        //
        //cl_int rC1 = previousMatrix.rowCount;
        //cl::Buffer memBuf2(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &rC1, &err);
        ////std::cout << "2 " << err << std::endl;
        //
        //cl_int cC1 = previousMatrix.colCount;
        //cl::Buffer memBuf3(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &cC1, &err);
        ////std::cout << "3 " << err << std::endl;
        //
        //cl::Buffer memBuf4(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(double) * a.rowCount * a.colCount, a.values, &err);
        ////std::cout << "4 " << err << std::endl;
        //
        //cl_int rC2 = a.rowCount;
        //cl::Buffer memBuf5(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &rC2, &err);
        ////std::cout << "5 " << err << std::endl;
        //
        //cl_int cC2 = a.colCount;
        //cl::Buffer memBuf6(Matrix2D::cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &cC2, &err);
        ////std::cout << "6 " << err << std::endl;
        //
        //cl::Buffer memBuf7(Matrix2D::cl_context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(double) * this->rowCount * this->colCount, this->values, &err);
        ////std::cout << "7 " << err << std::endl;
        //
        //err = Matrix2D::cl_multiply1Kernel.setArg(0, memBuf1());
        ////std::cout << "8 " << err << std::endl;
        //err = Matrix2D::cl_multiply1Kernel.setArg(1, memBuf2());
        ////std::cout << "9 " << err << std::endl;
        //err = Matrix2D::cl_multiply1Kernel.setArg(2, memBuf3());
        ////std::cout << "10 " << err << std::endl;
        //err = Matrix2D::cl_multiply1Kernel.setArg(3, memBuf4());
        ////std::cout << "11 " << err << std::endl;
        //err = Matrix2D::cl_multiply1Kernel.setArg(4, memBuf5());
        ////std::cout << "12 " << err << std::endl;
        //err = Matrix2D::cl_multiply1Kernel.setArg(5, memBuf6());
        ////std::cout << "13 " << err << std::endl;
        //err = Matrix2D::cl_multiply1Kernel.setArg(6, memBuf7());
        ////std::cout << "14 " << err << std::endl;
        //
        //cl::CommandQueue queue(Matrix2D::cl_context, Matrix2D::cl_device);
        //
        //constexpr int magicInt = 4096;
        //int local = min(magicInt, this->rowCount * this->colCount);
        //int global = (int)ceil((this->rowCount * this->colCount) * 1.0 / local) * local;
        //
        //queue.enqueueNDRangeKernel(Matrix2D::cl_multiply1Kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(local));
        //queue.finish();
        //queue.enqueueReadBuffer(memBuf7, CL_FALSE, 0, sizeof(double) * this->rowCount * this->colCount, this->values);
        //queue.finish();

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

    Matrix2D Matrix2D::hadamardProduct(const Matrix2D& a, const Matrix2D& b)
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

    Matrix2D Matrix2D::hadamardQuotient(const Matrix2D& a, const Matrix2D& b)
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
            productMatrix.values[index] = a.values[index] / b.values[index];
        }

        return productMatrix;
    }

    Matrix2D Matrix2D::random(uint16_t rows, uint16_t columns, double minValue, double maxValue)
    {
        Matrix2D::randomSeed += 1;
        std::srand(static_cast<std::time_t>(Matrix2D::randomSeed) * std::time(NULL) * std::chrono::steady_clock::now().time_since_epoch().count() * GetTickCount64());
        
        Matrix2D randomMatrix = Matrix2D::createUninitialized(rows, columns);
        
        for (int index = 0; index < rows * columns; index += 1)
        {
            double percent = (double) std::rand() / RAND_MAX;
            randomMatrix.values[index] = minValue + percent * (maxValue - minValue);
        }
        
        return randomMatrix;
    }

    void Matrix2D::randomize(double minValue, double maxValue)
    {
        Matrix2D::randomSeed += 1;
        std::srand(static_cast<std::time_t>(Matrix2D::randomSeed) * std::time(NULL) * std::chrono::steady_clock::now().time_since_epoch().count() * GetTickCount64());

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

    Matrix2D Matrix2D::exponential(const Matrix2D& a)
    {
        Matrix2D powerMatrix(a.rowCount, a.colCount);

        for (int index = 0; index < a.rowCount * a.colCount; index += 1)
        {
            powerMatrix.values[index] = exp(a.values[index]);
        }

        return powerMatrix;
    }

    Matrix2D Matrix2D::logarithmic(const Matrix2D& a)
    {
        Matrix2D powerMatrix(a.rowCount, a.colCount);

        for (int index = 0; index < a.rowCount * a.colCount; index += 1)
        {
            powerMatrix.values[index] = log(a.values[index]);
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

    Matrix2D Matrix2D::getVectorized()
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