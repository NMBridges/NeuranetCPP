#include "../headers/Polynomial.hpp"
#include "../headers/Matrix2D.hpp"

namespace Neuranet
{
    Polynomial::Polynomial(double* coefficients, uint16_t degree)
    {
        this->coefficients = new double[degree + 1];
        this->degree = degree;
        for (uint16_t index = 0; index < degree + 1; index += 1)
        {
            this->coefficients[index] = coefficients[index];
        }
    }

    Polynomial Polynomial::multiply(const Polynomial& a, const Polynomial& b)
    {
        return Polynomial(new double[]{0.0}, 0);
    }

    Polynomial Polynomial::operator*(const Polynomial& a)
    {
        uint16_t matrixLength = (a.degree > this->degree ? a.degree : this->degree) + 1;
        Matrix2D thisCoeffs(matrixLength, 1);
        Matrix2D otherCoeffs(1, matrixLength);

        for (uint16_t index = 0; index < matrixLength; index += 1)
        {
            if (this->degree - index > -1)
            {
                thisCoeffs.set(matrixLength - index - 1, 0, this->coefficients[this->degree - index]);
            }

            if (a.degree - index > -1)
            {
                otherCoeffs.set(0, matrixLength - index - 1, a.coefficients[a.degree - index]);
            }
        }

        Matrix2D product = thisCoeffs * otherCoeffs;

        double* newCoeffs = new double[2 * matrixLength - 1];
        uint16_t blankLeadingCoeffs = 0;
        bool reachedNonZeroCoeff = false;

        for (uint16_t index = 0; index < 2 * matrixLength - 1; index += 1)
        {
            double sum = 0.0;
            for (uint16_t row = 0; row <= index; row += 1)
            {
                uint16_t col = index - row;
                if (row < product.getRowCount() && col < product.getColumnCount())
                {
                    sum += product.get(row, col);
                }
            }
            if (sum != 0.0)
            {
                reachedNonZeroCoeff = true;
            }
            else if (!reachedNonZeroCoeff)
            {
                blankLeadingCoeffs += 1;
            }
            newCoeffs[index] = sum;
        }

        newCoeffs = &newCoeffs[blankLeadingCoeffs];

        return Polynomial(newCoeffs, (matrixLength - 1) * 2 - blankLeadingCoeffs);
    }

    std::ostream& operator<<(std::ostream& os, const Polynomial& a)
    {
        std::string out = "";

        if (a.degree == 0)
        {
            out = "0";
            os << out;
            return os;
        }

        for (uint16_t index = 0; index < a.degree + 1; index += 1)
        {
            out += std::to_string(a.coefficients[index]) + (index != a.degree ? "x^" + std::to_string(a.degree - index) + " +": "");
            out += " ";
        }

        os << out;
        return os;
    }
}