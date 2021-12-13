#include <iostream>
#include <string>

namespace Neuranet
{
    /**
     * @brief Class representing a single-variable polynomial.
     * @author Nolan Bridges
     * @version 1.0.0
     */
    class Polynomial
    {
    private:
        /**
         * @brief The coefficients of the polynomial, from the highest degree to 0.
         *        Contains 0.0s for powers that aren't in the polynomial.
         */
        double* coefficients;

        /**
         * @brief The highest degree (exponent) in the polynomial.
         */
        uint16_t degree;

        /**
         * @brief Multiplies two polynomials together.
         * 
         * @param a The first polynomial.
         * @param b The second polynomial.
         * @return The product. 
         */
        static Polynomial multiply(const Polynomial& a, const Polynomial& b);
        
    public:
        /**
         * @brief Construct a new Polynomial object
         * 
         * @param coefficients The coefficients of the polynomial, from the highest degree to 0.
         *                     Contains 0.0s for powers that aren't in the polynomial.
         * @param degree The highest degree (exponent) in the polynomial.
         */
        Polynomial(double* coefficients, uint16_t degree);

        /**
         * @brief Overrides the multiplication operator such that two polynomials can be multiplied.
         * 
         * @param a The polynomial to multiply to the current one.
         * @return The product polynomial. 
         */
        Polynomial operator*(const Polynomial& a);

        /**
         * @brief Overrides the bitshifting operator such that the state
         *        of the object as a string is printed.
         */
        friend std::ostream& operator<<(std::ostream& os, const Polynomial& a);

    };
}