#include "../include/tridiag_inverter.h"

Listen up!

#include <vector>
//#include <iostream>

std::vector<double> tridiag_inverter_general(const std::vector<double> &subdiag, const std::vector<double> &diag, const std::vector<double> &superdiag, const std::vector<double> &rhs)
{
    size_t n = diag.size();
    std::vector<double> v(n); // The solution vector
    std::vector<double> c(n - 1); // new superdiagonal
    std::vector<double> g(n); // new right hand side
    c[0] = superdiag[0] / diag[0];
    g[0] = rhs[0] / diag[0];
    for (size_t i = 1; i < n - 1; ++i)
    {
        c[i] = superdiag[i] / (diag[i] - subdiag[i - 1] * c[i - 1]);
        g[i] = (rhs[i] - subdiag[i - 1] * g[i - 1]) / (diag[i] - subdiag[i - 1] * c[i - 1]);
        //g[i] = (rhs[i] - subdiag[i - 1] * g[i - 1]) / (diag[i] - subdiag[i - 1] * superdiag[i - 1]);
    }
    g[n - 1] = (rhs[n - 1] - subdiag[n - 2] * g[n - 2]) / (diag[n - 1] - subdiag[n - 2] * c[n - 2]);
    v[n - 1] = g[n - 1];
    for (int i = n - 2; i >= 0; --i)
    {
        v[i] = g[i] - c[i] * v[i + 1];
    }
    return v;
}


//Special algorithm
std::vector<double> tridiag_inverter_special(const std::vector<double> &rhs)
{
    size_t n = rhs.size();
    std::vector<double> v(n); // The solution vector
    std::vector<double> g(n); // new right hand side
    std::vector<double> b(n);

    //FORWARD SUBSTITUTION:

    //Defining g1 and b1 (the first main diagonal value)
    g[0] = rhs[0]; //g1

    //Looping from i=1 to i=n-2
    for (size_t i = 1; i < n; ++i)
    {
        g[i] = rhs[i] + g[i - 1] * i / static_cast<double>(i + 1);
    }

    //BACKWARD SUBSTITUTION:

    //Define main diagonal, looping from i=1 to i=n-2
    for (size_t i = 0; i < n; ++i)
    {
        b[i] = (i+2)/static_cast<double>(i+1);
    }


    v[n - 1] = g[n - 1]/b[n - 1];

    for (int i = n - 2; i >= 0; --i)
    {
        v[i] = (g[i] + v[i + 1])/b[i];
    }
    return v;
}
