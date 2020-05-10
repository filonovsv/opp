#pragma once
#include <vector>
using namespace std;

double operator*(vector<double> lhs, vector<double> rhs);
vector<double> operator*(double lhs, vector<double> rhs);
vector<double> operator-(vector<double> lhs, vector<double> rhs);
vector<double> operator+(vector<double> lhs, vector<double> rhs);
double norma(vector<double> a);
