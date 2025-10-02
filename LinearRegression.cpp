#include "LinearRegression.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>

using namespace std;

// Dataset Implementation
Dataset::Dataset() : x_label("X"), y_label("Y") {}

void Dataset::loadFromCSV(const string& filename) {
    x_values.clear();
    y_values.clear();
    
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }
    
    string line;
    // Skip header if exists
    if (getline(file, line)) {
        istringstream header_stream(line);
        string x_header, y_header;
        if (getline(header_stream, x_header, ',') && 
            getline(header_stream, y_header, ',')) {
            x_label = x_header;
            y_label = y_header;
        } else {
            // Reset to read first line as data
            file.clear();
            file.seekg(0);
        }
    }
    
    while (getline(file, line)) {
        istringstream ss(line);
        string x_str, y_str;
        
        if (getline(ss, x_str, ',') && getline(ss, y_str, ',')) {
            try {
                double x = stod(x_str);
                double y = stod(y_str);
                x_values.push_back(x);
                y_values.push_back(y);
            } catch (const exception& e) {
                cerr << "Warning: Invalid data in line: " << line << endl;
            }
        }
    }
    
    if (x_values.empty()) {
        throw runtime_error("No valid data found in file: " + filename);
    }
}

void Dataset::addDataPoint(double x, double y) {
    x_values.push_back(x);
    y_values.push_back(y);
}

const vector<double>& Dataset::getXValues() const {
    return x_values;
}

const vector<double>& Dataset::getYValues() const {
    return y_values;
}

size_t Dataset::getSize() const {
    return x_values.size();
}

void Dataset::setLabels(const string& x_label, const string& y_label) {
    this->x_label = x_label;
    this->y_label = y_label;
}

string Dataset::getXLabel() const {
    return x_label;
}

string Dataset::getYLabel() const {
    return y_label;
}

void Dataset::displaySummary() const {
    cout << "\n=== Dataset Summary ===" << endl;
    cout << "Size: " << getSize() << " data points" << endl;
    cout << "X Label: " << x_label << endl;
    cout << "Y Label: " << y_label << endl;
    
    if (!x_values.empty()) {
        auto x_minmax = minmax_element(x_values.begin(), x_values.end());
        auto y_minmax = minmax_element(y_values.begin(), y_values.end());
        
        cout << "X Range: [" << *x_minmax.first << ", " << *x_minmax.second << "]" << endl;
        cout << "Y Range: [" << *y_minmax.first << ", " << y_minmax.second << "]" << endl;
    }
}

// RegressionModel Implementation
RegressionModel::RegressionModel() : slope(0), intercept(0), mse(0) {}

double RegressionModel::predict(double x) const {
    return slope * x + intercept;
}

double RegressionModel::calculateMSE(const Dataset& dataset) const {
    const auto& x_vals = dataset.getXValues();
    const auto& y_vals = dataset.getYValues();
    
    if (x_vals.size() != y_vals.size() || x_vals.empty()) {
        return 0.0;
    }
    
    double sum_squared_errors = 0.0;
    for (size_t i = 0; i < x_vals.size(); ++i) {
        double prediction = predict(x_vals[i]);
        double error = y_vals[i] - prediction;
        sum_squared_errors += error * error;
    }
    
    return sum_squared_errors / x_vals.size();
}

double RegressionModel::getSlope() const {
    return slope;
}

double RegressionModel::getIntercept() const {
    return intercept;
}

double RegressionModel::getMSE() const {
    return mse;
}

string RegressionModel::getEquation() const {
    stringstream ss;
    ss << fixed << setprecision(4);
    ss << "y = " << slope << " * x + " << intercept;
    return ss.str();
}

void RegressionModel::displayResults() const {
    cout << "\n=== Regression Results ===" << endl;
    cout << "Equation: " << getEquation() << endl;
    cout << "Slope: " << slope << endl;
    cout << "Intercept: " << intercept << endl;
    cout << "Mean Squared Error: " << mse << endl;
}

// GradientDescentModel Implementation
GradientDescentModel::GradientDescentModel(double lr, int max_iter, double tol) 
    : learning_rate(lr), max_iterations(max_iter), tolerance(tol) {}

void GradientDescentModel::train(const Dataset& dataset) {
    const auto& x_vals = dataset.getXValues();
    const auto& y_vals = dataset.getYValues();
    
    if (x_vals.empty()) {
        throw runtime_error("Dataset is empty");
    }
    
    // Initialize parameters
    slope = 0.0;
    intercept = 0.0;
    
    int n = x_vals.size();
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        double slope_gradient = 0.0;
        double intercept_gradient = 0.0;
        
        // Calculate gradients
        for (int i = 0; i < n; ++i) {
            double prediction = slope * x_vals[i] + intercept;
            double error = prediction - y_vals[i];
            
            slope_gradient += (2.0 / n) * error * x_vals[i];
            intercept_gradient += (2.0 / n) * error;
        }
        
        // Update parameters
        double new_slope = slope - learning_rate * slope_gradient;
        double new_intercept = intercept - learning_rate * intercept_gradient;
        
        // Check for convergence
        if (abs(new_slope - slope) < tolerance && abs(new_intercept - intercept) < tolerance) {
            break;
        }
        
        slope = new_slope;
        intercept = new_intercept;
    }
    
    mse = calculateMSE(dataset);
}

void GradientDescentModel::setParameters(double lr, int max_iter, double tol) {
    learning_rate = lr;
    max_iterations = max_iter;
    tolerance = tol;
}

// LeastSquaresModel Implementation
void LeastSquaresModel::train(const Dataset& dataset) {
    const auto& x_vals = dataset.getXValues();
    const auto& y_vals = dataset.getYValues();
    
    if (x_vals.empty()) {
        throw runtime_error("Dataset is empty");
    }
    
    int n = x_vals.size();
    
    // Calculate means
    double x_mean = accumulate(x_vals.begin(), x_vals.end(), 0.0) / n;
    double y_mean = accumulate(y_vals.begin(), y_vals.end(), 0.0) / n;
    
    // Calculate slope and intercept using least squares formula
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (int i = 0; i < n; ++i) {
        numerator += (x_vals[i] - x_mean) * (y_vals[i] - y_mean);
        denominator += (x_vals[i] - x_mean) * (x_vals[i] - x_mean);
    }
    
    slope = numerator / denominator;
    intercept = y_mean - slope * x_mean;
    
    mse = calculateMSE(dataset);
}

// LinearRegression Implementation
LinearRegression::LinearRegression() : is_trained(false) {}

void LinearRegression::loadData(const string& filename) {
    dataset.loadFromCSV(filename);
    is_trained = false;
}

void LinearRegression::addDataPoint(double x, double y) {
    dataset.addDataPoint(x, y);
    is_trained = false;
}

void LinearRegression::useGradientDescent(double lr, int max_iter, double tol) {
    model = make_unique<GradientDescentModel>(lr, max_iter, tol);
    is_trained = false;
}

void LinearRegression::useLeastSquares() {
    model = make_unique<LeastSquaresModel>();
    is_trained = false;
}

void LinearRegression::trainModel() {
    if (!model) {
        throw runtime_error("No regression model selected. Use useGradientDescent() or useLeastSquares() first.");
    }
    
    if (dataset.getSize() < 2) {
        throw runtime_error("Insufficient data for training. Need at least 2 data points.");
    }
    
    cout << "Training model..." << endl;
    model->train(dataset);
    is_trained = true;
    cout << "Training completed!" << endl;
}

double LinearRegression::predict(double x) const {
    if (!is_trained || !model) {
        throw runtime_error("Model not trained. Call trainModel() first.");
    }
    return model->predict(x);
}

void LinearRegression::displayResults() const {
    if (!is_trained || !model) {
        throw runtime_error("Model not trained. Call trainModel() first.");
    }
    model->displayResults();
}

void LinearRegression::displayDatasetSummary() const {
    dataset.displaySummary();
}

bool LinearRegression::isModelTrained() const {
    return is_trained;
}