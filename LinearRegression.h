#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <vector>
#include <string>
#include <memory>

using namespace std;

class Dataset {
private:
    vector<double> x_values;
    vector<double> y_values;
    string x_label;
    string y_label;

public:
    Dataset();
    void loadFromCSV(const string& filename);
    void addDataPoint(double x, double y);
    const vector<double>& getXValues() const;
    const vector<double>& getYValues() const;
    size_t getSize() const;
    void setLabels(const string& x_label, const string& y_label);
    string getXLabel() const;
    string getYLabel() const;
    void displaySummary() const;
};

class RegressionModel {
protected:
    double slope;
    double intercept;
    double mse;

public:
    RegressionModel();
    virtual ~RegressionModel() = default;
    
    virtual void train(const Dataset& dataset) = 0;
    virtual double predict(double x) const;
    virtual double calculateMSE(const Dataset& dataset) const;
    
    double getSlope() const;
    double getIntercept() const;
    double getMSE() const;
    virtual string getEquation() const;
    virtual void displayResults() const;
};

class GradientDescentModel : public RegressionModel {
private:
    double learning_rate;
    int max_iterations;
    double tolerance;

public:
    GradientDescentModel(double lr = 0.01, int max_iter = 1000, double tol = 1e-6);
    void train(const Dataset& dataset) override;
    void setParameters(double lr, int max_iter, double tol);
};

class LeastSquaresModel : public RegressionModel {
public:
    void train(const Dataset& dataset) override;
};

class LinearRegression {
private:
    unique_ptr<RegressionModel> model;
    Dataset dataset;
    bool is_trained;

public:
    LinearRegression();
    ~LinearRegression() = default;
    
    void loadData(const string& filename);
    void addDataPoint(double x, double y);
    void useGradientDescent(double lr = 0.01, int max_iter = 1000, double tol = 1e-6);
    void useLeastSquares();
    void trainModel();
    double predict(double x) const;
    void displayResults() const;
    void displayDatasetSummary() const;
    bool isModelTrained() const;
};

#endif