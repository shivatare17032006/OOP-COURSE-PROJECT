#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <map>
#include <direct.h>

using namespace std;

// Dataset Class
class Dataset {
private:
    vector<double> x_values;
    vector<double> y_values;
    string x_label;
    string y_label;

public:
    Dataset() : x_label("X"), y_label("Y") {}
    
    void loadFromCSV(const string& filename) {
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
    
    void addDataPoint(double x, double y) {
        x_values.push_back(x);
        y_values.push_back(y);
    }
    
    const vector<double>& getXValues() const { return x_values; }
    const vector<double>& getYValues() const { return y_values; }
    size_t getSize() const { return x_values.size(); }
    void setLabels(const string& x_label, const string& y_label) {
        this->x_label = x_label;
        this->y_label = y_label;
    }
    string getXLabel() const { return x_label; }
    string getYLabel() const { return y_label; }
    
    void displaySummary() const {
        cout << "\n*** Dataset Summary ***" << endl;
        cout << "Size: " << getSize() << " data points" << endl;
        cout << "X Label: " << x_label << endl;
        cout << "Y Label: " << y_label << endl;
        
        if (!x_values.empty()) {
            auto x_minmax = minmax_element(x_values.begin(), x_values.end());
            auto y_minmax = minmax_element(y_values.begin(), y_values.end());
            
            cout << "X Range: [" << *x_minmax.first << ", " << *x_minmax.second << "]" << endl;
            cout << "Y Range: [" << *y_minmax.first << ", " << *y_minmax.second << "]" << endl;
        }
    }
};

// RegressionModel Base Class
class RegressionModel {
protected:
    double slope;
    double intercept;
    double mse;

public:
    RegressionModel() : slope(0), intercept(0), mse(0) {}
    virtual ~RegressionModel() = default;
    
    virtual void train(const Dataset& dataset) = 0;
    
    double predict(double x) const {
        return slope * x + intercept;
    }
    
    double calculateMSE(const Dataset& dataset) const {
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
    
    double getSlope() const { return slope; }
    double getIntercept() const { return intercept; }
    double getMSE() const { return mse; }
    
    string getEquation() const {
        stringstream ss;
        ss << fixed << setprecision(4);
        ss << "y = " << slope << " * x + " << intercept;
        return ss.str();
    }
    
    void displayResults() const {
        cout << "\n*** Regression Results ***" << endl;
        cout << "Equation: " << getEquation() << endl;
        cout << "Slope: " << slope << endl;
        cout << "Intercept: " << intercept << endl;
        cout << "Mean Squared Error: " << mse << endl;
    }
};

// GradientDescentModel Class
class GradientDescentModel : public RegressionModel {
private:
    double learning_rate;
    int max_iterations;
    double tolerance;

public:
    GradientDescentModel(double lr = 0.01, int max_iter = 1000, double tol = 1e-6) 
        : learning_rate(lr), max_iterations(max_iter), tolerance(tol) {}
    
    void train(const Dataset& dataset) override {
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
    
    void setParameters(double lr, int max_iter, double tol) {
        learning_rate = lr;
        max_iterations = max_iter;
        tolerance = tol;
    }
};

// LeastSquaresModel Class
class LeastSquaresModel : public RegressionModel {
public:
    void train(const Dataset& dataset) override {
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
};

// LinearRegression Main Class
class LinearRegression {
private:
    unique_ptr<RegressionModel> model;
    Dataset dataset;
    bool is_trained;

public:
    LinearRegression() : is_trained(false) {}
    
    void loadData(const string& filename) {
        dataset.loadFromCSV(filename);
        is_trained = false;
    }
    
    void addDataPoint(double x, double y) {
        dataset.addDataPoint(x, y);
        is_trained = false;
    }
    
    void useGradientDescent(double lr = 0.01, int max_iter = 1000, double tol = 1e-6) {
        model = make_unique<GradientDescentModel>(lr, max_iter, tol);
        is_trained = false;
    }
    
    void useLeastSquares() {
        model = make_unique<LeastSquaresModel>();
        is_trained = false;
    }
    
    void trainModel() {
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
    
    double predict(double x) const {
        if (!is_trained || !model) {
            throw runtime_error("Model not trained. Call trainModel() first.");
        }
        return model->predict(x);
    }
    
    void displayResults() const {
        if (!is_trained || !model) {
            throw runtime_error("Model not trained. Call trainModel() first.");
        }
        model->displayResults();
    }
    
    void displayDatasetSummary() const {
        dataset.displaySummary();
    }
    
    bool isModelTrained() const {
        return is_trained;
    }
    
    // ADD THIS MISSING METHOD
    const Dataset& getDataset() const {
        return dataset;
    }
};

// Category definitions with all datasets
map<string, pair<string, string>> categories = {
    {"1", {"Education", "Study hours vs Exam scores"}},
    {"2", {"Real Estate", "House size vs Price"}},
    {"3", {"Business", "Advertising budget vs Sales"}},
    {"4", {"Healthcare", "Treatment duration vs Recovery rate"}},
    {"5", {"Sports", "Training hours vs Performance score"}},
    {"6", {"Salary Prediction", "Years of experience vs Salary"}},
    {"7", {"Temperature Analysis", "Temperature vs Ice Cream Sales"}},
    {"8", {"Car Valuation", "Car age vs Price"}},
    {"9", {"Custom", "Your own dataset"}}
};

// Function to display categories
void displayCategories() {
    cout << "\n*** SELECT CATEGORY ***" << endl;
    cout << "==========================================" << endl;
    for (const auto& category : categories) {
        cout << category.first << ". " << category.second.first << endl;
        cout << "   - " << category.second.second << endl;
    }
    cout << "==========================================" << endl;
}

// Function to get category-specific prompt
pair<string, string> getCategoryPrompts(const string& categoryId) {
    map<string, pair<string, string>> prompts = {
        {"1", {"study hours", "exam score"}},
        {"2", {"house size (sqft)", "price ($)"}},
        {"3", {"advertising budget ($)", "sales amount ($)"}},
        {"4", {"treatment duration (days)", "recovery rate (%)"}},
        {"5", {"training hours", "performance score"}},
        {"6", {"years of experience", "salary ($)"}},
        {"7", {"temperature (C)", "ice cream sales"}},
        {"8", {"car age (years)", "price ($)"}},
        {"9", {"input value", "output value"}}
    };
    
    return prompts[categoryId];
}

// Function to create sample CSV for a category
void createSampleCSV(const string& categoryId, const string& filename) {
    ofstream file(filename);
    
    if (categoryId == "1") {
        // Education sample
        file << "Study_Hours,Exam_Score" << endl;
        file << "1,45" << endl;
        file << "2,55" << endl;
        file << "3,65" << endl;
        file << "4,75" << endl;
        file << "5,85" << endl;
        file << "6,80" << endl;
        file << "7,90" << endl;
        file << "8,95" << endl;
        file << "9,92" << endl;
        file << "10,98" << endl;
    }
    else if (categoryId == "2") {
        // Real Estate sample
        file << "Size_sqft,Price" << endl;
        file << "800,250000" << endl;
        file << "1000,300000" << endl;
        file << "1200,350000" << endl;
        file << "1500,400000" << endl;
        file << "1800,450000" << endl;
        file << "2000,500000" << endl;
        file << "2200,520000" << endl;
        file << "2500,580000" << endl;
    }
    else if (categoryId == "3") {
        // Business sample
        file << "Advertising_Budget,Sales" << endl;
        file << "500,3000" << endl;
        file << "1000,5000" << endl;
        file << "1500,6500" << endl;
        file << "2000,8000" << endl;
        file << "2500,9500" << endl;
        file << "3000,12000" << endl;
        file << "4000,15000" << endl;
        file << "5000,18000" << endl;
    }
    else if (categoryId == "4") {
        // Healthcare sample
        file << "Treatment_Days,Recovery_Rate" << endl;
        file << "3,20" << endl;
        file << "5,30" << endl;
        file << "7,40" << endl;
        file << "10,50" << endl;
        file << "15,65" << endl;
        file << "20,75" << endl;
        file << "25,80" << endl;
        file << "30,85" << endl;
    }
    else if (categoryId == "5") {
        // Sports sample
        file << "Training_Hours,Performance_Score" << endl;
        file << "5,40" << endl;
        file << "10,60" << endl;
        file << "15,65" << endl;
        file << "20,75" << endl;
        file << "25,80" << endl;
        file << "30,85" << endl;
        file << "35,88" << endl;
        file << "40,90" << endl;
    }
    else if (categoryId == "6") {
        // Salary sample
        file << "Years_Experience,Salary" << endl;
        file << "1,35000" << endl;
        file << "2,40000" << endl;
        file << "3,45000" << endl;
        file << "4,50000" << endl;
        file << "5,55000" << endl;
        file << "6,60000" << endl;
        file << "7,65000" << endl;
        file << "8,70000" << endl;
    }
    else if (categoryId == "7") {
        // Temperature sample
        file << "Temperature,Ice_Cream_Sales" << endl;
        file << "15,100" << endl;
        file << "18,120" << endl;
        file << "20,150" << endl;
        file << "22,180" << endl;
        file << "25,220" << endl;
        file << "28,260" << endl;
        file << "30,300" << endl;
        file << "32,320" << endl;
    }
    else if (categoryId == "8") {
        // Car valuation sample
        file << "Car_Age,Price" << endl;
        file << "0,30000" << endl;
        file << "1,27000" << endl;
        file << "2,24000" << endl;
        file << "3,22000" << endl;
        file << "4,20000" << endl;
        file << "5,18000" << endl;
        file << "6,16000" << endl;
        file << "7,14000" << endl;
    }
    else if (categoryId == "9") {
        // Custom sample
        file << "Input,Output" << endl;
        file << "1,10" << endl;
        file << "2,20" << endl;
        file << "3,30" << endl;
        file << "4,40" << endl;
        file << "5,50" << endl;
        file << "6,60" << endl;
        file << "7,70" << endl;
        file << "8,80" << endl;
    }
    
    file.close();
    cout << "*** SUCCESS: Created sample file: " << filename << endl;
    cout << "*** INFO: Sample data created with realistic values for " << categories[categoryId].first << endl;
}

// Function to check if file exists
bool fileExists(const string& filename) {
    ifstream file(filename);
    return file.good();
}

// Function to display dataset suggestions
void displayDatasetSuggestions(const string& categoryId) {
    map<string, string> suggestions = {
        {"1", "student_data.csv, education_data.csv, marks_data.csv"},
        {"2", "housing_data.csv, real_estate_data.csv, property_data.csv"},
        {"3", "business_data.csv, sales_data.csv, advertising_data.csv"},
        {"4", "healthcare_data.csv, medical_data.csv, recovery_data.csv"},
        {"5", "sports_data.csv, training_data.csv, performance_data.csv"},
        {"6", "salary_data.csv, experience_data.csv, income_data.csv"},
        {"7", "temperature_data.csv, weather_data.csv, sales_data.csv"},
        {"8", "car_data.csv, vehicle_data.csv, auto_data.csv"},
        {"9", "Any CSV file with two columns (input,output)"}
    };
    
    cout << "*** SUGGESTION: " << suggestions[categoryId] << endl;
}

// Function to show current directory (Windows)
void showCurrentDirectory() {
    char buffer[1024];
    if (_getcwd(buffer, sizeof(buffer)) != NULL) {
        cout << "*** Current directory: " << buffer << endl;
    }
}

// Function to create all sample datasets at once
void createAllSampleDatasets() {
    cout << "\n*** CREATING ALL SAMPLE DATASETS ***" << endl;
    createSampleCSV("1", "student_data.csv");
    createSampleCSV("2", "housing_data.csv");
    createSampleCSV("3", "business_data.csv");
    createSampleCSV("4", "healthcare_data.csv");
    createSampleCSV("5", "sports_data.csv");
    createSampleCSV("6", "salary_data.csv");
    createSampleCSV("7", "temperature_data.csv");
    createSampleCSV("8", "car_data.csv");
    createSampleCSV("9", "custom_data.csv");
    cout << "*** SUCCESS: All sample datasets created successfully!" << endl;
}

// Main category-based workflow
void runCategoryWorkflow() {
    LinearRegression lr;
    string categoryId;
    string currentCategory;
    bool datasetsCreated = false;
    
    // Show current directory
    showCurrentDirectory();
    
    // Step 1: Ask about creating datasets FIRST
    cout << "\nWould you like to create all sample datasets first? (y/n): ";
    char createChoice;
    cin >> createChoice;
    
    if (createChoice == 'y' || createChoice == 'Y') {
        createAllSampleDatasets();
        datasetsCreated = true;
        cout << "\n*** NOTE: Sample datasets are now available in your current directory ***" << endl;
    }
    
    // Step 2: Category Selection
    displayCategories();
    cout << "Enter category number (1-9): ";
    cin >> categoryId;
    
    if (categories.find(categoryId) == categories.end()) {
        cout << "*** ERROR: Invalid category selection!" << endl;
        return;
    }
    
    currentCategory = categories[categoryId].first;
    cout << "\n*** SELECTED: " << currentCategory << endl;
    cout << "*** DESCRIPTION: " << categories[categoryId].second << endl;
    
    // Step 3: File Path Handling - SMART LOGIC
    string filepath;
    map<string, string> defaultFiles = {
        {"1", "student_data.csv"},
        {"2", "housing_data.csv"},
        {"3", "business_data.csv"},
        {"4", "healthcare_data.csv"},
        {"5", "sports_data.csv"},
        {"6", "salary_data.csv"},
        {"7", "temperature_data.csv"},
        {"8", "car_data.csv"},
        {"9", "custom_data.csv"}
    };
    
    if (datasetsCreated) {
        // If datasets were created, automatically use the corresponding file
        filepath = defaultFiles[categoryId];
        cout << "\n*** AUTOMATICALLY USING: " << filepath << endl;
    } else {
        // If no datasets created, ask for file path
        cout << "\n*** ENTER CSV FILE PATH ***" << endl;
        displayDatasetSuggestions(categoryId);
        
        // Suggest the default file for this category
        cout << "Suggested file: " << defaultFiles[categoryId] << endl;
        cout << "Enter the path to your CSV file: ";
        cin >> filepath;
        
        // If file doesn't exist, offer to create sample
        if (!fileExists(filepath)) {
            cout << "*** ERROR: File not found: " << filepath << endl;
            cout << "Would you like to create a sample dataset? (y/n): ";
            char choice;
            cin >> choice;
            
            if (choice == 'y' || choice == 'Y') {
                createSampleCSV(categoryId, filepath);
            } else {
                cout << "*** ERROR: Please provide a valid CSV file path." << endl;
                return;
            }
        }
    }
    
    // Step 4: Load Data
    try {
        cout << "\n*** LOADING DATA FROM: " << filepath << endl;
        lr.loadData(filepath);
        lr.displayDatasetSummary();
    } catch (const exception& e) {
        cout << "*** ERROR loading file: " << e.what() << endl;
        return;
    }
    
    // Step 5: Model Selection
    cout << "\n*** SELECT REGRESSION METHOD ***" << endl;
    cout << "1. Gradient Descent (Better for large datasets)" << endl;
    cout << "2. Least Squares (Faster for small datasets)" << endl;
    cout << "Enter choice (1 or 2): ";
    
    string modelChoice;
    cin >> modelChoice;
    
    if (modelChoice == "1") {
        double lr_rate;
        int max_iter;
        
        // Get learning rate with validation
        cout << "Enter learning rate (0.001 to 1.0, default 0.01): ";
        cin >> lr_rate;
        
        if (lr_rate <= 0 || lr_rate > 1.0) {
            cout << "*** WARNING: Invalid learning rate. Using default 0.01" << endl;
            lr_rate = 0.01;
        }
        
        cout << "Enter max iterations (100 to 100000, default 1000): ";
        cin >> max_iter;
        
        if (max_iter < 100 || max_iter > 100000) {
            cout << "*** WARNING: Invalid iterations. Using default 1000" << endl;
            max_iter = 1000;
        }
        
        lr.useGradientDescent(lr_rate, max_iter);
        cout << "*** SUCCESS: Using Gradient Descent" << endl;
    } else if (modelChoice == "2") {
        lr.useLeastSquares();
        cout << "*** SUCCESS: Using Least Squares" << endl;
    } else {
        cout << "*** WARNING: Invalid choice. Using Least Squares by default." << endl;
        lr.useLeastSquares();
    }
    
    // Step 6: Train Model
    try {
        cout << "\n*** TRAINING MODEL ***" << endl;
        lr.trainModel();
        cout << "*** SUCCESS: Model trained successfully!" << endl;
        lr.displayResults();
    } catch (const exception& e) {
        cout << "*** ERROR Training failed: " << e.what() << endl;
        return;
    }
    
    // Step 7: Prediction Loop
    auto prompts = getCategoryPrompts(categoryId);
    string inputPrompt = prompts.first;
    string outputLabel = prompts.second;
    
    cout << "\n*** PREDICTION MODE ***" << endl;
    cout << "=======================" << endl;
    cout << "I can predict " << outputLabel << " based on " << inputPrompt << endl;
    cout << "*** WARNING: Predictions are most accurate within the training data range ***" << endl;
    cout << "Enter -1 to exit prediction mode" << endl;
    
    while (true) {
        double inputValue;
        
        cout << "\nEnter " << inputPrompt << " (or -1 to exit): ";
        cin >> inputValue;
        
        if (inputValue == -1) {
            break;
        }
        
        try {
            double prediction = lr.predict(inputValue);
            cout << "*** PREDICTION RESULT ***" << endl;
            cout << "For " << inputPrompt << ": " << inputValue << endl;
            cout << "Predicted " << outputLabel << ": " << prediction << endl;
            
            // Add category-specific validation
            if (categoryId == "1") { // Education
                if (prediction > 100) {
                    cout << "*** WARNING: Predicted score exceeds 100 marks!" << endl;
                    cout << "*** REALISTIC ESTIMATE: Maximum possible score is ~100" << endl;
                }
                
                if (prediction >= 90) cout << "*** ANALYSIS: Excellent score!" << endl;
                else if (prediction >= 75) cout << "*** ANALYSIS: Good score!" << endl;
                else if (prediction >= 60) cout << "*** ANALYSIS: Average score" << endl;
                else cout << "*** ANALYSIS: Needs improvement" << endl;
            }
            else if (categoryId == "2") { // Real Estate
                if (prediction < 0) {
                    cout << "*** WARNING: Negative price predicted!" << endl;
                    cout << "*** REALISTIC ESTIMATE: Minimum price should be > 0" << endl;
                }
                cout << "*** ANALYSIS: Estimated property value" << endl;
            }
            else if (categoryId == "3") { // Business
                if (prediction < 0) {
                    cout << "*** WARNING: Negative sales predicted!" << endl;
                }
                cout << "*** ANALYSIS: Expected sales revenue" << endl;
            }
            else if (categoryId == "4") { // Healthcare
                if (prediction > 100) {
                    cout << "*** WARNING: Recovery rate exceeds 100%!" << endl;
                }
                if (prediction >= 80) cout << "*** ANALYSIS: High recovery rate!" << endl;
                else if (prediction >= 60) cout << "*** ANALYSIS: Good recovery rate" << endl;
                else cout << "*** ANALYSIS: Continuing treatment needed" << endl;
            }
            else if (categoryId == "5") { // Sports
                if (prediction > 100) {
                    cout << "*** WARNING: Performance score exceeds 100!" << endl;
                }
                if (prediction >= 90) cout << "*** ANALYSIS: Elite performance!" << endl;
                else if (prediction >= 80) cout << "*** ANALYSIS: Great performance!" << endl;
                else if (prediction >= 70) cout << "*** ANALYSIS: Good performance" << endl;
                else cout << "*** ANALYSIS: Keep training!" << endl;
            }
            else if (categoryId == "6") { // Salary
                if (prediction < 0) {
                    cout << "*** WARNING: Negative salary predicted!" << endl;
                }
                cout << "*** ANALYSIS: Estimated annual salary" << endl;
            }
            else if (categoryId == "7") { // Temperature
                cout << "*** ANALYSIS: Expected ice cream sales" << endl;
            }
            else if (categoryId == "8") { // Car valuation
                if (prediction < 0) {
                    cout << "*** WARNING: Negative car price predicted!" << endl;
                }
                cout << "*** ANALYSIS: Estimated car value" << endl;
            }
            else { // Custom
                cout << "*** ANALYSIS: Predicted output based on input" << endl;
            }
            
        } catch (const exception& e) {
            cout << "*** ERROR Prediction error: " << e.what() << endl;
        }
    }
    
    cout << "\n*** Thank you for using Linear Regression Predictor!" << endl;
    cout << "*** Category: " << currentCategory << endl;
}

int main() {
    cout << "*** LINEAR REGRESSION PREDICTION SYSTEM ***" << endl;
    cout << "===========================================" << endl;
    cout << "Predict outcomes based on your data!" << endl;
    
    runCategoryWorkflow();
    
    return 0;
}