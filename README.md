# Wine Classification Project
## 📝 Overview
This project implements a binary Logistic Regression classifier for the Wine Dataset using scikit-learn and NumPy. The goal is to demonstrate a custom implementation of logistic regression for wine classification.

## 🎯 Project Objectives
- Implement a custom Logistic Regression algorithm from scratch
- Classify wine samples using binary classification (specifically, classifying wines from the first class)
- Demonstrate machine learning workflow including:
  - Data preprocessing
  - Model training
  - Model evaluation
  - Loss curve visualization
  - Confusion matrix generation

## 🛠 Technologies Used
- Python
- NumPy
- scikit-learn
- Matplotlib (for visualization)
- Logistic Regression (custom implementation)

## 📊 Dataset
- Source: scikit-learn's built-in Wine Dataset
- Features: 13 numerical features describing wine characteristics
- Original Classes: 3 wine classes
- Modified to: Binary classification (first class vs. others)

## 🚀 Key Features
- Custom Logistic Regression implementation
- Gradient descent optimization
- Sigmoid activation function
- Standard feature scaling
- Loss curve tracking
- Comprehensive model evaluation metrics
- Visualization of results

## 🔍 Model Workflow
1. Data Loading
2. Binary Classification Conversion
3. Train/Test Split
4. Feature Scaling (StandardScaler)
5. Model Training
   - Gradient descent
   - Cost history tracking
6. Prediction
7. Performance Evaluation
8. Result Visualization

## 📈 Performance Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report
- Loss Curve Plot

## 🏁 How to Run
```bash
# Clone the repository
git clone https://github.com/nguyensinhloc/WineClassification.git

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
```
## 🧠 Custom Logistic Regression Implementation
The LogisticRegression class includes:
- Sigmoid activation
- Gradient descent optimization
- Cost computation
- Parameter initialization
- Prediction method
## 🔬 Customization Options
- Adjust ```learning_rate```
- Modify ```num_iterations```
- Experiment with different binary classification scenarios
## 📋 Requirements
- Python 3.12+
- NumPy
- scikit-learn
## 🤝 Contributing
Contributions, issues, and feature requests are welcome!
## 📜 License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/nguyensinhloc/WineClassification/blob/ac47e24e61e5a0ccb0f9c266ef005976fa6cf4e7/LICENSE) file for details.
## 🙏 Acknowledgments
- Scikit-learn for the Wine Dataset
- Open-source machine learning community
## Note
This project is for educational purposes and demonstrates a custom machine learning algorithm implementation.
