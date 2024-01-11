# Credit Card Eligibility Prediction

## Project Overview

This project develops a machine learning model to predict credit card eligibility for new customers based on their personal and financial information. It aims to streamline the credit card approval process and assist financial institutions in making informed decisions.

## Key Features

Data-driven decision-making: Leverages customer data to predict credit card eligibility.
Personalized credit assessments: Provides tailored predictions for individual customers.
Potential for reduced risk: Assists in identifying creditworthy applicants, potentially minimizing defaults.
Enhanced customer experience: Enables efficient and informed credit card approval processes.
## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn (including numpy, pandas, scikit-learn, and others)

## Project Structure

- README.md
- CreditCardEligibility.ipynb (Jupyter Notebook containing project code)
- customer_data.csv (sample dataset)
- application_record.csv (sample dataset)
- requirements.txt (list of required Python libraries)
## Usage

- Clone this repository.
- Install required libraries using pip install -r requirements.txt.
- Open creditapproval.ipynb in Jupyter Notebook.
- Run the code cells to execute the project.
## Model Details

In the development of this credit card eligibility prediction model, various machine learning algorithms were explored, including:

- **KNN (K-Nearest Neighbors)**
- **Vector Machine (Linear SVC)**
- **Naive Bayes**

The model assessment involved hyperparameter tuning to optimize performance. The key features considered for prediction encompassed factors such as marital status, homeownership, and credit history.

### Evaluation Metrics

The model performance was rigorously assessed using the following evaluation metrics:

- **Accuracy:** Measures the overall correctness of the predictions.
- **Precision:** Evaluates the accuracy of positive predictions.
- **Recall:** Measures the ability of the model to capture all relevant instances.
- **F1-score:** Harmonic mean of precision and recall, providing a balance between the two.

These metrics collectively offer a comprehensive understanding of the model's effectiveness in predicting credit card eligibility. The Jupyter Notebook (`CreditCardEligibility.ipynb`) contains the detailed implementation and assessment of each model.

## Contributing

We welcome contributions and feedback! Please feel free to open issues or pull requests.

## License

This project is licensed under the MIT License.
