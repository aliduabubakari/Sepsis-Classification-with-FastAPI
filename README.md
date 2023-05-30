# Sepsis-Classification-with-FastAPI

This project is focused on the accurate and efficient classification of sepsis cases using the FastAPI framework. Sepsis is a critical medical condition that requires prompt identification and treatment. This project aims to provide a streamlined solution for healthcare professionals to classify sepsis cases quickly and effectively.

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Data](#data)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Future Work](#future-work)
- [Contact](#contact)

## Project Overview

The "Sepsis Classification with FastAPI" project aims to develop an accurate and efficient classification system for sepsis cases using the FastAPI framework. Sepsis is a life-threatening condition that requires immediate medical attention. This project addresses the critical need for timely identification and classification of sepsis cases to facilitate prompt treatment and improve patient outcomes.

The objectives of the project are as follows:

1. Train a machine learning model on a diverse dataset of sepsis cases to accurately predict the likelihood of sepsis in patients.

2. Utilize the FastAPI framework to create a user-friendly and efficient web interface for healthcare professionals to interact with the sepsis classification model.

3. Improve diagnostic capabilities by achieving high accuracy, sensitivity, and specificity in sepsis classification.

4. Provide a comprehensive and scalable solution that can be easily deployed in real-time healthcare environments.

Key challenges in this project include acquiring and preprocessing a reliable sepsis dataset, selecting an appropriate machine learning algorithm, optimizing the model's performance, and deploying the system in a secure and efficient manner.


## Summary
| Code      | Name        | Published Article |  Deployed App |
|-----------|-------------|:-------------:|------:|
| LP6 | Sepsis Prediction App with fastapi and Streamlit|  [https://medium.com/@alidu143/building-a-sales-prediction-app-with-streamlit-and-machine-learning-31746625d6ca](/) | [https://huggingface.co/spaces/Abubakari/Sales_Prediction#sales-prediction-app](/) |


## Getting Started

To set up the project environment, follow these steps:

1. Clone the repository: git clone https://github.com/your-username/sepsis-classification.git
2. Install the required dependencies: pip install -r requirements.txt
3. Create a virtual environment:
  Windows: python -m venv venv; venv\Scripts\activate
  Linux & MacOS: python3 -m venv venv; source venv/bin/activate
  Upgrade Pip: python -m pip install -q --upgrade pip
4. Install the required packages: python -m pip install -qr requirements.txt


## Data

The data used in this project consists of a diverse collection of sepsis cases obtained from [Sepsis](https://www.kaggle.com/datasets/chaunguynnghunh/sepsis?select=README.md).

### Data Fields

| Column Name | Features | Description                                      |
|-------------|-----------------|--------------------------------------------------|
| ID          | N/A             | Unique number to represent patient ID             |
| PRG         | Attribute 1     | Plasma glucose                                   |
| PL          | Attribute 2     | Blood Work Result-1 (mu U/ml)                    |
| PR          | Attribute 3     | Blood Pressure (mm Hg)                           |
| SK          | Attribute 4     | Blood Work Result-2 (mm)                         |
| TS          | Attribute 5     | Blood Work Result-3 (mu U/ml)                    |
| M11         | Attribute 6     | Body mass index (weight in kg/(height in m)^2)   |
| BD2         | Attribute 7     | Blood Work Result-4 (mu U/ml)                    |
| Age         | Attribute 8     | Patient's age (years)                            |
| Insurance   | N/A             | If a patient holds a valid insurance card         |
| Sepsis      | Target          | Positive: if a patient in ICU will develop sepsis,<br> Negative: otherwise |




## Exploratory Data Analysis


During the exploratory data analysis (EDA) phase of the project, a thorough investigation of the sepsis dataset was conducted to gain insights into the data through different types of analyses.

First, univariate analysis was performed to examine each variable individually. Summary statistics such as mean, median, standard deviation, and quartiles were calculated to understand the central tendency and spread of the data.

Next, bivariate analysis was carried out to explore the relationships between pairs of variables. This analysis aimed to identify patterns and potential predictor variables that could be useful for sepsis classification.

Furthermore, multivariate analysis was conducted to examine relationships among multiple variables simultaneously, allowing for a deeper understanding of their interactions and impact on sepsis.

In addition to the exploratory analyses, hypotheses were formulated based on prior knowledge and existing research. These hypotheses were tested using statistical tests like t-tests, chi-square tests, or ANOVA tests, depending on the nature of the variables. The results of these tests helped validate or refute the formulated hypotheses and provided further insights into the relationships between variables.

Hypotheses:

- Hypothesis 1: Higher plasma glucose levels (PRG) are associated with an increased risk of developing sepsis.

- Hypothesis 2: Abnormal blood work results, such as high values of PL, SK, and BD2, are indicative of a higher likelihood of sepsis.

- Hypothesis 3: Older patients are more likely to develop sepsis compared to younger patients.

- Hypothesis 4: Patients with higher body mass index (BMI) values (M11) have a lower risk of sepsis.

- Hypothesis 5: Patients without valid insurance cards are more likely to develop sepsis.

These hypotheses, along with the results of the EDA, contribute to a deeper understanding of the dataset and provide valuable insights for further analysis and model development.


## Modeling

During the modeling phase, considering the imbalanced nature of the data, the evaluation of models was focused on metrics that are robust to imbalanced datasets. The main metrics of interest were F1 score and AUC score, which provide a balanced assessment of model performance.

The following models were evaluated:

- Decision Tree: The Decision Tree model achieved an F1 score of 0.602 and an AUC score of 0.725.

- Logistic Regression: The Logistic Regression model demonstrated improved performance with an F1 score of 0.634 and an AUC score of 0.750.

- Naive Bayes: The Naive Bayes model obtained an F1 score of 0.575 and an AUC score of 0.692.

- Support Vector Machines (SVM): The SVM model yielded an F1 score of 0.564 and an AUC score of 0.717.

- Random Forest: The Random Forest model achieved an F1 score of 0.548 and an AUC score of 0.683.



## Evaluation

Given the imbalanced nature of the data, the models' performance was assessed using the F1 score, which considers both precision and recall, providing a balanced measure of accuracy. Additionally, the AUC score was considered to evaluate the models' ability to distinguish between positive and negative cases.
Hyperparameter tuning was also implemented to optimize the performance of the models. By fine-tuning the hyperparameters, it was possible to identify the best combination of parameter values that yielded the highest performance for each model

## Deployment

### Fastapi deployment 

1. Make sure you have FastAPI and any necessary dependencies installed. You can install FastAPI using pip:

```bash
pip install fastapi
```

2. Open a terminal or command prompt and navigate to the directory where your main.py file is located.

3. Run the FastAPI application using the uvicorn command, specifying the module and application name:
```bash
uvicorn main:app --reload
```
4. fter running the command, you should see output indicating that the FastAPI application is running and listening on a specific address (e.g., http://127.0.0.1:8000). This address represents the API endpoint where you can access your application.

5. Open a web browser or use an API testing tool (e.g., Postman) to interact with your deployed FastAPI application. Use the API endpoint provided in the terminal to make requests and receive responses.

### Containerized deployment 

To run the Docker container based on the provided Dockerfile, follow these steps:

1. Make sure you have Docker installed on your system.

2. Create a new file named Dockerfile (without any file extension) in the root directory of your project.

3. Copy the content of the Dockerfile you provided into the newly created Dockerfile.

4. Open a terminal or command prompt and navigate to the directory where the Dockerfile is located.

5. Build the Docker image by running the following command:

```bash
docker build -t your-image-name .
```

6. Replace your-image-name with the desired name for your Docker image. The . at the end denotes the current directory as the build context.

7. Once the image is built, you can run a Docker container based on that image using the following command:

```bash
docker run -d -p host-port:container-port your-image-name
```
Replace host-port with the port number on your host machine that you want to map to the container's port, and replace container-port with the port number specified in the Dockerfile's EXPOSE instruction (in this case, it's 8000).

For example, if you want to map the container's port 8000 to port 8080 on your host machine, the command would be:

```bash
docker run -d -p 8080:8000 your-image-name
```

8. After running the command, the Docker container will start, and your FastAPI application will be running inside the container.

9. You can access your application by visiting http://localhost:host-port in your web browser or using an API testing tool.
For example, if you mapped the container's port 8000 to your host's port 8080, you would access the application at http://localhost:8080.

### Streamlit deployment 

[Instructions for deploying the model, including any necessary setup steps and requirements]

## Future Work

[Potential areas for future development and improvement]

## Contact
`Alidu Abubakari`

`Data Analyst`
`Azubi Africa`

- [![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/alidu-abubakari-2612bb57/) 


