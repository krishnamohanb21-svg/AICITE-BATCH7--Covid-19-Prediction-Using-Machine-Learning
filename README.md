# Covid19-Prediction-Using-Machine-Learning
# Problem Statement:
The outbreak of COVID-19 has created a global health crisis, affecting millions of people worldwide. Early detection of infected
individuals plays a crucial role in preventing the spread of the virus and ensuring timely medical intervention.
Traditional diagnostic methods such as RT-PCR tests require laboratory facilities, time, and medical resources. In situations where
testing resources are limited, there is a need for an intelligent system that can assist in predicting the likelihood of COVID-19
infection based on symptoms, medical history, and exposure factors.
This project aims to develop a Machine Learning model that predicts whether a person is COVID-19 positive or negative using the
following input features:
Symptoms:
Breathing Problem, Fever, Dry Cough, Sore Throat, Running Nose, Headache, Fatigue, Gastrointestinal Issues
Pre-existing Medical Conditions:
Asthma, Chronic Lung Disease, Heart Disease, Diabetes, Hypertension
Exposure & Travel History:
Abroad Travel, Contact with COVID Patient, Attended Large Gathering, Visited Public Exposed Places, Family Working in Public
Exposed Places
Preventive Measures:
Wearing Masks, Sanitization from Market
The target variable is:
COVID-19 (Positive / Negative)

# Proposed System of Covid19 Prediction Using Machine Learning
The proposed system aims to develop a Machine Learning-based predictive model to determine whether a person is COVID-19 positive or negative based on symptoms, medical conditions, exposure history, and preventive measures.
The dataset contains binary values:
0 = No
1 = Yes
The target variable:
COVID-19 (0 = Negative, 1 = Positive)

# 1) Data Collection
The dataset is collected in CSV format.
Each record represents an individual with the following features:
Symptoms
Breathing Problem
Fever
Dry Cough
Sore Throat
Running Nose
Headache
Fatigue
Gastrointestinal

# Pre-existing Medical Conditions
Asthma
Chronic Lung Disease
Heart Disease
Diabetes
Hypertension
Exposure & Travel History
Abroad Travel
Contact with COVID Patient
Attended Large Gathering
Visited Public Exposed Places
Family Working in Public Exposed Places
Preventive Measures
Wearing Masks
Sanitization from Market
Target
COVID-19 (Positive / Negative)

# 2) Data Preprocessing
Since the dataset uses binary values (0 and 1):
No categorical encoding is required.
Check for missing or null values.
Split dataset into:
80% Training Data
20% Testing Data

# 3) Machine Learning Algorithm
Classification algorithms are used to predict COVID-19 status:
1)Logistic Regression
2)Decision Tree
3)Random Forest
4)Support Vector Machine (SVM)
5) K-Nearest Neighbors (KNN)
The selected model learns patterns between symptoms, exposure factors, and infection status.
The best-performing algorithm is chosen based on evaluation metrics.

# 4) Deployment
(User Interface – Flask)
The trained model is deployed using Flask web framework.
Working:
User enters 0 (No) or 1 (Yes) for each feature.
Flask application receives input.
Input data is converted into numerical format.
Trained ML model predicts output.
Result is displayed as:
COVID-19 Positive
COVID-19 Negative
The system provides a simple and user-friendly interface for prediction.

# 5) Evaluation
The model is evaluated using:
1) Accuracy
2) Precision
3) Recall
4) F1-Score
5) Confusion Matrix
These metrics measure how effectively the model predicts COVID-19 cases.
6) Result
The final system successfully predicts whether a person is COVID-19 positive or negative based on input features.
The model provides:
1)Fast prediction
2)Preliminary screening support
3) Assistance in early risk identification

# Software Approach
HARDWARE REQUIREMENTS :
The hardware requirements for running this website and model  are:
RAM – 8.00 GB
Operating System – Windows 11 
Processor – Intel(R) Core(TM) i3-1115g4
Processor speed – 3.00 GHz
SOFTWARE REQUIREMENTS: 
The programming language used to develop this application is Python and the IDE used is Jupyter Notebook. Front end is made using HTML and is integrated with  flask.
Programming Language – Python
Python IDE – Jupyter Notebook
Python Libraries: Flask

# Libraries to build the model
1)Pandas 
2) matplotlib
3) seaborn 
4) sklearn 
5) Flask

# Algorithm 
1) Logistic Regression
2) Decision Tree Classifier 
3) Support Vector Machine
4) Random Forest Classifier 
5) K Neighbors Classifier

# Logistic Regression
This is the one of the most common model used in ML ,Logistic Regression is often applied in the actual manufacturing context the fields such as data mining, automatic disease diagnosis and economic prediction. For our model,   use Logistic regression to know the risk factors for heart disease and forecast the probability of disease    occurrence based on risk factors. This model is most frequently applied for classification, primarily two-category issues (that is, there are only two types of output, each representing one category), and can indicate the probability of occurrence of each classification event. Logistic regression model is shown below: This technique used is also known as sigmoid function .Sigmoid function helps in the easy representation in graphs. Logistic regression also provides better accuracy. By using equation the logistic regression algorithm is represented in the graphs showing the difference between the attributes.
Where Y refers to binary dependent variable (Y is equal to 1 if event happens; Y=0 otherwise), e stands for the foundation of natural logarithms and Z  means
with constant β0 ,coefficients β j and predictors X j , for p  predictors(j=1,2,3,.....,p)
with constant β0 ,coefficients β j and predictors X j , for p  predictors(j=1,2,3,.....,p)
The process of modeling the probability of a discrete outcome given an input variable is known as the Logistic Regression. The most common logistic regression, as its name suggests is not regression rather it is a classification algorithm that classifies something that can take two values such as true/false, yes/no, and so on. Logistic regression identifies a hyperplane in a manner that when it is passes through a function whose value ranges between 0 and 1 (typically we use sigmoidal), it optimizes cost function. Bases on closeness to 0 or 1, it predicts a Boolean output. Here vector parameters is used for training. σ(.) is usually a sigmoid function, with output between 0 and 1.

# Features of Logistic Regression

Multinomial logistic regression is the type of regression which uses the softmax function to compute probabilities.
● We use loss function to learn weights(vector w and bias b) from a labeled training. we perform such activity to minimize the cross-entropy loss.
● Iterative algos like gradient descent are used to get the weight(optimal).while minimizing the loss function the type of convex optimization problem.
● To avoid overfitting regularization is used.
● Logistic regression has the ability to transparently study the importance of individual features.

# Advantages  and Disadvantages of Logistic Regression 

Advantages 
● This technique is perform well and fast where we have to classify unknown records.
● This is not limited to binary classification we can easily extend it to  multinomial regression.
Disadvantages 
● Logistic regression will not perform well If the number of observations is lesser than the number of features in such condition it may lead to overfitting.
● Logistic regression constructs the linear boundaries. In the logistic function equation, x is the input variable. Let's feed in values −20 to 20 into the logistic function. As illustrated in Figure the inputs have been transferred to between 0 and  1.

<img width="1317" height="714" alt="image" src="https://github.com/user-attachments/assets/53db4453-65dd-4572-81bb-54bb68a8eb3a" />
<img width="1446" height="929" alt="image" src="https://github.com/user-attachments/assets/54343a23-3dc4-4ef6-a7d1-022b668ec021" />
<img width="1165" height="763" alt="image" src="https://github.com/user-attachments/assets/1c0b3a0e-cfae-47d7-bae6-33acb810577e" />
<img width="918" height="762" alt="image" src="https://github.com/user-attachments/assets/69dbd852-aeea-4e62-a3b1-a587301e788a" />
Accuracy 96.96 % 

# Decision Tree Classifier
A decision tree contains a flowchart-like structure. In the structure of DT(Decision Tree) in a test, an attribute is represented using an internal node. The result of the test is represented using the branch. To represent a class label, a leaf node is used. To represent classification rules paths from root to leaf are used. This is the type of analysis in which closely related influence diagrams are used for a visual and analytical root. Where the expected values of competing alternatives are calculated.
Architecture of Decision Tree.

<img width="926" height="520" alt="image" src="https://github.com/user-attachments/assets/eb9595a7-3f99-4e1f-83cc-794845a152dd" />
<img width="1987" height="965" alt="image" src="https://github.com/user-attachments/assets/6d62e07d-b216-46b0-8b65-02b739735efe" />
<img width="1231" height="811" alt="image" src="https://github.com/user-attachments/assets/12861d01-9bc0-40d0-8d31-d673cc1f5184" />
<img width="770" height="788" alt="image" src="https://github.com/user-attachments/assets/ebcdf026-cc11-4431-b8ba-6adbeffbd301" />
 Accuracy 97.52 %

# Random Forest Classifier 
A Random Forest Classifier is a method of ensemble learning applied to classification problems. It constructs multiple decision trees and combines their predictions to enhance accuracy and avoid overfitting. Each tree is trained on a random subset of data and attributes, hence making the model stable, robust, and effective in handling complex datasets.

<img width="840" height="546" alt="image" src="https://github.com/user-attachments/assets/1c2e12d9-ef8a-4de5-85ac-231c758ef020" />

# Features of Random Forest 
●	It runs Efficiently in scenarios where database is very huge.
●	We can perform classification on Thousands of Input variables.
●	Using this we can find which variable is useful for our classification.
● Using this we can easily calculate the missing data and also maintain the good accuracy. Even though it has missing data.

<img width="1973" height="998" alt="image" src="https://github.com/user-attachments/assets/ff8cc1c8-ebd3-4fb3-9d3e-7487186f416a" />
<img width="845" height="714" alt="image" src="https://github.com/user-attachments/assets/c62b33f9-c9dd-4f5d-9f2d-a0557374bd94" />
<img width="1081" height="733" alt="image" src="https://github.com/user-attachments/assets/d6be5012-dec9-4398-8edd-b4d062f3ff6c" />
Accuracy 97.52 %

# Support Vector Machine 
Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It works by finding the optimal hyperplane that best separates data into different classes. The main goal of SVM is to maximize the margin between different classes while minimizing classification error.

# Features of SVM 
1.Margin Maximization: SVM finds the hyperplane that maximizes the margin between classes, which helps improve generalization and robustness.
2.Support Vectors: Only a few important data points (called support vectors) are used to define the hyperplane, making the model efficient.
3.Effective in High Dimensions: SVM performs well even when the number of features is very large (high-dimensional spaces).
4. Kernel Trick: Allows SVM to solve non-linear classification problems by transforming the input data into higher-dimensional space using kernel functions (e.g., RBF, polynomial).
5.Versatile: Can be used for binary as well as multi-class classification problems. 
6. Regularization: SVM includes a regularization parameter (C) that balances the trade-off between maximizing the margin and minimizing the classification error.
7. Robust to Overfitting (in high-dimensional space):Due to its focus on margin and support vectors, SVM can generalize well, especially when data is not too noisy.

<img width="1959" height="975" alt="image" src="https://github.com/user-attachments/assets/50a9f7d0-583b-4dd7-bb00-e50d199dfccd" />
<img width="892" height="714" alt="image" src="https://github.com/user-attachments/assets/7f51ce46-3027-4cc8-ba1a-934f885b5b1a" />
<img width="1074" height="758" alt="image" src="https://github.com/user-attachments/assets/cedae137-5d74-4237-9fd7-035cd2da990d" />
Accuracy 97.79 %

# K Neighbors Classifier
K Neighbors  is a classification and regression online (lazy) learning algorithm. It's a non-parametric approach to the extent that it doesn't make any assumption regarding the distribution of the data.
How It Works:
Input Data: To predict a given new data point, KNN computes the similarity between this data point and the entire training data set.
Distance Metric: Any distance metric such as Euclidean, Manhattan, or others can be employed based on the problem.
Finding Neighbors: It finds the k training samples nearest to the input point.
Prediction: For classification, the most common label among the k neighbors is returned. For regression, the mean value of the k neighbors is taken as the prediction.
Model Training & Prediction (Using sklearn):
Training: Employ the fit() function of K Neighbors Classifier or Kneighbors Regressor from sklearn .neighbors.
Prediction: Employ the predict() function on the test set.
Selection of k value:
Selecting k (number of neighbors) is important:
A low k may produce overfitting (high variance).
A high k can smoothen the decision boundary but may lead to underfitting (high bias).
For selection of the best k, vary k and check the model performance using cross-validation.
The location at which adding k no longer further enhances accuracy appreciably is referred to as the knee point.

<img width="1830" height="950" alt="image" src="https://github.com/user-attachments/assets/a06fd428-dcda-4123-91e2-8cf4cd090035" />
<img width="1078" height="713" alt="image" src="https://github.com/user-attachments/assets/98ec4d36-d1b6-4420-93d7-bc4e91307660" />
<img width="810" height="714" alt="image" src="https://github.com/user-attachments/assets/463d432d-8796-42cd-ba7b-c2df8e211c0a" />
Accuracy : 97.06 %

# Deployment (Working Model Diagram)
<img width="1507" height="714" alt="image" src="https://github.com/user-attachments/assets/87ed3fa8-8fa7-4bd7-a057-9e98e47497da" />
<img width="1500" height="713" alt="image" src="https://github.com/user-attachments/assets/5b4e4c57-8d5c-472c-935c-a758ad150bcc" />
<img width="1509" height="713" alt="image" src="https://github.com/user-attachments/assets/d5bb58d7-aab6-4dfa-a924-b0521134cc20" />
<img width="1492" height="713" alt="image" src="https://github.com/user-attachments/assets/3c677137-b429-489a-adf3-1c5ce6b3d29c" />

# Results
<img width="766" height="507" alt="image" src="https://github.com/user-attachments/assets/feee967e-a2d9-43ee-a4a0-875de87d7331" />
<img width="766" height="554" alt="image" src="https://github.com/user-attachments/assets/ff17d57f-1478-4464-a4d0-44065e8a3534" />
<img width="825" height="811" alt="image" src="https://github.com/user-attachments/assets/0905dddb-fb59-4ebe-9657-8f9d03f8387f" />
<img width="953" height="811" alt="image" src="https://github.com/user-attachments/assets/a209c08e-a021-4da3-af2e-48c5bf38f69b" />
<img width="800" height="714" alt="image" src="https://github.com/user-attachments/assets/3958abc8-9944-40d0-87b8-47ae39fea0e2" />
<img width="881" height="714" alt="image" src="https://github.com/user-attachments/assets/84760f0d-9bcd-4ec0-a9c0-45c123c4ec6d" />
<img width="929" height="838" alt="image" src="https://github.com/user-attachments/assets/43bcab03-b632-4e3c-9e66-4c9748385eac" />
<img width="979" height="823" alt="image" src="https://github.com/user-attachments/assets/a454d280-1e25-48a8-9407-f905267a8e90" />
<img width="922" height="845" alt="image" src="https://github.com/user-attachments/assets/094d43f0-6af4-48cd-8c42-a20b00824bfc" />
<img width="806" height="824" alt="image" src="https://github.com/user-attachments/assets/b7a95abb-ed9f-4b7f-8505-bbc683070cf0" />
<img width="797" height="714" alt="image" src="https://github.com/user-attachments/assets/3c863192-631a-4072-bcd8-2d1ecff90339" />
<img width="1147" height="822" alt="image" src="https://github.com/user-attachments/assets/25333808-d9b7-4122-b1f4-fe599cbff304" />
<img width="797" height="714" alt="image" src="https://github.com/user-attachments/assets/f7c6cf69-97f8-4744-a450-1395314adebe" />
<img width="1159" height="882" alt="image" src="https://github.com/user-attachments/assets/bbe2a0f8-181d-492a-b07f-08e5aa21d917" />
<img width="802" height="756" alt="image" src="https://github.com/user-attachments/assets/1dce8d82-4aa4-4e82-9b02-221452e87d78" />
<img width="874" height="756" alt="image" src="https://github.com/user-attachments/assets/d76460eb-33c1-4543-8f6a-0450cc9354ff" />
<img width="1014" height="714" alt="image" src="https://github.com/user-attachments/assets/c1ea2a12-8d56-4b07-8df3-25bf39d65b01" />
<img width="913" height="733" alt="image" src="https://github.com/user-attachments/assets/56b205da-1cf1-47f9-802d-eed4d4e649f4" />
<img width="1003" height="1099" alt="image" src="https://github.com/user-attachments/assets/fafbade4-0b05-4c8b-9681-3cf484d1dd40" />

# Conclusion
The COVID-19 prediction system developed using machine learning leverages a set of clinical symptoms, pre-existing health conditions, lifestyle factors, and exposure risks—including breathing problems, fever, dry cough, sore throat, running nose, asthma, chronic lung disease, headache, heart disease, diabetes, hypertension, fatigue, gastrointestinal issues, travel history, contact with COVID patients, attendance at large gatherings, visits to public-exposed places, family exposure, mask usage, and market sanitization habits—to accurately predict the likelihood of infection. By applying machine learning algorithms to this dataset, the system is able to identify patterns and correlations between symptoms, health conditions, and exposure factors, allowing for timely risk assessment. This approach aids in early detection and preventive measures, helping individuals and healthcare authorities take informed actions to reduce the spread of COVID-19.Overall, the study demonstrates that machine learning models can serve as effective predictive tools in public health, complementing traditional testing methods and supporting data-driven decision-making during pandemic situations.

# Future Scope
The COVID-19 prediction system, which uses symptoms (breathing problems, fever, dry cough, sore throat, running nose, fatigue, headache, gastrointestinal issues), pre-existing conditions (asthma, chronic lung disease, heart disease, diabetes, hypertension), and exposure factors (abroad travel, contact with COVID patients, attendance at large gatherings, visits to public-exposed places, family exposure, mask usage, and market sanitization), has significant potential for future enhancement. Future improvements could include integrating real-time health monitoring through wearable devices or mobile apps, enabling early alerts for high-risk individuals. Models could be extended to predict disease severity, hospitalization needs, and recovery outcomes, helping healthcare systems allocate resources efficiently. Additionally, incorporating larger, more diverse datasets and analyzing patterns related to emerging variants can enhance predictive accuracy. Overall, this approach can support data-driven public health strategies, early intervention, and pandemic preparedness for future outbreaks.






























