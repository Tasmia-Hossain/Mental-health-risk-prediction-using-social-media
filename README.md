# Mental Health Risk Prediction Using Social Media Sentiment Analysis

This repository contains a complete machine learning framework for predicting mental-health risk from social-media text. The system integrates sentiment analysis, linguistic feature extraction, TF-IDF vectorization, and multiple supervised learning models. CodeCarbon is used to measure model-specific CO₂ emissions to support Green AI evaluation.

This work was completed as part of the Pattern Recognition and Machine Learning (PRML) Lab.

---

## Project Files Included

This repository contains the following files:

### **1. Jupyter Notebook**
**`mental-health-risk-prediction-using-social-media-sentiment-analysis.ipynb`**  
Contains the full implementation:
- Preprocessing  
- Sentiment feature extraction  
- TF-IDF vectorization  
- Model training  
- Performance evaluation  
- Carbon emission tracking  

### **2. Research Report**
**`Report_Mental_Health_Risk_Prediction_Using_Social_Media_Sentiment_Analysis.pdf`**  
A detailed academic report covering:
- Literature review  
- Dataset analysis  
- Methodology  
- Experimental results  
- Green AI analysis  
- Conclusion  

### **3. Presentation Slide**
**`Mental Health Risk Prediction Using Social Media Sentiment Analysis.pptx`**  
The project presentation used for academic evaluation:
- Problem motivation  
- Key insights  
- Model outcomes  
- Sustainability analysis  

---

## Dataset

Dataset used: **Sentiment Analysis for Mental Health**  
Dataset source (Kaggle):  
(https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

Dataset details:
- Total samples: **52,266**
- Original classes: **7**
- Converted to binary:
  - **High-risk:** Depression, Suicidal, Anxiety, Bipolar, Stress, Personality disorder  
  - **Low-risk:** Normal  

All preprocessing steps are included inside the notebook.

---

## Feature Engineering

The system extracts four major feature categories:

### **1. TF-IDF Features**
- 1,000 unigram features  
- Stopword removal  
- min_df = 2, max_df = 0.9  

### **2. Sentiment Features**
From VADER:
- compound  
- positive  
- neutral  
- negative  

From TextBlob:
- polarity  
- subjectivity  

### **3. Linguistic Features**
- Text length  
- Word count  
- Capitalization ratio  
- Exclamation / question frequency  
- Average word length  

### **4. Keyword Features**
- Mental health risk keywords  
- Positive emotion keywords  

Final feature vector size: **1009 dimensions**

---

## Machine Learning Models

The following six models were trained and compared:

1. Logistic Regression  
2. Random Forest  
3. Gradient Boosting  
4. Linear SVM  
5. Naive Bayes  
6. Neural Network (MLP)

All models include:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC AUC  
- Training time  
- CO₂ emission via CodeCarbon  

---

## Results Summary

### **Highest Accuracy**
**Neural Network (MLP)**  
- Accuracy: **93.0%**  
- ROC AUC: **0.974**

### **Most Environmentally Efficient (Green AI)**
**Naive Bayes**
- CO₂ emission: **0.000001 kg**  
- Training time: **0.3 seconds**

### **Best Practical Trade-Off**
**Random Forest**  
Strong accuracy with low carbon footprint.

This demonstrates the balance between performance and sustainability.

---

## Green AI Analysis

Using **CodeCarbon**, the following were measured:

- Energy consumption  
- CO₂ emissions  
- Training time  
- Accuracy-to-emission efficiency score  

Efficiency ranking:
1. Naive Bayes  
2. Random Forest  
3. Neural Network  
4. Linear SVM  
5. Logistic Regression  
6. Gradient Boosting  

---

## How to Run the Project

### 1. Install dependencies

pip install -r requirements.txt


### 2. Open Jupyter Notebook


jupyter notebook


### 3. Run the notebook


mental-health-risk-prediction-using-social-media-sentiment-analysis.ipynb


---

## Technologies Used

- Python  
- Scikit-learn  
- VADER Sentiment Analyzer  
- TextBlob  
- CodeCarbon  
- NumPy, Pandas  
- Matplotlib, Seaborn  

---

## Authors

- **Tasmia Hossain**  
- **Abu Dojana**  
- **Md. Ahnaf Ahsan**

Ahsanullah University of Science and Technology (AUST)  
Department of Computer Science and Engineering  

---

## License

This project is intended for academic and research use only.
