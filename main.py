import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import io
import sys
from sklearn.preprocessing import LabelEncoder


# Data Preprocessing
# Converting the response variables 
def wine_quality_labelling(data):
    data['quality'] = data['quality'].map({3: 'bad', 4: 'bad', 5: 'bad', 6: 'good', 7: 'good', 8: 'good'})
    return data

# Converting the categorical variables into numerical values
def categorical_to_numerical(data):
    le = LabelEncoder()
    data['quality'] = le.fit_transform(data['quality'])
    return data

# Splitting the data into dependent and independent variables and train test split
def train_test_spliting(data):
    X = data.drop('quality', axis=1)
    y = data['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test


# Function for model training for logistic regression
def logistic_regression(data):
    data = categorical_to_numerical(data)
    X_train, X_test, y_train, y_test = train_test_spliting(data)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred
    

# Function for model training for SGD Classifier
def SGD_classifier(data):
    data = categorical_to_numerical(data)
    X_train, X_test, y_train, y_test = train_test_spliting(data)
    model = SGDClassifier(penalty=None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred

# Function for model training for Decision Tree
def decision_tree(data):
    data = categorical_to_numerical(data)
    X_train, X_test, y_train, y_test = train_test_spliting(data)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred

# Function for model training for Random Forest
def random_forest(data):
    data = categorical_to_numerical(data)
    X_train, X_test, y_train, y_test = train_test_spliting(data)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred


def main():
    data = pd.read_csv('winequality-red.csv')
    st.title('Wine Quality Prediction')
    original_data = data.copy()
    # Converting the response variables
    wine_quality_labelling(data)

    #drop down for data and info
    option = st.selectbox('Select Option', ['Data','Describe-Data','Info', 'Number of Good and Bad Wine', 'Original Data'])
    if option == 'Data':
        st.write(data)
    elif option == 'Describe-Data':
        st.write(data.describe())
    elif option == 'Info':
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    elif option == 'Number of Good and Bad Wine':
        st.write(data['quality'].value_counts())
    elif option == 'Original Data':
        st.write(original_data)

    # divider
    st.markdown('<hr style="border:3px solid grey;">', unsafe_allow_html=True)


    st.sidebar.title('Model Selection')
    model_option = st.sidebar.selectbox('Select Model', ['Logistic Regression','SGD Classifier', 'Decision Tree', 'Random Forest'])
    


    # Define input ranges for each attribute
    alcohol = st.slider('Alcohol', min_value=8.4, max_value=14.9, value=10.42, step=0.01, format="%.2f", help="Range: 8.4 - 14.9")
    fixed_acidity = st.slider('Fixed Acidity', min_value=4.6, max_value=15.9, value=8.32, step=0.01, format="%.2f", help="Range: 4.6 - 15.9")
    volatile_acidity = st.slider('Volatile Acidity', min_value=0.12, max_value=1.58, value=0.53, step=0.01, format="%.2f", help="Range: 0.12 - 1.58")
    citric_acid = st.slider('Citric Acid', min_value=0.0, max_value=1.0, value=0.27, step=0.01, format="%.2f", help="Range: 0.0 - 1.0")
    residual_sugar = st.slider('Residual Sugar', min_value=0.9, max_value=15.5, value=2.54, step=0.01, format="%.2f", help="Range: 0.9 - 15.5")
    chlorides = st.slider('Chlorides', min_value=0.012, max_value=0.611, value=0.087, step=0.001, format="%.3f", help="Range: 0.012 - 0.611")
    free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', min_value=1, max_value=72, value=16, step=1, help="Range: 1 - 72")
    total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', min_value=6, max_value=289, value=46, step=1, help="Range: 6 - 289")
    density = st.slider('Density', min_value=0.99, max_value=1.004, value=0.997, step=0.001, format="%.3f", help="Range: 0.99 - 1.004")
    pH = st.slider('pH', min_value=2.74, max_value=4.01, value=3.31, step=0.01, format="%.2f", help="Range: 2.74 - 4.01")
    sulphates = st.slider('Sulphates', min_value=0.33, max_value=2.0, value=0.66, step=0.01, format="%.2f", help="Range: 0.33 - 2.0")

    # Logistic Regression
    if model_option == 'Logistic Regression':
        model, X_train, X_test, y_train, y_test, y_pred = logistic_regression(data)
        st.sidebar.write('Training Score :', model.score(X_train, y_train))
        st.sidebar.write('Testing Score :', model.score(X_test, y_test))

        # Predict buttons
        if st.button('Predict Wine Quality', key=1):
            user_input = [alcohol, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates]
            prediction = model.predict([user_input])
            if prediction == [0]:
                st.write('<p style="font-size:40px;">The wine quality is <b style="color:red">bad<b></p>', unsafe_allow_html=True)
            else:
                st.write('<p style="font-size:40px;">The wine quality is <b style="color:green">good<b></p>', unsafe_allow_html=True)

        st.markdown('---')

        # generate a report as dropdown
        report = st.selectbox('Select Report', ['Classification Report', 'Confusion Matrix'], key=10)
        if report == 'Classification Report':
            st.image('Resources/Logistic_regression_reports/classification_report.png')
        elif report == 'Confusion Matrix':
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, cmap = 'viridis', fmt = '3.0f')
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            st.pyplot(fig)

    # SGD Classifier
    elif model_option == 'SGD Classifier':
        model, X_train, X_test, y_train, y_test, y_pred = SGD_classifier(data)
        st.sidebar.write('Training Score :', model.score(X_train, y_train))
        st.sidebar.write('Testing Score :', model.score(X_test, y_test))

        if st.button('Predict Wine Quality', key=2):
            user_input = [alcohol, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates]
            prediction = model.predict([user_input])
            if prediction == [0]:
                st.write('<p style="font-size:40px;">The wine quality is <b style="color:red">bad<b></p>', unsafe_allow_html=True)
            else:
                st.write('<p style="font-size:40px;">The wine quality is <b style="color:green">good<b></p>', unsafe_allow_html=True)

        st.markdown('---')

        # generate a report as dropdown
        report = st.selectbox('Select Report', ['Classification Report', 'Confusion Matrix'], key=11)
        if report == 'Classification Report':
            st.image('Resources/SGD_classifier_reports/classifictaion_report.png')
        elif report == 'Confusion Matrix':
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, cmap = 'viridis', fmt = '3.0f')
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            st.pyplot(fig)

    # Decision Tree
    elif model_option == 'Decision Tree':
        model, X_train, X_test, y_train, y_test, y_pred = decision_tree(data)
        st.sidebar.write('Training Score :', model.score(X_train, y_train))
        st.sidebar.write('Testing Score :', model.score(X_test, y_test))
        st.sidebar.write('Max Depth :', model.get_depth())
        st.sidebar.write('Min Samples Split :', model.min_samples_split)
        st.sidebar.write('Min Samples Leaf :', model.get_n_leaves())
        st.sidebar.write('Max Features :', model.max_features)
        st.sidebar.write('Cross Validation Score :', cross_val_score(model, X_train, y_train, cv=10).mean())

        if st.button('Predict Wine Quality', key=4):
            user_input = [alcohol, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates]
            prediction = model.predict([user_input])
            if prediction == [0]:
                st.write('<p style="font-size:40px;">The wine quality is <b style="color:red">bad<b></p>', unsafe_allow_html=True)
            else:
                st.write('<p style="font-size:40px;">The wine quality is <b style="color:green">good<b></p>', unsafe_allow_html=True)

        st.markdown('---')

        # generate a report as dropdown
        report = st.selectbox('Select Report', ['Classification Report', 'Confusion Matrix'], key=13)
        if report == 'Classification Report':
            st.image('Resources/Decision_tree_reports/classification_report.png')
        elif report == 'Confusion Matrix':
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, cmap = 'viridis', fmt = '3.0f')
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            st.pyplot(fig)

    # Random Forest
    elif model_option == 'Random Forest':
        model, X_train, X_test, y_train, y_test, y_pred = random_forest(data)
        st.sidebar.write('Training Score :', model.score(X_train, y_train))
        st.sidebar.write('Testing Score :', model.score(X_test, y_test))
        st.sidebar.write('Cross Validation Score :', cross_val_score(model, X_train, y_train, cv=10).mean())

        if st.button('Predict Wine Quality', key=5):
            user_input = [alcohol, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates]
            prediction = model.predict([user_input])
            if prediction == [0]:
                st.write('<p style="font-size:40px;">The wine quality is <b style="color:red">bad<b></p>', unsafe_allow_html=True)
            else:
                st.write('<p style="font-size:40px;">The wine quality is <b style="color:green">good<b></p>', unsafe_allow_html=True)

        st.markdown('---')

        # generate a report as dropdown
        report = st.selectbox('Select Report', ['Classification Report', 'Confusion Matrix'], key=14)
        if report == 'Classification Report':
            st.image('Resources/Random_forest_reports/classification_report.png')
        elif report == 'Confusion Matrix':
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, cmap = 'viridis', fmt = '3.0f')
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            st.pyplot(fig)


    # divider
    st.markdown('<hr style="border:3px solid grey;">', unsafe_allow_html=True)

    # Data Visualization
    st.title('Data Visualization')

    # drop down for data visualization
    data_visualization = st.selectbox('Relation of wine with various attributes', ['Correlation Matrix',
                                                                    'Relation of fixed acidity with wine (Scatter plot)', 
                                                                    'Relation of alcohol with wine (Bar plot)', 
                                                                    'Relation of citric acid with wine (Bar plot)' , 
                                                                    'Relation of residual sugar with wine (Bar plot)', 
                                                                    'Relation of chlorides with wine (Bar plot)',
                                                                    'Relation of free sulfur dioxide with wine (Bar plot)',
                                                                    'Relation of sulphates with wine (Bar plot)',
                                                                    ])
    if data_visualization == 'Relation of fixed acidity with wine (Scatter plot)':
        fig, ax = plt.subplots()
        sns.scatterplot(x='quality', y='fixed acidity', data=original_data, ax=ax)
        ax.set_title('Relation of fixed acidity with wine')
        ax.set_xlabel('Quality')
        ax.set_ylabel('Fixed Acidity')
        st.pyplot(fig)
    elif data_visualization == 'Relation of alcohol with wine (Bar plot)':
        fig, ax = plt.subplots()
        plt.bar(original_data['quality'], original_data['alcohol'], color = 'maroon')
        ax.set_title('Relation of alcohol with wine')
        st.pyplot(fig)
    elif data_visualization == 'Relation of citric acid with wine (Bar plot)':
        fig, ax = plt.subplots()
        sns.barplot(x='quality', y='citric acid', data=original_data, ax=ax)
        ax.set_title('Relation of citric acid with wine')
        ax.set_xlabel('Quality')
        ax.set_ylabel('Citric Acid')
        st.pyplot(fig)
    elif data_visualization == 'Relation of residual sugar with wine (Bar plot)':
        fig, ax = plt.subplots()
        sns.barplot(x='quality', y='residual sugar', data=original_data, ax=ax)
        ax.set_title('Relation of residual sugar with wine')
        ax.set_xlabel('Quality')
        ax.set_ylabel('Residual Sugar')
        st.pyplot(fig)
    elif data_visualization == 'Relation of chlorides with wine (Bar plot)':
        fig, ax = plt.subplots()
        sns.barplot(x='quality', y='chlorides', data=original_data, ax=ax)
        ax.set_title('Relation of chlorides with wine')
        ax.set_xlabel('Quality')
        ax.set_ylabel('Chlorides')
        st.pyplot(fig)
    elif data_visualization == 'Relation of free sulfur dioxide with wine (Bar plot)':
        fig, ax = plt.subplots()
        sns.barplot(x='quality', y='free sulfur dioxide', data=original_data, ax=ax)
        ax.set_title('Relation of free sulfur dioxide with wine')
        ax.set_xlabel('Quality')
        ax.set_ylabel('Free Sulfur Dioxide')
        st.pyplot(fig)
    elif data_visualization == 'Relation of sulphates with wine (Bar plot)':
        fig, ax = plt.subplots()
        sns.barplot(x='quality', y='sulphates', data=original_data, ax=ax)
        ax.set_title('Relation of sulphates with wine')
        ax.set_xlabel('Quality')
        ax.set_ylabel('Sulphates')
        st.pyplot(fig)
    elif data_visualization == 'Correlation Matrix':
        f, ax = plt.subplots(figsize=(10, 10))
        corr = original_data.corr()
        sns.heatmap(corr, annot=True, ax=ax, cmap=sns.diverging_palette(220, 10, as_cmap=True))
        ax.set_title('Correlation Matrix')
        st.pyplot(f)



if __name__ == '__main__':
    main()