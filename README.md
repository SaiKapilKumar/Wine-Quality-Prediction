# Wine Quality Prediction

This project is about predicting the quality of wine based on various features. The project uses several machine learning models from the sklearn library.

## Dependencies

The project requires the following Python libraries:

- streamlit
- pandas
- seaborn
- matplotlib
- sklearn
- io
- sys

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

## File Description

`main.py`: This is the main file of the project. It contains the following functions:

- `wine_quality_labelling(data)`: This function takes a DataFrame as input and maps the 'quality' column to 'bad' or 'good' based on the quality score.

- `categorical_to_numerical(data)`: This function converts the categorical 'quality' column into numerical values using LabelEncoder.

- `train_test_spliting(data)`: This function splits the data into training and testing sets. It also scales the features using StandardScaler.

## Usage

To run the project, navigate to the project directory and run the following command:

```bash
streamlit run main.py