import ipywidgets as widgets
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("./generated_dataset.csv")

from sklearn.metrics import accuracy_score

# Predict on the test set
# y_pred = model.predict(X_test)


df.head()

df.columns

df.info()

df["Card Type"].unique()

df["POS Entry Mode"].unique()

df["Transaction Type"].unique()

df.isnull()

df.isnull().sum()

df.shape

# Splitting the data into features and target variable
X = df.drop(["Fraud"], axis=1)  # Features
y = df["Fraud"]  # Target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numeric and categorical features
numeric_features = ["Card Number", "Transaction Amount"]
categorical_features = ["Card Type", "Currency", "Transaction Date and Time", "Merchant Name", "Transaction Status"]

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)]
)

# Random Forest Classifier pipeline
model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42))])

# Train the model
model.fit(X_train, y_train)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset with the new transaction
new_transaction_df = pd.concat([X_test, pd.DataFrame({"Fraud": y_test})], axis=1)

# Predictions on test set
y_pred = model.predict(X_test)

y_pred

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# prompt: np.argmax(scores)

scores = [0.1, 0.2, 0.3, 0.4, 0.5]
np.argmax(scores)

# Convert boolean predictions to numeric values
y_pred_numeric = np.where(y_pred == True, 1, 0)


import pandas as pd

data = [
    [
        "Michael Howell",
        "3525145677277322",
        "Diners Club / Carte Blanche",
        "11/31",
        886.18,
        "THB",
        "03/09/2024 00:56",
        "Hall and Sons",
        "8e4d554f-0081-463e-a38e-7b7dca25a03e",
        "57a6df6d-ba68-418e-b225-d3abfb3442b0",
        "Declined",
        "6d42dafa-fd19-4f70-a6e9-95faab423ed9",
        "Chip",
        "Purchase",
        897,
        "Not Matched",
        "Not Required",
        "Settled",
    ]
]

columns = [
    "Cardholder Name",
    "Card Number",
    "Card Type",
    "Card Expiry Date",
    "Transaction Amount",
    "Currency",
    "Transaction Date and Time",
    "Merchant Name",
    "Merchant ID",
    "Authorization Code",
    "Authorization Response",
    "Terminal ID",
    "POS Entry Mode",
    "Transaction Type",
    "CVV/CVC",
    "AVS Data",
    "Customer Signature",
    "Transaction Status",
]

df = pd.DataFrame(data, columns=columns)

# Assuming you have defined your pipeline as 'pipe'
predictions = model.predict(df)

predictions

import pandas as pd
import ipywidgets as widgets
from IPython.display import display


# Define function for prediction
def predict_transaction(request_data):
    # Create DataFrame from form data
    df = pd.DataFrame([request_data])

    # Assuming you have defined your model as 'model'
    return model.predict(df)[0]


# Define textboxes for user input
cardholder_name_text = widgets.Text(placeholder="Enter cardholder name")
card_number_text = widgets.Text(placeholder="Enter card number")
card_type_text = widgets.Text(placeholder="Enter card type")
card_expiry_date_text = widgets.Text(placeholder="Enter card expiry date")
transaction_amount_text = widgets.Text(placeholder="Enter transaction amount")
currency_text = widgets.Text(placeholder="Enter currency")
transaction_date_time_text = widgets.Text(placeholder="Enter transaction date and time")
merchant_name_text = widgets.Text(placeholder="Enter merchant name")
merchant_id_text = widgets.Text(placeholder="Enter merchant ID")
authorization_code_text = widgets.Text(placeholder="Enter authorization code")
authorization_response_text = widgets.Text(placeholder="Enter authorization response")
terminal_id_text = widgets.Text(placeholder="Enter terminal ID")
pos_entry_mode_text = widgets.Text(placeholder="Enter POS entry mode")
transaction_type_text = widgets.Text(placeholder="Enter transaction type")
cvv_cvc_text = widgets.Text(placeholder="Enter CVV/CVC")
avs_data_text = widgets.Text(placeholder="Enter AVS data")
customer_signature_text = widgets.Text(placeholder="Enter customer signature")
transaction_status_text = widgets.Text(placeholder="Enter transaction status")

# Define predict button
predict_button = widgets.Button(description="Predict")

# Define output area
output = widgets.Output()


# Define function to handle button click event
def on_predict_button_clicked(request_data):
    with output:
        output.clear_output()

    return predict_transaction(request_data)


# Attach button click event
predict_button.on_click(on_predict_button_clicked)

# Display widgets
display(
    widgets.VBox(
        [
            cardholder_name_text,
            card_number_text,
            card_type_text,
            card_expiry_date_text,
            transaction_amount_text,
            currency_text,
            transaction_date_time_text,
            merchant_name_text,
            merchant_id_text,
            authorization_code_text,
            authorization_response_text,
            terminal_id_text,
            pos_entry_mode_text,
            transaction_type_text,
            cvv_cvc_text,
            avs_data_text,
            customer_signature_text,
            transaction_status_text,
            predict_button,
            output,
        ]
    )
)

# Confusion matrix heatmap
# def plot_confusion_matrix(y_true, y_pred):
#     cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
#     plt.title('Confusion Matrix')
#     plt.show()

# plot_confusion_matrix(y_test, y_pred)

# Pie chart for fraud distribution
# def plot_fraud_distribution(y):
#     fraud_counts = y.value_counts()
#     plt.figure(figsize=(6, 6))
#     plt.pie(fraud_counts, labels=fraud_counts.index, autopct='%1.1f%%', startangle=140)
#     plt.title('Fraud Distribution')
#     plt.axis('equal')
#     plt.show()

# plot_fraud_distribution(df['Fraud'])


# Boxplot for numeric features
# def plot_numeric_boxplot(df, numeric_features):
#     plt.figure(figsize=(10, 4))
#     for i, feature in enumerate(numeric_features, 1):
#         plt.subplot(1, len(numeric_features), i)
#         sns.boxplot(y=feature, data=df, palette='viridis')
#         plt.title(f'Boxplot of {feature}')
#     plt.tight_layout()
#     plt.show()

# plot_numeric_boxplot(df, numeric_features)

# Scatter plot for numeric features
# def plot_numeric_scatter(df, numeric_features, target_feature):
#     plt.figure(figsize=(10, 4))
#     for i, feature in enumerate(numeric_features, 1):
#         plt.subplot(1, len(numeric_features), i)
#         sns.scatterplot(x=feature, y=target_feature, data=df, alpha=0.5)
#         plt.title(f'{feature} vs {target_feature}')
#     plt.tight_layout()
#     plt.show()

# plot_numeric_scatter(df, numeric_features, 'Fraud')
