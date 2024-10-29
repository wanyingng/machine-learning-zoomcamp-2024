# Load the model
import pickle

input_file = 'model_C=1.0.bin'

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

# Turn this customer into feature matrix
X = dv.transform([customer])
# Get the probability that this particular customer is going to churn
y_pred = model.predict_proba(X)[0, 1]

print('Input', customer)
print('Churn probability', y_pred)
