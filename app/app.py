from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

def validate_input(data):
    ranges = {
        'Alcohol': (0,20),
        'Hepatitis B': (0,100),
        'Measles': (0,100000),
        'Polio': (0,100),
        'Diphtheria': (0,100),
        'HIV/AIDS': (0,50),
    }
    errors = []
    for feature, value in data.items():
        if feature in ranges:
            min_val, max_val = ranges[feature]
            if value < min_val or value > max_val:
                errors.append(f'{feature}: Value must be between {min_val} and {max_val}')
    return errors

@app.route('/',methods=['GET','POST'])
def home():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        alochol = float(request.form['alcohol'])
        hepatitis_b = float(request.form['hepatitis_b'])
        measles = int(request.form['measles'])
        polio = float(request.form['polio'])
        diphtheria = float(request.form['diphtheria'])
        hiv_aids = float(request.form['hiv_aids'])

        # Prepare the input data for prediction
        input_data = {
            'Alcohol': alochol,
            'Hepatitis B': hepatitis_b,
            'Measles': measles,
            'Polio': polio,
            'Diphtheria': diphtheria,
            'HIV/AIDS': hiv_aids
            }

        # Validate the input data
        errors =validate_input(input_data)

        if errors:
            return render_template('index.html', errors=errors)
    
        # Convert the input data to a numpy array
        input_data = np.array(list(input_data.values())).reshape(1,-1)

        # Use the loaded model to make the prediction
        prediction = model.predict(input_data)[0]

        # Redirect to the result page with the prediction
        return redirect(url_for('result',prediction=float(prediction)))

    # If it is a GET request, render the index page
    return render_template('index.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html',prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)










