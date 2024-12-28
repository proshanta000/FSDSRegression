from flask import Flask, request, render_template, redirect, url_for
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        try:
            data = CustomData(
                carat=float(request.form.get('carat')),
                depth=float(request.form.get('depth')),
                table=float(request.form.get('table')),
                x=float(request.form.get('x')),
                y=float(request.form.get('y')),
                z=float(request.form.get('z')),
                cut=request.form.get('cut'),
                color=request.form.get('color'),
                clarity=request.form.get('clarity')
            )

            final_new_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)

            if pred is None or not pred:
                return render_template('results.html', error="Prediction failed. Please check your input data or model.")

            results = round(pred[0], 2)
            # Corrected: Render the template directly, no redirect needed
            return render_template('results.html', results=results)

        except ValueError as e:
            return render_template('results.html', error=f"Invalid input: {e}")
        except Exception as e:
            return render_template('results.html', error=f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)