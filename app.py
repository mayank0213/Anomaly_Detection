from flask import Flask, render_template, request, jsonify,redirect,url_for
import pandas as pd
from rishab_adarsh_pranjya import predict_transaction, on_predict_button_clicked

app = Flask(__name__, template_folder="template")

# Handle requests for favicon.ico
@app.route("/favicon.ico")
def favicon():
    return app.send_static_file("favicon.ico")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive form data
        request_data = request.form.to_dict()
        # df = pd.DataFrame([data])
        # Make prediction using your ML model
        prediction = on_predict_button_clicked(request_data)
        # Return prediction result as JSON

        if prediction:
            return jsonify({"prediction": "true"})

        return jsonify({"prediction": "false"})
    except Exception as e:
        # Log the error
        print("Error:", e)
        # Return an error response
        return jsonify({"error": "An error occurred during prediction"})


if __name__ == "__main__":
    app.run(debug=True)
