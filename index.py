from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model
with open("LinearRegression.pkl", "rb") as f:
    model = pickle.load(f)

with open("bike_model.pkl", "rb") as h:
    model4 = pickle.load(h)


with open("laptop_price_model.pkl", "rb") as g:
    model3 = pickle.load(g)

model2 = pickle.load(open("Mobile_price_model.pkl", "rb"))
# Load the DataFrame containing the car data
car = pd.read_csv("Final_dataset.csv")

Bike = pd.read_csv("Bike_Final.csv")

laptop = pd.read_csv("Final_dataset_laptop.csv")

# Assuming 'car' is the DataFrame containing your dataset
# Replace 'car' with your actual DataFrame if it has a different name
def get_unique_values():
    names = car["name"].unique().tolist()
    companies = car["company"].unique().tolist()
    fuel_types = car["fuel_type"].unique().tolist()

    return {"names": names, "companies": companies, "fuel_types": fuel_types}



def get_unique_values2():
    companies = laptop["Company"].unique().tolist()
    type_names = laptop["TypeName"].unique().tolist()
    cpu_brands = laptop["Cpu brand"].unique().tolist()
    gpu_brands = laptop["Gpu brand"].unique().tolist()

    return {"companies": companies, "type_names": type_names, "cpu_brands": cpu_brands, "gpu_brands": gpu_brands}

def get_unique_values3():
    bike_name = Bike["bike_name"].unique().tolist()
    city = Bike["city"].unique().tolist()
    owner = Bike["owner"].unique().tolist()
    brand = Bike["brand"].unique().tolist()

    return {"bike_name": bike_name, "city": city, "owner": owner, "brand": brand}


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/car_form')
def car_form():
    return render_template('car_form.html')

@app.route('/mobile_form')
def mobile_form():
    return render_template('mobile_form.html')


@app.route('/laptop_form')
def laptop_form():
    return render_template('laptop_form.html')



@app.route('/bike_form')
def bike_form():
    return render_template('bike_form.html')

@app.route('/vehicles')
def vehicles():
    return render_template('vehicles.html')

@app.route('/electronics')
def electronics():
    return render_template('electronics.html')


@app.route("/get_dropdown_options")
def get_dropdown_options():
    options = get_unique_values()
    return jsonify(options)


@app.route("/get_dropdown_options2")
def get_dropdown_options2():
    options = get_unique_values2()
    return jsonify(options)

@app.route("/get_dropdown_options3")
def get_dropdown_options3():
    options = get_unique_values3()
    return jsonify(options)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert the input data into a DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # Use the model to make predictions
    predicted_price = model.predict(input_data)

    return jsonify({"price": predicted_price[0]})

@app.route('/predictmobile', methods=['POST'])
def predictmobile():
    data = request.form.to_dict()
    features = np.array(list(data.values())).reshape(1, -1).astype(float)
    prediction = model2.predict(features)
    return jsonify(prediction.tolist())

@app.route("/predictlaptop", methods=["POST"])
def predictlaptop():
    data = request.get_json()

    # Convert the input data into a DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # Use the model to make predictions
    predicted_price = model3.predict(input_data)

    return jsonify({"price": predicted_price[0]})


@app.route("/predictbike", methods=["POST"])
def predictbike():
    data = request.get_json()

    # Convert the input data into a DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # Use the model to make predictions
    predicted_price = model4.predict(input_data)

    return jsonify({"price": predicted_price[0]})




if __name__ == "__main__":
    app.run(debug=True)
