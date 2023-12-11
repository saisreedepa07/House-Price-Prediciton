from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import plotly.graph_objs as go

app = Flask(__name__)
CORS(app)

#considering highest accuracy gained model
with open('xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)
#with open('random_forest.pkl', 'rb') as file:
    #rf_model = pickle.load(file)
#with open('GBR_model.pkl', 'rb') as file:
   #GBR_model = pickle.load(file)

#for dropdown menu options, mapping them with the correct key
method_mapping = {'S': 'Method_S', 'SP': 'Method_SP', 'VB': 'Method_VB', 'SA': 'Method_SA'}
property_mapping = {'House': ('Type_t', 'Type_u', 0, 0),
                    'Townhouse': ('Type_t', 'Type_u', 1, 0),
                    'Unit/Apartment': ('Type_t', 'Type_u', 0, 1)}

def process_input(data):
    processed_data = []
    #considering the columns that are used in the models for predicition
    for key in sorted(data.keys()):
        if key in ['Rooms', 'Postcode', 'Bathroom', 'Car', 'YearBuilt', 'Propertycount', 
                   'Type_t', 'Type_u', 'Method_S', 'Method_SA', 'Method_SP', 'Method_VB', 
                   'Regionname_Eastern Victoria', 'Regionname_Northern Metropolitan', 
                   'Regionname_Northern Victoria', 'Regionname_South-Eastern Metropolitan', 
                   'Regionname_Southern Metropolitan', 'Regionname_Western Metropolitan', 
                   'Regionname_Western Victoria', 'Year', 'Month', 'Day', 'BuildingAge', 
                   'Suburb_encoded', 'SellerG_encoded', 'CouncilArea_encoded']:
            processed_data.append(int(data[key]))
        elif key in ['Distance', 'Landsize', 'BuildingArea', 'Lattitude', 'Longtitude', 'Price_per_sqm']:
            processed_data.append(float(data[key]))
        elif key == 'Method':
            # For 'Method' field assigning selected value to 1 and others to 0 
            for method, field in method_mapping.items():
                processed_data.append(1 if data[key] == method else 0)
        elif key == 'PropertyType':
            # For 'PropertyType' field assigning selected value to 1 and others to 0 
            mapping_values = property_mapping.get(data[key])
            if mapping_values:
                for val in mapping_values[2:]:
                    processed_data.append(val)
        elif key == 'Region':
            # Update the region fields
            for region_key in data.keys():
                if region_key.startswith('Regionname_'):
                    if region_key == f"Regionname_{data[key]}":
                        processed_data.append(1)
                    else:
                        processed_data.append(0)
        else:
            raise ValueError(f"Unexpected data type for field: {key}")

    return processed_data
@app.route('/')
def base_page():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict(flat=True)
    exclude_keys = ['Region', 'PropertyType', 'Method']
    data = {key: value for key, value in data.items() if key not in exclude_keys}

    print("Received data:", data)
    model_name = data.pop('model', 'xgboost')  

    #if model_name == 'random_forest':
       # model = rf_model
    #elif model_name == 'GBR':
        #model = GBR_model
    #else:
    model = xgb_model

    processed_data = process_input(data)
    prediction = model.predict([processed_data])
    prediction = prediction.tolist()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(port=5000,debug=True)
