from django.shortcuts import render
import pickle
import sklearn
print(sklearn.__version__)

# Create your views here.

# Define a global variable to hold the loaded model
loaded_model = None

def load_model():
    global loaded_model
    # Load the trained model from disk
    with open('myhftapp/savedmodels/random_forest_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

load_model()

def home(request):
    return render(request,'home.html')

def predict_view(request):
    if request.method == 'POST':
        input_data = request.POST.get('input_data')
        if loaded_model:
            # Perform prediction using the loaded model
            prediction = loaded_model.predict(input_data)
            return render(request, 'result.html', {'prediction': prediction})
        else:
            # Render error message if model is not loaded
            error_message = 'Model not loaded'
            return render(request, 'predict.html', {'error_message': error_message})
    return render(request, 'predict.html')