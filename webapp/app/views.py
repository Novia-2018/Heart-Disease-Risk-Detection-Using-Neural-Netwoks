from django.shortcuts import render
import joblib
# Create your views here.
from .forms import InputForm
import pandas as pd
import numpy as np

knn_from_joblib = joblib.load('models/mlp_model.pkl')
sc_model = joblib.load('models/sc_model.pkl')
X_train = joblib.load('models/data.pkl')


def home(request) :
    return render(request,'heart/home1.html')

def forms(request) :
    return render(request,'heart/HForm.html')

def quality (request) :
        # if this is a POST request we need to process the form data
        if request.method == 'POST' :
            # create a form instance and populate it with data from the request:
            form = InputForm ( request.POST )
            # check whether it's valid:
            if form.is_valid ( ) :
                # process the data in form.cleaned_data as required
                number2 = form.cleaned_data.get ( 'number2' )
                number3 = form.cleaned_data.get ( 'number3' )
                number4 = form.cleaned_data.get ( 'number4' )
                number5 = form.cleaned_data.get ( 'number5' )
                number6 = form.cleaned_data.get ( 'number6' )
                number7 = form.cleaned_data.get ( 'number7' )
                number8 = form.cleaned_data.get ( 'number8' )
                number9 = form.cleaned_data.get ( 'number9' )
                number10 = form.cleaned_data.get ( 'number10' )
                number11 = form.cleaned_data.get ( 'number11' )
                number13 = form.cleaned_data.get ( 'number13' )
                number14 = form.cleaned_data.get ( 'number14' )
                number15 = form.cleaned_data.get ( 'number15' )
                number16 = form.cleaned_data.get ( 'number16' )
                number17 = form.cleaned_data.get ( 'number16' )
                a = [ number2,number3,number4,number5,number6,number7,number8,number9,number10,number11,number17,number13,number14,number15,number16]
                a = np.array ( a )
                a = a.reshape ( len ( a ), 1 )
                a = a.reshape ( 1, -1 )
                p = sc_model.fit_transform(X_train)
                a = sc_model.transform(a)
                y_pred = knn_from_joblib.predict(a)
                if y_pred == [0]:
                    return render ( request, 'heart/result1.html' )

                else:
                     return  render ( request, 'heart/result2.html' )


                #return render ( request, 'heart/result.html', context )

        # if a GET (or any other method) we'll create a blank form
        return render ( request, 'heart/result3.html')