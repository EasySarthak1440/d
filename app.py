from flask import Flask, render_template,request 
import numpy as np
import pickle

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result',method=['POST'])
def result():
    a=request.form.get('pregnancies')
    b=request.form.get('glucose')
    c=request.form.get('bp')
    d=request.form.get('skinthickness')
    e=request.form.get('insulin')
    f=request.form.get('bmi')
    g=float(request.form.get('pedigree'))
    h=request.form.get('age')

    userdata=(a,b,c,d,e,f,g,h)
    userarray=np.asarray(userdata)
    reshaped=userarray.reshape(1,-1)
    prediction=model.predict(reshaped)
    if prediction==0:
        result='diabetes absent'
    else:
        result='diabetes present'
    return render_template('result.html',answer=result)

app.debug=1
app.run()