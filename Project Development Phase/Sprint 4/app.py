import json
import pandas as pd 
import numpy as np 
from flask import Flask, request, render_template
import pickle 
from sklearn.preprocessing import LabelEncoder 
import pickle

app=Flask(__name__)

model_rand = pickle.load(open('model.pkl','rb'))

@app.route('/') 
def index():
   return render_template('resalepredict.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
   if request.method == "POST":
      with open('mapping.json', 'r') as file:
         mapping = json.load(file)
         
      vehicle_type = mapping['vehicleType'][request.form.get('vehicletype')]
      years_old = 2022 - int(request.form['regyear']) 
      gearbox = mapping['gearbox'][request.form.get('gearbox')] 
      powerps = float(request.form['powerps'])
      kms = float(request.form['kms']) 
      fuelType = mapping['fuelType'][request.form.get('fuel')]
      brand = mapping['brand'][request.form.get('brand')]
      damage = mapping['notRepairedDamage'][request.form.get('damage')] 
      
      data = [[vehicle_type, years_old, gearbox, powerps, kms, fuelType, brand, damage]] 
      pred = model_rand.predict(data) 
      print(pred) 
   return render_template('output.html', pred='The resale value predicted is ${:.2f}'.format(pred[0]))
