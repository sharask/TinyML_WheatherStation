import csv
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sklearn.metrics
import tensorflow as tf

from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import activations
from tensorflow.keras import layers


BATCH_SIZE = 64
MIN_SNOW_CM = 5 # Above this value, we consider it as snow
NUM_EPOCHS = 20
OUTPUT_DATASET_FILE = "snow_dataset.csv"
TFL_MODEL_FILE = "snow_model.tflite"
TFL_MODEL_HEADER_FILE = "model.h"
TF_MODEL = "snow_forecast"


import calendar
import requests

api_key = '4a52b68225ab4690925121546252305'
city = 'canazei'

t_list = []
h_list = []
s_list = []

for year in range(2011, 2021):
  for month in range(1, 13):
    num_days_month = calendar.monthrange(year, month)[1]
    start_date='{year}-{month}-01'.format(year=year, month=month)
    end_date='{year}-{month}-{last_day}'.format(year=year, month=month, last_day=num_days_month)

    url_base = 'http://api.worldweatheronline.com/premium/v1/past-weather.ashx'
    api_url = url_base + "?key={key}&q={city}&format=json&date={start_date}&enddate={end_date}&tp=1".format(key=api_key,
                                                                                                            city=city,
                                                                                                            start_date=start_date,
                                                                                                            end_date=end_date)

    print(api_url)

    response = requests.get(api_url)

    if response.status_code == 200:
      json = response.json()

      for x in json['data']['weather']:
        snow_in_cm = float(x['totalSnow_cm'])
        for y in x['hourly']:
          t = float(y['tempC'])
          h = float(y['humidity'])
          t_list.append(t)
          h_list.append(h)
          s_list.append(snow_in_cm)


t_bin_list = []
h_bin_list = []

for snow, t, h in zip(s_list, t_list, h_list):
  if snow > MIN_SNOW_CM:
    t_bin_list.append(t)
    h_bin_list.append(h)

plt.figure(dpi=100)
sc = plt.scatter(t_bin_list, h_bin_list, c='#000000', label="Snow")
plt.grid(color = '#AAAAAA', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.title("Snowfall")
plt.xlabel("Temperature - °C")
plt.ylabel("Humidity - %")
plt.show()