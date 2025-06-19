####################################################################
# This script prepares a machine learning model for snow forecasting
# using weather data from a CSV file. It includes data extraction, etc.
# Duomenys jau atsiųsti ir išsaugoti į CSV failą, todėl šis žingsnis yra praleistas. (žr. Get_Wheather_Open_Meteo.py)
####################################################################


import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sklearn.metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = viskas, 1 = INFO (numatytasis), 2 = WARNING, 3 = ERROR
import tensorflow as tf

from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from tensorflow.keras import activations
from tensorflow.keras import layers, Input


# Constants for Snow Forecasting Model
BATCH_SIZE = 64
MELTING_TEMPERATURE = 2
MIN_SNOW_CM = 0.0 # Above this value, we consider it as snow (Originaliai buvo 0.5, bet mazai duomenu)
NUM_EPOCHS = 20
OUTPUT_DATASET_FILE = "snow_dataset.csv"
TFL_MODEL_FILE = "snow_forecast_model.tflite"
TFL_MODEL_HEADER_FILE = "snow_forecast_model.h"
TF_MODEL = "snow_forecast"


######################################################
# Importing the CSV file with weather data

import pandas as pd
csv_file_path = "open_meteo_data/canazei_2011-01-01_to_2020-12-31.csv" # <--- IMPORTANT: Update this to your CSV file's path

try:
    df = pd.read_csv(csv_file_path, sep=',')

    # Extracting data using your actual column names:
    print("**************************************")
    print("\n\nExtracting data from CSV file...")
    print(f"Columns available in your CSV: {df.columns.tolist()}")

    t_list = df['temperature_2m'].astype(float).to_list()
    h_list = df['relativehumidity_2m'].astype(float).to_list()
    s_list = df['snowfall'].astype(float).to_list()
    #s_list = df['snow_depth'].astype(float).to_list()

    # Now, t_list, h_list, and s_list contain your data.
    print(f"\nSuccessfully extracted {len(t_list)} temperature readings.")
    print(f"Successfully extracted {len(h_list)} humidity readings.")
    print(f"Successfully extracted {len(s_list)} snowfall readings.")

    # You can print some values to verify:
    if len(t_list) > 5:
        print("\nFirst 5 temperature values (t_list):", t_list[:5])
        print("First 5 humidity values (h_list):", h_list[:5])
        print("First 5 snowfall values (s_list):", s_list[:5])

except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found. Please ensure the path is correct.")
except KeyError as e:
    print(f"Error: Column {e} was not found in the CSV file.")
    print("Please check if the column names in the script match exactly with those in your CSV.")
    if 'df' in locals(): # Check if df was loaded before the KeyError
        print(f"Available columns in your CSV are: {df.columns.tolist()}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
#End of CSV file import


##########################################
# Explore the extracted physical quantities in a 2D scatter chart

# Snow is considered significant if it exceeds MIN_SNOW_CM (originaliai buvo 0.5 cm, tačiau tokiu atveju yra labai mažai duomenų)
def binarize(snow, threshold):
  if snow > threshold:
    return 1
  else:
    return 0

s_bin_list = [binarize(snow, MIN_SNOW_CM) for snow in s_list]

#cm = plt.cm.get_cmap('gray_r')
cm = plt.colormaps['gray_r']  #atnaujinta versija
plt.figure(dpi=150)
sc = plt.scatter(t_list, h_list, c=s_bin_list, cmap=cm, label="Snow")
plt.colorbar(sc)
plt.legend()
plt.grid(True)
plt.title("Snow(T, H)")
plt.xlabel("Temperature - °C")
plt.ylabel("Humidity - %")
#plt.show()


##########################################
# Generate the output labels (Yes and No)
# Nustatomos žymos, ar yra sniego, atsižvelgiant į temperatūrą (<2C) ir sniego kiekį (>0.0 cm)
def gen_label(snow, temperature):
  if snow > MIN_SNOW_CM and temperature < MELTING_TEMPERATURE:
    return "Yes"
  else:
    return "No"

snow_labels = [gen_label(snow, temp) for snow, temp in zip(s_list, t_list)]


##################################################
# Build the dataset for training
# Temp0/Humi0: Temperature and humidity at time t = t0 - 2
# Temp1/Humi1: Temperature and humidity at time t = t0 - 1
# Temp2/Humi2: Temperature and humidity at time t = t0
# Snow: Label reporting whether it will snow at time t = t0
csv_header = ["Temp0", "Temp1", "Temp2", "Humi0", "Humi1", "Humi2", "Snow"]
df_dataset = pd.DataFrame(list(zip(t_list[:-2], t_list[1:-1], t_list[2:], h_list[:-2], h_list[1:-1], h_list[2:], snow_labels[2:])), columns = csv_header)


####################################################
# Balance the dataset by undersampling the majority class
# We will undersample the "No" class to balance the dataset - No sudaro ~70% duomenų, todėl reikia sumažinti iki 50%

df0 = df_dataset[df_dataset['Snow'] == "No"]
df1 = df_dataset[df_dataset['Snow'] == "Yes"]

num_nosnow_samples_old = round((len(df0.index) / (len(df_dataset.index))) * 100, 2)
num_snow_samples_old   = round((len(df1.index) / (len(df_dataset.index))) * 100, 2)

# Random subsampling of the majority class to guarantee 50% split
if len(df1.index) < len(df0.index):
  df0_sub = df0.sample(len(df1.index))
  df_dataset = pd.concat([df0_sub, df1])
else:
  df1_sub = df1.sample(len(df0.index))
  df_dataset = pd.concat([df1_sub, df0])

df0 = df_dataset[df_dataset['Snow'] == "No"]
df1 = df_dataset[df_dataset['Snow'] == "Yes"]

num_nosnow_samples_new = round((len(df0.index) / (len(df_dataset.index))) * 100, 2)
num_snow_samples_new = round((len(df1.index) / (len(df_dataset.index))) * 100, 2)

# Show number of samples
df_samples_results = pd.DataFrame.from_records(
                [["% No Snow", num_nosnow_samples_old, num_nosnow_samples_new],
                ["% Snow", num_snow_samples_old, num_snow_samples_new]],
            columns = ["Class", "Before - %", "After - %"], index="Class").round(2)

#display(df_samples_results) #special Jupyter command to display DataFrame nicely
print("\nDataset samples before and after balancing:")
print(df_samples_results)


###############################################
# Scale the input features with Z-score independently
# Get all values
t_list = df_dataset['Temp0'].tolist()
h_list = df_dataset['Humi0'].tolist()
t_list = t_list + df_dataset['Temp2'].tail(2).tolist()
h_list = h_list + df_dataset['Humi2'].tail(2).tolist()

# Calculate mean and standard deviation
t_avg = mean(t_list)
h_avg = mean(h_list)
t_std = std(t_list)
h_std = std(h_list)

print("\nCOPY ME!")
print("Temperature - [MEAN, STD]  ", round(t_avg, 5), round(t_std, 5))
print("Humidity - [MEAN, STD]     ", round(h_avg, 5), round(h_std, 5))

# Scaling with Z-score function
def scaling(val, avg, std):
  return (val - avg) / (std)

df_dataset['Temp0'] = df_dataset['Temp0'].apply(lambda x: scaling(x, t_avg, t_std))
df_dataset['Temp1'] = df_dataset['Temp1'].apply(lambda x: scaling(x, t_avg, t_std))
df_dataset['Temp2'] = df_dataset['Temp2'].apply(lambda x: scaling(x, t_avg, t_std))
df_dataset['Humi0'] = df_dataset['Humi0'].apply(lambda x: scaling(x, h_avg, h_std))
df_dataset['Humi1'] = df_dataset['Humi1'].apply(lambda x: scaling(x, h_avg, h_std))
df_dataset['Humi2'] = df_dataset['Humi2'].apply(lambda x: scaling(x, h_avg, h_std))


######################################################
# Visualize raw/scaled input features distributions
t_norm_list = df_dataset['Temp0'].tolist()
h_norm_list = df_dataset['Humi0'].tolist()
t_norm_list = t_norm_list + df_dataset['Temp2'].tail(2).tolist()
h_norm_list = h_norm_list + df_dataset['Humi2'].tail(2).tolist()

fig, ax=plt.subplots(1,2)
plt.subplots_adjust(wspace = 0.4)
#sns.distplot(t_list, ax=ax[0])
sns.histplot(t_list, ax=ax[0], kde=True) # Pakeista iš sns.distplot į sns.histplot
ax[0].set_title("Un-normalized temperature")
#sns.distplot(h_list, ax=ax[1])
sns.histplot(h_list, ax=ax[1], kde=True) # Pakeista iš sns.distplot į sns.histplot
ax[1].set_title("Un-normalized humidity")

fig, ax=plt.subplots(1,2)
plt.subplots_adjust(wspace = 0.5)
#sns.distplot(t_norm_list, ax=ax[0])
sns.histplot(t_norm_list, ax=ax[0], kde=True) # Pakeista iš sns.distplot į sns.histplot
ax[0].set_title("Normalized temperature")
#sns.distplot(h_norm_list, ax=ax[1])
sns.histplot(h_norm_list, ax=ax[1], kde=True) # Pakeista iš sns.distplot į sns.histplot
ax[1].set_title("Normalized humidity")

# Uncomment the following line to show the plots
#plt.show()
print("\nUncomment plt.show() to visualize the distributions of raw and scaled input features.")


##############################################
# Export to CSV file
try:
   df_dataset.to_csv(OUTPUT_DATASET_FILE, index=False)
   print(f"\nDataset successfully saved to {OUTPUT_DATASET_FILE}")
except Exception as e:
   print(f"Error saving dataset to CSV: {e}")



##############################################
# Training the ML model with TF
##############################################

# Extract the input features and output labels from the df_dataset Pandas DataFrame
f_names = df_dataset.columns.values[0:6]
l_name  = df_dataset.columns.values[6:7]
x = df_dataset[f_names]
y = df_dataset[l_name]

# Encode the labels to numerical values
labelencoder = LabelEncoder()
labelencoder.fit(y.Snow)
y_encoded = labelencoder.transform(y.Snow)

# Split the dataset into train, validation, and test datasets
# Split 1 (85% vs 15%)
x_train, x_validate_test, y_train, y_validate_test = train_test_split(x, y_encoded, test_size=0.15, random_state = 1)
# Split 2 (50% vs 50%)
x_test, x_validate, y_test, y_validate = train_test_split(x_validate_test, y_validate_test, test_size=0.50, random_state = 3)
     
# Create the model with Keras API
# model = tf.keras.Sequential()
# model.add(layers.Dense(12, activation='relu', input_shape=(len(f_names),)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.summary()
print("\nBuilding the model...")
# Rekomenduojamas būdas apibrėžti modelį su Input sluoksniu
model = tf.keras.Sequential([
    tf.keras.Input(shape=(len(f_names),), name="input_layer"), # Naudojame tf.keras.Input
    layers.Dense(12, activation='relu', name="hidden_layer_1"),
    layers.Dropout(0.2, name="dropout_layer"),
    layers.Dense(1, activation='sigmoid', name="output_layer")
])
model.summary()

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
print("\nTraining the model...")
history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_validate, y_validate))
    
# Analyze the accuracy and loss after each training epoch
loss_train = history.history['loss']
loss_val   = history.history['val_loss']
acc_train  = history.history['accuracy']
acc_val    = history.history['val_accuracy']
epochs     = range(1, NUM_EPOCHS + 1)

def plot_train_val_history(x, y_train, y_val, type_txt):
  plt.figure(figsize = (10,7))
  plt.plot(x, y_train, 'g', label='Training'+type_txt)
  plt.plot(x, y_val, 'b', label='Validation'+type_txt)
  plt.title('Training and Validation'+type_txt)
  plt.xlabel('Epochs')
  plt.ylabel(type_txt)
  plt.legend()
  #plt.show()

plot_train_val_history(epochs, loss_train, loss_val, "Loss")
plot_train_val_history(epochs, acc_train, acc_val, "Accuracy")
plt.show()

# Export the model to SavedModel format, suitable for TFLite conversion.
# The TF_MODEL constant ("snow_forecast") will be the directory name for the SavedModel.
print(f"\nExporting model to SavedModel format at: {TF_MODEL}")
model.export(TF_MODEL)


###################################################
# Evaluating the model effectiveness

# Visualize the confusion matrix
y_test_pred = model.predict(x_test)

y_test_pred = (y_test_pred > 0.5).astype("int32")

cm = sklearn.metrics.confusion_matrix(y_test, y_test_pred)

index_names  = ["Actual No Snow", "Actual Snow"]
column_names = ["Predicted No Snow", "Predicted Snow"]

df_cm = pd.DataFrame(cm, index = index_names, columns = column_names)

plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True, fmt='d', cmap="Blues")
plt.show()

# Calculate Recall, Precision, and F-score performance metrics
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

accuracy = (TP + TN) / (TP + TN + FN + FP)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f_score = (2 * recall * precision) / (recall + precision)

print("\nModel evaluation metrics:")
print("Accuracy:  ", round(accuracy, 3))
print("Recall:    ", round(recall, 3))
print("Precision: ", round(precision, 3))
print("F-score:   ", round(f_score, 3))


###############################################
# Quantizing the model with TFLite converter
###############################################

# Select a few hundred of samples randomly from the test dataset to calibrate the quantization
def representative_data_gen():
  for i_value in tf.data.Dataset.from_tensor_slices(x_test).batch(1).take(100):
    i_value_f32 = tf.dtypes.cast(i_value, tf.float32)
    yield [i_value_f32]

# Import the TensorFlow SavedModel directory into TensorFlow Lite Converter
converter = tf.lite.TFLiteConverter.from_saved_model(TF_MODEL)

# Initialize TensorFlow Lite converter for the 8-bit quantization
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert the model to TensorFlow Lite file format (FlatBuffers) as save it as .tflite
tflite_model_quant = converter.convert()
open(TFL_MODEL_FILE, "wb").write(tflite_model_quant)

# Convert the TFLite model to C header file
def convert_tflite_to_header(tflite_model_file, header_file):
    with open(tflite_model_file, "rb") as f:
        tflite_model = f.read()

    with open(header_file, "w") as f:
        f.write("// Auto-generated TFLite model header file\n")
        f.write("#ifndef TFL_MODEL_H\n")
        f.write("#define TFL_MODEL_H\n\n")
        f.write(f"alignas(8) const unsigned char snow_model_tflite[] = {{\n")

        for i in range(0, len(tflite_model), 12):
            f.write("  " + ", ".join(f"0x{byte:02x}" for byte in tflite_model[i:i+12]) + ",\n")

        f.write("};\n\n")
        f.write(f"unsigned int snow_model_tflite_len = {len(tflite_model)};\n\n")
        f.write("#endif // TFL_MODEL_H\n")

convert_tflite_to_header(TFL_MODEL_FILE, TFL_MODEL_HEADER_FILE)


# Get the TensorFlow model size in bytes to estimate the program memory usage
size_tfl_model = len(tflite_model_quant)
print(f"\nQuantized TFLite model size: {size_tfl_model} bytes\n")
#print(len(tflite_model_quant), "bytes")




