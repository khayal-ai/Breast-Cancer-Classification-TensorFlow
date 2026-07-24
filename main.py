import numpy as np
import pandas as pd #create dataframe
import matplotlib.pyplot as plt #visulaize training 
from sklearn.model_selection import train_test_split #split data into training and testing
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report,  ConfusionMatrixDisplay

df = pd.read_csv("data.csv")

df = df.drop(columns=['id', 'Unnamed: 32'])

df['label'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop(columns=['diagnosis', 'label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = tf.keras.Sequential([
    keras.layers.Input(shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

scaler= StandardScaler()

X_train_std= scaler.fit_transform(X_train)
X_test_std= scaler.transform(X_test)

history = model.fit(
    X_train_std,
    y_train,
    validation_split=0.1,
    epochs=10
)

# Visulize loss func and accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Training Data', 'Validation Data'], loc='lower right')
plt.savefig("images/accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'], loc= 'upper right')
plt.savefig("images/loss.png", dpi=300, bbox_inches="tight")
plt.show()

#Accuarcy of model on test data
loss, accuarcy= model.evaluate(X_test_std, y_test)
print("Accuracy of model in test data: ", accuarcy)

#model.predict() -> gives probabilty of each class for the data point, the higher probablity means it is blenogs to category of either 0 or 1
#argmax func -> give the label of highest probablity 

y_pred= model.predict(X_test_std)
print(y_pred)
#convert prediction probablity to class labels 0 / 1

y_pred_labels = [np.argmax(i) for i in y_pred]

print(y_pred_labels)

input_data= (13.03,18.42,82.61,523.8,0.08983,0.03766,0.02562,0.02923,0.1467,0.05863,0.1839,2.342,1.17,14.16,0.004352,0.004899,0.01343,0.01164,0.02671,0.001777,13.3,22.81,84.46,545.9,0.09701,0.04619,0.04833,0.05013,0.1987,0.06169
)

#change input data to numpy array
input_data_asarray= np.asarray(input_data)

#reshape numpy array for predicting data point
reshaped_data= input_data_asarray.reshape(1,-1)

#standarize input data
input_data_std= scaler.transform(reshaped_data)

pred= model.predict(input_data_std)
print(pred)

pred_label = [np.argmax(pred)]
print(pred_label)

if pred_label == 1:
    print("The tumor is Malignant")
else:
    print("The tumor is Benign")

#Evaluate the model
print(classification_report(y_test, y_pred_labels, target_names=["Benign", "Malignant"]))
print(confusion_matrix(y_test, y_pred_labels))

y_pred = model.predict(X_test_std)
y_pred_labels = np.argmax(y_pred, axis=1)   
cm = confusion_matrix(y_test, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Benign", "Malignant"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()