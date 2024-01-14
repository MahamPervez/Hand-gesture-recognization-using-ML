import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Print shapes to debug
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

score=accuracy_score(y_predict,y_test)
print('{}% of samples were classified correctly '.format(score*100))
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)

# Plot confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()