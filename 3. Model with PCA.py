#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import shap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

label_encoder = LabelEncoder()

# Prepare input and output
data = pd.read_csv(r"C:\Users\PC\Desktop\IR\Updated_HPLC_280.csv")
X = data.iloc[:, 6:4224].values  # Exclude the group column
print(X[0])
y = label_encoder.fit_transform(data["Group"])  # Encode group labels as integers
print(y[0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f'X_train.shape: {X_train.shape}')
print(f'y_train.shape: {y_train.shape}')
print(f'X_test.shape: {X_test.shape}')
print(f'y_test.shape: {y_test.shape}')

#%%
wavenumbers = np.arange(1280, 2700800, 640)
plt.figure(figsize=(10, 6))
plt.plot(wavenumbers, X_train[1], label="Sample 1", alpha=0.7)
plt.plot(wavenumbers, X_train[0], label="Sample 2", alpha=0.7)
plt.xlabel("Wavenumber (cm^-1)")
plt.ylabel("Intensity")
plt.legend()
plt.title("IR Spectrum Preprocessing Example")
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardizing the data before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying PCA 
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"X_pca shape: {X_pca.shape}")
print(f"X_test_pca shape: {X_test_pca.shape}")
# Plot PCA results
plt.figure(figsize=(8, 6))
for class_label in np.unique(y):
    plt.scatter(X_pca[y_train == class_label, 0], X_pca[y_train == class_label, 1], label=f'Group {class_label}', alpha=0.7)

plt.xlabel('PCA-1')
plt.ylabel('PCA-2')
plt.title('PCA Visualization of IR spectra')
plt.legend()
plt.grid(True)
plt.show()

#%% Explained variance ratio
explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()

#%% Train XGBoost model
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)

xgb_model = xgb.XGBClassifier(
    max_depth=6, 
    learning_rate=0.1, 
    n_estimators=100, 
    objective="multi:softprob",  
    num_class=len(label_encoder.classes_),  
    eval_metric="mlogloss"
)
xgb_model.fit(X_pca, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_pca)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
class_names = [str(cls) for cls in label_encoder.classes_]
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#%%
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Train SVM model
svm_model = SVC(kernel='rbf', C=100, gamma='scale')  # Adjust C and gamma for optimal performance
svm_model.fit(X_pca, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_pca)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train SVM model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_pca, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_pca)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#%%
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

# Combine XGBoost, Random Forest, and SVM
xgb_clf = xgb.XGBClassifier(n_estimators=100)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(probability=True, C = 100)

voting_clf = VotingClassifier(estimators=[
    ('xgb', xgb_clf),
    ('rf', rf_clf),
    ('svm', svm_clf)
], voting='soft')

# Train Voting Classifier
voting_clf.fit(X_pca, y_train)

# Evaluate
y_pred = voting_clf.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#%%
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

# Define model checkpoint callback
checkpoint = ModelCheckpoint(
    r"C:\Users\PC\Desktop\IR\best_cnn_model_test.h5",   # Save model to this file
    monitor="val_accuracy", # Monitor validation accuracy
    save_best_only=True,    # Save only the best model
    mode="max",             # Maximize accuracy
    verbose=1
)

# Build 1D CNN
cnn_model = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_pca.shape[1], 1)),  # Use kernel_size=2
    MaxPooling1D(pool_size=1),  # Reduce pool size

    Conv1D(64, kernel_size=2, activation='relu'),
    MaxPooling1D(pool_size=1),  # Reduce pool size

    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(y_train.max() + 1, activation='softmax')
])


# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape data for CNN input
X_train_cnn = X_pca[..., np.newaxis]
X_test_cnn = X_test_pca[..., np.newaxis]

# Convert labels to one-hot encoding
y_train_one_hot = to_categorical(y_train, num_classes=4)
y_test_one_hot = to_categorical(y_test, num_classes=4)

# Train the model
cnn_model.fit(X_train_cnn, y_train_one_hot, epochs=50, batch_size=16, validation_split=0.2, callbacks=[checkpoint] )

# Evaluate the model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

best_model = load_model(r"C:\Users\PC\Desktop\IR\best_cnn_model_test.h5")
loss, accuracy = best_model.evaluate(X_test_cnn, y_test_one_hot, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict the test set
y_pred = best_model.predict(X_test_cnn)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
print("Classification Report:\n", classification_report(y_test_classes, y_pred_classes, target_names=class_names))
# Compute confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(1, 5), yticklabels=np.arange(1, 5))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import shap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

label_encoder = LabelEncoder()
wavelengths = list(range(190, 710, 10))
for i in wavelengths:
    # Prepare input and output
    data = pd.read_csv(f"C:/Users/PC/Desktop/IR/Updated_HPLC_{i}.csv")
    X = data.iloc[:, 6:4224].values  # Exclude the group column
    y = label_encoder.fit_transform(data["Group"])  # Encode group labels as integers
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)    
    # Standardizing the data before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Applying PCA to reduce dimensions to 2
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)
    X_test_pca = pca.transform(X_test_scaled)    
    # Combine XGBoost, Random Forest, and SVM
    xgb_clf = xgb.XGBClassifier(n_estimators=100)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_clf = SVC(probability=True, C = 100)
    stacking_clf = StackingClassifier(estimators=[
        ('xgb', xgb_clf),
        ('rf', rf_clf),
        ('svm', svm_clf)
    ], final_estimator=RandomForestClassifier())
    
    # Train Voting Classifier
    stacking_clf.fit(X_pca, y_train)
    # Evaluate
    y_pred = stacking_clf.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Wavelength: {i} nm. Test Accuracy: {accuracy * 100:.2f}%")