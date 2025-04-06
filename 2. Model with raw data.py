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
label_encoder = LabelEncoder()

# Prepare input and output
data = pd.read_csv(r"C:\Users\PC\Desktop\IR\Updated_HPLC_280.csv")
# X = data.iloc[:, 3:6].values  # Use CM, CA and CAL levels as input variables
X = data.iloc[:, 6:4224].values  # Use full HPLC as input variables
y = label_encoder.fit_transform(data["Group"])  # Encode group labels as integers

print(X[0])
print(y[0])
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f'X_train.shape: {X_train.shape}')
print(f'y_train.shape: {y_train.shape}')
print(f'X_test.shape: {X_test.shape}')
print(f'y_test.shape: {y_test.shape}')

#%%
# X_df = data.iloc[:, 3:6]
X_df = data.iloc[:, 6:4224]
feature_names = X_df.columns.tolist()
print(feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

#%%
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef

def evaluate(y_test, y_pred, y_pred_proba=None):
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall (macro)': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'Precision (macro)': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'F1-score (macro)': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'MCC (mean over targets)': matthews_corrcoef(y_test.ravel(), y_pred.ravel())  # flatten for binary MCC
    }

    try:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        metrics['AUC (macro)'] = auc
    except Exception as e:
        metrics['AUC (macro)'] = f"Not computable: {e}"

    return metrics
#%%
wavenumbers = np.arange(600, 4001, 2)
plt.figure(figsize=(10, 6))
plt.plot(wavenumbers, X_train[1], label="Sample 1", alpha=0.7)
plt.plot(wavenumbers, X_train[0], label="Sample 2", alpha=0.7)
plt.xlabel("Wavenumber (cm^-1)")
plt.ylabel("Intensity")
plt.legend()
plt.title("IR Spectrum Preprocessing Example")
plt.show()

#%%
#from imblearn.over_sampling import SVMSMOTE
#svm_smote = SVMSMOTE(random_state=42)
#X_train, y_train = svm_smote.fit_resample(X_train, y_train)
#print(X_train.shape)
#print(X_val.shape)
#print(X_test.shape)

#%% Train XGBoost model
xgb_model = xgb.XGBClassifier(
    max_depth=6, 
    learning_rate=0.1, 
    n_estimators=100, 
    objective="multi:softprob",  
    num_class=len(label_encoder.classes_), 
    eval_metric="mlogloss"  
)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)

results = evaluate(y_test, y_pred, y_pred_proba)
print(results)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
class_names = [str(cls) for cls in label_encoder.classes_]
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Explain model with SHAP
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test_df)
mean_shap = np.abs(shap_values.values).mean(axis=0).mean(axis=1)
shap_importance = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_shap
})
shap_importance.to_csv(r"C:\Users\PC\Desktop\IR\SHAP.csv", index=False)

#%%
import seaborn as sns
import matplotlib.pyplot as plt

shap_importance = pd.read_csv(r"C:\Users\PC\Desktop\IR\SHAP.csv")

# Plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(
        data=shap_importance,
        y='mean_abs_shap',
        x='time (min)',
        palette='viridis'
)

ax.set_xticklabels([str(i) for i in range(1, 47)])

plt.xlabel('Time (min)')
plt.ylabel('Mean Absolute SHAP Value')
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.show()

#%%
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Train SVM model
svm_model = SVC(kernel='rbf', C=100, gamma='scale', probability=1)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)
y_pred_proba = svm_model.predict_proba(X_test)

results = evaluate(y_test, y_pred, y_pred_proba)
print(results)

# Evaluate model
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Test Accuracy: {accuracy * 100:.2f}%")
#print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

#cm = confusion_matrix(y_test, y_pred)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
#disp.plot(cmap=plt.cm.Blues)
#plt.show()

# Explain model with SHAP
#background = shap.sample(X_train, 50)
#explainer = shap.KernelExplainer(svm_model.predict, background)
#shap_values = explainer.shap_values(X_test)
#shap.summary_plot(shap_values, X_test)

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

results = evaluate(y_test, y_pred, y_pred_proba)
print(results)

# Evaluate model
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Test Accuracy: {accuracy * 100:.2f}%")
#print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

#cm = confusion_matrix(y_test, y_pred)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
#disp.plot(cmap=plt.cm.Blues)
#plt.show()

# Explain model with SHAP
#explainer = shap.Explainer(rf_model)
#shap_values = explainer.shap_values(X_test)
#shap.summary_plot(shap_values, X_test)

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
voting_clf.fit(X_train, y_train)

# Evaluate
y_pred = voting_clf.predict(X_test)
y_pred_proba = voting_clf.predict_proba(X_test)

results = evaluate(y_test, y_pred, y_pred_proba)
print(results)

#accuracy = accuracy_score(y_test, y_pred)
#print(f"Test Accuracy: {accuracy * 100:.2f}%")
#print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

#cm = confusion_matrix(y_test, y_pred)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
#disp.plot(cmap=plt.cm.Blues)
#plt.show()

# Explain model with SHAP
#background = shap.sample(X_train, 10)
#explainer = shap.KernelExplainer(voting_clf.predict, background)
#shap_values = explainer.shap_values(X_test)
#shap.summary_plot(shap_values, X_test)

#%%
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

checkpoint = ModelCheckpoint(
    r"C:\Users\PC\Desktop\IR\best_cnn_model_raw.h5",   
    monitor="val_accuracy", 
    save_best_only=True,    
    mode="max",            
    verbose=1
)

# Build 1D CNN
cnn_model = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),  
    MaxPooling1D(pool_size=1), 

    Conv1D(64, kernel_size=2, activation='relu'),
    MaxPooling1D(pool_size=1), 

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
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

# Convert labels to one-hot encoding
y_train_one_hot = to_categorical(y_train, num_classes=4)
y_test_one_hot = to_categorical(y_test, num_classes=4)

# Train the model
cnn_model.fit(X_train_cnn, y_train_one_hot, epochs=50, batch_size=16, validation_split=0.2, callbacks=[checkpoint] )

# Evaluate the model
loss, accuracy = cnn_model.evaluate(X_test_cnn, y_test_one_hot, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

#%%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

best_model = load_model(r"C:\Users\PC\Desktop\IR\best_cnn_model_raw.h5")

# Predict the test set
y_pred = best_model.predict(X_test_cnn)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the predicted class indices
results = evaluate(y_test, y_pred_classes, y_pred)
print(results)

#%%
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
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
stacking_clf.fit(X_train, y_train)

# Evaluate
y_pred = stacking_clf.predict(X_test)
y_pred_proba = stacking_clf.predict_proba(X_test)

results = evaluate(y_test, y_pred, y_pred_proba)
print(results)