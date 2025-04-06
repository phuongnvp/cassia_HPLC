import pandas as pd
import os

folder_path = r"C:\Users\PC\Desktop\IR\Data"  

summary_df = pd.read_csv(r"C:\Users\PC\Desktop\IR\HPLC_data.csv")

# Function to process each file
def process_file(file_path, filename):
    df = pd.read_csv(file_path)
    
    if '280' in df.columns:
        values = df['280'].tolist()

        match_index = summary_df[summary_df['File name'] == filename].index
        
        if not match_index.empty:

            for i, value in enumerate(values):
                col_name = f'value_{i+1}' 

                if col_name not in summary_df.columns:
                    summary_df[col_name] = pd.NA

                summary_df.loc[match_index[0], col_name] = value

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(filename)
        process_file(file_path, filename)

# Save the updated summary file
summary_df.to_csv(r"C:\Users\PC\Desktop\IR\Updated_HPLC_280.csv", index=False)
print("Processing complete. Updated file saved as 'Updated_HPLC_280.csv'")