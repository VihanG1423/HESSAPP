import pandas as pd

def clean_csv(input_file, output_file):
    
    # Load original data
    df = pd.read_csv(input_file)

    # Make copy to keep original safe.
    df_cleaned = df.copy()

    # Convert 'Time' to datetime
    df_cleaned['Time'] = pd.to_datetime(df_cleaned['Time'], format='%d-%b-%Y %H:%M:%S')

    # Split into new columns
    df_cleaned['Year'] = df_cleaned['Time'].dt.year
    df_cleaned['Month'] = df_cleaned['Time'].dt.month
    df_cleaned['Date'] = df_cleaned['Time'].dt.day
    df_cleaned['ClockTime'] = df_cleaned['Time'].dt.strftime('%H:%M:%S')

    # Drop 'Time' column
    df_cleaned = df_cleaned.drop(columns=['Time'])

    # Rename other columns
    df_cleaned = df_cleaned.rename(columns={
        'P_in_W': 'power_watts',
        'V_in_V': 'voltage_volts',
        'I_in_A': 'current_amperes',
        'T_Bat_in_C': 'temp_battery_celsius',
        'T_Room_in_C': 'temp_room_celsius',
        'Interpolated': 'interpolated'
    })

    # Reorder columns
    new_order = [
        'Year', 'Month', 'Date', 'ClockTime',
        'power_watts', 'voltage_volts', 'current_amperes',
        'temp_battery_celsius', 'temp_room_celsius', 'interpolated'
    ]
    df_cleaned = df_cleaned[new_order]

        # Save to new file
    df_cleaned.to_csv(output_file, index=False)
    print(f"Cleaned Data saved into {output_file}")



# Example usage (can be called from another script or main)
if __name__ == "__main__":
    original_path = "C:/Users/Vihan/Downloads/newv1HESSAPP/data/2022_10_System_ID_01.csv"
    cleaned_path = "C:/Users/Vihan/Downloads/newv1HESSAPP/aftercleaning.csv"
    clean_csv(original_path, cleaned_path)
    