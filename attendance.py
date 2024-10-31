import pandas as pd
from datetime import datetime

# load csv files into dataframes
jan = pd.read_csv('USS daily waiting time/download_universal-studios-singapore_jan_18521_2024_10_31_15_23_50.csv')
feb = pd.read_csv('USS daily waiting time/download_universal-studios-singapore_feb_18521_2024_10_31_15_23_46.csv')
mar = pd.read_csv('USS daily waiting time/download_universal-studios-singapore_mar_18521_2024_10_31_15_23_41.csv')
apr = pd.read_csv('USS daily waiting time/download_universal-studios-singapore_apr_18521_2024_10_31_15_23_37.csv')
may = pd.read_csv('USS daily waiting time/download_universal-studios-singapore_may_18521_2024_10_31_15_23_34.csv')
jun = pd.read_csv('USS daily waiting time/download_universal-studios-singapore_jun_18521_2024_10_31_15_37_28.csv')
jul = pd.read_csv('USS daily waiting time/download_universal-studios-singapore_jul_18521_2024_10_31_15_23_26.csv')
aug = pd.read_csv('USS daily waiting time/download_universal-studios-singapore_aug_18521_2024_10_31_15_23_21.csv')
sep = pd.read_csv('USS daily waiting time/download_universal-studios-singapore_sep_18521_2024_10_31_15_23_10.csv')
oct = pd.read_csv('USS daily waiting time/download_universal-studios-singapore_oct_18521_2024_10_31_15_23_05.csv')
nov = pd.read_csv('USS daily waiting time/download_universal-studios-singapore_nov_18521_2024_10_31_15_22_53.csv')
dec = pd.read_csv('USS daily waiting time/download_universal-studios-singapore_dec_18521_2024_10_31_14_44_27.csv')

list_of_df = [jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec]

combined_df = pd.concat(list_of_df, ignore_index=True)

combined_df['Date/Time'] = pd.to_datetime(combined_df['Date/Time']) # Convert 'Date/Time' column to datetime
combined_df['Month'] = combined_df['Date/Time'].dt.month
combined_df['Day'] = combined_df['Date/Time'].dt.day
combined_df['Time'] = combined_df['Date/Time'].dt.strftime('%H:%M')

combined_df = combined_df.drop(columns=['Park', 'Date/Time'])

print(combined_df)



