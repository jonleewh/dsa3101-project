import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

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
combined_df['Day of Week'] = combined_df['DateTime'].dt.day_name()
combined_df = combined_df.drop(columns=['Park'])


jan['Date/Time'] = pd.to_datetime(jan['Date/Time']) # Convert 'Date/Time' column to datetime
jan['Month'] = jan['Date/Time'].dt.month
jan['Day'] = jan['Date/Time'].dt.day
jan['Time'] = jan['Date/Time'].dt.strftime('%H:%M')
jan['Day of Week'] = jan['DateTime'].dt.day_name()
jan = jan.drop(columns=['Park'])

print(jan)

# Set the DateTime column as the index
combined_df.set_index('Date/Time', inplace=True)

# # Plotting the Wait Time
# plt.figure(figsize=(10, 5))
# plt.plot(combined_df.index, combined_df['Wait Time'], marker='o')

# # Adding titles and labels
# plt.title('Trend of Wait Time')
# plt.xlabel('Date and Time')
# plt.ylabel('Wait Time (minutes)')
# plt.xticks(rotation=45)
# plt.grid()

# # Show the plot
# plt.tight_layout()  # Adjust layout for better viewing
# plt.show()

# # Create a scatter plot
# plt.figure(figsize=(10, 5))
# plt.scatter(combined_df.index, combined_df['Wait Time'], color='b', marker='o')

# # Add titles and labels
# plt.title('Scatter Plot of Wait Time')
# plt.xlabel('Date and Time')
# plt.ylabel('Wait Time (minutes)')
# plt.xticks(rotation=45)
# plt.grid()

# # Show the plot
# plt.tight_layout()  # Adjust layout for better viewing
# plt.show()



#############
# # Set the DateTime column as the index
# jan.set_index('Date/Time', inplace=True)

# # Plotting the Wait Time
# plt.figure(figsize=(10, 5))
# plt.plot(jan['Time'], jan['Wait Time'], marker='o')

# # Adding titles and labels
# plt.title('Trend of Wait Time in Jan')
# plt.xlabel('Date and Time')
# plt.ylabel('Wait Time (minutes)')
# plt.xticks(rotation=45)
# plt.grid()

# # Show the plot
# plt.tight_layout()  # Adjust layout for better viewing
# plt.show()

########
jan_monday = jan[jan['Day of Week'] == 'Monday']

# Set the DateTime column as the index
jan_monday.set_index('Date/Time', inplace=True)

# Plotting the Wait Time
plt.figure(figsize=(10, 5))
plt.plot(jan_monday['Time'], jan_monday['Wait Time'], marker='o')

# Adding titles and labels
plt.title('Trend of Wait Time on Mondays in Jan')
plt.xlabel('Date and Time')
plt.ylabel('Wait Time (minutes)')
plt.xticks(rotation=45)
plt.grid()

# Show the plot
plt.tight_layout()  # Adjust layout for better viewing
plt.show()
