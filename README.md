Navigating the Uber Dataset: Unraveling Insights with EDA and Data Preprocessing

In today's data-driven world, businesses and analysts rely heavily on the power of data to make informed decisions. One of the initial steps in this journey is the process of Exploratory Data Analysis (EDA) and Data Preprocessing (DPL). In this blog, we embark on an exciting journey through the world of EDA and DPL using a real-world dataset from Uber. Our goal is to unravel hidden patterns, cleanse the data, and prepare it for in-depth analysis or for machine learning applications.

Introduction.

Uber, a global giant in the ridesharing industry, generates an enormous amount of data every day. This data is not just about routes and fares; it's a treasure trove of valuable information about transportation, customer behavior, and urban mobility. In this blog, we'll delve into an Uber dataset, exploring the significance of EDA and DPL in harnessing the potential within such datasets.

Data Description

Before we embark on our journey, it's essential to understand the dataset at hand:

- Source: Our dataset is sourced from Uber's ride records in a major area of New York for year 2016-2017.
- Size: The dataset is substantial, containing 4169 rows and 19 columns. 
- Data Type: It's structured data, comprising information about rides, such as 
Trip pickup datetime and drop-off , passenger count, trip distance, pickup longitude, pickup latitude, ratecode-Id,toll amount, trip amount, fare amount and more.

Exploratory Data Analysis (EDA)

EDA is the compass that guides us through the labyrinth of data. Let's begin with the first steps:

 1. Data Cleaning

The cleanliness of the data can significantly impact the insights we can extract from it. Data cleaning is the process of identifying and rectifying issues like missing values, outliers, or duplicate records. Python's pandas library becomes our trusted companion in this phase.

python
import pandas as pd

# Loading the Uber dataset
uber_data = pd.read_csv('uber.csv')

# Investigating missing values
missing_values = uber_data.isnull().sum()
```

2. Data Visualization

Data visualization is our flashlight in the dark cave of data. It helps us see trends, patterns, and anomalies more clearly. Let's light it up with a couple of examples:

 A. Riders Count by Hour by vendors.

To get a sense of when Uber riders  per vendor are most in demand, we can visualize the ride count by hour. 

Python

import matplotlib.pyplot as plt
import seaborn as sns

#Plotting bar plot
plt.subplot(1, 2, 2)
data['VendorID'].value_counts().sort_index().plot(kind='bar', color='lightseagreen')
plt.title('Number of Rides per Vendor')
plt.xlabel('Vendor ID')
plt.ylabel('Number of Rides')

plt.tight_layout()
plt.show()
```
<img width="137" alt="image" src="https://github.com/RiyaGupta122/Blog-on-EDA--DPL-on-uber-dataset/assets/149296023/1ac51e98-cc05-4102-8a42-b716f62b23ca">

 B. Fare Distribution

Visualizing the fare distribution gives us insights into how much passengers typically pay for Uber rides.


import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('/content/updated_vendor_dataset_dpl_eda.csv')

#Define the mapping for payment type
payment_mapping = {1: 'Cash', 2: 'UPI', 3: 'Apple Pay'}
data['payment_type_label'] = data['payment_type'].map(payment_mapping)

#Plotting bar plot
plt.figure(figsize=(8, 5))
data['payment_type_label'].value_counts().plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('Distribution of Payment Types')
plt.xlabel('Payment Type')
plt.ylabel('Number of Transactions')
plt.show()

<img width="277" alt="image" src="https://github.com/RiyaGupta122/Blog-on-EDA--DPL-on-uber-dataset/assets/149296023/f1167d9f-2da0-4b32-819e-34c5d38598c9">



C. CORRELATION MATRIX.

A correlation matrix for Uber refers to a statistical analysis that examines the relationships between various factors or variables within Uber's operations or business data. This matrix provides valuable insights into how different variables are related to each other, whether positively or negatively. 

#Compute the correlation matrix 

correlation_matrix=data[['trip_distance','fare_amount','tip_amount','total_amount']].corr()
correlation_matrix

D. SCATTER PLOT.

Now we will see the density of pickup in NYC using scatter plot (this will give us an idea of where we shall place more of our Vendors and where we shall increase the prices per ride)


import plotly.express as px 

#scatter plot of pickup locations 
fig = px.scatter(data, x='pickup_longitude', y='pickup_latitude',                                                                                .                          title='Density of Pickups in NYC',                 
                           opacity=0.1, height=600, width=800) #Define the boundries 
fig.update_xaxes(range=[-74.10, -73.70]) 
fig.update_yaxes(range=[40.58, 40.90])
 fig.show()

<img width="300" alt="image" src="https://github.com/RiyaGupta122/Blog-on-EDA--DPL-on-uber-dataset/assets/149296023/53e26aa5-148b-409b-9fb0-ffc88d4b5311">

E. MOST TRAVELLED/ RUSH TIME.

Now we will see the hourly ride count in NYC using time series plot using Plotly to see at which time do the people travel the most

#Extract the hour
data['pickup_hour'] = pd.to_datetime(data['tpep_pickup_datetime']).dt.hour

#Aggregate data
hourly_rides = data.groupby('pickup_hour').size().reset_index(name='ride_count')

#time series plot using Plotly
fig = px.line(hourly_rides, x='pickup_hour', y='ride_count', title='Hourly Ride Counts in NYC')

fig.show()

<img width="906" alt="image" src="https://github.com/RiyaGupta122/Blog-on-EDA--DPL-on-uber-dataset/assets/149296023/08a19aee-c48c-4490-a7b5-95a02a3af3af">


Data Preprocessing (DPL)

Now that we have illuminated the path through EDA, we can move on to Data Preprocessing. This phase prepares our data for modeling or more in-depth analysis.

1. Feature Engineering

Feature engineering is like crafting a masterpiece out of raw materials. It involves creating new features or transforming existing ones to improve the dataset's suitability for analysis. In the Uber dataset, we can engineer features like the day of the week or the distance between pickup and drop-off points.

 2. Data Transformation

Data preprocessing also encompasses steps like encoding categorical variables and scaling numerical features. Let's perform some of these transformations:

A.	Using Data processing we will first normalise the data only the numeric columns (both integers and floats) from the dataset, as normalization is typically applied to numeric data.

from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('/content/updated_vendor_dataset_dpl_eda.csv')

#Instantiate the scaler
scaler = MinMaxScaler()

#Apply the scaler
normalized_data = pd.DataFrame(scaler.fit_transform(data.select_dtypes(include=['float64', 'int64'])),
                               columns=data.select_dtypes(include=['float64', 'int64']).columns)

#Replace the original columns with the normalized one
data[normalized_data.columns] = normalized_data

data.to_csv('path_to_save_normalized_dataset.csv', index=False)

B.	 Filtering trips with a specific passenger count in an Uber dataset involves selecting and isolating the rows that match the passenger count you are interested in.

       single_passenger_trips = data[data['passenger_count'] >=5]
       print(single_passenger_trips)

C.	Filtering trips in an Uber dataset based on fare amount involves selecting and isolating the rows that meet specific criteria for fare amounts.

        expensive_trips = data[data['fare_amount'] > 200]
        print(expensive_trips)
        

D.	Dropping NULL (missing) values from a dataset and saving the cleaned dataset as a new dataset is a common data preprocessing step in data analysis. This process ensures that your data is free from missing or incomplete information, making it more suitable for analysis or modeling.

         data = pd.read_csv('/content/updated_vendor_dataset_dpl_eda.csv')

         #Drop rows where 'tolls_amount' is 0 (representing null values)
         cleaned_data = data[data['tolls_amount'] != 0]

        cleaned_data.to_csv('path_to_save_cleaned_dataset.csv', index=False)

        
E.	Replacing missing (NULL) values in a dataset with the mean values of non-null entries is a common technique for handling missing data. This approach helps maintain the overall data structure and can be useful when the number of missing values is relatively small or when the missing values are missing at random

data = pd.read_csv('/content/updated_vendor_dataset_dpl_eda.csv')

#Calculate the mean
mean_tolls = data[data['tolls_amount'] != 0]['tolls_amount'].mean()

#Replace the calculated mean
data['tolls_amount'] = data['tolls_amount'].replace(0, mean_tolls)

data.to_csv('path_to_save_modified_dataset.csv', index=False)


F.	Plotting a heatmap is a useful way to visualize the difference between two datasets, such as the original dataset and a cleaned dataset with missing values replaced by column means. Heatmaps provide a graphical representation of the data, showing variations in values through colors. In our case, it can help you compare the distribution of values before and after cleaning.


 #Load the dataset with dropped null values
data_dropped = pd.read_csv('/content/path_to_save_cleaned_dataset.csv')

#Load the dataset with replaced null values with mean
data_replaced = pd.read_csv('/content/path_to_save_modified_dataset.csv')

#Ensure the datasets are aligned
data_dropped.set_index('trip_distance', inplace=True)
data_replaced.set_index('trip_distance', inplace=True)

#Calculate the difference in 'tolls_amount' between the two datasets
difference = data_dropped['tolls_amount'] - data_replaced['tolls_amount']

#Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(difference.values.reshape(-1, 1), cmap='coolwarm', cbar_kws={'label': 'Difference in tolls amount'})
plt.title('Difference in tolls amount between Dropped and Replaced Datasets')
plt.xlabel('Dataset Difference')
plt.ylabel('Entries')
plt.show()

<img width="328" alt="image" src="https://github.com/RiyaGupta122/Blog-on-EDA--DPL-on-uber-dataset/assets/149296023/efc4f704-d839-4132-bad0-46c64805e6c8">


G.	Creating a scatter plot to visualize the difference between two datasets can be a useful way to compare the values of corresponding data points in the two datasets. This type of plot allows you to identify discrepancies between the datasets, revealing how data points deviate from a perfect one-to-one correspondence.


#Load the dataset with replaced null values with mean
data_replaced = pd.read_csv('/content/path_to_save_modified_dataset.csv')

#Ensure the datasets are aligned
data_dropped.set_index('trip_distance', inplace=True)
data_replaced.set_index('trip_distance', inplace=True)

#Align both datasets by their index to ensure they have the same size
aligned_dropped, aligned_replaced = data_dropped.align(data_replaced, axis=0)

#Scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(aligned_dropped['tolls_amount'], aligned_replaced['tolls_amount'], alpha=0.5)
plt.plot([aligned_dropped['tolls_amount'].min(), aligned_dropped['tolls_amount'].max()],
         [aligned_dropped['tolls_amount'].min(), aligned_dropped['tolls_amount'].max()],
         'r', linewidth=2)
plt.title('Comparison between Dropped and Replaced Datasets')
plt.xlabel('Tolls Amount (Dropped Null Values)')
plt.ylabel('Tolls Amount (Replaced with Mean)')
plt.grid(True)
plt.show()

<img width="320" alt="image" src="https://github.com/RiyaGupta122/Blog-on-EDA--DPL-on-uber-dataset/assets/149296023/4668b954-7795-4e01-9188-420e5d033a95">


H.	WE TRAIN AND TEST THE MODEL

Split the dataset into training and testing sets

from sklearn.model_selection import train_test_split

#Select 'trip_distance' as the feature and 'fare_amount' as the target variable
X = data[['trip_distance']]
y = data['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


I.	Applying a Linear Regression model involves a series of steps to build, train, evaluate, and use the model to make predictions


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)




 Conclusion

Exploratory Data Analysis and Data Preprocessing are the pillars upon which insightful data analysis is built. By meticulously cleaning the data, creating informative visualizations, and preparing it for modeling or deeper analysis, we uncover the true potential hidden within the data.

In this journey through an Uber dataset, we've touched upon the essence of EDA and DPL. These techniques are not limited to ride-sharing data; they can be applied to a multitude of datasets across various domains. The ability to extract knowledge from data is a valuable skill in today's data-driven world. So, dive into your datasets, apply EDA and DPL techniques, and unlock the rich insights waiting to be discovered. 
