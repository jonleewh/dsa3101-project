# Enhancing Guest Experience through Data-Driven Journey Mapping and Analysis

<p style="text-align:center;">A project for NUS DSA3101: Data Science in Practice (AY2024/25 Semester 1)</p>

<p style="text-align:center;">Video | <a href="https://github.com/jonleewh/dsa3101-project/wiki">Wiki</a> | Dashboard (Links to be attached)</p>



## About the Project

Attractions and entertainment venues often struggle to provide consistently excellent guest experiences due to a lack of comprehensive understanding of the guest journey, preferences, and pain points. Traditional methods of gathering and analyzing guest data often fall short in providing actionable insights to improve operations, marketing strategies, and overall guest satisfaction.

This project aims to develop a data-driven system that maps and analyzes the entire guest journey, from pre-visit planning to post-visit feedback. By leveraging advanced data analysis, machine learning, and predictive modeling techniques, the project seeks to identify bottlenecks, optimize guest flow, personalize experiences, and ultimately boost guest satisfaction while potentially increasing revenue and operational efficiency.

## Getting Started
Instructions for Setting Up the Environment and Running the Code
### 1. Prerequisites
- **Python 3.8+**
- **pip** (Python package installer)
- **Git**

### 2. Clone the Repository
Clone the repository:
```bash
git clone https://github.com/jonleewh/dsa3101-project.git
```

### 3. Navigate to the project directory
Create and activate a virtual environment:
```bash
cd dsa3101-project
```

## How to use this repository
Summary of Key Folders and Files
### Group A
data/: Contains survey data for theme park analysis.

src/: Jupyter Notebooks for visualizations and analysis, including a Sankey diagram and weather based dynamic pricing analysis.

### Group B
data/: Includes simulation outputs, weather data, and theme park data (edges, nodes, and facilities).

src/: Python scripts and Jupyter Notebooks for various tasks, including alternative park layouts, dynamic queue simulations, and survey data synthesis.


### README.md
Provides an overview of the project, setup instructions, and usage.

## Data Sources:
Survey Data: Theme Parks Survey Responses 4 Nov.csv (contains responses from a theme park visitor survey).
Simulation Data: Generated simulation output files (e.g., simulation_output_2024-11-10_22-11-01).
Weather Data: Collected daily and hourly weather data (e.g., daily_weather_data.csv, hourly_weather_data.xlsx).
Park Layout Data: Includes CSV files describing park nodes and edges (e.g., theme_park_nodes.csv, theme_park_edges.csv).

### Data Preparation:
Clean and preprocess the raw survey data (e.g., remove missing values, standardize date formats).
Merge weather data with simulation data for analysis (e.g., combine temperature, attendance, and queue times).
Process and normalize park layout data to build graph-based representations of park paths.

## Data Dictionary
### Group A
  
**dynamic_pricing_weather.ipynb**

Satisfaction Score: A calculated score based on weather conditions (0–100)
  
Dynamic Price (SGD): Adjusted ticket price based on satisfaction score on weather.
  
Possible Extra Revenue (SGD): Revenue generated from dynamic pricing adjustments.

### Group B

**daily_weather_data.csv**

Rainfall (mm): Maximum rainfall in a window (mm).
  
Daily Rainfall Total (mm): Total rainfall for the day (mm).
  
Mean Temperature (°C): Average temperature for the day (°C).
  
Max Wind Speed (km/h): Maximum recorded wind speed for the day (km/h).


### 


## Project Team





## Acknowledgements




