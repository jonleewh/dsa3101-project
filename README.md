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
Group A/src/Subgroup A - Sankey Diagram.ipynb

This data dictionary provides descriptions of the variables used in the analysis, based on the three code chunks for building Sankey diagrams. These include survey data, guest segmentation, and journey patterns through Universal Studios Singapore (USS).

### 1. Survey Data Columns (Used Across Code Chunks)

| **Variable Name** | **Description** |
|-------------------|-----------------|
| `Timestamp`       | The date and time when the survey response was recorded. |
| `Nationality`     | The nationality of the respondent. It could be either "Singaporean/PR" or "Foreigner". |
| `Age_Group`       | The age group of the respondent. Possible values are: "Below 18", "18-25", "26-35", "36-59", "60 and above". |
| `express_pass_purchase` | Indicates if the respondent purchased an express pass during their visit to USS. Possible values are "Yes" or "No". |
| `Rank the following theme park zones based on the order in which you visited them. [Hollywood]` | The rank of the Hollywood zone as per the respondent's visit, where "1st" indicates the first zone visited. |
| `Rank the following theme park zones based on the order in which you visited them. [New York]` | The rank of the New York zone as per the respondent's visit. |
| `Rank the following theme park zones based on the order in which you visited them. [Sci-Fi City]` | The rank of the Sci-Fi City zone as per the respondent's visit. |
| `Rank the following theme park zones based on the order in which you visited them. [Ancient Egypt]` | The rank of the Ancient Egypt zone as per the respondent's visit. |
| `Rank the following theme park zones based on the order in which you visited them. [The Lost World]` | The rank of The Lost World zone as per the respondent's visit. |
| `Rank the following theme park zones based on the order in which you visited them. [Far Far Away]` | The rank of the Far Far Away zone as per the respondent's visit. |
| `Rank the following theme park zones based on the order in which you visited them. [Madagascar]` | The rank of the Madagascar zone as per the respondent's visit. |

### 2. Variables for Sankey Diagram: Guest Segmentation and Journey Patterns

These variables are used in the Sankey diagrams to visualize the relationship between age groups, express pass usage, and theme park zones.

#### Guest Segmentation (Age Group)

| **Variable Name** | **Description** |
|-------------------|-----------------|
| `Age Group`       | The categorization of guests by age group. Possible values are: "Below 18", "18-25", "26-35", "36-59", "60 and above". |
| `Nationality`     | The nationality of the guest, either "Singaporean/PR" or "Foreigner". |
| `Express Pass`    | Whether the guest purchased an express pass. The possible values are "Yes" or "No". |
| `Theme Park Zones` | The rank in which guests visited different theme park zones like "Hollywood", "New York", "Sci-Fi City", etc. |

#### Sankey Flow Data (First Code Chunk)

| **Flow Source**  | **Flow Target**  | **Description** |
|------------------|------------------|-----------------|
| `Age Group`      | `Express Pass`   | The flow from guest age group to whether they purchased an express pass. |
| `Express Pass`   | `Theme Park Zones` | The flow from express pass usage to the theme park zone visited first. |

#### Sankey Flow Data (Second Code Chunk)

| **Flow Source**  | **Flow Target**  | **Description** |
|------------------|------------------|-----------------|
| `Age Group`      | `Theme Park Zones` | The flow from guest age group to the first theme park zone they visited. |

### 3. Sankey Diagram for "Foreigner" Guests

This part focuses on foreigner guests and their journey from express pass usage to visiting theme park zones.

#### Sankey Flow Data for Foreigner Guests

| **Flow Source**  | **Flow Target**  | **Description** |
|------------------|------------------|-----------------|
| `Foreigner`      | `Express Pass`   | The flow from the nationality (Foreigner) to whether they purchased an express pass. |
| `Express Pass`   | `Theme Park Zones` | The flow from express pass usage to the theme park zone visited first. |

### 4. Labels for Sankey Diagram

The labels in the Sankey diagram represent different categories or variables within the survey data. The labels used in the Sankey diagrams are:

| **Label**        | **Description** |
|------------------|-----------------|
| `Below 18`       | Age group "Below 18" |
| `Ages 18-25`     | Age group "18-25" |
| `Ages 26-35`     | Age group "26-35" |
| `Ages 36-59`     | Age group "36-59" |
| `Ages 60 and above` | Age group "60 and above" |
| `Hollywood`      | Hollywood theme park zone |
| `New York`       | New York theme park zone |
| `Sci-Fi City`    | Sci-Fi City theme park zone |
| `Ancient Egypt`  | Ancient Egypt theme park zone |
| `The Lost World` | The Lost World theme park zone |
| `Far Far Away`   | Far Far Away theme park zone |
| `Madagascar`     | Madagascar theme park zone |
| `Express Pass`   | Guests who purchased an express pass |
| `No Express Pass`| Guests who did not purchase an express pass |
| `Foreigner`      | Non-Singaporean guests (foreigners) |

### 5. Data Sources

- **Survey Data**: The survey responses used to generate the Sankey diagrams are stored in an Excel file (`Cleaned_Survey_Responses.xlsx`), which includes information about guest demographics, theme park preferences, and experiences.

### Conclusion

This data dictionary describes the key variables and their relationships as visualized in the Sankey diagrams, which illustrate the flow of guests from their demographic characteristics (such as age and nationality) to their behavior and preferences in the theme park.


### 


## Project Team





## Acknowledgements




