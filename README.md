# GrantScope the Grant Data Exploration Dashboard

## Overview

GrantScope is an interactive tool designed to aid grant analysts, grant writers, and individuals in comprehensively understanding and analyzing complex grant data. This tool leverages advanced data processing and visualization techniques to extract actionable insights from grant datasets, facilitating easier identification of funding opportunities, understanding funding trends, and enhancing grant writing and analysis efforts.

## Features

### Interactive Data Visualizations

- **Data Summary**: Offers a concise overview of the dataset, including total counts of unique grants, funders, and recipients. Visualize the top funders by total grant amount and understand the distribution of grants by funder type.

- **Grant Amount Visualizations**: Explore grant amounts through various lenses using interactive charts. Users can examine the distribution of grant amounts across different USD clusters, observe trends over time with scatter plots, and analyze grant amounts across geographical regions and subject areas using heatmaps.

- **Word Clouds**: Visualize the most common themes and keywords in grant descriptions across different segments, providing insights into the focus areas of funders and the nature of funded projects.

- **Treemaps**: Investigate the allocation of grant amounts by subject, population, and strategy. Treemaps allow for a hierarchical exploration of how funds are distributed among different categories.

### Detailed Analysis Tools

- **Univariate Analysis**: Perform detailed statistical analysis on numeric columns to understand the distribution, variability, and central tendencies of grant amounts and other numerical data points.

- **Grant Relationship Analysis**: Explore the relationships between funders and recipients, grant amounts, and project subjects or populations served. This feature allows users to uncover patterns and connections within the grant ecosystem.

- **Grant Descriptions Deep Dive**: Dive into the details of grant descriptions using text analysis tools. Identify frequently used terms, analyze sentiment, and extract thematic clusters to gain a deeper understanding of grant narratives.

## User Roles and Customization

The dashboard supports different user roles, providing tailored experiences for grant analysts/writers and general users:

- **Grant Analyst/Writer**: Access to advanced analytics features, including detailed relationship analysis, trend analysis over time, and custom data filters for in-depth research.

- **Normal Grant User**: A simplified interface focusing on key visualizations such as data summaries, basic grant amount distributions, and word clouds, suitable for users seeking a general overview of the grant landscape.

## Downloadable Reports and Data

Users can download customized reports and data extracts based on their analysis, enabling offline review and integration into grant proposals or reports. This feature supports Excel and CSV formats for easy use in various applications.

## How to Use the Dashboard

- Start by uploading your grant data file or using the preloaded dataset.
- Select your user role to customize the dashboard experience according to your needs.
- Explore the dashboard sections to visualize and analyze grant data. Utilize filters and interactive elements to tailor the analysis.
- Download data extracts and reports for further use.

## Technology Stack

This dashboard is built using Streamlit, enabling an interactive web application experience. Data visualization is powered by Plotly and Matplotlib for dynamic and static charts, respectively. Pandas and NumPy are used for data manipulation and analysis, while advanced text processing features leverage natural language processing libraries.

## Web version available at grantscope.streamlit.app

## Run on your own resources

1. Clone the repository to your local machine.
2. Install the required dependencies listed in the `requirements.txt` file.
3. Prepare your grant data in the required JSON format. You can either provide a file path to the JSON file or upload the file through the dashboard's user interface.
4. Run the Streamlit app using the command `streamlit run streamlit_app.py`.
5. Access the dashboard through your web browser at the provided URL.

## Usage

1. Select your user role (Grant Analyst/Writer or Normal Grant User) from the sidebar.
2. Choose the desired chart or analysis from the options provided based on your user role.
3. Interact with the visualizations, apply filters, and explore the grant data.
4. Use the download buttons to export the data for further analysis or reporting.

## Contributing

We welcome contributions to enhance the Grant Analysis Dashboard. If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. Let's collaborate to make this tool even more valuable for the grant analysis community!

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

---

We hope that GrantScope empowers grant analysts and writers to uncover valuable insights, identify potential funding opportunities, and crafting compelling grant proposals. Happy exploring and grant writing!

