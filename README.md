# GrantScope: The Grant Data Exploration Dashboard

## Overview
GrantScope is an interactive tool designed to aid grant analysts, grant writers, and individuals in comprehensively understanding and analyzing complex grant data. This tool leverages advanced data processing and visualization techniques to extract actionable insights from grant datasets, facilitating easier identification of funding opportunities, understanding funding trends, and enhancing grant writing and analysis efforts.

## Features
### Interactive Data Visualizations
- **Data Summary**: Offers a concise overview of the dataset, including total counts of unique grants, funders, and recipients. Visualize the top funders by total grant amount and understand the distribution of grants by funder type.
- **Grant Amount Visualizations**: Explore grant amounts through various lenses using interactive charts. Users can examine the distribution of grant amounts across different USD clusters, observe trends over time with scatter plots, and analyze grant amounts across geographical regions and subject areas using heatmaps.
- **Word Clouds**: Visualize the most common themes and keywords in grant descriptions across different segments, providing insights into the focus areas of funders and the nature of funded projects.
- **Treemaps**: Investigate the allocation of grant amounts by subject, population, and strategy. Treemaps allow for a hierarchical exploration of how funds are distributed among different categories.

### AI-Assisted Analysis
- **Contextual Prompts**: Each visualization and analysis section is equipped with an AI-assisted chat feature that generates custom prompts based on the specific context of the data being explored.
- **Predefined Questions**: Users can select from a set of predefined questions relevant to each chart or analysis, guiding them in uncovering key insights and patterns.
- **Custom Questions**: Users have the flexibility to ask their own questions related to the data, leveraging the power of natural language processing to obtain meaningful responses.
- **Focused Insights**: The AI provides focused and relevant insights based on the specific chart or analysis being viewed, avoiding confusion between different sections of the dashboard.

### Detailed Analysis Tools
- **Univariate Analysis**: Perform detailed statistical analysis on numeric columns to understand the distribution, variability, and central tendencies of grant amounts and other numerical data points.
- **Grant Relationship Analysis**: Explore the relationships between funders and recipients, grant amounts, and project subjects or populations served. This feature allows users to uncover patterns and connections within the grant ecosystem.
- **Grant Descriptions Deep Dive**: Dive into the details of grant descriptions using text analysis tools. Identify frequently used terms, analyze sentiment, and extract thematic clusters to gain a deeper understanding of grant narratives.

### User Roles and Customization
The dashboard supports different user roles, providing tailored experiences for grant analysts/writers and general users:
- **Grant Analyst/Writer**: Access to advanced analytics features, including detailed relationship analysis, trend analysis over time, and custom data filters for in-depth research.
- **Normal Grant User**: A simplified interface focusing on key visualizations such as data summaries, basic grant amount distributions, and word clouds, suitable for users seeking a general overview of the grant landscape.

### Downloadable Reports and Data
Users can download customized reports and data extracts based on their analysis, enabling offline review and integration into grant proposals or reports. This feature supports Excel and CSV formats for easy use in various applications.

## How to Use the Dashboard
1. Start by uploading your grant data file or using the preloaded dataset.
2. Select your user role to customize the dashboard experience according to your needs.
3. Explore the dashboard sections to visualize and analyze grant data. Utilize filters and interactive elements to tailor the analysis.
4. Engage with the AI-assisted chat feature to ask questions and gain insights specific to each chart or analysis.
5. Download data extracts and reports for further use.

## Technology Stack
This dashboard is built using Streamlit, enabling an interactive web application experience. Data visualization is powered by Plotly and Matplotlib for dynamic and static charts, respectively. Pandas and NumPy are used for data manipulation and analysis, while advanced text processing features leverage natural language processing libraries. The AI-assisted chat feature is implemented using OpenAI's GPT-4 and the LlamaIndex library.

## Usage
1. Access the web version of the dashboard at [grantscope.streamlit.app](https://grantscope.streamlit.app).

### Run on your own resources
1. Clone the repository to your local machine.
2. Install the required dependencies listed in the `requirements.txt` file.
3. Set up your OpenAI API key either as an environment variable or enter it through the dashboard's user interface.
4. Prepare your grant data in the required JSON format. You can either provide a file path to the JSON file or upload the file through the dashboard's user interface.
5. Run the Streamlit app using the command `streamlit run app.py`.
6. Access the dashboard through your web browser at the provided URL.

## Contributing
We welcome contributions to enhance the Grant Analysis Dashboard. If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. Let's collaborate to make this tool even more valuable for the grant analysis community!

## License
This project is licensed under the GNU General Public License v3.0.

We hope that GrantScope empowers grant analysts and writers to uncover valuable insights, identify potential funding opportunities, and craft compelling grant proposals. Happy exploring and grant writing!
