import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import openai
import uvicorn
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np

# Load the environment variables

# Function to analyze data
def analyze_data(df):
    # Summary statistics
    summary = df.describe(include='all').transpose()

    # Missing values count
    missing_values = df.isnull().sum()

    # Correlation matrix
    correlations = df.corr()

    # Detect outliers using the IQR method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

    # Perform KMeans clustering (arbitrarily setting k=3 for demonstration)
    numeric_df = df.select_dtypes(include='number').dropna()
    if len(numeric_df.columns) > 0:  # Ensure there are numeric columns for clustering
        kmeans = KMeans(n_clusters=3, random_state=1)
        cluster_labels = kmeans.fit_predict(numeric_df)
        df['Cluster'] = cluster_labels  # Add cluster labels to the dataset
    else:
        cluster_labels = []

    # Hierarchical clustering (using a subset for efficiency)
    if len(numeric_df.columns) > 1:
        sample_df = numeric_df.sample(min(100, len(numeric_df)), random_state=1)  # Limit to 100 rows for dendrogram
        hierarchy = linkage(sample_df, method='ward')
    else:
        hierarchy = None

    return {
        "summary": summary,
        "missing_values": missing_values,
        "correlations": correlations,
        "outliers": outliers,
        "clusters": cluster_labels,
        "hierarchy": hierarchy,
    }

import os
import openai
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, before_log, after_log
import logging
# Configure logging for retries
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before=before_log(logger, logging.INFO),
    after=after_log(logger, logging.INFO),
)
def execute_code_with_retry(code_to_run):
    """
    Executes the given Python code and retries upon failure.
    """
    exec_globals = {}
    try:
        exec(code_to_run, exec_globals)
    except Exception as e:
        logger.error(f"Error during code execution: {e}")
        raise  # Trigger retry
    return exec_globals

load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# Set the OpenAI API key. Use AIPROXY_TOKEN if available, otherwise fallback to environment variable
openai.api_key = AIPROXY_TOKEN if AIPROXY_TOKEN else os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable or provide AIPROXY_TOKEN")

openai.api_base = "https://aiproxy.sanand.workers.dev/openai/"

def interact_with_llm(filename, df, max_tokens=1500):
    # Prepare context for the LLM
    column_info = df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"})
    summary_stats = df.describe(include='all').transpose()
    sample_data = df.head(5).to_dict(orient="records")

    # Missing values count---------------------------
    missing_values = df.isnull().sum()

    """# Correlation matrix
    correlations = df.corr(numeric_only=True)

    # Detect outliers using the IQR method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

    # Perform KMeans clustering (arbitrarily setting k=3 for demonstration)
    numeric_df = df.select_dtypes(include='number').dropna()
    if len(numeric_df.columns) > 0:  # Ensure there are numeric columns for clustering
        kmeans = KMeans(n_clusters=3, random_state=1)
        cluster_labels = kmeans.fit_predict(numeric_df)
        df['Cluster'] = cluster_labels  # Add cluster labels to the dataset
    else:
        cluster_labels = []

    # Hierarchical clustering (using a subset for efficiency)
    if len(numeric_df.columns) > 1:
        sample_df = numeric_df.sample(min(100, len(numeric_df)), random_state=1)  # Limit to 100 rows for dendrogram
        hierarchy = linkage(sample_df, method='ward')
    else:
        hierarchy = None
        #--------------------------"""
    context = {
        "filename": filename,
        "columns": column_info.to_dict(orient="records"),
        "summary_statistics": summary_stats.to_dict(),
        "example_values": sample_data,
        
    }
    """missing_values": missing_values,
    "correlations": correlations,
    "outliers": outliers,
    "clusters": cluster_labels,
    "hierarchy": hierarchy
    - Correlations: {context["correlations"]}
    - Outliers: {context["outliers"]}
    - Clusters: {context["cluster_labels"]}
    - Hierarchical Clustering: {context["hierarchy"]}"""
    # Compose the initial prompt
    prompt = f"""
    You are an expert in data analysis and visualization. I am providing you the structure, summary statistics, and sample data of a dataset to analyze.

    Dataset Information:
    - Filename: {filename}
    - Column Details (Name and Type): {context["columns"]}
    - Summary Statistics: {context["summary_statistics"]}
    - Sample Records (first 5 rows): {context["example_values"]}


    Your tasks:
    Important: Your code should NOT print out anything.
    Also ensure that your code does NOT generate any plots / figures (but your code can give aids for visualisation of data, e.g. corr matrix instead of heatmap)
    Do generic analysis that will apply to all datasets. For example, summary statistics, counting missing values, correlation matrices, outliers, clustering, hierarchy detection, etc.
    Use chardet library appropriately to handle character encoding related errors while your code tries reading the dataset
    1. Analyze the dataset to uncover meaningful insights. Focus on:
      - Outlier and anomaly detection (e.g., errors, high-impact opportunities)
      - Correlation and regression analysis (e.g., identifying relationships and key predictors)(Use df.corr() with numeric_only=True)
      - Feature importance analysis (e.g., identifying the most influential variables)
      - Clustering and group detection (e.g., natural groupings in data)
      - Time-series analysis (e.g., identifying patterns for prediction, if relevant)
      - Any other analysis that could provide actionable insights.
    2. For each suggested analysis:
      - Explain why it is relevant to this dataset and the kind of insights it can provide.
      - Provide Python code that can run without modifications and works reliably on the given dataset.
    3. Ensure all analyses are general and applicable to datasets with similar structures. 

    Additional Notes:
    - Include code for appropriately preprocessing the dataset (it may contain null values, incoherent values etcc... also use proper imputation). Use the summary statistics and sample values for context.
    - Prioritize analyses that can improve decision-making or highlight high-value opportunities.
    - Make sure the Python code you provide handles common issues like missing values, non-numeric data, and scaling.
    """

    # Send the prompt to the LLM
    url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"  # Replace with the actual proxy URL

    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",  # Specify the required model
        "messages": [
            #{"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens":max_tokens
    }

    # Send the request
    response = requests.post(url, json=data, headers=headers)

    # Check for errors
    if response.status_code != 200:
        raise RuntimeError(f"API request failed: {response.status_code}, {response.text}")

    """response = openai.Completion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,

    )"""

    llm_reply = response.json()["choices"][0]["message"]["content"]
    print("\n=== LLM Suggestions ===\n")
    print(llm_reply)

    # Extract and run code suggestions cautiously
    print("\n*********************************\n=== Autocode Execution ===\n")
 
    suggestions = llm_reply.split("```")
    executed_results = {}
    code_to_run=""
    for i, code in enumerate(suggestions):
        if code.startswith("python"):
            code_to_run += '\n'+code[7:].strip()
    try:
        # Attempt to execute the code with retry on failure
        executed_results["analysis_results"] = execute_code_with_retry(code_to_run)
        print(f"Code executed successfully.")
    except Exception as e:
        print(f"All retry attempts failed. Error: {e}")
        #executed_results = {"error": str(e)}

    executed_results["summary"] = summary_stats
    executed_results["missing_values"] = missing_values
    return executed_results
folder=""
def visualize_results(analysis_results):
    visualizations = []
    global folder
    if "correlation_matrix" in analysis_results:
        plt.figure(figsize=(6, 6))  # Set size to 512x512 px
        sns.heatmap(analysis_results["correlation_matrix"], annot=True, cmap="coolwarm", fmt=".2f")
        correlation_image = folder+"/correlation_matrix.png"
        plt.savefig(correlation_image, dpi=96)  # Ensure 512x512 px resolution
        visualizations.append(correlation_image)
        plt.close()

    if "outliers" in analysis_results:
        plt.figure(figsize=(6, 6))  # Set size to 512x512 px
        sns.boxplot(data=analysis_results["outliers"])
        outliers_image = folder+"/outliers.png"
        plt.savefig(outliers_image, dpi=96)  # Ensure 512x512 px resolution
        visualizations.append(outliers_image)
        plt.close()

    if "clusters" in analysis_results:
        plt.figure(figsize=(6, 6))  # Set size to 512x512 px
        sns.scatterplot(
            x=analysis_results["clusters"]["x"],
            y=analysis_results["clusters"]["y"],
            hue=analysis_results["clusters"]["labels"]
        )
        clusters_image = folder+"/clusters.png"
        plt.savefig(clusters_image, dpi=96)  # Ensure 512x512 px resolution
        visualizations.append(clusters_image)
        plt.close()

    return visualizations


def generate_story_from_analysis(filename, analysis_results):
    # Extract analysis details
    
    summary_stats = analysis_results.get("summary")
    missing_values = analysis_results.get("missing_values")
    correlations = analysis_results.get("correlations").to_dict() if analysis_results.get("correlations") is not None else "N/A"
    outliers = analysis_results.get("outliers").to_dict() if analysis_results.get("outliers") is not None else "N/A"
    clusters = analysis_results.get("clusters") if analysis_results.get("clusters") else "No clusters identified."
    hierarchy = "Generated hierarchical clustering dendrogram." if analysis_results.get("hierarchy") is not None else "No hierarchical clustering performed."

    # Construct the prompt for the LLM
    prompt = f"""
    You are a storytelling expert with knowledge in data analysis. Based on the following analysis results, generate a compelling story:

    Dataset: {filename}

    1. Summary Statistics:
    {summary_stats.to_dict()}

    2. Missing Values:
    {missing_values.to_dict()}

    3. Correlation Analysis:
    {correlations}

    4. Outlier Detection:
    {outliers}

    5. Clustering:
    {clusters}

    6. Hierarchical Analysis:
    {hierarchy}

    Using the above, write a narrative covering:
    - An overview of the dataset and its characteristics.
    - Key insights derived from the analyses.
    - Potential implications or actionable recommendations based on these insights.
    - A conversational, engaging tone suitable for non-technical readers.
    """

    # Send the prompt to the LLM
    url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"  # Replace with the actual proxy URL

    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",  # Specify the required model
        "messages": [
            {"role": "system", "content": "You are a storytelling expert with knowledge in data analysis. "},
            {"role": "user", "content": prompt}
        ],
        "max_tokens":1500
    }

    # Send the request
    response = requests.post(url, json=data, headers=headers)

    # Check for errors
    if response.status_code != 200:
        raise RuntimeError(f"API request failed: {response.status_code}, {response.text}")

    # Send the prompt to the LLM
    

    story = response.json()["choices"][0]["message"]["content"].strip()
    return story
import sys
import os
import pandas as pd
import base64
import chardet
# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]

    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    with open(filename, 'rb') as file:
        result = chardet.detect(file.read())  # Read a portion of the file
        detected_encoding = result['encoding']
    # Load the dataset
    try:
        df = pd.read_csv(filename,encoding=detected_encoding)
        global folder
        folder = os.path.dirname(filename)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Step 1: Analyze the data using LLM interaction
    try:
        print(f"****************Interacting with the LLM***********************")
        analysis_results = interact_with_llm(filename, df)
    except Exception as e:
        print(f"Error during LLM interaction: {e}")
        sys.exit(1)

    # Step 2: Visualize the results
    try:
        print(f"**************** Visualisations underway ***********************")
        visualizations = visualize_results(analysis_results['analysis_results'])
    except Exception as e:
        print(f"Error generating visualizations--: {e}")
        sys.exit(1)

    # Step 3: Generate a story and save it in README.md
 

    try:
        print(f"**************** Generating the story ***********************")

        readme_content = generate_story_from_analysis(filename, analysis_results)
        readme_content += "\n\n## Visualizations\n"

        
        # Embed Correlation Matrix as Base64
        if os.path.isfile(folder+"/correlation_matrix.png"):
            with open(folder+"/correlation_matrix.png", "rb") as img_file:
                correlation_matrix_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            readme_content += "### Correlation Matrix\n"
            readme_content += f"![Correlation Matrix](data:image/png;base64,{correlation_matrix_base64})\n\n"

        # Embed Outliers Detection as Base64
        if os.path.isfile(folder+"/outliers.png"):
            with open(folder+"/outliers.png", "rb") as img_file:
                outliers_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            readme_content += "### Outlier Detection\n"
            readme_content += f"![Outlier Detection](data:image/png;base64,{outliers_base64})\n\n"

        with open(folder+"/README.md", "w") as readme_file:
            readme_file.write(readme_content)
        
        print("Analysis completed successfully.")
        print("Results saved to README.md with embedded images.")
    except Exception as e:
        print(f"Error generating the story: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

