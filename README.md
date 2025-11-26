# CryptoStreamPredictor

This project leverages Apache Spark Structured Streaming to analyze real-time cryptocurrency data from Binance, predict market trends using a trained RandomForestClassifier model, and monitor key metrics.

## Project Overview

In the fast-paced world of cryptocurrency, identifying emerging trends quickly can be crucial. This project sets up a live streaming data pipeline to ingest cryptocurrency ticker data, clean and preprocess it, and then apply a machine learning model to predict potential trends. The results are continuously monitored and saved, demonstrating a robust real-time analytical solution.

## Features

*   **Real-time Data Ingestion**: Continuously fetches 24-hour ticker data for top USDT trading pairs from the Binance API.
*   **Spark Structured Streaming**: Utilizes Spark's powerful streaming capabilities to process incoming data in micro-batches.
*   **Data Preprocessing**: Cleans and prepares raw data, handling missing values and ensuring correct data types.
*   **Trend Prediction**: Employs a RandomForestClassifier to predict if a cryptocurrency pair is showing a 'trend' (based on significant price change or volume).
*   **Dynamic Output**: Stores real-time predictions to CSV files and allows for on-the-fly SQL queries to analyze the latest processed data.
*   **Key Metric Monitoring**: Provides insights into top performing assets by price change and volume, as well as the distribution of predicted trends.

## Technologies Used

*   **Apache Spark (PySpark)**: For distributed data processing and structured streaming.
*   **Pandas**: For initial data handling and utility functions.
*   **Requests**: To fetch data from the Binance API.
*   **Matplotlib / Seaborn (Optional for visualization)**: For data visualization (if added).
*   **Jupyter/Colab Notebook**: For interactive development and execution.

## Setup and Installation

To run this project, you'll need a Python environment with PySpark and other dependencies. A Google Colab environment is recommended for ease of setup.

1.  **Install Dependencies**:

    ```bash
    !apt-get update -qq
    !apt-get install -y openjdk-8-jdk-headless -qq
    !wget -q https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
    !tar xf spark-3.5.0-bin-hadoop3.tgz
    !pip install -q findspark pyspark requests pandas matplotlib seaborn
    ```

2.  **Configure Environment Variables**:

    ```python
    import os, findspark
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
    os.environ["SPARK_HOME"] = "/content/spark-3.5.0-bin-hadoop3"
    findspark.init()
    ```

3.  **Initialize Spark Session**:

    ```python
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("CryptoLiveStreaming") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    ```

## Usage

1.  **Fetch Initial Data**: The `fetch_binance_tickers` function retrieves current cryptocurrency data.
2.  **Train Model**: A RandomForestClassifier is trained on historical (or snapshot) data to identify trends.
3.  **Start Streaming**: A Spark Structured Streaming job monitors a specified input directory for new data files.
4.  **Real-time Prediction**: As new data arrives, the model makes predictions, and the results are saved and displayed.
5.  **Post-Stream Analysis**: SQL queries can be run against the aggregated stream data to extract insights.

## Data Flow

Binance API -> Raw Data (CSV) -> Spark Structured Stream -> Feature Engineering -> RandomForest Model -> Real-time Predictions (CSV) -> Analysis

## Model Details

The RandomForestClassifier is trained on features such as `lastPrice`, `priceChangePercent`, and `volume`. The target variable `trend` is defined as `1` if `priceChangePercent > 0.5` or `volume > 1,000,000`, and `0` otherwise.

## Output

The streaming output includes CSV files in the `/content/predictions_only` directory, each containing predictions for a micro-batch of incoming data. The project also demonstrates SQL queries to analyze this output, showing top movers and trend distributions.

## Future Enhancements

*   Integrate with a message broker (e.g., Kafka) for more robust real-time data ingestion.
*   Implement a more sophisticated trend detection algorithm.
*   Build a real-time dashboard for visualization.
*   Expand feature set for machine learning model (e.g., technical indicators).
*   Deploy the Spark application to a cluster for scalability.
