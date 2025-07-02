"""
Webapp: Model Loading from S3 and Webapp Python Script
"""

import logging
from dotenv import load_dotenv
import os
import boto3
import pickle
import numpy as np
from typing import Any, Optional
import streamlit as st
import yaml
import pandas as pd
import io

# Load .env file
load_dotenv()

# Logging Setup
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "webapp_interface.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("webapp_interface")


@st.cache_resource
def load_config(path: str = "config/config_webapp.yaml") -> dict:
    """
    Load configuration YAML file

    Args:
        path (str): path to the configuration YAML file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    logger.info("Configuration file loaded from %s", path)
    return config_data


@st.cache_data
def load_rfm_from_s3(bucket: str, key: str):
    """
    Load RFM clustering results from a CSV file in S3.

    Args:
        bucket (str): Name of the S3 bucket.
        key (str): Path to the RFM CSV file in the S3 bucket.

    Returns:
        pd.DataFrame: DataFrame containing RFM segmentation and CLV data.
    """
    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read()
    df = pd.read_csv(io.BytesIO(content))
    logger.info("RFM data loaded from S3: s3://%s/%s", bucket, key)
    return df


@st.cache_data
def load_data_recom(bucket: str, key: str) -> pd.DataFrame:
    """
    Load raw order history data from S3 and filter for restaurant vendors only.

    Args:
        bucket (str): Name of the S3 bucket.
        key (str): Path to the order history CSV in the S3 bucket.

    Returns:
        pd.DataFrame: Filtered DataFrame with only restaurant order.
    """
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read()
    df = pd.read_csv(io.BytesIO(content))
    return df[df["vendor_category_en"] == "Restaurants"]

@st.cache_resource
def load_model_from_s3(bucket: str, key: str):
    """
    Load collaborative filtering model from S3.

    Args:
        bucket (str): Name of the S3 bucket.
        key (str): Path to the model pickle file in the S3 bucket.

    Returns:
        Any: Load Surprise model
    """
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    model = pickle.load(io.BytesIO(response["Body"].read()))
    logger.info("Model %s loaded from s3://%s/%s", key, bucket, key)
    return model


def run_app():
    """
    Main function to run streamlit webapp
    """
    st.set_page_config(
        page_title="Restaurant Clustering and Recommendation System", layout="centered"
    )
    st.title("Restaurant Clustering and Recommendation System")

    config = load_config()
    bucket_name = config.get("bucket_name")
    rfm_key = config.get("clustering_key")
    order_key = config.get("recom_data")

    rfm_df = load_rfm_from_s3(bucket_name, rfm_key)
    order_df = load_data_recom(bucket_name, order_key)

    st.subheader("Choose Model and Customer")
    col1, col2 = st.columns(2)

    model_options = list(config["models"].keys())
    with col1:
        customer_ids = order_df["customer_id"].unique()
        selected_customer = st.selectbox("Customer ID", customer_ids)

    with col2:
        model_choice = st.selectbox("Recommendation Model", model_options)

    model_key = config["models"][model_choice]["s3_key"]
    model = load_model_from_s3(bucket_name, model_key)

    row = rfm_df[rfm_df["customer_id"] == selected_customer]
    if not row.empty:
        st.markdown(f"**Customer Segment:** {row['Segment_x'].values[0]}")
        st.markdown(f"**Food Segment:** {row['Segment_y'].values[0]}")
        st.markdown(f"**Estimated CLV (30 days):** ${row['CLV_30'].values[0]:.2f}")
    else:
        st.warning("Customer not found in RFM data.")

    k = st.slider("Top-N Recommendations", 1, 20, 5)

    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            all_vendors = order_df["vendor_id"].unique()
            customer_vendors = order_df[
                order_df["customer_id"] == selected_customer
            ]["vendor_id"].unique()
            vendor_to_predict = [
                v for v in all_vendors if v not in customer_vendors
            ]

            predictions = [
                model.predict(selected_customer, vendor) for vendor in vendor_to_predict
            ]
            top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:k]

            rec_df = pd.DataFrame(
                [(pred.iid, pred.est) for pred in top_n],
                columns=["Vendor ID", "Estimated Score"],
            )

        st.success(f"Top {k} recommendations for customer {selected_customer}:")
        st.dataframe(rec_df)


if __name__ == "__main__":
    run_app()
