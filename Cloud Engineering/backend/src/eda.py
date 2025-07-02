import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import boto3
import logging
import os
from io import BytesIO
from dotenv import load_dotenv

# Create log directory if it doesn't exist
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logger
logger = logging.getLogger('eda')
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("log/eda.log")
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info("EDA module initialized")

def perform_eda(df, bucket_name, s3_prefix, aws_region):
    """
    Performs EDA and uploads plots directly to S3 without saving locally.

    Args:
        df (pd.DataFrame): Input data.
        bucket_name (str): Target S3 bucket.
        s3_prefix (str): S3 prefix for uploaded files.
        aws_region (str): AWS region for boto3 session.
    """
    logger.info(f"Starting EDA with {len(df)} rows of data")
    
    # Use AWS credentials from environment variables
    load_dotenv()
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    
    # Create session with credentials from env vars
    session = boto3.Session(
        region_name=aws_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    
    s3_client = session.client("s3")

    def upload_plot_to_s3(fig, filename):
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        s3_client.upload_fileobj(buffer, bucket_name, f"{s3_prefix}{filename}")
        logger.info(f"Uploaded {filename} to s3://{bucket_name}/{s3_prefix}{filename}")

    # 1. Correlation heatmap
    logger.info("Generating correlation heatmap")
    corr = df.select_dtypes(include='number').corr()
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, ax=ax1)
    ax1.set_title('Correlation Matrix')
    upload_plot_to_s3(fig1, 'correlation_heatmap.png')
    plt.close(fig1)

    # 2. Box plot: total spend per customer
    if 'customer_id' in df.columns and 'grand_total' in df.columns:
        logger.info("Generating total spend per customer box plot")
        spend_per_customer = df.groupby('customer_id')['grand_total'].sum().reset_index()
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=spend_per_customer['grand_total'], ax=ax2)
        ax2.set_title('Total Spend per Customer')
        ax2.set_xlabel('Total Spend')
        upload_plot_to_s3(fig2, 'box_total_spend_per_customer.png')
        plt.close(fig2)
    else:
        logger.warning("Required columns 'customer_id' and 'grand_total' not found for spend plot.")

    # 3. Box plot: order count per customer
    if 'customer_id' in df.columns:
        logger.info("Generating order counts per customer box plot")
        order_counts = df['customer_id'].value_counts().reset_index(drop=True)
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=order_counts, ax=ax3)
        ax3.set_title('Box Plot of Order Counts per Customer')
        ax3.set_xlabel('Order Count')
        upload_plot_to_s3(fig3, 'box_order_counts_per_customer.png')
        plt.close(fig3)
    else:
        logger.warning("Column 'customer_id' not found for order count boxplot.")
    
    logger.info("EDA process completed successfully")
