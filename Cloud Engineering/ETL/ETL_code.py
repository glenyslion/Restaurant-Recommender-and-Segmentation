import os
import sys
import json
import boto3
import pandas as pd
import numpy as np
import io
import logging
from collections import Counter

# Add to the system path for Lambda Layers
sys.path.append('/opt')
os.environ['LD_LIBRARY_PATH'] = '/opt/python/numpy.libs:/opt/python/pandas.libs:/opt/python'

# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3 = boto3.client('s3')

def lambda_handler(event, context):
    bucket_name = "ce-raw-datasets"
    customer_key = "train_customers.csv"
    location_key = "train_locations.csv"
    order_key = "orders.csv"
    vendor_key = "vendors.csv"
    output_key = "processed_data/order_clean_join_all.csv"

    try:
        logger.info("Starting Lambda function execution...")
        logger.info(f"NumPy Version: {np.__version__}")
        logger.info(f"Pandas Version: {pd.__version__}")

        customers = read_csv_from_s3(bucket_name, customer_key)
        locations = read_csv_from_s3(bucket_name, location_key)
        orders = read_csv_from_s3(bucket_name, order_key)
        vendors = read_csv_from_s3(bucket_name, vendor_key)

        cleaned_customers = clean_customers(customers)
        cleaned_locations = clean_locations(locations)
        cleaned_orders = clean_orders(orders)
        cleaned_vendors = clean_vendors(vendors)

        final_data = merge_datasets(cleaned_customers, cleaned_locations, cleaned_orders, cleaned_vendors)

        upload_to_s3(final_data, bucket_name, output_key)

        logger.info(f"Data processing complete. File saved to {output_key}")

        return {
            "statusCode": 200,
            "message": f"Data processing complete. File saved to {output_key}"
        }

    except Exception as e:
        logger.error(f"Error in Lambda function execution: {str(e)}")
        return {
            "statusCode": 500,
            "message": str(e)
        }

def read_csv_from_s3(bucket, key):
    try:
        logger.info(f"Reading {key} from bucket {bucket}")
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read()
        return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        logger.error(f"Error reading {key} from {bucket}: {str(e)}")
        raise

def upload_to_s3(df, bucket, key):
    try:
        logger.info(f"Uploading data to {key} in bucket {bucket}")
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
        logger.info(f"Data successfully uploaded to {key}")
    except Exception as e:
        logger.error(f"Error uploading to {key}: {str(e)}")
        raise

def clean_customers(df):
    logger.info("Cleaning customers dataset")
    try:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['updated_at'] = pd.to_datetime(df['updated_at'])
        df = df.loc[df.groupby('akeed_customer_id')['updated_at'].idxmax()].reset_index(drop=True)
        df.drop(['created_at', 'updated_at', 'status', 'verified', 'language'], axis=1, inplace=True, errors='ignore')
        df['gender'] = df['gender'].str.strip().str.title().fillna('Unknown')
        df.loc[~df['gender'].isin(['Male', 'Female']), 'gender'] = 'Unknown'
        df['dob'] = pd.to_numeric(df['dob'], errors='coerce')
        df.loc[df['dob'] > 2020, 'dob'] = np.nan
        df.loc[df['dob'] < 1945, 'dob'] = np.nan
        df = df.rename(columns={'akeed_customer_id': 'customer_id'})
        return df
    except Exception as e:
        logger.error(f"Error cleaning customers dataset: {str(e)}")
        raise

def clean_locations(df):
    logger.info("Cleaning locations dataset")
    try:
        df = df.dropna(subset=['latitude', 'longitude'])
        df['location_type'] = df['location_type'].fillna('Other')
        df = df.rename(columns={'location_number': "LOCATION_NUMBER", 'location_type': "LOCATION_TYPE"})
        return df
    except Exception as e:
        logger.error(f"Error cleaning locations dataset: {str(e)}")
        raise

def clean_orders(df):
    logger.info("Cleaning orders dataset")
    try:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['promo_code'] = df['promo_code'].notna().astype(int)
        df['promo_code_discount_percentage'] = pd.to_numeric(df['promo_code_discount_percentage'], errors='coerce').fillna(0)
        df['item_count'] = pd.to_numeric(df['item_count'], errors='coerce')
        df['item_count'] = df.groupby('customer_id')['item_count'].transform(lambda x: x.fillna(x.median()))
        df['is_favorite'] = df['is_favorite'].fillna("No")
        df['vendor_rating'] = pd.to_numeric(df['vendor_rating'], errors='coerce').fillna(0)
        df = df.drop('delivery_time', axis=1, errors='ignore')
        df['LOCATION_TYPE'] = df['LOCATION_TYPE'].fillna('Other')
        return df
    except Exception as e:
        logger.error(f"Error cleaning orders dataset: {str(e)}")
        raise

def clean_vendors(df):
    logger.info("Cleaning vendors dataset")
    try:
        vendor_cols = ['id', 'latitude', 'longitude', 'vendor_category_en', 'delivery_charge', 'vendor_tag_name']
        df = df[vendor_cols]
        df = df.rename(columns={
            'latitude': 'latitude_vendor',
            'longitude': 'longtitude_vendor',
            'id': 'vendor_id'
        })
        return df
    except Exception as e:
        logger.error(f"Error cleaning vendors dataset: {str(e)}")
        raise

def merge_datasets(customers, locations, orders, vendors):
    logger.info("Merging datasets")
    try:
        df_order_cust = pd.merge(orders, customers, on='customer_id', how='left')
        df_loc = locations.rename(columns={'location_number': "LOCATION_NUMBER"})
        df_order_full = pd.merge(df_order_cust, df_loc, on=['customer_id', 'LOCATION_NUMBER'], how='left')

        vendor_cols = ['vendor_id', 'latitude_vendor', 'longtitude_vendor', 'vendor_category_en', 'delivery_charge', 'vendor_tag_name']
        vendor_useful = vendors[vendor_cols]
        df_order_full = pd.merge(df_order_full, vendor_useful, on='vendor_id', how='left')

        return df_order_full
    except Exception as e:
        logger.error(f"Error merging datasets: {str(e)}")
        raise
