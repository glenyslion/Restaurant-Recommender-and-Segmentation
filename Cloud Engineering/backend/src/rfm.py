"""
RFM Analysis Module.

This module performs RFM (Recency, Frequency, Monetary) analysis on customer data.
"""
import logging
import yaml
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine # type: ignore
from sqlalchemy.exc import SQLAlchemyError # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger('rfm')
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("log/rfm.log")
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info("RFM module initialized")

class RFM:
    def __init__(self, df=None, config_path=None):
        """Initializes the RFM system by loading configuration and dataset."""
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
            self.rfm_config = config["rfm"]
            self.snapshot_date = pd.to_datetime(self.rfm_config["snapshot_date"]).date()
            self.cluster_mapping = self.rfm_config["cluster_mapping"]
            if df is not None:
                self.df = df.copy()
                logger.info(f"Loaded data, shape: {self.df.shape}")
        except Exception as error:
            logger.error("Failed to initialize RFM class: %s", error)
            raise

    def preprocess(self):
        """Select relevant columns and clean data."""
        try:
            self.df = self.df[["customer_id", "created_at", "akeed_order_id", "grand_total"]]
            self.df['order_date'] = pd.to_datetime(self.df['created_at']).dt.date
            rfm = self.df.groupby('customer_id').agg({
                    'order_date': lambda x: (self.snapshot_date - x.max()).days,
                    'akeed_order_id': 'count',
                    'grand_total': 'sum'
                })
            rfm.rename(columns={
                    'order_date': 'Recency',
                    'akeed_order_id': 'Frequency',
                    'grand_total': 'Monetary'
                }, inplace=True)
            self.rfm_df = rfm.reset_index()
        except Exception as error:
            logger.error("Error in preprocessing data: %s", error)
            raise


    def _train_model(self):
        """Train KMeans clustering on RFM features."""
        try:
            logger.info("Training RFM clustering model")
            rfm = self.rfm_df.copy()

            epsilon = 1e-10
            rfm['Log_Monetary'] = np.log(np.maximum(rfm['Monetary'], epsilon))
            
            features = rfm[['Recency', 'Frequency', 'Log_Monetary']]
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(features)
            kmeans = KMeans(
                n_clusters=self.rfm_config["n_clusters"],
                random_state=self.rfm_config["random_state"]
            )
            rfm['K-Means'] = kmeans.fit_predict(rfm_scaled)
            rfm['Segment'] = rfm['K-Means'].map(self.cluster_mapping)
            self.rfm_labeled = rfm.reset_index()
            logger.info("Model training complete")
        except Exception as e:
            logger.error("Error training model: %s", e)
            raise

    def clv_calculation(self):
        """Calculate CLV per segment using average monthly orders and average order value."""
        try:
            logger.info("Calculating Customer Lifetime Value (CLV_30)")
            # Merge original orders with segment labels
            df = self.df.copy()
            df = pd.merge(df, self.rfm_labeled[['customer_id', 'Segment']], on='customer_id', how='left')
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            df['year_month'] = df['order_date'].dt.to_period('M')
            # Step 1: Monthly avg orders per user per segment
            monthly_orders = df.groupby(['Segment', 'year_month'])['akeed_order_id'].count().reset_index()
            monthly_users = df.groupby(['Segment', 'year_month'])['customer_id'].nunique().reset_index()
            monthly_orders = monthly_orders.merge(monthly_users, on=['Segment', 'year_month'], how='left')
            monthly_orders['Avg_Orders_Per_User_Per_Month'] = (
                monthly_orders['akeed_order_id'] / monthly_orders['customer_id']
            )
            avg_monthly_orders_per_segment = monthly_orders.groupby('Segment')[
                'Avg_Orders_Per_User_Per_Month'
            ].mean().reset_index()
            avg_monthly_orders_per_segment.columns = ['Segment', 'Avg_Orders_Per_Month_Per_User']
            # Step 2: Calculate average order value from RFM
            rfm = self.rfm_labeled.copy()
            cluster_stats = rfm.groupby('K-Means').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'K-Means': 'count'
            }).rename(columns={'K-Means': 'Count'}).reset_index()
            cluster_stats['Segment'] = cluster_stats['K-Means'].map(self.cluster_mapping)
            cluster_stats['avg_order_value'] = cluster_stats['Monetary'] / cluster_stats['Frequency']
            # Step 3: Combine
            clv_df = pd.merge(
                avg_monthly_orders_per_segment,
                cluster_stats[['Segment', 'avg_order_value']],
                how='left',
                on='Segment'
            )
            clv_df['CLV_30'] = clv_df['Avg_Orders_Per_Month_Per_User'] * clv_df['avg_order_value']
            self.clv_df = clv_df
            self.rfm_labeled = pd.merge(
                self.rfm_labeled[['customer_id', 'Segment']],
                clv_df[['Segment', 'CLV_30']],
                on='Segment',
                how='left'
            )
        except Exception as e:
            logger.error("Error during CLV_30 calculation: %s", e)
            raise

    def run(self):
        """Run the complete RFM analysis, return final rfm dataframe with segment and clv"""
        try:
            self.preprocess()
            self._train_model()
            self.clv_calculation()
            return self.rfm_labeled
        except Exception as e:
            logger.error("Failed to complete RFM: %s", e)
            raise


