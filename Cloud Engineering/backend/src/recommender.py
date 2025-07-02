"""
Restaurant Recommendation System Module.

This module provides functionality to build and train restaurant recommendations models
based on customer order history using collaborative filtering and upload them to S3.
"""
import logging
import os
import pickle
import tempfile
import pandas as pd
import yaml
import boto3
from surprise import Dataset, Reader, SVD, NMF, SVDpp, KNNBasic
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger('recommender')
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("log/recommender.log")
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info("Recommender module initialized")

class Recommender:
    """
    Restaurant recommendation system using collaborative filtering.

    This class handles building recommendation models and uploading them to S3.
    """

    def __init__(self, config, df):
        """
        Initializes the Recommender system by loading configuration and dataset.
        """
        self.model_params = config["models"]
        self.aws_config = config["aws_rs"]
        self.data = df
        self.data_frame = self._data_preprocess()

    def _data_preprocess(self):
        """
        Loads data from the database and preprocesses it for restaurant recommendations.

        Returns:
            pd.DataFrame: Aggregated DataFrame with columns [customer_id, vendor_id, rating]
        """
        try:
            df = self.data[['customer_id', 'vendor_id', 'vendor_category_en']].copy()
            df.loc[:, 'rating'] = 1
            restaurant_df = df[df['vendor_category_en'] == 'Restaurants'][
                ['customer_id', 'vendor_id', 'rating']]
            grouped = restaurant_df.groupby(
                ['customer_id', 'vendor_id'], as_index=False)['rating'].sum()
            logger.info("Data loaded successfully with %d rows", len(grouped))
            return grouped
        except SQLAlchemyError as db_err:
            logger.error("Database error: %s", db_err)
            raise
        except Exception as error:
            logger.error("Unexpected error while loading data: %s", error)
            raise

    def _build_model(self, model_type):
        """
        Builds a recommendation model using parameters from config.

        Args:
            model_type (str): One of ['svd', 'nmf', 'svdpp', 'user_knn', 'item_knn']

        Returns:
            A Surprise model instance
        """
        try:
            params = self.model_params.get(model_type, {})
            if model_type == 'svd':
                return SVD(**params)
            if model_type == 'nmf':
                return NMF(**params)
            if model_type == 'svdpp':
                return SVDpp(**params)
            if model_type in ['user_knn', 'item_knn']:
                return KNNBasic(sim_options=params.get("sim_options", {}))

            raise ValueError(f"Unknown model type: {model_type}")
        except Exception as error:
            logger.error("Failed to build model: %s", error)
            raise

    def upload_models_to_s3(self):
        """
        Trains all available recommendation models and uploads them to S3.
        
        This function builds each model in the available models list, trains it on
        the current dataset, and uploads the serialized model to the specified S3 bucket
        under the configured prefix folder.
        
        Returns:
            dict: Status of each model upload {'model_name': success_status}
        """
        try:
            if not self.aws_config.get("upload", False):
                logger.info("AWS upload disabled, skipping recommender model upload")
                return {"upload_enabled": False}
            
            # Use AWS credentials from environment variables
            load_dotenv()
            aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
            
            s3_client = boto3.client(
                's3',
                region_name=self.aws_config['region'],
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )

            bucket_name = self.aws_config['bucket_name']
            prefix = self.aws_config['prefix']

            reader = Reader(rating_scale=(
                self.data_frame['rating'].min(),
                self.data_frame['rating'].max()
            ))
            data = Dataset.load_from_df(
                self.data_frame[['customer_id', 'vendor_id', 'rating']],
                reader
            )
            trainset = data.build_full_trainset()
            
            # Import needed for stdout redirection
            import sys
            from io import StringIO
            
            upload_status = {}
            for model_type in ["svd", "nmf", "svdpp", "item_knn"]:
                logger.info(f"Training {model_type} model...")
                model = self._build_model(model_type)
                
                # Redirect stdout to suppress similarity matrix computation messages
                if model_type in ["user_knn", "item_knn"]:
                    # Save original stdout
                    original_stdout = sys.stdout
                    # Redirect stdout to a string buffer
                    sys.stdout = StringIO()
                    
                    try:
                        # Fit the model - output will be captured
                        model.fit(trainset)
                    finally:
                        # Restore original stdout
                        sys.stdout = original_stdout
                else:
                    # For other models, no need to redirect
                    model.fit(trainset)
                
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    pickle.dump(model, tmp)
                    tmp_name = tmp.name
                
                try:
                    logger.info(f"Uploading {model_type} model to S3...")
                    s3_key = f"{prefix}{model_type}_model.pkl"
                    s3_client.upload_file(tmp_name, bucket_name, s3_key)
                    upload_status[model_type] = True
                    logger.info(f"Successfully uploaded {model_type} model to S3")
                except Exception as upload_error:
                    logger.error(f"Failed to upload {model_type} model: {upload_error}")
                    upload_status[model_type] = False
                finally:
                    if os.path.exists(tmp_name):
                        os.remove(tmp_name)
            
            return upload_status
        
        except Exception as error:
            logger.error(f"Failed to upload models to S3: {error}")
            raise
