import argparse
import logging
import yaml
import os
from dotenv import load_dotenv

import src.acquire_data as ad
import src.eda as eda
from src.rfm import RFM
from src.food import FOOD
from src.upload_s3 import upload_clustering_to_s3
from src.recommender import Recommender

load_dotenv()

log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logger
logger = logging.getLogger('pipeline')
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("log/pipeline.log")
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info("Pipeline initialized")

if __name__ == "__main__":
    # CLI argument: --config
    parser = argparse.ArgumentParser(description="EDA pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # Load config.yaml
    with open(args.config, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error("Failed to load config: %s", args.config)
            raise e
        else:
            logger.info("Loaded config from %s", args.config)

    # Step 1: Acquire data
    logger.info("Step 1: Acquiring data from RDS")
    df = ad.acquire_data_rds(config)

    # Step 2: Perform EDA and upload directly to S3
    aws_config = config["aws_eda"]  
    if aws_config["upload"]:
        logger.info("Step 2: Performing EDA and uploading plots to S3")
        eda.perform_eda(
            df,
            bucket_name=aws_config["bucket_name"],
            s3_prefix=aws_config["prefix"],
            aws_region=aws_config["region"]
        )
    else:
        logger.warning("AWS upload disabled. Skipping EDA.")

    logger.info("EDA upload completed.")
    
    # Step 3: Run rfm and food clusterings, merge, and save to S3
    rfm=RFM(df=df, config_path=args.config)
    rfm_df= rfm.run()
    food=FOOD(df=df, config_path=args.config)
    food_df = food.run()
    logger.info("Step 3: Merging RFM and FOOD dataframes")
    joined_df = rfm_df.merge(food_df, on="customer_id", how="left")
    logger.info("Merged dataframe shape: %s", joined_df.shape)
    # Save the merged dataframe to S3
    s3_config = config["aws_clustering"]
    if s3_config["upload"]:
        logger.info("Uploading merged dataframe to S3")
        upload_clustering_to_s3(
            df=joined_df,
            bucket_name=s3_config["bucket_name"],
            s3_prefix=s3_config["prefix"],
            aws_region=s3_config["region"],
        )
    else:
        logger.warning("AWS clustering upload disabled. Skipping upload of merged dataframe.")

# Step 4: Build and upload recommender models to S3
    try:    
        logger.info("Step 4: Building and uploading recommender models")
        rs = Recommender(config, df)
        upload_status = rs.upload_models_to_s3()
        
        if upload_status:
            if upload_status.get("upload_enabled", True) is False:
                logger.warning("Recommender models trained but upload was disabled in configuration")
            else:
                if all(status for status in upload_status.values()):
                    logger.info("All recommender models were successfully uploaded")
                else:
                    logger.warning("Some recommender models failed to upload to S3")
        else:
            logger.warning("No upload status returned from recommender")    
    except Exception as e:
        logger.error(f"Failed to upload recommender models: {e}")
        logger.error("Recommender pipeline failed.")