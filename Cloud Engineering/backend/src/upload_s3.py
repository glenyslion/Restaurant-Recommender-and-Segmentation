import boto3
from io import StringIO
import logging
import datetime
import os
from dotenv import load_dotenv

logger = logging.getLogger("clustering_s3_upload")

def upload_clustering_to_s3(df, bucket_name, s3_prefix, aws_region="us-east-1", filename=None):
    """Uploads a pandas DataFrame clustering CSV to an S3 bucket."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = filename or f"clustering_{timestamp}.csv"
        s3_key = f"{s3_prefix.rstrip('/')}/{file_name}"

        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)

        # Use AWS credentials from environment variables
        load_dotenv()
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        
        s3 = boto3.client(
            "s3",
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )

        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue())

        s3_uri = f"s3://{bucket_name}/{s3_key}"
        logger.info("Uploaded DataFrame to S3: %s", s3_uri)
        return s3_uri

    except Exception as e:
        logger.error("Failed to upload DataFrame to S3: %s", e)
        raise