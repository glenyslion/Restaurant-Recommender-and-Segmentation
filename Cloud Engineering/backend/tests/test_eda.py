import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock, ANY
import sys
sys.path.append('../')  # Add parent directory to path
from src.eda import perform_eda

class TestEDA(unittest.TestCase):
    def setUp(self):
        # Create a simple test DataFrame
        self.test_df = pd.DataFrame({
            'customer_id': [1, 1, 2, 2, 3],
            'vendor_id': [101, 102, 101, 103, 102],
            'grand_total': [50.0, 30.0, 25.0, 40.0, 60.0],
            'item_count': [2, 1, 1, 2, 3]
        })
        
        # Mock S3 client
        self.mock_s3_client = MagicMock()
        self.mock_boto3_session = MagicMock()
        self.mock_boto3_session.client.return_value = self.mock_s3_client

    @patch('matplotlib.pyplot.savefig')  # Prevent actual figure saving
    @patch('boto3.Session')
    @patch('os.environ.get')  # Mock environment variable access
    def test_perform_eda(self, mock_environ_get, mock_boto3_session, mock_savefig):
        # Configure the mocks
        mock_boto3_session.return_value = self.mock_boto3_session
        # Make os.environ.get return None for AWS credentials
        mock_environ_get.return_value = None
        
        # Test the function
        perform_eda(
            df=self.test_df,
            bucket_name='test-bucket',
            s3_prefix='test/',
            aws_region='us-east-1'
        )
        
        mock_boto3_session.assert_called()
        
        # Verify uploads occurred (at least 3 plots should be generated and uploaded)
        self.assertGreaterEqual(self.mock_s3_client.upload_fileobj.call_count, 3)

    @patch('matplotlib.pyplot.savefig')
    @patch('boto3.Session')
    def test_perform_eda_handles_missing_columns(self, mock_boto3_session, mock_savefig):
        # Configure the mock
        mock_boto3_session.return_value = self.mock_boto3_session
        
        # Create a DataFrame missing some columns
        df_missing_columns = pd.DataFrame({
            'customer_id': [1, 2, 3],
        })
        
        # Test the function - should not raise an exception
        perform_eda(
            df=df_missing_columns,
            bucket_name='test-bucket',
            s3_prefix='test/',
            aws_region='us-east-1'
        )
        
        # Should still create at least 1 plot (correlation heatmap might be empty though)
        self.assertGreaterEqual(self.mock_s3_client.upload_fileobj.call_count, 1)

if __name__ == '__main__':
    unittest.main()
