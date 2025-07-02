import unittest
import pandas as pd
from unittest.mock import patch, MagicMock, ANY
import sys
sys.path.append('../')
from src.upload_s3 import upload_clustering_to_s3

class TestUploadS3(unittest.TestCase):
    def setUp(self):
        # Create a simple test DataFrame
        self.test_df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'segment': ['A', 'B', 'A'],
            'score': [0.8, 0.6, 0.9]
        })
        
        # Mock boto3 client
        self.mock_s3_client = MagicMock()
        
    @patch('boto3.client')
    @patch('os.environ.get')  # Mock environment variable access
    def test_upload_clustering_to_s3(self, mock_environ_get, mock_boto3_client):
        # Configure the mocks
        mock_boto3_client.return_value = self.mock_s3_client
        # Make os.environ.get return None for AWS credentials
        mock_environ_get.return_value = None
        
        # Test with default filename
        s3_uri = upload_clustering_to_s3(
            df=self.test_df,
            bucket_name='test-bucket',
            s3_prefix='test/',
            aws_region='us-east-1'
        )
        
        # Use ANY for parameters
        mock_boto3_client.assert_called_with(
            's3',
            region_name='us-east-1',
            aws_access_key_id=ANY,
            aws_secret_access_key=ANY
        )
        
        # Verify put_object was called
        self.mock_s3_client.put_object.assert_called_once()
        
        # Verify S3 URI format
        self.assertTrue(s3_uri.startswith('s3://test-bucket/test/clustering_'))
        self.assertTrue(s3_uri.endswith('.csv'))
        
    @patch('boto3.client')
    def test_upload_with_custom_filename(self, mock_boto3_client):
        # Configure the mock
        mock_boto3_client.return_value = self.mock_s3_client
        
        # Test with custom filename
        s3_uri = upload_clustering_to_s3(
            df=self.test_df,
            bucket_name='test-bucket',
            s3_prefix='test/',
            aws_region='us-east-1',
            filename='custom_file.csv'
        )
        
        # Verify correct S3 key
        expected_s3_uri = 's3://test-bucket/test/custom_file.csv'
        self.assertEqual(s3_uri, expected_s3_uri)
        
        # Verify put_object was called with correct key
        call_kwargs = self.mock_s3_client.put_object.call_args[1]
        self.assertEqual(call_kwargs['Bucket'], 'test-bucket')
        self.assertEqual(call_kwargs['Key'], 'test/custom_file.csv')
    
    @patch('boto3.client')
    def test_upload_error_handling(self, mock_boto3_client):
        # Configure the mock to raise an exception
        mock_boto3_client.return_value = self.mock_s3_client
        self.mock_s3_client.put_object.side_effect = Exception("Upload failed")
        
        # Test error handling
        with self.assertRaises(Exception):
            upload_clustering_to_s3(
                df=self.test_df,
                bucket_name='test-bucket',
                s3_prefix='test/',
                aws_region='us-east-1'
            )

if __name__ == '__main__':
    unittest.main()
