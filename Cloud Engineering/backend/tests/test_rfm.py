import unittest
import pandas as pd
from datetime import datetime, date, timedelta
import os
import yaml
from unittest.mock import patch, MagicMock, mock_open
import sys
sys.path.append('../')
from src.rfm import RFM

class TestRFM(unittest.TestCase):
    def setUp(self):
        # Create a mock config
        self.mock_config = {
            "rfm": {
                "snapshot_date": "2023-12-31",
                "n_clusters": 2,
                "random_state": 42,
                "cluster_mapping": {
                    0: "regular_user",
                    1: "super_user"
                }
            }
        }
        
        # Mock yaml.safe_load to return our test config
        self.yaml_patch = patch('yaml.safe_load', return_value=self.mock_config)
        
        # Create a simple test DataFrame
        today = date(2023, 12, 31)
        self.test_df = pd.DataFrame({
            'customer_id': [1, 1, 2, 2, 3],
            'created_at': [
                today - timedelta(days=5),
                today - timedelta(days=10),
                today - timedelta(days=20),
                today - timedelta(days=25),
                today - timedelta(days=100)
            ],
            'akeed_order_id': [101, 102, 103, 104, 105],
            'grand_total': [50.0, 30.0, 25.0, 40.0, 60.0]
        })
        
        # Convert dates to strings for proper testing
        self.test_df['created_at'] = self.test_df['created_at'].astype(str)

    def test_init(self):
        # Test initialization with mock config
        with patch('builtins.open', mock_open()):
            with self.yaml_patch:
                rfm = RFM(df=self.test_df, config_path="mock_config.yaml")
                self.assertEqual(rfm.snapshot_date, date(2023, 12, 31))
                self.assertEqual(rfm.cluster_mapping, {0: "regular_user", 1: "super_user"})
                self.assertIsNotNone(rfm.df)
    
    def test_preprocess(self):
        # Test preprocessing
        with patch('builtins.open', mock_open()):
            with self.yaml_patch:
                rfm = RFM(df=self.test_df, config_path="mock_config.yaml")
                rfm.preprocess()
                
                # Check if rfm_df was created
                self.assertIsNotNone(rfm.rfm_df)
                
                # Check if it has the right columns
                self.assertIn('Recency', rfm.rfm_df.columns)
                self.assertIn('Frequency', rfm.rfm_df.columns)
                self.assertIn('Monetary', rfm.rfm_df.columns)
                
                # Check if customer aggregation is correct
                self.assertEqual(len(rfm.rfm_df), 3)  # 3 unique customers
    
    def test_train_model(self):
        # Test model training
        with patch('builtins.open', mock_open()):
            with self.yaml_patch:
                rfm = RFM(df=self.test_df, config_path="mock_config.yaml")
                rfm.preprocess()
                rfm._train_model()
                
                # Check if rfm_labeled was created
                self.assertIsNotNone(rfm.rfm_labeled)
                
                # Check if cluster columns exist
                self.assertIn('K-Means', rfm.rfm_labeled.columns)
                self.assertIn('Segment', rfm.rfm_labeled.columns)
    
    def test_run(self):
        # Test the full pipeline
        with patch('builtins.open', mock_open()):
            with self.yaml_patch:
                rfm = RFM(df=self.test_df, config_path="mock_config.yaml")
                result = rfm.run()
                
                # Check if result is a DataFrame
                self.assertIsInstance(result, pd.DataFrame)
                
                # Check if it has expected columns
                self.assertIn('customer_id', result.columns)
                self.assertIn('Segment', result.columns)
                self.assertIn('CLV_30', result.columns)

if __name__ == '__main__':
    unittest.main()
