import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
sys.path.append('../')
from src.acquire_data import acquire_data_rds

class TestAcquireData(unittest.TestCase):
    def setUp(self):
        # Mock config
        self.test_config = {
            "database": {
                "host": "test-host",
                "port": 5432,
                "name": "test-db",
                "user": "test-user",
                "password": "test-password"
            }
        }
        
        # Mock DataFrame
        self.test_df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'vendor_id': [101, 102, 103],
            'grand_total': [50.0, 30.0, 25.0]
        })
    
    @patch('src.acquire_data.create_engine')  # Patch the exact import path
    @patch('pandas.read_sql')
    def test_acquire_data_rds(self, mock_read_sql, mock_create_engine):
        # Setup mocks
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Create a mock DataFrame with a mock shape property
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_df.shape = (3, 3)  # This works on a MagicMock but not on a real DataFrame
        mock_read_sql.return_value = mock_df
        
        # Test the function
        result = acquire_data_rds(self.test_config)
        
        # Verify create_engine was called with correct URL format
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args[0][0]
        self.assertTrue('postgresql://' in call_args)
        
        # Verify read_sql was called
        mock_read_sql.assert_called_with('SELECT * FROM order_clean_join_all', mock_engine)
        
        # Verify the result
        self.assertEqual(result, mock_df)
    
    @patch('pandas.read_sql')
    @patch('sqlalchemy.create_engine')
    def test_acquire_data_error_handling(self, mock_create_engine, mock_read_sql):
        # Setup mocks to raise an exception
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        # Make read_sql raise the exception instead of create_engine
        mock_read_sql.side_effect = Exception("Connection error")
        
        # Test the function with exception
        with self.assertRaises(Exception):
            acquire_data_rds(self.test_config)

if __name__ == '__main__':
    unittest.main()
