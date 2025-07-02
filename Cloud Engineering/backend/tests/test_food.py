import unittest
import pandas as pd
import numpy as np
import os
import yaml
from unittest.mock import patch, MagicMock, mock_open
import sys
sys.path.append('../')  # Add parent directory to path
from src.food import FOOD

class TestFOOD(unittest.TestCase):
    def setUp(self):
        # Create a mock config
        self.mock_config = {
            "food": {
                "columns": ["customer_id", "Italian", "Chinese", "Mexican"],
                "food_mapping": {
                    "Italian": ["Italian", "Pasta", "Pizza"],
                    "Asian": ["Chinese"],
                    "Mexican": ["Mexican"]
                },
                "num_clusters": 2,
                "random_state": 42,
                "cluster_mapping": {
                    0: "Western",
                    1: "Asian"
                }
            }
        }
        
        # Mock yaml.safe_load to return our test config
        self.yaml_patch = patch('yaml.safe_load', return_value=self.mock_config)
        
        # Create a simple test DataFrame
        self.test_df = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'vendor_tag_name': [
                'Italian,Pizza', 
                'Chinese', 
                'Mexican,Italian', 
                'Pizza,Pasta', 
                'Chinese,Mexican'
            ]
        })

    def test_init(self):
        # Test initialization with mock config
        with patch('builtins.open', mock_open()):
            with self.yaml_patch:
                food = FOOD(df=self.test_df, config_path="mock_config.yaml")
                self.assertEqual(food.columns, ["customer_id", "Italian", "Chinese", "Mexican"])
                self.assertEqual(food.n_clusters, 2)
                self.assertEqual(food.cluster_mapping, {0: "Western", 1: "Asian"})
                self.assertIsNotNone(food.df)
    
    def test_preprocess(self):
        # Test preprocessing
        with patch('builtins.open', mock_open()):
            with self.yaml_patch:
                food = FOOD(df=self.test_df, config_path="mock_config.yaml")
                food.preprocess()
                
                # Check if food_df was created
                self.assertIsNotNone(food.food_df)
                
                # Check expanded columns
                for col in ["customer_id", "Italian", "Chinese", "Mexican"]:
                    self.assertIn(col, food.food_df.columns)
    
    def test_aggregate_cuisines(self):
        # Test cuisine aggregation
        with patch('builtins.open', mock_open()):
            with self.yaml_patch:
                food = FOOD(df=self.test_df, config_path="mock_config.yaml")
                food.preprocess()
                food.aggregate_cuisines()
                
                # Check if aggregated_df was created
                self.assertIsNotNone(food.aggregated_df)
                
                # Check if aggregated columns exist
                self.assertIn('customer_id', food.aggregated_df.columns)
                self.assertIn('Italian', food.aggregated_df.columns)
                self.assertIn('Asian', food.aggregated_df.columns)
                self.assertIn('Mexican', food.aggregated_df.columns)
    
    def test_cluster(self):
        # Test clustering
        with patch('builtins.open', mock_open()):
            with self.yaml_patch:
                food = FOOD(df=self.test_df, config_path="mock_config.yaml")
                food.preprocess()
                food.aggregate_cuisines()
                food.cluster()
                
                # Check if food_labeled was created
                self.assertIsNotNone(food.food_labeled)
                
                # Check if it has the expected columns
                self.assertIn('customer_id', food.food_labeled.columns)
                self.assertIn('Segment', food.food_labeled.columns)
    
    def test_run(self):
        # Test the full pipeline
        with patch('builtins.open', mock_open()):
            with self.yaml_patch:
                food = FOOD(df=self.test_df, config_path="mock_config.yaml")
                result = food.run()
                
                # Check if result is a DataFrame
                self.assertIsInstance(result, pd.DataFrame)
                
                # Check if it has expected columns
                self.assertIn('customer_id', result.columns)
                self.assertIn('Segment', result.columns)
                
                # Check if all customers are present
                self.assertEqual(len(result), 5)

if __name__ == '__main__':
    unittest.main()
