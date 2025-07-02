import logging
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
import os

log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger('food')
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("log/food.log")
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info("Food clustering module initialized")

class FOOD:
    def __init__(self, df, config_path='config.yaml'):
        """Initializes the FOOD class with config and in-memory data."""
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
                food_cfg = config['food']
                self.columns = food_cfg['columns']
                self.food_mapping = food_cfg['food_mapping']
                self.cluster_mapping = food_cfg['cluster_mapping']
                self.n_clusters = food_cfg['num_clusters']
                self.random_state = food_cfg['random_state']
                logger.info("Configuration loaded successfully.")
                self.df = df.copy()
        except Exception as error:
            logger.error("Failed to initialize FOOD class: %s", error)
            raise

    def preprocess(self):
        """Expands vendor_tag_name into dummy columns and selects predefined columns."""
        try:
            temp = self.df['vendor_tag_name'].str.get_dummies(sep=',')
            df = pd.concat([self.df.drop(columns=['vendor_tag_name']), temp], axis=1)
            self.food_df = df[self.columns].copy()
            logger.info("Vendor tag data expanded and filtered. Final shape: %s", self.food_df.shape)
        except Exception as error:
            logger.error("Error in preprocessing data: %s", error)
            raise

    def aggregate_cuisines(self):
        """Aggregates cuisine data based on the food mapping."""
        try:
            columns_to_keep = set(sum(self.food_mapping.values(), []))
            available_cols = [col for col in self.food_df.columns if col in columns_to_keep]
            missing_cols = list(columns_to_keep - set(available_cols))
            if missing_cols:
                logger.warning("Missing cuisine tag columns: %s", missing_cols)

            filtered_df = self.food_df[['customer_id'] + available_cols]
            agg = pd.DataFrame()
            agg['customer_id'] = filtered_df['customer_id']

            for category, tags in self.food_mapping.items():
                valid_tags = [tag for tag in tags if tag in filtered_df.columns]
                agg[category] = filtered_df[valid_tags].sum(axis=1) if valid_tags else 0

            self.aggregated_df = agg
            logger.info("Cuisine aggregation completed. Shape: %s", agg.shape)
        except Exception as error:
            logger.error("Error during cuisine aggregation: %s", error)
            raise

    def cluster(self):
        """Clusters the aggregated data using KMeans."""
        try:
            tfidf_transformer = TfidfTransformer()
            numerical_cols = [col for col in self.aggregated_df.columns if col != 'customer_id']
            tfidf_scaled = tfidf_transformer.fit_transform(self.aggregated_df[numerical_cols])
            tfidf_scaled_dense = pd.DataFrame(tfidf_scaled.toarray(), columns=numerical_cols)

            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            cluster_labels = kmeans.fit_predict(tfidf_scaled_dense)

            self.aggregated_df['food_cluster'] = cluster_labels
            self.aggregated_df['Segment'] = self.aggregated_df['food_cluster'].map(self.cluster_mapping)
            self.food_labeled = self.aggregated_df[['customer_id', 'Segment']]
            logger.info("KMeans clustering completed. Cluster counts: %s", dict(self.aggregated_df['Segment'].value_counts()))
        except Exception as error:
            logger.error("Error during clustering: %s", error)
            raise

    def run(self):
        """Runs the full food clustering pipeline and returns labeled results."""
        try:
            self.preprocess()
            self.aggregate_cuisines()
            self.cluster()
            return self.food_labeled
        except Exception as error:
            logger.error("Error in run(): %s", error)
            raise