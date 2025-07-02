from sqlalchemy import create_engine
import pandas as pd

def acquire_data_rds(config):
    """
    Connects to PostgreSQL RDS using credentials from config.
    Loads data from the 'order_clean_join_all' table and returns as DataFrame.

    Args:
        config (dict): Configuration dictionary with DB credentials.

    Returns:
        pd.DataFrame: Data from the 'order_clean_join_all' table.
    """
    # Pull credentials from config
    db_config = config["database"]

    host = db_config["host"]
    port = db_config["port"]
    dbname = db_config["name"]
    username = db_config["user"]
    password = db_config["password"]

    # Build SQLAlchemy connection string
    db_url = f'postgresql://{username}:{password}@{host}:{port}/{dbname}'
    engine = create_engine(db_url)

    # Run SQL query
    query = 'SELECT * FROM order_clean_join_all'
    df = pd.read_sql(query, engine)

    return df
