# Restaurant Recommendation, Customer Segmentation & Cloud Engineering  

## Overview
Choosing a place to eat can be overwhelming with so many options out there. Our project helps solve that by:
- Understanding customer behavior with **segmentation**
- Suggesting restaurants with a **recommendation system** built from real order history
- Deploying the system on AWS using a **cloud engineering pipeline**

This helps users discover restaurants they'll enjoy, while helping restaurants target the right customers. Our cloud deployment ensures scalability, reliability, and easy access for end users.

---

## Project Features

### Customer Segmentation
We used **RFM analysis** (Recency, Frequency, Monetary) along with **Customer Lifetime Value (CLV)** to segment users based on purchase behavior and estimated spending potential.

Segments include:
- **Super Users** — frequent and high spenders  
  *Strategy: Loyalty programs, upselling, exclusive offers*
- **Regular Users** — moderate spending and engagement  
  *Strategy: Personalized discounts to increase frequency*
- **Churn Users** — recently inactive  
  *Strategy: Targeted reactivation campaigns*
- **Lost Users** — long inactive or disengaged  
  *Strategy: Win-back promotions or surveys*

We also applied clustering on food preferences (e.g., vendor tags) to identify cuisine-based segments:
- American food lovers
- Asian food fans
- Balanced eaters
- Arabic cuisine loyalists
- Breakfast-focused users

**Why CLV matters:**  
By estimating monthly spend potential, CLV supports **cost-effective, personalized marketing**. High CLV users can be prioritized for premium perks, while lower CLV segments receive cost-sensitive offers.

---

### Recommendation System
We implemented:
- **Memory-Based Collaborative Filtering**  
  (User-User & Item-Item)
- **Model-Based Collaborative Filtering**  
  using Matrix Factorization (SVD, NMF, SVD++) and Deep Learning
- **Hybrid Models**  
  Weighted and Stacked (Random Forest, Neural Network)

We addressed the **cold start problem** by letting new users select favorite cuisines when they join.

---

### Cloud Engineering on AWS
We engineered and deployed the full pipeline on AWS to demonstrate real-world readiness. This part of the project includes:

- **ETL Pipelines**  
  Using AWS services to ingest and preprocess data
- **Backend**  
  APIs built to serve recommendations to clients
- **Frontend**  
  Simple user interface to showcase recommendations
- **Deployment**  
  - Containerized microservices with Docker
  - Orchestrated and deployed on AWS ECS
  - Storage in AWS RDS and S3
  - Logging and monitoring via CloudWatch

This design ensures **scalability, maintainability, and ease of integration** with production systems.

---

## Key Findings
- Complex models (like deep learning) don’t always beat simpler ones
- Stacked hybrid models had the best RMSE
- Hybrid approaches improved diversity in recommendations
- Vendor tags were highly predictive of user preferences

---

## Future Work
- Make user segmentation dynamic and real-time
- Integrate segmentation into recommendations directly
- Evaluate with live business metrics (CTR, conversions)
- Perform A/B testing in a production environment

---

## Repository Structure
```text
├── Clustering.ipynb                      # User clustering based on cuisine preferences
├── EDA.ipynb                             # Exploratory Data Analysis of the dataset
├── RFM Analysis.ipynb                    # Customer segmentation with RFM and CLV
├── Recommendation System.ipynb           # Collaborative filtering and hybrid recommender models
├── presentation.pdf                      # Data mining presentation
├── Cloud Engineering/                    # AWS deployment part
│   ├── ETL/                              # Data ingestion and preprocessing
│   ├── backend/                          # Backend APIs
│   ├── frontend/                         # Frontend app
│   └── Project Presentation.pdf          # Cloud Engineering presentation
└── README.md                             # Project overview and documentation
```

## Team Credits
Project developed by:
- Fuqian Zou
- Glenys Lion
- Iris Lee
- Liana Bergman-Turnbull

## Thank you!
This project was developed as part of a data mining course. We appreciate your interest and welcome feedback!
