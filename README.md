# Restaurant Recommendation & Customer Segmentation  


## Overview
Choosing a place to eat can be overwhelming with so many options out there. Our project helps solve that by:
- Understanding customer behavior using **segmentation**
- Suggesting restaurants using a **recommendation system** built from real order history

This helps users find restaurants they'll enjoy, and helps restaurants reach the right customers.

---

## Project Features

### 📊 Customer Segmentation
We used **RFM analysis** (Recency, Frequency, Monetary) along with **Customer Lifetime Value (CLV)** to segment users based on their purchase behavior and estimated spending potential.

Segments include:
- **Super Users** — frequent and high spenders  
  *Strategy: Loyalty programs, upselling, exclusive offers*
- **Regular Users** — moderate spending and engagement  
  *Strategy: Personalized discounts to increase frequency*
- **Churn Users** — recently inactive  
  *Strategy: Targeted reactivation campaigns*
- **Lost Users** — long inactive or disengaged  
  *Strategy: Win-back promotions or surveys*

In addition, we applied clustering on food preferences (e.g., vendor tags) to identify cuisine-based segments:
- American food lovers
- Asian food fans
- Balanced eaters
- Arabic cuisine loyalists
- Breakfast-focused users

**Why CLV matters:**  
By estimating how much a user is likely to spend monthly, CLV helps us design **cost-effective, personalized campaigns**. For example, high CLV users can be prioritized for premium perks, while lower CLV segments can be targeted with cost-sensitive offers.

### 🤖 Recommendation System
We implemented several approaches:
- **Memory-Based Collaborative Filtering**  
  (User-User & Item-Item)
- **Model-Based Collaborative Filtering**  
  using Matrix Factorization (SVD, NMF, SVD++) and Deep Learning
- **Hybrid Models**  
  Weighted and Stacked (Random Forest, Neural Network)

We tackled the **cold start problem** by letting new users select their favorite food types to get started.

---

## Key Findings
- Complex models (like deep learning) don’t always beat simpler ones
- Stacked models performed best (lowest RMSE)
- Hybrid models offer better diversity in recommendations
- Vendor tags (like “Burgers” or “Salads”) are strong signals of user preferences

---

## Future Work
- Make user segmentation dynamic
- Combine segmentation directly into recommendations
- Use real-world metrics (CTR, conversion rates) beyond just RMSE
- Run A/B testing in a live app for better tuning

---

## Repository Structure
```text
├── Clustering.ipynb                # User clustering based on cuisine preferences
├── EDA.ipynb                       # Exploratory Data Analysis of the dataset
├── RFM Analysis.ipynb              # Customer segmentation using RFM and CLV analysis
├── Recommendation System.ipynb     # Collaborative filtering and hybrid recommender models
├── presentation.pdf                # Final presentation slides summarizing the project
└── README.md                       # Project overview and documentation
```

## Team Credits
Project developed by:
- Fuqian Zou
- Glenys Lion
- Iris Lee
- Liana Bergman-Turnbull

## Thank you!
This project was developed as part of a data mining course. We appreciate your interest and welcome feedback!
