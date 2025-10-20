![Open Food Facts Logo](https://static.openfoodfacts.org/images/logos/off-logo-horizontal-light.svg)

# FoodLens: An End-to-End MLOps Demo for Nutri-Score Prediction Using AWS SageMaker Notebooks 
This project implements a production-ready, end-to-end Machine Learning Operations (MLOps) demo pipeline on AWS to predict the nutritional quality ("Nutri-Score") of food products. The system automatically trains, evaluates, deploys, and monitors an XGBoost regression model built on the [Open Food Facts dataset](https://world.openfoodfacts.org/) from a parquet download provided on [Hugging Face](https://huggingface.co/openfoodfacts). The goal is to provide a continuous nutrition score prediction, and to build a robust, automated, and observable system around that model.

This was developed as a final project for AAI-540: Machine Learning Operations at the University of San Diego.

## Project Overview

In today's fast-paced world, consumers often lack the time to read and compare complex nutrition labels. Even when they do, it can be difficult to determine which products are healthy without prior nutritional knowledge. This project aims to solve this problem by creating a machine learning model that predicts a continuous, intuitive nutrition score for food products. This score can then be categorized into a simple (A-E) grade, similar to the Nutri-Score system, empowering consumers to make healthier decisions quickly. For a retailer like Whole Foods, this solution could be integrated into their e-commerce platform and digital shelf tags. This would allow customers to filter and compare products with ease, strengthening brand loyalty and trust.

## How to Run
1.  **Set up AWS:** Ensure you have an AWS account and the necessary IAM permissions for SageMaker, S3, Step Functions, and CloudWatch.
2. **Upload Raw Dataset to S3:** Download the parquet file from [Hugging Face](https://huggingface.co/openfoodfacts) and upload to S3.
3.  **Clone the repository in SageMaker Studio:**
    ```bash
    git clone https://github.com/apmalinsky/AAI-540-FinalProject.git
    ```
4.  **Run Notebooks:** Execute the Jupyter notebooks in numerical order.
      * `01` - `04` will prepare the data and walk through pipeline components.
      * `05_ModelMonitoring.ipynb` will deploy the real-time endpoint and the monitoring schedules.
      * `06_CI-CD-Pipeline.ipynb` will walk through implementing a full pipeline demo.

## Team
  * Andy Malinsky
  * Ahmad Milad 
  * Olga Pospelova
