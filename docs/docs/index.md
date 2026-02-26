# TalentLens documentation!

## Description
TalentLens: Job Market Intelligence using Data Science / Machine Learning

## Purpose
I wanted to create this project not only as a learning purpose but I feel like it tackles on a real-world problem that many students will encounter when looking for opportuntites on job boards. This project applies a end-to-end pipeline integrating modern tools such as RAG, vector DBs and LLMs.

## Problem Statement
Job boards have hidden patterns that are invisible to the human eye and will waste valuable time for those looking for a job.

## Research Questions
1) Are jobs actually hiring? Ghost jobs have roles posted typically to collect resumes or makes the the company look like it's growing without intent to hire.
Method: Analyze the job description, # of reposts, company size, etc.
Use: NLP Classification
2) How has the job market changed over the years. Back in the day 

## Pipeline
1) Problem Definition - How can we predict a job's posting's role (SWE, DS, AI/ML/GenAI), what skills are most in-demand per role, and how have they trended over time?
2) Data Acquistion - We are going to choose a dataset from Kaggle that has > 50000 data entries
3) Preprocessing / Data Cleaning - We will clean the raw data by putting it in a usable format. 
How will we clean the data? We will look at errors, missing values, duplicates, (nulls), inconsistencies within the dataset. 
4) Feature Engineering - We will standarize the format, encode categorical variables, normalize features and engineer now features that will represent the problem.
5) Data Modeling using ML - Once we have a good understanding of the data, we will choose one / many ML models (Supervised / Unsupervised / Reinforced) to build a predictive / descripive model. We will split our data into 3 categories: Training, Validation and Test sets, The three-way split will be 70/15/15.
6) Model Evaluation - We will use metrics such as accuracy, pricision, recall, RMSE, or AUC. Depending how the model generalizes to new data or meets success criteria.


## Commands

The Makefile contains the central entry points for common tasks related to this project.

