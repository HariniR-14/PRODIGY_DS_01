import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
train = pd.read_csv('C:/Users/harin/Downloads/train.csv')


# Display the first few rows of the dataset
print(train.head())

# Basic information about the dataset
print(train.info())

# Summary statistics of the dataset
print(train.describe())

# Check for missing values
print(train.isnull().sum())

# Data Cleaning

# Fill missing Age values with the median age
train['Age'].fillna(train['Age'].median(), inplace=True)

# Fill missing Embarked values with the most common port
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# Drop the Cabin column as it has too many missing values
train.drop(columns=['Cabin'], inplace=True)

# Verify that there are no more missing values
print(train.isnull().sum())

# Convert Categorical Variables

# Convert 'Sex' to numerical values: 0 for male, 1 for female
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' to numerical values: 0 for S, 1 for C, 2 for Q
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Verify the changes
print(train.head())

# Exploratory Data Analysis (EDA) and Visualization

# 1. Univariate Analysis

# Histogram of Age distribution
plt.figure(figsize=(8, 6))
sns.histplot(train['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Histogram of Fare distribution
plt.figure(figsize=(8, 6))
sns.histplot(train['Fare'], bins=30, kde=True)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

# Countplot of Survival
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=train)
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# 2. Bivariate Analysis

# Survival Rate by Gender
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=train)
plt.title('Survival Rate by Gender')
plt.xlabel('Sex (0 = Male, 1 = Female)')
plt.ylabel('Survival Rate')
plt.show()

# Survival Rate by Passenger Class
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survived', data=train)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Age vs. Fare, colored by Survival
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=train)
plt.title('Age vs Fare, Colored by Survival')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# 3. Multivariate Analysis

# Pairplot to show relationships between several variables
sns.pairplot(train[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']], hue='Survived', height=2.5)
plt.suptitle('Pairplot of Various Features', y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = train.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
