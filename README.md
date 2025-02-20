# Required Files

- **Train File**:  
  [Download Train File](https://drive.usercontent.google.com/download?id=1_WqUew2CdIfAY2oPh7kOZqgtXDtLa6CN&export=download&authuser=0)
  
- **Test File**:  
  [Download Test File](https://datahack-prod.s3.amazonaws.com/test_file/test_Rj9YEaI.csv)

---

## **Windows Instructions**

### Step 1: Download the Train File
1. Click the following link to open the download page:  
   [Download Train File](https://drive.usercontent.google.com/download?id=1_WqUew2CdIfAY2oPh7kOZqgtXDtLa6CN&export=download&authuser=0)
2. The file will begin downloading automatically. If prompted, choose the location to save the file.

### Step 2: Download the Test File
1. Click the following link to download the test file:  
   [Download Test File](https://datahack-prod.s3.amazonaws.com/test_file/test_Rj9YEaI.csv)
2. The file will begin downloading automatically. If prompted, choose the location to save the file.

---

## **Mac Instructions**

### Step 1: Download the Train File
1. Click the following link to open the download page:  
   [Download Train File](https://drive.usercontent.google.com/download?id=1_WqUew2CdIfAY2oPh7kOZqgtXDtLa6CN&export=download&authuser=0)
2. The file will begin downloading automatically. If prompted, choose the location to save the file.

### Step 2: Download the Test File
1. Click the following link to download the test file:  
   [Download Test File](https://datahack-prod.s3.amazonaws.com/test_file/test_Rj9YEaI.csv)
2. The file will begin downloading automatically. If prompted, choose the location to save the file.

---

# Move downloaded files into project

## Create data folder in the project directory
1.  Create data/ folder in the project directory
2.  Unzip the train folder into your data/ folder
3.  Move the test csv into the data folder

## Your Data Folder Should Look Like This

project_directory/<br>
│── data/<br>
│   ├── image_data/<br>
│   ├── bbox_train.csv<br>
│   ├── train.csv<br>
│   ├── test.csv<br>


# Setting up the Environment and Installing Dependencies

Follow these instructions to set up a **virtual environment** and install the required libraries for the project using `pip`.

---

## **For Windows:**

### Step 1: Open Command Prompt
1. Press `Win + R`, type `cmd`, and press Enter to open Command Prompt.

### Step 2: Navigate to Your Project Directory
Use the `cd` command to navigate to the folder where your project and `requirements.txt` file are located. 

For example:
```bash
cd path\to\your\project

## **For Mac/Linux:**

### Step 1: Open Terminal
1. `Press Cmd + Space`, type `Terminal`, and press Enter to open the Terminal application.

### Step 2: Navigate to Your Project Directory
```bash
cd /path/to/your/project

# Activate your environment

### Step 1: Activate your environment
To activate your python environment do
```bash
python -m venv venv
venv\Scripts\activate

### Step 2: Install necessary packages
To install necessary packages run
```bash
pip install -r requirements.txt
