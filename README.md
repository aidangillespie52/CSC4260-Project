# 📂 Required Files

Install the following required files and store them in the **data/** folder.

---

## 📥 Download Files

### **Train File**  
[📥 Download Train File](https://drive.usercontent.google.com/download?id=1_WqUew2CdIfAY2oPh7kOZqgtXDtLa6CN&export=download&authuser=0)

### **Test File**  
[📥 Download Test File](https://datahack-prod.s3.amazonaws.com/test_file/test_Rj9YEaI.csv)

---

# 💻 Windows Instructions

## **Step 1: Download the Train File**
1. Click the following link:  
   [📥 Download Train File](https://drive.usercontent.google.com/download?id=1_WqUew2CdIfAY2oPh7kOZqgtXDtLa6CN&export=download&authuser=0)
2. The file will begin downloading automatically. If prompted, choose a location to save it.

## **Step 2: Download the Test File**
1. Click the following link:  
   [📥 Download Test File](https://datahack-prod.s3.amazonaws.com/test_file/test_Rj9YEaI.csv)
2. The file will begin downloading automatically. If prompted, choose a location to save it.

---

# 🍏 Mac/Linux Instructions

## **Step 1: Download the Train File**
1. Click the following link:  
   [📥 Download Train File](https://drive.usercontent.google.com/download?id=1_WqUew2CdIfAY2oPh7kOZqgtXDtLa6CN&export=download&authuser=0)
2. The file will begin downloading automatically. If prompted, choose a location to save it.

## **Step 2: Download the Test File**
1. Click the following link:  
   [📥 Download Test File](https://datahack-prod.s3.amazonaws.com/test_file/test_Rj9YEaI.csv)
2. The file will begin downloading automatically. If prompted, choose a location to save it.

---

# 📁 Move Downloaded Files into the Project

## **Step 1: Create the `data/` Folder**
1. Create a `data/` folder inside your project directory.
2. Unzip the **train** folder into the `data/` folder.
3. Move the **test CSV** into the `data/` folder.

---

## ✅ Your Data Folder Should Look Like This:

```plaintext
project_directory/
│── data/
│   ├── image_data/
│   ├── bbox_train.csv
│   ├── train.csv
│   ├── test.csv
```

# ⚙️ Setting up the Environment and Installing Dependencies

Follow these instructions to set up a **virtual environment** and install the required libraries for the project using `pip`.

---

## 🖥️ For Windows:

### **Step 1: Open Command Prompt**
1. Press `Win + R`, type `cmd`, and press **Enter** to open Command Prompt.

### **Step 2: Navigate to Your Project Directory**
Use the `cd` command to navigate to the folder where your project and `requirements.txt` file are located.

For example:
```bash
cd path\to\your\project
```

## 🍏 Mac/Linux Instructions

### **Step 1: Open Terminal**
1. `Press Cmd + Space`, type `Terminal`, and press Enter to open the Terminal application.

### **Step 2: Navigate to Your Project Directory**
```bash
cd /path/to/your/project
```

# 🔧 Environment Set Up

### **Step 1: Activate Your Environment**
To activate your Python environment, run:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux


### Step 2: Install necessary packages
To install necessary packages run
```bash
pip install -r requirements.txt
```