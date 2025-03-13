# Lung Cancer Prediction Web App

## Project Overview
This project is a **Lung Cancer Prediction Web Application** designed to take user input via a web interface and predict lung cancer likelihood along with a possible associated biomarker. The application utilizes a trained machine learning model and is deployed using Flask.

## Contributors
This project was developed by **Babaloloa Toluwanyemi** and **Priscillia Adindeh** from the **Department of Biomedical Engineering Federal  University of Technology, Owerri, Nigeria**.

**Ephraim Agumbada** contributed by assisting in:
- Fine-tuning the model.
- Ensuring data consistency for input labels.
- Connecting the trained model to a Flask web application.
- Implementing a seamless user input system via the web app.

### GitHub Repository
The core model training and extensive data analysis are conducted in the main project repository:
👉 **[Lung Cancer Model - Main Repository](https://github.com/babalolatoluwayemi/lung_cancer_mode/tree/main-branch)**

Toluwanyemi and Priscillia retain **full rights** to the project and its extensions.

---

## Features
✅ Accepts user input through an interactive web form.  
✅ Uses a pre-trained **Random Forest Classifier** for lung cancer prediction.  
✅ Predicts the **most likely biomarker** associated with lung cancer cases.  
✅ Flask-based backend for handling model inference.  
✅ **Asynchronous form submission** with a loading indicator for better user experience.  
✅ Deployed locally and can be extended to cloud platforms.  

---

## Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/babalolatoluwayemi/lung_cancer_mode.git
cd lung_cancer_mode
```

### 2. Install Dependencies
Ensure you have Python 3 installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Flask App
```bash
python app.py
```
The app will be accessible at **http://127.0.0.1:5000/**.

---

## File Structure
```
├── app.py  # Flask backend
├── templates/
│   ├── index.html  # Frontend UI
├── static/
│   ├── styles.css  # Styling
├── models/
│   ├── cancer_model.joblib  # Trained lung cancer model
│   ├── biomarker_model.joblib  # Biomarker prediction model
│   ├── scaler.joblib  # Feature scaler
│   ├── label_encoder.joblib  # Encodes biomarker labels
├── requirements.txt  # Required Python libraries
└── README.md  # Project Documentation
```

---

## Usage
1. **Open the Web App**: Run the Flask server and open the app in your browser.
2. **Enter Your Health Data**: Fill out the form with the required inputs.
3. **Submit the Form**: Click the submit button to get the prediction.
4. **Receive Results**: The app displays whether lung cancer is detected and, if so, the likely associated biomarker.

---

## Biomarker Mapping
If lung cancer is detected, the app predicts one of the **five most common lung cancer biomarkers in Africa**:

| Biomarker ID | Biomarker Name |
|-------------|----------------|
| 1           | EGFR           |
| 2           | ALK            |
| 3           | KRAS           |
| 4           | BRAF           |
| 5           | ROS1           |

If no cancer is detected, the biomarker result will be **"N/A"**.

---

## Future Enhancements
- 🔹 Deploying the web app to a cloud-based service (AWS, Heroku, etc.).
- 🔹 Expanding the dataset for improved model accuracy.
- 🔹 Adding more advanced feature selection techniques.

---

## License
This project is fully owned by **Babaloloa Toluwanyemi** and **Priscillia Adindeh**. It is shared under their discretion, and all rights are reserved.

**Contributor:** Ephraim Agumbada

---
### 🚀 For more details, visit the [GitHub Repository](https://github.com/babalolatoluwayemi/lung_cancer_mode/tree/main-branch)

