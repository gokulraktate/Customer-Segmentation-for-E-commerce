# Customer Segmentation for E-commerce

This project implements customer segmentation using machine learning techniques to group e-commerce customers based on their purchase behavior, demographics, and browsing activity. The goal is to offer personalized recommendations and insights for better business strategies.

## 💡 Project Description

This project is developed under the subject Data Mining and Warehousing (DMW).

Customer segmentation is essential for understanding diverse customer needs and targeting the right group with personalized marketing. In this project, we use clustering algorithms to analyze customer data and divide them into distinct segments.

Built with:
- Python
- Streamlit (for the interactive web interface)
- Scikit-learn (for clustering)
- Pandas, NumPy (for data processing)
- Matplotlib & Seaborn (for visualizations)

---

## 🧩 Features

✅ Upload CSV customer dataset  
✅ Preprocess & normalize data  
✅ Choose number of clusters  
✅ Apply K-Means clustering  
✅ Visualize clusters in 2D  
✅ View clustered data  
✅ Download results  
✅ View personalized recommendations  

---

## 📁 Project Structure

```
customer_segmentation/
│
├── ui/
│   └── app.py                  # Main Streamlit app
│
├── segmentation/
│   ├── __init__.py
│   ├── clustering.py           # KMeans clustering & visualization
│   └── distance_metrics.py     # Cosine similarity & Minkowski distance
│
├── utils/
│   ├── __init__.py
│   └── preprocess.py           # Preprocessing logic
│
├── customer_segmentation_perfect_50.csv  # Sample dataset
├── requirements.txt
└── README.md
```

---

## 📊 Sample Dataset Columns

Ensure your uploaded CSV file has the following columns:

- Age  
- Gender  
- Annual Income (k$)  
- Spending Score  
- Pages Visited  
- Time on Site  

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/gokulraktate/Customer-Segmentation-for-E-commerce.git
cd customer-segmentation
```

### 2. Set up virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate    # On Windows
source venv/bin/activate # On Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run ui/app.py
```

---

## 🧠 Techniques Used

- K-Means Clustering (unsupervised learning)
- Cosine Similarity for group recommendations
- Minkowski Distance for optional similarity analysis
- PCA for dimensionality reduction & visualization

---

## 📥 Sample Dataset

You can use the sample CSV provided (`customer_segmentation_perfect_50.csv`) to test the system.

---

## 📌 Author

👨‍💻 Krishna Shelar
👨‍💻 Gokul Raktate
👨‍💻 Prasad Parjane
👨‍💻 Rohit Wakchaure  
🎓 Computer Science Student, Sanjivani College of Engineering, Kopargaon

---

## 📜 License

This project is licensed for academic use. Feel free to fork and modify.
