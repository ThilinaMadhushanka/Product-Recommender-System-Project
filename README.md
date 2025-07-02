# 🛍️ Multi-Category Product Recommendation System

A modern, interactive web application for discovering personalized product recommendations across multiple categories, powered by real datasets and a Flask backend.

---

## 🚀 Features
- **Personalized Recommendations**: Get top-rated, price-based, popular, or random product suggestions.
- **Multi-Category Support**: E-Commerce, Electronics, Mobile, PC, Vehicles, Clothing, Food, Toys, and more.
- **Live Currency Conversion**: Product prices are shown in Sri Lankan Rupees (LKR), auto-converted from USD using a real-time exchange rate API.
- **Product Details Modal**: Click any product to view full details and a direct "Buy Now" link to popular e-commerce sites.
- **Responsive UI**: Beautiful, mobile-friendly design with category tabs and intuitive controls.

---

## 🖼️ Screenshots
> _Add your screenshots here!_

| Home Page | Product Modal |
|-----------|--------------|
|![22](https://github.com/user-attachments/assets/7f6192e4-795f-4199-9e03-5d452ffff10f) |  ![23](https://github.com/user-attachments/assets/587b00d3-c467-4e26-aefa-cd987cbdda46) |

---

## ⚙️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. **Install dependencies:**
   ```bash
   pip install -r recommendation_system/requirements.txt
   ```
3. **Add your dataset:**
   - Place your `products_1000.csv` (or similar) in `recommendation_system/Data/`.
   - Ensure it has columns: `name`, `price`, `rating`, `category`, `brand`, `details`/`specs`/`description`.
4. **Run the Flask app:**
   ```bash
   cd recommendation_system
   python app.py
   ```
5. **Open your browser:**
   - Go to [http://localhost:5001](http://localhost:5001)

---

## 📝 Usage Guide
- **Select a category** using the tabs at the top.
- **Adjust filters** (price range, number of results, recommendation type).
- **Click "Get Recommendations"** to view personalized suggestions.
- **Click any product card** to see full details and a "Buy Now" link.

---

## 🔌 API Endpoints
- `GET /` — Main web interface
- `GET /api/products?category=<name>&limit=<n>` — Get products for a category
- `GET /api/info` — Backend health/info check

---

## 📊 Dataset Info
- All products are loaded from a single CSV (e.g., `products_1000.csv`) in the `Data` folder.
- The `category` column is used for filtering. Add new categories by updating your CSV and the frontend mapping.

---

## 🎨 Customization
- **Add new categories:** Update the category tabs and `categoryMap` in `templates/index.html`.
- **Change currency:** Edit the exchange rate logic in the frontend JS.
- **Style/UI tweaks:** Modify the CSS in `index.html` for a unique look.

---

## 🙏 Credits
- Built with [Flask](https://flask.palletsprojects.com/), [Pandas](https://pandas.pydata.org/), and [scikit-learn](https://scikit-learn.org/).
- Datasets from [Kaggle](https://www.kaggle.com/) and other open sources.
- UI inspired by modern e-commerce platforms.

---

> _Made with ❤️ for the AI & Data Science community._
