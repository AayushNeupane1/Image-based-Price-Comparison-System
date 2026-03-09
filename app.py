from flask import Flask, request, jsonify, render_template
import pytesseract
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import pickle
from fuzzywuzzy import fuzz
import os
import warnings
warnings.filterwarnings('ignore')

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Load model, vectorizer, dataset ──
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

df = pd.read_csv('products_all_fixed.csv')

# ── Functions ──
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
    return thresh

def extract_text(processed_img):
    pil_img = Image.fromarray(processed_img)
    config = '--oem 3 --psm 6'
    text = pytesseract.image_to_string(pil_img, config=config)
    text = text.lower().strip()
    text = ''.join(c if c.isalnum() or c == ' ' else ' ' for c in text)
    text = ' '.join(text.split())
    return text

# def predict_category(text):
#     text_vec = vectorizer.transform([text])
#     category = model.predict(text_vec)[0]
#     confidence = round(model.predict_proba(text_vec).max() * 100, 1)
#     return category, confidence
# Brand to category mapping
BRAND_CATEGORY = {
    # Personal Care keywords
    'silver':     'Personal Care',
    'shield':     'Personal Care',
    'germ':       'Personal Care',
    'protection': 'Personal Care',
    'stronger':   'Personal Care',
    'soap':       'Personal Care',
    'bodywash':   'Personal Care',
    'lotion':     'Personal Care',
    'haircare':   'Personal Care',

    # Food keywords  
    'noodles':    'Food & Beverages',
    'instant':    'Food & Beverages',
    'coffee':     'Food & Beverages',
    'biscuit':    'Food & Beverages',
    'chocolate':  'Food & Beverages',
    'juice':      'Food & Beverages',
    'cereal':     'Food & Beverages',

    # Household keywords
    'toilet':     'Household',
    'floor':      'Household',
    'bleach':     'Household',
    'mosquito':   'Household',
    'garbage':    'Household',
    'tissue':     'Household',

    'garnier':    'Cosmetics',
    'maybelline': 'Cosmetics',
    'loreal':     'Cosmetics',
    'lakme':      'Cosmetics',
    'nivea':      'Cosmetics',
    'neutrogena': 'Cosmetics',
    'olay':       'Cosmetics',
    'dove':       'Personal Care',
    'dettol':     'Personal Care',
    'colgate':    'Personal Care',
    'lifebuoy':   'Personal Care',
    'pantene':    'Personal Care',
    'gillette':   'Personal Care',
    'vaseline':   'Personal Care',
    'himalaya':   'Personal Care',
    'nestle':     'Food & Beverages',
    'nescafe':    'Food & Beverages',
    'maggi':      'Food & Beverages',
    'lays':       'Food & Beverages',
    'pepsi':      'Food & Beverages',
    'coca':       'Food & Beverages',
    'kelloggs':   'Food & Beverages',
    'ariel':      'Household',
    'surf':       'Household',
    'vim':        'Household',
    'harpic':     'Household',
    'febreze':    'Household',
    'gatorade':   'Sports',
    'pocari':     'Sports',
    'powerade':   'Sports',
    'myprotein':  'Sports',
    'nike':       'Sports',
    'adidas':     'Sports',
    'micellar':   'Cosmetics',
    'cleansing':  'Cosmetics',
    'foundation': 'Cosmetics',
    'lipstick':   'Cosmetics',
    'mascara':    'Cosmetics',
    'moisturizer':'Cosmetics',
    'shampoo':    'Personal Care',
    'toothpaste': 'Personal Care',
    'deodorant':  'Personal Care',
    'bodywash':   'Personal Care',
    'detergent':  'Household',
    'dishwash':   'Household',
    'noodles':    'Food & Beverages',
    'biscuit':    'Food & Beverages',
    'chocolate':  'Food & Beverages',
    'protein':    'Sports',
    'creatine':   'Sports',
}

def predict_category(text):
    # Check known brands AND product keywords in OCR text
    words = text.lower().split()
    for word in words:
        if word in BRAND_CATEGORY:
            return BRAND_CATEGORY[word], 99.0

    # Also check partial matches for brands
    for brand, category in BRAND_CATEGORY.items():
        if brand in text.lower():
            return category, 95.0

    # Fallback to ML model
    short_text = ' '.join(text.split()[:10])
    text_vec = vectorizer.transform([short_text])
    category = model.predict(text_vec)[0]
    confidence = round(model.predict_proba(text_vec).max() * 100, 1)
    return category, confidence

# def match_product(text, category):
#     df_filtered = df[df['category'] == category]
#     scores = []
#     for product in df_filtered['product_name'].unique():
#         score = fuzz.partial_ratio(text.lower(), product.lower())
#         scores.append((product, score))
#     scores = sorted(scores, key=lambda x: x[1], reverse=True)
#     best_match, best_score = scores[0]
#     return best_match, best_score

# def match_product(text, category):
#     # First try within predicted category
#     df_filtered = df[df['category'] == category]
#     scores = []
#     for product in df_filtered['product_name'].unique():
#         score = fuzz.partial_ratio(text.lower(), product.lower())
#         scores.append((product, score))
#     scores = sorted(scores, key=lambda x: x[1], reverse=True)
#     best_match, best_score = scores[0]

#     # If score is too low, search entire dataset
#     if best_score < 60:
#         scores_all = []
#         for product in df['product_name'].unique():
#             score = fuzz.partial_ratio(text.lower(), product.lower())
#             scores_all.append((product, score))
#         scores_all = sorted(scores_all, key=lambda x: x[1], reverse=True)
#         best_match, best_score = scores_all[0]

#     return best_match, best_score
def match_product(text, category):
    # Try within predicted category first
    df_filtered = df[df['category'] == category]
    scores = []
    for product in df_filtered['product_name'].unique():
        score = fuzz.token_set_ratio(text.lower(), product.lower())
        scores.append((product, score))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    best_match, best_score = scores[0]

    # If score too low — search entire dataset
    if best_score < 70:
        scores_all = []
        for product in df['product_name'].unique():
            score = fuzz.token_set_ratio(text.lower(), product.lower())
            scores_all.append((product, score))
        scores_all = sorted(scores_all, key=lambda x: x[1], reverse=True)
        best_match, best_score = scores_all[0]

    return best_match, best_score
# def compare_prices(product_name):
#     product_df = df[df['product_name'] == product_name].copy()
#     product_df = product_df.sort_values('price_NPR')
#     results = []
#     for _, row in product_df.iterrows():
#         results.append({
#             'seller':    row['seller'],
#             'price_NPR': round(row['price_NPR'], 2),
#         })
#     best_price  = product_df['price_NPR'].min()
#     worst_price = product_df['price_NPR'].max()
#     you_save    = round(worst_price - best_price, 2)
#     return results, you_save
def compare_prices(product_name):
    product_df = df[df['product_name'] == product_name].copy()

    # Keep only lowest price per seller
    product_df = product_df.groupby('seller', as_index=False)['price_NPR'].min()

    product_df = product_df.sort_values('price_NPR')
    results = []
    for _, row in product_df.iterrows():
        results.append({
            'seller':    row['seller'],
            'price_NPR': round(row['price_NPR'], 2),
        })
    best_price  = product_df['price_NPR'].min()
    worst_price = product_df['price_NPR'].max()
    you_save    = round(worst_price - best_price, 2)
    return results, you_save

# ── Flask App ──
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['image']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        processed = preprocess_image(img)
        ocr_text = extract_text(processed)

        if not ocr_text:
            return jsonify({'error': 'Could not extract text from image.'}), 400

        category, confidence = predict_category(ocr_text)
        matched_product, match_score = match_product(ocr_text, category)
        prices, you_save = compare_prices(matched_product)

        return jsonify({
            'ocr_text':        ocr_text,
            'category':        category,
            'confidence':      confidence,
            'matched_product': matched_product,
            'match_score':     match_score,
            'prices':          prices,
            'you_save':        you_save,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)