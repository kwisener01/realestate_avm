# Real Estate AVM (Automated Valuation Model)

AI-powered tool for analyzing real estate investment opportunities. Estimates After Repair Value (ARV) using machine learning trained on actual sales data.

## Features

- **Dual ARV Models**: Location-based and ML-based estimates
- **Confidence Scoring**: Identifies high-confidence deals where both models agree
- **Google Sheets Integration**: Automatic updates to your property sheets
- **Training on Real Sales**: Uses actual comparable sales data
- **Deal Analysis**: Automatically flags properties meeting the 50% rule

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/realestate-avm.git
cd realestate-avm

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up Google Sheets API credentials
# Follow instructions in SETUP_API_CREDENTIALS.md

# 5. Run the analysis
python scripts/train_arv_model_with_comps.py
```

## Project Structure

```
realestate_avm/
├── app/                    # Web application (Bolt integration)
├── data/                   # Data directory (gitignored)
├── models/                 # Trained ML models
├── scripts/               # Analysis scripts
│   ├── arv_model.py       # Location-based ARV model
│   ├── train_arv_model_with_comps.py  # ML model with comps
│   ├── create_hybrid_model.py         # Hybrid analysis
│   └── upload_hybrid_to_sheets.py     # Google Sheets upload
├── README.md
├── requirements.txt
└── SETUP_API_CREDENTIALS.md
```

## How It Works

### 1. Location-Based Model
- Uses distance from city center, days on market, and area premiums
- Multipliers range from 1.5x to 2.3x
- Fast and doesn't require training data

### 2. ML Model
- Trained on actual sales comps
- Features: Square footage, beds, baths, assessed value, location
- Uses Gradient Boosting Regressor
- More accurate but requires training data

### 3. Hybrid Approach
- Shows both estimates side-by-side
- Flags high-confidence deals where both models agree
- Provides three ARV values: Location, ML, and Average

## Results

From 487 Atlanta metro properties:
- **20 HIGH confidence deals** (both models agree)
- **181 MEDIUM confidence** (one model identifies as deal)
- Average potential profit on high-confidence deals: $260k-$350k

## Google Sheets Output

Adds 5 columns to your sheet:
1. **Deal Status** - Which model(s) identified it as a deal
2. **ARV (Location)** - Market-based estimate
3. **ARV (ML Model)** - Machine learning estimate
4. **ARV (Average)** - Average of both
5. **Confidence** - Agreement level (HIGH/MEDIUM/LOW)

## API Setup

See [SETUP_API_CREDENTIALS.md](SETUP_API_CREDENTIALS.md) for detailed instructions on:
- Creating Google Cloud project
- Enabling APIs
- Downloading credentials
- Sharing your sheet with the service account

## Usage

### Analyze a Google Sheet

```python
from scripts.analyze_sheet import analyze_google_sheet

# Analyze and update sheet
results = analyze_google_sheet(
    sheet_id='YOUR_SHEET_ID',
    training_data='path/to/comps.csv'
)

print(f"Found {results['high_confidence_deals']} high-confidence deals")
```

### Train Custom Model

```python
from scripts.train_arv_model_with_comps import ImprovedARVModel

model = ImprovedARVModel()
X, y, df = model.prepare_training_data(your_sales_data)
model.train_model(X, y)
predictions = model.predict_arv(target_properties)
```

## Web Application

Coming soon: Web interface for easy analysis without coding!

1. Paste Google Sheet URL
2. Upload training data (optional)
3. Get instant ARV analysis
4. Download results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file

## Credits

Built with Claude Code
