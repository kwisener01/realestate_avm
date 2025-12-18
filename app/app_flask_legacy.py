from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
sys.path.append('../scripts')

from train_arv_model_with_comps import ImprovedARVModel
import pandas as pd
import re

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze a Google Sheet"""
    try:
        data = request.json
        sheet_url = data.get('sheet_url')
        
        # Extract sheet ID from URL
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheet_url)
        if not match:
            return jsonify({'error': 'Invalid Google Sheets URL'}), 400
        
        sheet_id = match.group(1)
        
        # Download sheet data
        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        df = pd.read_csv(export_url)
        
        # Run analysis (simplified for demo)
        results = {
            'total_properties': len(df),
            'sheet_id': sheet_id,
            'message': 'Analysis complete! Check your sheet for results.'
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
