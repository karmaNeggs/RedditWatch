#!/usr/bin/env python3
"""
Reddit Bot Analysis - Admin Dashboard Backend
Password-protected admin interface for manual data collection and analysis
Run: python3 app.py
Access: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import json
import os
import subprocess
import hashlib
import secrets
from datetime import datetime
from functools import wraps
import glob

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)

# Configuration
ADMIN_PASSWORD_HASH = hashlib.sha256('admin123'.encode()).hexdigest()  # Change this!
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
SCRIPTS_DIR = 'scripts'

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'authenticated' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Serve the admin dashboard"""
    return render_template('admin.html')

@app.route('/api/login', methods=['POST'])
def login():
    """Login endpoint"""
    data = request.json
    password = data.get('password', '')
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    if password_hash == ADMIN_PASSWORD_HASH:
        session['authenticated'] = True
        return jsonify({'success': True, 'message': 'Login successful'})
    else:
        return jsonify({'success': False, 'message': 'Invalid password'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout endpoint"""
    session.pop('authenticated', None)
    return jsonify({'success': True})

@app.route('/api/status', methods=['GET'])
@login_required
def get_status():
    """Get current system status"""
    # Check latest data
    data_files = glob.glob(os.path.join(DATA_DIR, 'reddit_data_*.csv'))
    latest_data = None
    if data_files:
        latest_data = max(data_files, key=os.path.getctime)
        latest_data_time = datetime.fromtimestamp(os.path.getctime(latest_data))
    else:
        latest_data_time = None
    
    # Check latest analysis
    analysis_files = glob.glob(os.path.join(OUTPUT_DIR, 'analysis_*.json'))
    latest_analysis = None
    if analysis_files:
        latest_analysis = max(analysis_files, key=os.path.getctime)
        latest_analysis_time = datetime.fromtimestamp(os.path.getctime(latest_analysis))
    else:
        latest_analysis_time = None
    
    return jsonify({
        'data_collected': latest_data_time.isoformat() if latest_data_time else None,
        'analysis_completed': latest_analysis_time.isoformat() if latest_analysis_time else None,
        'data_file': os.path.basename(latest_data) if latest_data else None,
        'analysis_file': os.path.basename(latest_analysis) if latest_analysis else None,
    })

@app.route('/api/collect-data', methods=['POST'])
@login_required
def collect_data():
    """Trigger data collection"""
    try:
        result = subprocess.run(
            ['python3', os.path.join(SCRIPTS_DIR, 'collect_data.py')],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'message': 'Data collection completed successfully',
                'output': result.stdout
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Data collection failed',
                'error': result.stderr
            }), 500
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'message': 'Data collection timed out (>10 minutes)'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/run-analysis', methods=['POST'])
@login_required
def run_analysis():
    """Trigger analysis"""
    try:
        result = subprocess.run(
            ['python3', os.path.join(SCRIPTS_DIR, 'analyze_data.py')],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'message': 'Analysis completed successfully',
                'output': result.stdout
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Analysis failed',
                'error': result.stderr
            }), 500
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'message': 'Analysis timed out (>5 minutes)'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/latest-analysis', methods=['GET'])
@login_required
def get_latest_analysis():
    """Get latest analysis results"""
    try:
        if os.path.exists(os.path.join(OUTPUT_DIR, 'analysis_latest.json')):
            with open(os.path.join(OUTPUT_DIR, 'analysis_latest.json'), 'r') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify({'error': 'No analysis available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-analysis', methods=['GET'])
@login_required
def download_analysis():
    """Download latest analysis as JSON"""
    try:
        file_path = os.path.join(OUTPUT_DIR, 'analysis_latest.json')
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name='reddit_bot_analysis.json')
        else:
            return jsonify({'error': 'No analysis available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-data', methods=['GET'])
@login_required
def download_data():
    """Download latest data as CSV"""
    try:
        file_path = os.path.join(DATA_DIR, 'reddit_data_latest.csv')
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name='reddit_data.csv')
        else:
            return jsonify({'error': 'No data available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    """Get list of all analysis runs"""
    try:
        analysis_files = glob.glob(os.path.join(OUTPUT_DIR, 'analysis_*.json'))
        history = []
        
        for file_path in sorted(analysis_files, reverse=True):
            filename = os.path.basename(file_path)
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    scores = data.get('unified_scores', {})
                    top_sub = max(scores.items(), key=lambda x: x[1]['final_score']) if scores else None
                    
                    history.append({
                        'filename': filename,
                        'timestamp': file_time.isoformat(),
                        'top_bot_sub': top_sub[0] if top_sub else None,
                        'top_bot_score': top_sub[1]['final_score'] if top_sub else None,
                    })
            except:
                pass
        
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("REDDIT BOT ANALYSIS - ADMIN DASHBOARD")
    print("="*80)
    print("\n🔐 Admin Dashboard running at: http://localhost:5000")
    print("📝 Default password: admin123 (CHANGE THIS IN PRODUCTION!)")
    print("\n" + "="*80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
