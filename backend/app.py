from flask import Flask, jsonify, request
from flask_cors import CORS
import uuid
import time

from routes.synthetic_routes import synthetic_bp
from routes.export_routes import export_bp
from routes.trial_routes import trial_bp
from routes.predictive_routes import predictive_bp
from routes.accelerated_routes import accelerated_bp

app = Flask(__name__)

# CORS Configuration
CORS(app, origins=['http://127.0.0.1:8000', 'http://localhost:8000'], supports_credentials=True)

# Register blueprints
app.register_blueprint(synthetic_bp)
app.register_blueprint(export_bp)
app.register_blueprint(trial_bp)
app.register_blueprint(predictive_bp)
app.register_blueprint(accelerated_bp)

# In-memory storage for trial data
trial_data_store = {}

@app.route("/")
def home():
    return {
        "status": "running",
        "api": "Virtual Clinical Trial Simulator",
        "endpoint": "/generate"
    }

# Trial data storage endpoints
@app.route('/store-trial-data', methods=['POST', 'OPTIONS'])
def store_trial_data():
    """Store large trial dataset on backend"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', 'http://127.0.0.1:8000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        data = request.json
        patients = data.get('patients', [])
        timestamp = data.get('timestamp', time.time())
        count = len(patients)
        
        session_id = str(uuid.uuid4())
        
        trial_data_store[session_id] = {
            'patients': patients,
            'timestamp': timestamp,
            'count': count,
            'expires': time.time() + 3600
        }
        
        cleanup_expired_sessions()
        
        print(f"✅ Stored {count} patients with session ID: {session_id}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': f'Stored {count} patients successfully'
        })
        
    except Exception as e:
        print(f"❌ Error storing trial data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/get-trial-data/<session_id>', methods=['GET', 'OPTIONS'])
def get_trial_data(session_id):
    """Retrieve large trial dataset from backend"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', 'http://127.0.0.1:8000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET')
        return response
    
    try:
        if session_id in trial_data_store:
            data = trial_data_store[session_id]
            if data['expires'] < time.time():
                del trial_data_store[session_id]
                return jsonify({
                    'success': False,
                    'error': 'Session expired'
                }), 410
            
            print(f"✅ Retrieved {data['count']} patients for session: {session_id}")
            
            return jsonify({
                'success': True,
                'patients': data['patients'],
                'count': data['count'],
                'timestamp': data['timestamp']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Session not found'
            }), 404
            
    except Exception as e:
        print(f"❌ Error retrieving trial data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def cleanup_expired_sessions():
    """Remove expired sessions from memory"""
    expired = []
    for session_id, data in trial_data_store.items():
        if data['expires'] < time.time():
            expired.append(session_id)
    
    for session_id in expired:
        del trial_data_store[session_id]
    
    if expired:
        print(f"🧹 Cleaned up {len(expired)} expired sessions")

@app.route('/trial-storage-status', methods=['GET'])
def storage_status():
    """Check how many sessions are currently stored"""
    return jsonify({
        'active_sessions': len(trial_data_store),
        'total_patients_stored': sum(data['count'] for data in trial_data_store.values())
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)