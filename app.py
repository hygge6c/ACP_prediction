import os
import json
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from config import Config
from model import TransformerClassifier
from utils import (
    validate_sequence, encode_sequence, calculate_sequence_properties,
    get_amino_acid_properties, calculate_amino_acid_composition,
    create_attention_heatmap, create_feature_importance_plot,
    create_amino_acid_composition_plot, create_batch_statistics_plot
)

app = Flask(__name__)
app.config.from_object(Config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

# Load performance metrics
PERFORMANCE_METRICS = {}
try:
    with open('model_performance.json') as f:
        PERFORMANCE_METRICS = json.load(f)
except:
    PERFORMANCE_METRICS = app.config['DEFAULT_PERFORMANCE']

def load_model():
    global model
    if model is None:
        cfg = app.config['MODEL_CONFIG']
        model = TransformerClassifier(
            vocab_size=cfg['vocab_size'], embed_dim=cfg['embed_dim'],
            num_heads=cfg['num_heads'], hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'], dropout=cfg['dropout'], max_len=cfg['max_len']
        )
        model_path = app.config['MODEL_PATH']
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model file not found: {model_path}. Using random weights (demo mode).")
        model.to(device)
        model.eval()
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        seq = data.get('sequence', '').strip().upper()
        is_valid, msg = validate_sequence(seq)
        if not is_valid:
            return jsonify({'error': msg}), 400

        model = load_model()
        encoded = encode_sequence(seq, app.config['VOCAB'], app.config['MODEL_CONFIG']['max_len']).to(device)
        with torch.no_grad():
            prob = model(encoded).item()

        threshold = app.config['PREDICTION_THRESHOLD']
        prediction = 'Anticancer Peptide' if prob >= threshold else 'Non-Anticancer Peptide'
        confidence = round(prob * 100, 2) if prediction == 'Anticancer Peptide' else round((1 - prob) * 100, 2)

        # Simulated attention and importance (for demo)
        L = len(seq)
        sim_attn = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                sim_attn[i, j] = 0.3 if i == j else 0.2 / (abs(i - j) + 1)
        sim_attn = sim_attn / sim_attn.sum(axis=1, keepdims=True)

        sim_importance = np.zeros(L)
        for i, aa in enumerate(seq):
            if aa in 'KRH':
                sim_importance[i] = np.random.uniform(0.7, 0.9)
            elif aa in 'DE':
                sim_importance[i] = np.random.uniform(0.5, 0.7)
            elif aa in 'FYW':
                sim_importance[i] = np.random.uniform(0.6, 0.8)
            else:
                sim_importance[i] = np.random.uniform(0.3, 0.6)

        composition = calculate_amino_acid_composition(seq)
        properties = calculate_sequence_properties(seq)

        response = {
            'success': True,
            'sequence': seq,
            'probability': round(prob, 4),
            'prediction': prediction,
            'confidence': confidence,
            'length': len(seq),
            'properties': properties,
            'amino_acid_composition': composition,
            'visualizations': {
                'attention_heatmap': json.loads(create_attention_heatmap(seq, sim_attn)),
                'feature_importance': json.loads(create_feature_importance_plot(seq, sim_importance)),
                'composition_plot': json.loads(create_amino_acid_composition_plot(composition))
            }
        }
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        text = data.get('sequences', '').strip()
        if not text:
            return jsonify({'error': 'No sequences provided'}), 400

        sequences = []
        for line in text.splitlines():
            line = line.strip()
            if ',' in line:
                sequences.extend([s.strip().upper() for s in line.split(',') if s.strip()])
            elif line:
                sequences.append(line.upper())

        if not sequences:
            return jsonify({'error': 'No valid sequences found'}), 400

        model = load_model()
        results = []
        valid_count = 0
        anticancer_count = 0

        for seq in sequences:
            valid, msg = validate_sequence(seq)
            if not valid:
                results.append({'sequence': seq, 'is_valid': False, 'error': msg})
                continue

            encoded = encode_sequence(seq, app.config['VOCAB'], app.config['MODEL_CONFIG']['max_len']).to(device)
            with torch.no_grad():
                prob = model(encoded).item()
            pred = 'Anticancer' if prob >= app.config['PREDICTION_THRESHOLD'] else 'Non-Anticancer'
            results.append({
                'sequence': seq, 'is_valid': True, 'length': len(seq),
                'probability': round(prob, 4), 'prediction': pred
            })
            valid_count += 1
            if pred == 'Anticancer':
                anticancer_count += 1

        stats = {
            'total_sequences': len(sequences),
            'valid_sequences': valid_count,
            'anticancer_count': anticancer_count,
            'non_anticancer_count': valid_count - anticancer_count,
            'anticancer_percentage': round(anticancer_count / valid_count * 100, 2) if valid_count else 0
        }
        stats_plot = json.loads(create_batch_statistics_plot(stats)) if valid_count else None
        return jsonify({
            'success': True,
            'results': results,
            'statistics': stats,
            'visualizations': {'batch_statistics': stats_plot}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'model_name': 'AntiCancerPeptide-Transformer',
        'architecture': 'Transformer',
        'parameters': app.config['MODEL_CONFIG'],
        'performance': PERFORMANCE_METRICS,
        'training_info': {
            'dataset_size': '423 sequences',
            'positive_ratio': '50%',
            'validation_method': 'Hold-out (80/20 split)',
            'optimizer': 'AdamW',
            'loss_function': 'Binary Cross Entropy'
        }
    })

@app.route('/api/amino_acid_info', methods=['GET'])
def amino_acid_info():
    return jsonify(get_amino_acid_properties())

@app.route('/api/example_sequences', methods=['GET'])
def example_sequences():
    return jsonify({
        'anticancer': ['GLFDIIKKIAESF', 'KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK', 'GIGKFLHSAKKFGKAFVGEIMNS'],
        'non_anticancer': ['SLDQINVTFLDLEYEMKKLEEAIKKLEESYIDLKEL', 'DISGINASVVNIQKEIDRLNEVAKNLNESLIDLQEL', 'EEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEE']
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None, 'device': str(device)})

if __name__ == '__main__':
    os.makedirs('best_models', exist_ok=True)
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)