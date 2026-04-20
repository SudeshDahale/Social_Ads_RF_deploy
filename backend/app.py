from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
classifier = joblib.load(os.path.join(BASE, 'model', 'classifier.pkl'))
sc = joblib.load(os.path.join(BASE, 'model', 'scaler.pkl'))

with open(os.path.join(BASE, 'model', 'stats.json')) as f:
    stats = json.load(f)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        age = float(data['age'])
        salary = float(data['salary'])

        # Exact code from notebook: feature scale then predict
        features = np.array([[age, salary]])
        features_scaled = sc.transform(features)
        prediction = int(classifier.predict(features_scaled)[0])
        proba = classifier.predict_proba(features_scaled)[0].tolist()

        return jsonify({
            'success': True,
            'prediction': prediction,
            'label': 'Purchased' if prediction == 1 else 'Not Purchased',
            'probability_not_purchased': round(proba[0], 4),
            'probability_purchased': round(proba[1], 4),
            'confidence': round(max(proba), 4)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/stats', methods=['GET'])
def model_stats():
    return jsonify(stats)


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    try:
        records = request.get_json()  # list of {age, salary}
        results = []
        for rec in records:
            features = np.array([[float(rec['age']), float(rec['salary'])]])
            features_scaled = sc.transform(features)
            pred = int(classifier.predict(features_scaled)[0])
            proba = classifier.predict_proba(features_scaled)[0].tolist()
            results.append({
                'age': rec['age'],
                'salary': rec['salary'],
                'prediction': pred,
                'label': 'Purchased' if pred == 1 else 'Not Purchased',
                'confidence': round(max(proba), 4)
            })
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'Random Forest Classifier', 'accuracy': stats['accuracy']})


# if __name__ == '__main__':
#     print(f"🌲 Social Network Ads - Random Forest API")
#     print(f"   Running on http://localhost:5000")
#     print(f"   Accuracy: {stats['accuracy']*100:.1f}%")
#     app.run(debug=True, port=5000)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)