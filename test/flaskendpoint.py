from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

@app.route('/api/interpret', methods=['POST'])
def interpret_face():
    try:
        data = request.get_json(force=True)
        print("=== Received Interpretation Text ===")
        print("Raw JSON:", data)

        if "text" not in data:
            return jsonify({
                'status': 'error',
                'error': 'Missing "text" field in JSON.'
            }), 400

        # Do something with the interpretation text
        interpretation_text = data["text"]
        print("LLM Interpretation:", interpretation_text)

        return jsonify({
            'status': 'success',
            'message': 'Text received and logged successfully.'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
