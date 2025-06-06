from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
import modeloutput_pb2
import textwrap


app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# initialise Ecal
ecal_core.initialize([], "LLM Answer")
pub = ProtoPublisher("from Flask",modeloutput_pb2.OUT)


def genECALdata(text):
    pub2ecal = modeloutput_pb2.OUT()
    pub2ecal.headline = "LLM Answer"
    pub2ecal.text = text
    pub.send(pub2ecal)


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
        wrapper = textwrap.TextWrapper(width=80, subsequent_indent='  ')
        formatted = "\n".join(wrapper.fill(line) if not line.startswith("-") else wrapper.fill(line) for line in interpretation_text.splitlines())

        # send Ecal data
        genECALdata(formatted)

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
    ecal_core.finalize()
