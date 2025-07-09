from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import spaces
from kokoro import KModel, KPipeline
import os
import random
import torch
import numpy as np
import io
import wave
import base64
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize models (same as original)
IS_DUPLICATE = not os.getenv('SPACE_ID', '').startswith('hexgrad/')
CUDA_AVAILABLE = torch.cuda.is_available()
if not IS_DUPLICATE:
    import kokoro
    import misaki

    print('DEBUG', kokoro.__version__, CUDA_AVAILABLE, misaki.__version__)

CHAR_LIMIT = None if IS_DUPLICATE else 5000
models = {gpu: KModel().to('cuda' if gpu else 'cpu').eval() for gpu in [False] + ([True] if CUDA_AVAILABLE else [])}
pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in 'ab'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

# Voice choices
CHOICES = {
    'af_heart': '🇺🇸 🚺 Heart ❤️',
    'af_bella': '🇺🇸 🚺 Bella 🔥',
    'af_nicole': '🇺🇸 🚺 Nicole 🎧',
    'af_aoede': '🇺🇸 🚺 Aoede',
    'af_kore': '🇺🇸 🚺 Kore',
    'af_sarah': '🇺🇸 🚺 Sarah',
    'af_nova': '🇺🇸 🚺 Nova',
    'af_sky': '🇺🇸 🚺 Sky',
    'af_alloy': '🇺🇸 🚺 Alloy',
    'af_jessica': '🇺🇸 🚺 Jessica',
    'af_river': '🇺🇸 🚺 River',
    'am_michael': '🇺🇸 🚹 Michael',
    'am_fenrir': '🇺🇸 🚹 Fenrir',
    'am_puck': '🇺🇸 🚹 Puck',
    'am_echo': '🇺🇸 🚹 Echo',
    'am_eric': '🇺🇸 🚹 Eric',
    'am_liam': '🇺🇸 🚹 Liam',
    'am_onyx': '🇺🇸 🚹 Onyx',
    'am_santa': '🇺🇸 🚹 Santa',
    'am_adam': '🇺🇸 🚹 Adam',
    'bf_emma': '🇬🇧 🚺 Emma',
    'bf_isabella': '🇬🇧 🚺 Isabella',
    'bf_alice': '🇬🇧 🚺 Alice',
    'bf_lily': '🇬🇧 🚺 Lily',
    'bm_george': '🇬🇧 🚹 George',
    'bm_fable': '🇬🇧 🚹 Fable',
    'bm_lewis': '🇬🇧 🚹 Lewis',
    'bm_daniel': '🇬🇧 🚹 Daniel',
}

# Pre-load voices
for voice_id in CHOICES.keys():
    pipelines[voice_id[0]].load_voice(voice_id)


@spaces.GPU(duration=30)
def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)


def generate_audio(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    """Generate audio from text"""
    text = text if CHAR_LIMIT is None else text.strip()[:CHAR_LIMIT]
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE

    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps) - 1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except Exception as e:
            if use_gpu:
                print(f"GPU error: {e}, retrying with CPU")
                audio = models[False](ps, ref_s, speed)
            else:
                raise e
        return audio.numpy(), ps
    return None, ''


def audio_to_wav_bytes(audio_array, sample_rate=24000):
    """Convert audio array to WAV bytes"""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        # Convert float32 to int16
        audio_int16 = (audio_array * 32767).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())
    buffer.seek(0)
    return buffer


# API Routes
@app.route('/', methods=['GET'])
def home():
    """API information"""
    return jsonify({
        "service": "Kokoro-TTS REST API",
        "version": "1.0.0",
        "description": "Text-to-Speech API using Kokoro-82M model",
        "endpoints": {
            "/synthesize": "POST - Generate speech from text",
            "/voices": "GET - List available voices",
            "/health": "GET - Health check",
            "/random-quote": "GET - Get a random quote"
        },
        "cuda_available": CUDA_AVAILABLE,
        "character_limit": CHAR_LIMIT
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cuda_available": CUDA_AVAILABLE,
        "models_loaded": len(models) > 0
    })


@app.route('/voices', methods=['GET'])
def get_voices():
    """Get available voices"""
    return jsonify({
        "voices": [
            {
                "id": voice_id,
                "name": voice_name,
                "language": "en-US" if voice_id.startswith('a') else "en-GB",
                "gender": "female" if voice_id[1] == 'f' else "male"
            }
            for voice_id, voice_name in CHOICES.items()
        ]
    })


@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Synthesize speech from text"""
    try:
        data = request.get_json()

        # Validate input
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' parameter"}), 400

        text = data['text']
        voice = data.get('voice', 'af_heart')
        speed = data.get('speed', 1.0)
        use_gpu = data.get('use_gpu', CUDA_AVAILABLE)
        output_format = data.get('format', 'audio')  # 'audio' or 'base64'

        # Validate parameters
        if voice not in CHOICES:
            return jsonify({"error": f"Invalid voice. Available: {list(CHOICES.keys())}"}), 400

        if not (0.5 <= speed <= 2.0):
            return jsonify({"error": "Speed must be between 0.5 and 2.0"}), 400

        if CHAR_LIMIT and len(text) > CHAR_LIMIT:
            return jsonify({"error": f"Text too long. Maximum {CHAR_LIMIT} characters"}), 400

        # Generate audio
        audio_array, tokens = generate_audio(text, voice, speed, use_gpu)

        if audio_array is None:
            return jsonify({"error": "Failed to generate audio"}), 500

        # Return based on format
        if output_format == 'base64':
            wav_buffer = audio_to_wav_bytes(audio_array)
            audio_b64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
            return jsonify({
                "audio_base64": audio_b64,
                "tokens": tokens,
                "sample_rate": 24000,
                "voice": voice,
                "speed": speed,
                "text": text
            })
        else:
            # Return audio file
            wav_buffer = audio_to_wav_bytes(audio_array)
            return send_file(
                wav_buffer,
                mimetype='audio/wav',
                as_attachment=True,
                download_name=f'kokoro_tts_{voice}_{int(datetime.now().timestamp())}.wav'
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/tokenize', methods=['POST'])
def tokenize():
    """Tokenize text without generating audio"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' parameter"}), 400

        text = data['text']
        voice = data.get('voice', 'af_heart')

        if voice not in CHOICES:
            return jsonify({"error": f"Invalid voice. Available: {list(CHOICES.keys())}"}), 400

        pipeline = pipelines[voice[0]]
        for _, ps, _ in pipeline(text, voice):
            return jsonify({
                "tokens": ps,
                "text": text,
                "voice": voice,
                "token_count": len(ps.split()) if ps else 0
            })

        return jsonify({"error": "Failed to tokenize text"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/random-quote', methods=['GET'])
def random_quote():
    """Get a random quote"""
    try:
        with open('en.txt', 'r') as r:
            quotes = [line.strip() for line in r if line.strip()]
        return jsonify({
            "quote": random.choice(quotes),
            "source": "random"
        })
    except FileNotFoundError:
        return jsonify({
            "quote": "The quick brown fox jumps over the lazy dog.",
            "source": "default"
        })


@app.route('/batch-synthesize', methods=['POST'])
def batch_synthesize():
    """Synthesize multiple texts at once"""
    try:
        data = request.get_json()

        if not data or 'texts' not in data:
            return jsonify({"error": "Missing 'texts' parameter (array of strings)"}), 400

        texts = data['texts']
        voice = data.get('voice', 'af_heart')
        speed = data.get('speed', 1.0)
        use_gpu = data.get('use_gpu', CUDA_AVAILABLE)

        if not isinstance(texts, list):
            return jsonify({"error": "'texts' must be an array"}), 400

        if len(texts) > 10:  # Limit batch size
            return jsonify({"error": "Maximum 10 texts per batch"}), 400

        results = []
        for i, text in enumerate(texts):
            try:
                audio_array, tokens = generate_audio(text, voice, speed, use_gpu)
                if audio_array is not None:
                    wav_buffer = audio_to_wav_bytes(audio_array)
                    audio_b64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
                    results.append({
                        "index": i,
                        "success": True,
                        "audio_base64": audio_b64,
                        "tokens": tokens,
                        "text": text
                    })
                else:
                    results.append({
                        "index": i,
                        "success": False,
                        "error": "Failed to generate audio",
                        "text": text
                    })
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e),
                    "text": text
                })

        return jsonify({
            "results": results,
            "voice": voice,
            "speed": speed,
            "sample_rate": 24000
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Kokoro-TTS REST API...")
    print(f"📊 CUDA Available: {CUDA_AVAILABLE}")
    print(f"📝 Character Limit: {CHAR_LIMIT}")
    print(f"🎙️  Voices Loaded: {len(CHOICES)}")

    app.run(
        host='0.0.0.0',
        port=7860,
        debug=True,
        threaded=True
    )