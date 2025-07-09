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
import re
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


def parse_voice_blend(voice_string):
    """
    Parse voice blend string like "af_nicole(1)+am_michael(1)" or "af_heart(2)+bf_emma(1)+am_liam(0.5)"
    Returns list of tuples: [(voice_id, weight), ...]
    """
    if '+' not in voice_string:
        # Single voice, return as-is
        return [(voice_string, 1.0)]

    # Pattern to match voice(weight) format
    pattern = r'([a-z]{2}_[a-z]+)\(([0-9]*\.?[0-9]+)\)'
    matches = re.findall(pattern, voice_string)

    if not matches:
        raise ValueError(f"Invalid voice blend format: {voice_string}")

    voice_blend = []
    for voice_id, weight_str in matches:
        if voice_id not in CHOICES:
            raise ValueError(f"Unknown voice: {voice_id}")

        weight = float(weight_str)
        if weight <= 0:
            raise ValueError(f"Voice weight must be positive: {weight}")

        voice_blend.append((voice_id, weight))

    return voice_blend


def blend_voice_references(voice_blend, ps_length):
    """
    Blend multiple voice references based on weights
    Returns blended reference tensor
    """
    if len(voice_blend) == 1:
        # Single voice, no blending needed
        voice_id, _ = voice_blend[0]
        pipeline = pipelines[voice_id[0]]
        pack = pipeline.load_voice(voice_id)
        return pack[ps_length - 1]

    # Get all voice references
    voice_refs = []
    weights = []

    for voice_id, weight in voice_blend:
        pipeline = pipelines[voice_id[0]]
        pack = pipeline.load_voice(voice_id)
        ref_s = pack[ps_length - 1]
        voice_refs.append(ref_s)
        weights.append(weight)

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Blend the references
    blended_ref = None
    for ref_s, weight in zip(voice_refs, normalized_weights):
        if blended_ref is None:
            blended_ref = ref_s * weight
        else:
            blended_ref = blended_ref + (ref_s * weight)

    return blended_ref


@spaces.GPU(duration=30)
def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)


def generate_audio_with_blend(text, voice_blend, speed=1, use_gpu=CUDA_AVAILABLE):
    """Generate audio from text using voice blending"""
    text = text if CHAR_LIMIT is None else text.strip()[:CHAR_LIMIT]

    # Use the first voice's pipeline for tokenization (they should be compatible)
    primary_voice = voice_blend[0][0]
    pipeline = pipelines[primary_voice[0]]

    use_gpu = use_gpu and CUDA_AVAILABLE

    for _, ps, _ in pipeline(text, primary_voice, speed):
        # Blend voice references
        blended_ref = blend_voice_references(voice_blend, len(ps))

        try:
            if use_gpu:
                audio = forward_gpu(ps, blended_ref, speed)
            else:
                audio = models[False](ps, blended_ref, speed)
        except Exception as e:
            if use_gpu:
                print(f"GPU error: {e}, retrying with CPU")
                audio = models[False](ps, blended_ref, speed)
            else:
                raise e
        return audio.numpy(), ps
    return None, ''


def generate_audio(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    """Generate audio from text (supports both single voice and voice blending)"""
    try:
        # Parse voice blend
        voice_blend = parse_voice_blend(voice)
        return generate_audio_with_blend(text, voice_blend, speed, use_gpu)
    except ValueError as e:
        print(f"Voice parsing error: {e}")
        # Fallback to single voice
        if voice in CHOICES:
            voice_blend = [(voice, 1.0)]
            return generate_audio_with_blend(text, voice_blend, speed, use_gpu)
        else:
            raise e


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


def validate_voice_blend(voice_string):
    """Validate voice blend string and return parsed blend info"""
    try:
        voice_blend = parse_voice_blend(voice_string)

        # Check for language compatibility
        languages = set(voice_id[0] for voice_id, _ in voice_blend)
        if len(languages) > 1:
            return False, f"Cannot mix voices from different languages: {languages}"

        return True, voice_blend
    except ValueError as e:
        return False, str(e)


# API Routes
@app.route('/', methods=['GET'])
def home():
    """API information"""
    return jsonify({
        "service": "Kokoro-TTS REST API with Voice Blending",
        "version": "2.0.0",
        "description": "Text-to-Speech API using Kokoro-82M model with voice blending support",
        "voice_blending": {
            "description": "Mix multiple voices to create custom blended voices",
            "format": "voice1(weight1)+voice2(weight2)+...",
            "example": "af_nicole(1)+am_michael(1)",
            "limitations": "Voices must be from the same language (a* or b*)"
        },
        "endpoints": {
            "/synthesize": "POST - Generate speech from text (supports voice blending)",
            "/voices": "GET - List available voices",
            "/health": "GET - Health check",
            "/random-quote": "GET - Get a random quote",
            "/validate-blend": "POST - Validate voice blend string"
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
        "models_loaded": len(models) > 0,
        "voice_blending": "enabled"
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
        ],
        "voice_blending": {
            "description": "You can blend voices using the format: voice1(weight1)+voice2(weight2)",
            "example": "af_nicole(1)+am_michael(1)",
            "note": "Voices must be from the same language group"
        }
    })


@app.route('/validate-blend', methods=['POST'])
def validate_blend():
    """Validate voice blend string"""
    try:
        data = request.get_json()
        if not data or 'voice' not in data:
            return jsonify({"error": "Missing 'voice' parameter"}), 400

        voice_string = data['voice']
        is_valid, result = validate_voice_blend(voice_string)

        if is_valid:
            return jsonify({
                "valid": True,
                "voice_blend": [{"voice": v, "weight": w} for v, w in result],
                "total_voices": len(result),
                "language": result[0][0][0]  # First character indicates language
            })
        else:
            return jsonify({
                "valid": False,
                "error": result
            }), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Synthesize speech from text (supports voice blending)"""
    try:
        data = request.get_json()

        # Validate input
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' parameter"}), 400

        text = data['input']
        voice = data.get('voice', 'af_heart')
        speed = data.get('speed', 1.0)
        use_gpu = data.get('use_gpu', CUDA_AVAILABLE)
        output_format = data.get('response_format', 'audio')  # 'audio' or 'base64'

        # Validate voice blend
        is_valid, validation_result = validate_voice_blend(voice)
        if not is_valid:
            return jsonify({"error": f"Invalid voice blend: {validation_result}"}), 400

        # Validate other parameters
        if not (0.5 <= speed <= 2.0):
            return jsonify({"error": "Speed must be between 0.5 and 2.0"}), 400

        if CHAR_LIMIT and len(text) > CHAR_LIMIT:
            return jsonify({"error": f"Text too long. Maximum {CHAR_LIMIT} characters"}), 400

        # Generate audio with blending
        audio_array, tokens = generate_audio(text, voice, speed, use_gpu)

        if audio_array is None:
            return jsonify({"error": "Failed to generate audio"}), 500

        # Prepare response info
        voice_info = {
            "voice_string": voice,
            "voice_blend": [{"voice": v, "weight": w} for v, w in validation_result],
            "is_blended": len(validation_result) > 1
        }

        # Return based on format
        if output_format == 'base64':
            wav_buffer = audio_to_wav_bytes(audio_array)
            audio_b64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
            return jsonify({
                "audio_base64": audio_b64,
                "tokens": tokens,
                "sample_rate": 24000,
                "voice_info": voice_info,
                "speed": speed,
                "text": text
            })
        else:
            # Return audio file
            wav_buffer = audio_to_wav_bytes(audio_array)
            blend_name = voice.replace('(', '_').replace(')', '').replace('+', '_')
            return send_file(
                wav_buffer,
                mimetype='audio/wav',
                as_attachment=True,
                download_name=f'kokoro_tts_{blend_name}_{int(datetime.now().timestamp())}.wav'
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

        # Validate voice blend
        is_valid, validation_result = validate_voice_blend(voice)
        if not is_valid:
            return jsonify({"error": f"Invalid voice blend: {validation_result}"}), 400

        # Use primary voice for tokenization
        primary_voice = validation_result[0][0]
        pipeline = pipelines[primary_voice[0]]

        for _, ps, _ in pipeline(text, primary_voice):
            return jsonify({
                "tokens": ps,
                "text": text,
                "voice_info": {
                    "voice_string": voice,
                    "primary_voice": primary_voice,
                    "voice_blend": [{"voice": v, "weight": w} for v, w in validation_result]
                },
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
    """Synthesize multiple texts at once (supports voice blending)"""
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

        # Validate voice blend once
        is_valid, validation_result = validate_voice_blend(voice)
        if not is_valid:
            return jsonify({"error": f"Invalid voice blend: {validation_result}"}), 400

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
            "voice_info": {
                "voice_string": voice,
                "voice_blend": [{"voice": v, "weight": w} for v, w in validation_result],
                "is_blended": len(validation_result) > 1
            },
            "speed": speed,
            "sample_rate": 24000
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Kokoro-TTS REST API with Voice Blending...")
    print(f"📊 CUDA Available: {CUDA_AVAILABLE}")
    print(f"📝 Character Limit: {CHAR_LIMIT}")
    print(f"🎙️  Voices Loaded: {len(CHOICES)}")
    print("🎭 Voice Blending: ENABLED")
    print("   Example: af_nicole(1)+am_michael(1)")
    print("   Example: af_heart(2)+af_bella(1)+am_liam(0.5)")

    app.run(
        host='0.0.0.0',
        port=7860,
        debug=True,
        threaded=True
    )