"""
vision_engine.py — Image description
Priority: Gemini Vision (free) -> Claude Vision -> local BLIP
"""

import os
import base64
import logging

logger = logging.getLogger(__name__)

_blip_model = None
_blip_processor = None


def _get_blip():
    global _blip_model, _blip_processor
    if _blip_model is None:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            model_id = os.getenv("VISION_MODEL", "Salesforce/blip-image-captioning-base")
            logger.info(f"Loading BLIP: {model_id}")
            _blip_processor = BlipProcessor.from_pretrained(model_id)
            _blip_model = BlipForConditionalGeneration.from_pretrained(model_id)
            _blip_model.eval()
        except Exception as e:
            logger.error(f"Could not load BLIP: {e}")
            _blip_model = "unavailable"
    return _blip_model, _blip_processor


def describe_image_gemini(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """Describe image using Gemini Vision API (free tier)."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return {"error": "GEMINI_API_KEY not set."}
    try:
        import requests
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        prompt = (
            "Analyze this image and respond with ONLY a JSON object (no markdown, no backticks) like this:\n"
            '{"caption": "one sentence describing the image", '
            '"tags": ["tag1", "tag2", "tag3"], '
            '"details": "2-3 sentence detailed description"}'
        )
        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": b64}},
                    {"text": prompt}
                ]
            }]
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = raw.strip("```json").strip("```").strip()
        import json
        data = json.loads(raw)
        data["model"] = "Gemini Vision (free)"
        return data
    except Exception as e:
        logger.error(f"Gemini Vision error: {e}")
        return {"error": str(e)}


def describe_image_claude(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """Describe image using Claude Vision API."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set."}
    try:
        import anthropic, json
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "Analyze this image and respond with ONLY a JSON object (no markdown) like:\n"
            '{"caption": "one sentence", "tags": ["tag1","tag2","tag3"], "details": "2-3 sentences"}'
        )
        msg = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=300,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        raw = msg.content[0].text.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
        data["model"] = "Claude Vision API"
        return data
    except Exception as e:
        logger.error(f"Claude Vision error: {e}")
        return {"error": str(e)}


def describe_image_blip(image_bytes: bytes) -> dict:
    """Describe image using local BLIP model."""
    from PIL import Image
    import io, torch
    model, processor = _get_blip()
    if model == "unavailable" or model is None:
        return {"error": "BLIP model not available. Install transformers and torch."}
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()
        tags = []
        for prompt in ["a photo of", "the color is", "the scene shows"]:
            inp = processor(image, text=prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=10)
            tag = processor.decode(out[0], skip_special_tokens=True).replace(prompt, "").strip().split(".")[0].strip()
            if tag and len(tag) > 2:
                tags.append(tag)
        return {"caption": caption, "tags": tags[:3], "model": "BLIP (local)"}
    except Exception as e:
        logger.error(f"BLIP error: {e}")
        return {"error": str(e)}


def describe_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Smart dispatcher — priority: Gemini (free) -> Claude -> BLIP local
    """
    # Try Gemini first (free)
    if os.getenv("GEMINI_API_KEY"):
        result = describe_image_gemini(image_bytes, mime_type)
        if "error" not in result:
            return result
        logger.warning(f"Gemini Vision failed: {result['error']} — trying Claude...")

    # Try Claude
    if os.getenv("ANTHROPIC_API_KEY"):
        result = describe_image_claude(image_bytes, mime_type)
        if "error" not in result:
            return result
        logger.warning(f"Claude Vision failed: {result['error']} — trying BLIP...")

    # Local BLIP fallback
    return describe_image_blip(image_bytes)


def format_vision_response(result: dict) -> str:
    if "error" in result:
        return f"❌ Vision error: {result['error']}"
    caption = result.get("caption", "No caption generated.")
    tags = result.get("tags", [])
    details = result.get("details", "")
    model = result.get("model", "unknown")
    tag_str = " ".join(f"#{t.replace(' ', '_')}" for t in tags) if tags else "No tags"
    lines = ["🖼️ *Image Analysis*", "", f"📝 *Caption:* {caption}"]
    if details:
        lines += ["", f"🔍 *Details:* {details}"]
    lines += ["", f"🏷️ *Tags:* {tag_str}", "", f"_Model: {model}_"]
    return "\n".join(lines)
