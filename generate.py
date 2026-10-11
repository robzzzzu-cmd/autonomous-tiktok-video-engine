#!/usr/bin/env python3
"""
Autonomous 40-Second TikTok Video Generation Engine
Powered by Google Gemini 2.5 Flash, Hugging Face Flux.1-Schnell, Edge-TTS & FFmpeg.
Designed to run 100% autonomously in GitHub Actions.
"""

import argparse
import asyncio
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.parse
import wave
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_FPS = 30
TOTAL_SCENES = 5
TARGET_MIN_WORDS = 85
TARGET_MAX_WORDS = 95
VOICE_NAME = "en-US-AvaNeural"
VOICE_RATE = "-4%"
BGM_VOLUME_DB = "-22dB"

# Aesthetic anchor style applied to every scene prompt (Tasty / @sugarspiceddreams style)
AESTHETIC_STYLE_ANCHOR = (
    "Tasty style, hyperrealistic 8k editorial food photography, vertical 9:16 portrait framing, "
    "macro close-up, warm diffused natural studio window light, shallow depth of field f/1.8, 35mm lens, "
    "white marble countertop with soft neutral linen cloth and unbleached parchment paper, "
    "clear fluted glass mixing bowls, warm vintage brass utensils, wire cooling rack, mouthwatering and cozy"
)

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class SceneStoryboard:
    scene_number: int
    scene_name: str
    narration: str
    primary_visual_prompt: str = ""
    secondary_visual_prompt: str = ""
    flux_prompt: str = ""
    video_search_queries: List[str] = field(default_factory=list)
    duration: float = 0.0
    image_path: str = ""
    audio_path: str = ""
    video_path: str = ""

@dataclass
class VideoProject:
    recipe_title: str
    recipe_slug: str
    concept_description: str
    scenes: List[SceneStoryboard]
    total_words: int = 0
    total_duration: float = 0.0
    final_video_path: str = ""


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def log(tag: str, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{tag}] {message}", flush=True)

def sanitize_slug(text: str) -> str:
    text = text.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', text).strip('-')
    return slug or "viral-recipe"

def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+(?:'\w+)?\b", text))

def format_ass_timestamp(seconds: float) -> str:
    """Formats seconds into ASS timestamp: H:MM:SS.cs (centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ==============================================================================
# HISTORY MANAGEMENT
# ==============================================================================

def load_history(history_path: str) -> List[str]:
    """Loads previously generated recipe titles from history.json."""
    if not os.path.exists(history_path):
        log("HISTORY", f"History file {history_path} does not exist. Initializing empty history.")
        return []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            entries = data.get("history", [])
            titles = [e.get("title", "").strip() for e in entries if e.get("title")]
            log("HISTORY", f"Loaded {len(titles)} previously generated recipes from {history_path}.")
            return titles
    except Exception as e:
        log("HISTORY", f"Warning: Failed to parse history file ({e}). Starting fresh.")
        return []

def record_history(history_path: str, title: str, slug: str) -> None:
    """Records the newly generated recipe title and slug in history.json."""
    history_entries = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                history_entries = data.get("history", [])
        except Exception:
            history_entries = []

    # Check if duplicate title already exists
    if any(e.get("slug") == slug or e.get("title", "").lower() == title.lower() for e in history_entries):
        log("HISTORY", f"Entry '{title}' already recorded in history.")
        return

    new_entry = {
        "title": title,
        "slug": slug,
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    history_entries.append(new_entry)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({"history": history_entries}, f, indent=2, ensure_ascii=False)
    log("HISTORY", f"Recorded '{title}' ({slug}) to {history_path}.")


# ==============================================================================
# GEMINI API STORYBOARD & BRAINSTORMING
# ==============================================================================

def brainstorm_storyboard(
    gemini_api_key: str,
    history_titles: List[str],
    user_topic: Optional[str] = None
) -> VideoProject:
    """
    Uses Google Gemini 2.5 Flash to autonomously select a trending viral recipe
    (or use user_topic) and construct a strictly timed 5-scene storyboard.
    Enforces 85-95 words total narration.
    """
    log("GEMINI", "Initializing Google Gemini 2.5 Flash client...")

    # SDK resolution: preference for google-genai, fallback to google-generativeai
    use_new_sdk = False
    genai_client = None
    genai_module = None

    try:
        from google import genai
        from google.genai import types
        genai_client = genai.Client(api_key=gemini_api_key)
        use_new_sdk = True
        log("GEMINI", "Using modern google-genai SDK.")
    except Exception as e_new:
        log("GEMINI", f"google-genai SDK not available or initialized ({e_new}), falling back to google-generativeai.")
        try:
            import google.generativeai as gai
            gai.configure(api_key=gemini_api_key)
            genai_module = gai
            use_new_sdk = False
            log("GEMINI", "Using google-generativeai SDK.")
        except Exception as e_old:
            raise RuntimeError(f"Failed to load both google-genai and google-generativeai SDKs: {e_old}")

    past_titles_str = ", ".join(f'"{t}"' for t in history_titles[-25:]) if history_titles else "None"

    topic_instruction = (
        f"Generate the storyboard for the specific recipe requested by the user: '{user_topic}'."
        if user_topic and user_topic.strip()
        else (
            "AUTONOMOUSLY BRAINSTORM ONE viral, visually mesmerizing, decadent baking or comfort-food dessert/pastry. "
            "Examples of high-performing TikTok formats: stuffed NYC thick cookies with molten core, "
            "caramelized brown butter cinnamon rolls, molten lava chocolate cakes, artisan honeycomb focaccia, "
            "cloud matcha brioche toast, burnt Basque espresso cheesecake, or flaky pull-apart brioche. "
            f"CRITICAL: Do NOT pick or duplicate any of these previously generated recipes: [{past_titles_str}]."
        )
    )

    system_prompt = (
        "You are an elite food video director creating ultra-retentive 40-second TikTok videos "
        "in the cozy, mouthwatering aesthetic of Tasty and @sugarspiceddreams.\n\n"
        "STRICT REQUIREMENTS:\n"
        "1. Exactly 5 sequential scenes representing the recipe progression:\n"
        "   - Scene 1 (The Hook): Macro cross-section / bite / gooey center reveal.\n"
        "   - Scene 2 (Base Prep): Whisking / creaming warm ingredients into a glossy ribbon.\n"
        "   - Scene 3 (Dough / Mix-in): Folding chunks, velvety ganache, textures, or pouring sauces.\n"
        "   - Scene 4 (Cook / Bake): Oven rise, sizzling pan, bubbling crust, golden browning.\n"
        "   - Scene 5 (Finishing & Plating): Drizzle, flaky sea salt garnish, final aesthetic plating shot.\n\n"
        "2. STRICT 40-SECOND TIMING CONSTRAINT:\n"
        "   - TikTok retention drops if pacing drags. The TOTAL word count across all 5 scene narrations "
        "     combined MUST BE STRICTLY BETWEEN 85 AND 95 WORDS TOTAL.\n"
        "   - That is approximately 17 to 19 words per scene at conversational speed (38 to 40 seconds spoken).\n"
        "   - Count the words carefully! Less than 85 words or more than 95 words will fail validation.\n\n"
        "3. DYNAMIC MULTI-SHOT VISUAL AESTHETIC (POLLINATIONS.AI ENGINE):\n"
        "   - To generate lively, cinematic video clips matching the premise of the video, provide TWO distinct visual prompts per scene:\n"
        "     * `primary_visual_prompt`: The main culinary action / state of the scene (e.g. whisking into glossy ribbon, pouring batter into fluted pan, molten center pull reveal).\n"
        "     * `secondary_visual_prompt`: An extreme macro close-up texture or dynamic secondary angle (e.g. velvety batter cascading off whisk, bubbling caramel browning, flaky sea salt falling onto warm glaze).\n"
        "   - Maintain complete visual continuity: same clean white Carrara marble countertop, soft neutral folded linen, "
        "     clear fluted glass mixing bowls, warm brass utensils, warm diffused natural morning studio light, "
        "     shallow depth of field f/1.8, 35mm macro lens perspective.\n"
        "   - Also include `flux_prompt` matching `primary_visual_prompt` for backward compatibility.\n\n"
        "4. DYNAMIC KINETIC STOCK VIDEO SEARCH QUERIES:\n"
        "   - For each scene, provide an array `video_search_queries` with 2 to 3 concise search terms.\n\n"
        "Output ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "recipe_title": "String (e.g. Brown Butter Caramel Stuffed Cookies)",\n'
        '  "recipe_slug": "String (e.g. brown-butter-caramel-stuffed-cookies)",\n'
        '  "concept_description": "String summary",\n'
        '  "scenes": [\n'
        '    {\n'
        '      "scene_number": 1,\n'
        '      "scene_name": "The Hook",\n'
        '      "narration": "Spoken voiceover for scene 1 (approx 17-19 words)",\n'
        '      "primary_visual_prompt": "Ultra-detailed macro food visual prompt of the main culinary action",\n'
        '      "secondary_visual_prompt": "Ultra-detailed extreme close-up texture detail or secondary angle",\n'
        '      "flux_prompt": "Ultra-detailed prompt for fallback",\n'
        '      "video_search_queries": ["query 1", "query 2"]\n'
        '    },\n'
        '    ... exactly 5 scenes\n'
        '  ]\n'
        "}"
    )

    user_query = f"{topic_instruction}\n\nDeliver the storyboard now in strict JSON format."

    models_to_try = ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

    def call_gemini(query_text: str) -> str:
        last_err = None
        for model_name in models_to_try:
            try:
                log("GEMINI", f"Attempting generation with model '{model_name}'...")
                if use_new_sdk:
                    response = genai_client.models.generate_content(
                        model=model_name,
                        contents=query_text,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            temperature=0.7,
                        )
                    )
                    return response.text
                else:
                    model = genai_module.GenerativeModel(
                        model_name=model_name,
                        system_instruction=system_prompt,
                        generation_config={"response_mime_type": "application/json", "temperature": 0.7}
                    )
                    response = model.generate_content(query_text)
                    return response.text
            except Exception as e:
                log("GEMINI", f"Model '{model_name}' returned error: {e}. Trying next model...")
                last_err = e
        raise RuntimeError(f"All Gemini models failed: {last_err}")

    raw_response = call_gemini(user_query)
    log("GEMINI", "Received initial response from Gemini 2.5 Flash.")

    # Parse and validate JSON
    data = None
    try:
        # Strip potential markdown formatting if returned
        cleaned_json = re.sub(r"^```(?:json)?\s*", "", raw_response.strip(), flags=re.MULTILINE)
        cleaned_json = re.sub(r"\s*```$", "", cleaned_json.strip(), flags=re.MULTILINE)
        data = json.loads(cleaned_json)
    except Exception as e:
        log("GEMINI", f"JSON parsing failed ({e}). Attempting JSON regex recovery...")
        match = re.search(r"(\{.*\})", raw_response, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            raise RuntimeError(f"Could not parse JSON from Gemini response: {raw_response[:300]}")

    # Validate word count across scenes
    scenes_data = data.get("scenes", [])
    if len(scenes_data) != TOTAL_SCENES:
        log("GEMINI", f"Warning: Expected 5 scenes, received {len(scenes_data)}.")

    total_words = sum(count_words(s.get("narration", "")) for s in scenes_data)
    log("GEMINI", f"Initial total narration word count: {total_words} words (target: {TARGET_MIN_WORDS}-{TARGET_MAX_WORDS}).")

    # Word Count Adjustment Loop if needed
    if total_words < TARGET_MIN_WORDS or total_words > TARGET_MAX_WORDS:
        log("GEMINI", f"Word count ({total_words}) is outside 85-95 words. Requesting precision word count adjustment...")
        refine_prompt = (
            f"The total word count of your narration was {total_words} words, but it MUST BE STRICTLY "
            f"between {TARGET_MIN_WORDS} and {TARGET_MAX_WORDS} words total across all 5 scenes combined.\n\n"
            f"Here is the current draft:\n{json.dumps(data, indent=2)}\n\n"
            f"Please adjust the narration of each scene so the SUM of all words across the 5 scenes is strictly "
            f"between 85 and 95 words (approx 17-19 words per scene). Return the complete updated JSON."
        )
        try:
            refined_raw = call_gemini(refine_prompt)
            cleaned_refined = re.sub(r"^```(?:json)?\s*", "", refined_raw.strip(), flags=re.MULTILINE)
            cleaned_refined = re.sub(r"\s*```$", "", cleaned_refined.strip(), flags=re.MULTILINE)
            refined_data = json.loads(cleaned_refined)
            refined_words = sum(count_words(s.get("narration", "")) for s in refined_data.get("scenes", []))
            log("GEMINI", f"Refined word count: {refined_words} words.")
            data = refined_data
            total_words = refined_words
        except Exception as e:
            log("GEMINI", f"Refinement call failed ({e}). Proceeding with initial draft.")

    # Build Project
    recipe_title = data.get("recipe_title", "Viral Artisanal Pastry")
    recipe_slug = sanitize_slug(data.get("recipe_slug", recipe_title))
    concept_desc = data.get("concept_description", "")

    scenes: List[SceneStoryboard] = []
    for idx, s in enumerate(data.get("scenes", [])[:TOTAL_SCENES]):
        scene_num = idx + 1
        scene_name = s.get("scene_name", f"Scene {scene_num}")
        narration = s.get("narration", "").strip()
        primary_prompt = s.get("primary_visual_prompt", "").strip() or s.get("flux_prompt", "").strip()
        secondary_prompt = s.get("secondary_visual_prompt", "").strip() or primary_prompt
        flux_prompt = s.get("flux_prompt", "").strip() or primary_prompt

        # Enforce aesthetic anchor tokens in prompts
        if AESTHETIC_STYLE_ANCHOR not in primary_prompt:
            primary_prompt = f"{primary_prompt}. {AESTHETIC_STYLE_ANCHOR}"
        if AESTHETIC_STYLE_ANCHOR not in secondary_prompt:
            secondary_prompt = f"{secondary_prompt}. {AESTHETIC_STYLE_ANCHOR}"
        if AESTHETIC_STYLE_ANCHOR not in flux_prompt:
            flux_prompt = f"{flux_prompt}. {AESTHETIC_STYLE_ANCHOR}"

        raw_queries = s.get("video_search_queries", [])
        clean_queries = [str(q).strip() for q in raw_queries if str(q).strip()] if isinstance(raw_queries, list) else []

        scenes.append(SceneStoryboard(
            scene_number=scene_num,
            scene_name=scene_name,
            narration=narration,
            primary_visual_prompt=primary_prompt,
            secondary_visual_prompt=secondary_prompt,
            flux_prompt=flux_prompt,
            video_search_queries=clean_queries
        ))

    project = VideoProject(
        recipe_title=recipe_title,
        recipe_slug=recipe_slug,
        concept_description=concept_desc,
        scenes=scenes,
        total_words=total_words
    )

    log("GEMINI", f"Storyboard finalized: '{recipe_title}' ({recipe_slug}) | {len(scenes)} scenes | {total_words} words.")
    return project


# ==============================================================================
# VISUAL ASSET GENERATION (FLUX.1-SCHNELL + PEXELS + PIL FALLBACK)
# ==============================================================================

def generate_image_flux(prompt: str, output_path: str, hf_token: str) -> bool:
    """Attempts generation via Hugging Face Inference API for FLUX.1-schnell."""
    if not hf_token:
        return False

    log("VISUAL-FLUX", f"Attempting Flux.1-Schnell generation via Hugging Face Hub...")
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=hf_token, timeout=90)
        # 768x1360 is optimal portrait aspect ratio for Flux.1-schnell
        image = client.text_to_image(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell",
            width=768,
            height=1344
        )
        # Resize to standard TikTok 1080x1920
        image = image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
        image.save(output_path, "PNG", quality=95)
        log("VISUAL-FLUX", f"Successfully generated Flux.1-Schnell visual: {output_path}")
        return True
    except Exception as e:
        log("VISUAL-FLUX", f"InferenceClient call encountered issue ({e}). Trying direct HTTP endpoint...")

    # Fallback to direct HTTP post
    endpoints = [
        "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
        "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    ]
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"width": 768, "height": 1344}
    }

    for ep in endpoints:
        try:
            resp = requests.post(ep, headers=headers, json=payload, timeout=90)
            if resp.status_code == 200 and resp.content:
                img = Image.open(io.BytesIO(resp.content))
                img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                img.save(output_path, "PNG", quality=95)
                log("VISUAL-FLUX", f"Successfully fetched Flux image from {ep}")
                return True
            else:
                log("VISUAL-FLUX", f"Endpoint {ep} returned status {resp.status_code}: {resp.text[:120]}")
        except Exception as err:
            log("VISUAL-FLUX", f"Direct HTTP post to {ep} failed: {err}")

    return False

def generate_image_pexels(query: str, output_path: str, pexels_key: str) -> bool:
    """Fallback visual generation via Pexels portrait search."""
    if not pexels_key:
        return False

    log("VISUAL-PEXELS", f"Searching Pexels fallback with query: '{query}'...")
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": pexels_key}
    params = {
        "query": f"{query} food baking dessert",
        "orientation": "portrait",
        "per_page": 5
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            photos = data.get("photos", [])
            if photos:
                img_url = photos[0]["src"].get("large2x") or photos[0]["src"].get("portrait") or photos[0]["src"].get("original")
                img_resp = requests.get(img_url, timeout=30)
                if img_resp.status_code == 200:
                    img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                    # Crop and fit to 1080x1920
                    img_ratio = img.width / img.height
                    target_ratio = TARGET_WIDTH / TARGET_HEIGHT
                    if img_ratio > target_ratio:
                        new_width = int(img.height * target_ratio)
                        left = (img.width - new_width) // 2
                        img = img.crop((left, 0, left + new_width, img.height))
                    else:
                        new_height = int(img.width / target_ratio)
                        top = (img.height - new_height) // 2
                        img = img.crop((0, top, img.width, top + new_height))
                    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                    img.save(output_path, "PNG", quality=95)
                    log("VISUAL-PEXELS", f"Successfully saved Pexels fallback visual: {output_path}")
                    return True
        else:
            log("VISUAL-PEXELS", f"Pexels API returned status {resp.status_code}: {resp.text[:120]}")
    except Exception as e:
        log("VISUAL-PEXELS", f"Pexels request failed: {e}")

    return False

def generate_image_procedural(scene: SceneStoryboard, recipe_title: str, output_path: str) -> None:
    """
    Tertiary ultra-reliable fallback: Generates an aesthetic high-resolution editorial card
    with warm marble/linen tones, soft lighting vignette, and typography.
    """
    log("VISUAL-PROCEDURAL", f"Rendering procedural aesthetic visual for Scene {scene.scene_number}...")
    img = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), color=(248, 244, 238))
    draw = ImageDraw.Draw(img)

    # Gradient background: warm cream to soft warm almond
    for y in range(TARGET_HEIGHT):
        ratio = y / TARGET_HEIGHT
        r = int(250 - ratio * 20)
        g = int(246 - ratio * 25)
        b = int(240 - ratio * 32)
        draw.line([(0, y), (TARGET_WIDTH, y)], fill=(r, g, b))

    # Inner decorative gold/brass border
    margin = 70
    draw.rectangle(
        [(margin, margin), (TARGET_WIDTH - margin, TARGET_HEIGHT - margin)],
        outline=(212, 175, 55),
        width=3
    )

    # Subtle inner card
    card_margin = 120
    draw.rectangle(
        [(card_margin, 350), (TARGET_WIDTH - card_margin, TARGET_HEIGHT - 350)],
        fill=(255, 255, 255),
        outline=(230, 225, 215),
        width=2
    )

    # Typography
    try:
        font_large = ImageFont.truetype("Arial", 54)
        font_sub = ImageFont.truetype("Arial", 38)
        font_body = ImageFont.truetype("Arial", 30)
    except Exception:
        font_large = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_body = ImageFont.load_default()

    draw.text((TARGET_WIDTH // 2, 450), "✦ ARTISANAL KITCHEN ✦", fill=(180, 140, 70), font=font_sub, anchor="mm")
    draw.text((TARGET_WIDTH // 2, 540), recipe_title[:32], fill=(45, 35, 25), font=font_large, anchor="mm")
    draw.text((TARGET_WIDTH // 2, 620), f"— {scene.scene_name.upper()} —", fill=(130, 110, 90), font=font_sub, anchor="mm")

    # Wrapped narration preview
    narration_words = scene.narration.split()
    lines = []
    current_line = []
    for w in narration_words:
        current_line.append(w)
        if len(" ".join(current_line)) > 26:
            lines.append(" ".join(current_line))
            current_line = []
    if current_line:
        lines.append(" ".join(current_line))

    start_y = 780
    for line in lines[:5]:
        draw.text((TARGET_WIDTH // 2, start_y), line, fill=(80, 70, 60), font=font_body, anchor="mm")
        start_y += 48

    # Bottom watermark
    draw.text((TARGET_WIDTH // 2, TARGET_HEIGHT - 200), "SLOW-CRAFTED COMFORT FOOD", fill=(170, 150, 130), font=font_sub, anchor="mm")

    img.save(output_path, "PNG")
    log("VISUAL-PROCEDURAL", f"Procedural aesthetic card saved to {output_path}")

def build_aesthetic_food_prompt(prompt: str, recipe_title: str, scene_name: str) -> str:
    """
    Constructs an ultra-aesthetic, coherent macro food photography prompt for Pollinations.
    Maintains complete visual continuity (white Carrara marble, warm brass utensils, warm morning studio lighting).
    """
    clean = prompt.replace("\n", " ").strip()
    prefix = f"{recipe_title}: " if recipe_title.lower() not in clean.lower() else ""
    style = (
        "hyperrealistic 8k macro food b-roll photography, soft warm diffused morning window lighting, "
        "shallow depth of field f/1.8, 35mm macro lens, clean white Carrara marble countertop, "
        "warm brass utensils, folded neutral linen, cozy aesthetic, delicious golden tones, "
        "mouthwatering, no text, no watermark, no human face"
    )
    full_prompt = f"{prefix}{clean}, {style}"
    if len(full_prompt) > 230:
        full_prompt = full_prompt[:230].rsplit(" ", 1)[0]
    return full_prompt

def generate_image_pollinations(prompt: str, output_path: str, seed: Optional[int] = None) -> bool:
    """
    Generates high-resolution 1080x1920 visuals using Pollinations.ai.
    Free, fast, and captures the exact recipe visuals without requiring API tokens.
    """
    log("POLLINATIONS", f"Generating visual via Pollinations: '{prompt[:75]}...'")
    clean_prompt = prompt.replace("\n", " ").strip()
    if len(clean_prompt) > 220:
        clean_prompt = clean_prompt[:220].rsplit(" ", 1)[0]

    encoded = urllib.parse.quote(clean_prompt)
    if seed is None:
        seed = int(time.time() * 1000) % 100000

    endpoints = [
        f"https://image.pollinations.ai/prompt/{encoded}?width={TARGET_WIDTH}&height={TARGET_HEIGHT}&nologo=true&seed={seed}",
        f"https://image.pollinations.ai/prompt/{encoded}?width={TARGET_WIDTH}&height={TARGET_HEIGHT}&model=turbo&nologo=true&seed={seed}",
        f"https://image.pollinations.ai/prompt/{encoded}?width={TARGET_WIDTH}&height={TARGET_HEIGHT}&model=sana&nologo=true&seed={seed}"
    ]

    for url in endpoints:
        try:
            resp = requests.get(url, timeout=35)
            if resp.status_code == 200 and resp.content and len(resp.content) > 2000:
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                if img.size != (TARGET_WIDTH, TARGET_HEIGHT):
                    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                img.save(output_path, "PNG", quality=95)
                log("POLLINATIONS", f"Successfully generated visual: {output_path} ({img.size[0]}x{img.size[1]})")
                return True
            else:
                log("POLLINATIONS", f"Pollinations returned status {resp.status_code} (bytes: {len(resp.content) if resp.content else 0})")
        except Exception as e:
            log("POLLINATIONS", f"Pollinations request failed ({e})")

    return False

def generate_scene_clip_from_pollinations(
    scene: SceneStoryboard,
    recipe_title: str,
    output_path: str,
    target_duration: float,
    temp_dir: str,
    pollinations_api_key: Optional[str] = None
) -> bool:
    """
    Generates a dynamic multi-shot video clip for the scene using Pollinations.ai.
    Splits the scene into 2 distinct shots (Action + Macro Texture) to create
    snappy, retentive TikTok video clips that perfectly match the recipe premise.
    """
    log("CLIP-POLLINATIONS", f"Creating dynamic video clip for Scene {scene.scene_number} ({scene.scene_name})...")

    # If user provided a Pollinations API key, try the direct video endpoint first
    if pollinations_api_key:
        try:
            log("CLIP-POLLINATIONS", "Pollinations API key detected, attempting direct video endpoint...")
            v_prompt = build_aesthetic_food_prompt(scene.primary_visual_prompt or scene.flux_prompt, recipe_title, scene.scene_name)
            v_encoded = urllib.parse.quote(v_prompt[:180])
            video_url = f"https://gen.pollinations.ai/video/{v_encoded}?key={pollinations_api_key}"
            v_resp = requests.get(video_url, timeout=60)
            if v_resp.status_code == 200 and v_resp.content and len(v_resp.content) > 10000:
                raw_temp = output_path.replace(".mp4", "_pollinations_raw.mp4")
                with open(raw_temp, "wb") as f:
                    f.write(v_resp.content)
                filter_graph = f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,crop={TARGET_WIDTH}:{TARGET_HEIGHT},setsar=1,fps={TARGET_FPS}"
                norm_cmd = [
                    "ffmpeg", "-y", "-i", raw_temp,
                    "-vf", filter_graph, "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-t", f"{target_duration:.3f}", output_path
                ]
                subprocess.run(norm_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=40, check=True)
                if os.path.exists(raw_temp):
                    os.remove(raw_temp)
                log("CLIP-POLLINATIONS", f"Successfully generated direct AI video clip: {output_path}")
                return True
        except Exception as e_v:
            log("CLIP-POLLINATIONS", f"Direct video endpoint attempt failed ({e_v}). Continuing with multi-shot Ken Burns engine...")

    # Multi-shot Ken Burns dynamic clip engine (100% Free, prompt-matched, fast)
    motion_pairs = {
        1: ("zoom_in", "pan_down"),      # The Hook: Molten core push-in + drip tilt
        2: ("pan_down", "diagonal"),     # Base Prep: Bowl entry + ribbon whisking
        3: ("diagonal", "zoom_in"),      # Dough/Mix-in: Folding pan + chunk zoom
        4: ("pan_up", "zoom_in"),        # Cook/Bake: Rising crust + golden bubble zoom
        5: ("zoom_in", "zoom_out")       # Finishing: Drizzle push-in + plating reveal
    }
    motion1, motion2 = motion_pairs.get(scene.scene_number, ("zoom_in", "zoom_out"))

    dur1 = round(target_duration / 2.0, 2)
    dur2 = round(target_duration - dur1, 2)

    prompt1 = build_aesthetic_food_prompt(
        scene.primary_visual_prompt or scene.flux_prompt,
        recipe_title,
        scene.scene_name
    )
    prompt2 = build_aesthetic_food_prompt(
        scene.secondary_visual_prompt or f"{scene.scene_name} extreme close-up texture detail",
        recipe_title,
        scene.scene_name
    )

    img1_path = os.path.join(temp_dir, f"scene_{scene.scene_number}_shot1.png")
    img2_path = os.path.join(temp_dir, f"scene_{scene.scene_number}_shot2.png")

    shot1_ok = generate_image_pollinations(prompt1, img1_path, seed=scene.scene_number * 100 + 1)
    if not shot1_ok:
        return False

    shot2_ok = generate_image_pollinations(prompt2, img2_path, seed=scene.scene_number * 100 + 2)

    shot1_video = os.path.join(temp_dir, f"scene_{scene.scene_number}_shot1.mp4")
    shot2_video = os.path.join(temp_dir, f"scene_{scene.scene_number}_shot2.mp4")

    if shot1_ok and shot2_ok:
        render_ken_burns_shot(img1_path, dur1, motion1, shot1_video)
        render_ken_burns_shot(img2_path, dur2, motion2, shot2_video)
        concatenate_scenes([shot1_video, shot2_video], output_path, temp_dir)
        log("CLIP-POLLINATIONS", f"Successfully produced 2-shot dynamic clip for Scene {scene.scene_number} ({target_duration:.2f}s).")
        return True
    elif shot1_ok:
        render_ken_burns_shot(img1_path, target_duration, motion1, output_path)
        log("CLIP-POLLINATIONS", f"Produced single-shot dynamic clip for Scene {scene.scene_number} ({target_duration:.2f}s).")
        return True

    return False

def fetch_scene_video_clip(
    scene: SceneStoryboard,
    recipe_title: str,
    output_path: str,
    target_duration: float,
    pexels_key: str
) -> bool:
    """
    Searches Pexels Video API for authentic stock video footage matching the premise and kinetic action of the scene.
    Downloads, normalizes, crops to 1080x1920 portrait (9:16) @ 30fps, and trims to the exact scene narration duration.
    """
    if not pexels_key:
        return False

    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": pexels_key}

    # Curate high-yield action fallback queries per scene archetype
    scene_defaults = {
        1: ["cutting cake", "cake slice fork", "dessert bite", "pastry close up", "cutting dessert"],
        2: ["whisking batter", "mixing bowl cooking", "baking prep", "whisking eggs", "mixing dough"],
        3: ["pouring batter", "folding dough", "pouring sauce", "chocolate chips", "pouring melted chocolate"],
        4: ["baking oven", "cake baking oven", "bread oven", "oven cooking", "cookies baking"],
        5: ["drizzling sauce dessert", "dusting powdered sugar", "plating dessert", "pastry presentation", "serving dessert"]
    }

    # Build search query queue: scene's custom queries first, then archetype defaults
    candidate_queries = []
    if scene.video_search_queries:
        for q in scene.video_search_queries:
            if q and q not in candidate_queries:
                candidate_queries.append(q)

    # Also add recipe title + scene action
    recipe_keyword = recipe_title.split()[-1].lower() if recipe_title else "dessert"
    candidate_queries.append(f"{recipe_keyword} {scene.scene_name.lower()}")

    for dq in scene_defaults.get(scene.scene_number, ["baking food", "dessert"]):
        if dq not in candidate_queries:
            candidate_queries.append(dq)

    for q in candidate_queries:
        log("VIDEO-PEXELS", f"Searching real video clip for Scene {scene.scene_number} ({scene.scene_name}): '{q}'...")
        try:
            # First try portrait orientation
            resp = requests.get(url, headers=headers, params={"query": q, "orientation": "portrait", "per_page": 8}, timeout=15)
            videos = []
            if resp.status_code == 200:
                videos = resp.json().get("videos", [])

            # If no portrait footage found, search across all orientations
            if not videos:
                resp = requests.get(url, headers=headers, params={"query": q, "per_page": 8}, timeout=15)
                if resp.status_code == 200:
                    videos = resp.json().get("videos", [])

            if videos:
                # Select best video with appropriate duration (prefer >= 4.0s)
                chosen_video = videos[0]
                for v in videos:
                    if v.get("duration", 0) >= 4.0:
                        chosen_video = v
                        break

                video_files = chosen_video.get("video_files", [])
                best_file = None
                # First priority: Vertical HD (height > width and height >= 720)
                for vf in video_files:
                    w, h = vf.get("width", 0), vf.get("height", 0)
                    if h > w and h >= 720:
                        best_file = vf
                        break
                # Second priority: Standard HD (width >= 1280 or quality == "hd")
                if not best_file:
                    for vf in video_files:
                        if vf.get("quality") == "hd" or vf.get("width", 0) >= 1280:
                            best_file = vf
                            break
                # Fallback to any file
                if not best_file and video_files:
                    best_file = video_files[0]

                if best_file and best_file.get("link"):
                    dl_url = best_file["link"]
                    raw_temp = output_path.replace(".mp4", "_raw.mp4")
                    log("VIDEO-PEXELS", f"Found stock footage (ID: {chosen_video.get('id')}). Downloading clip...")
                    dl_resp = requests.get(dl_url, stream=True, timeout=50)
                    if dl_resp.status_code == 200:
                        with open(raw_temp, "wb") as f:
                            for chunk in dl_resp.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)

                        raw_duration = get_media_duration(raw_temp)
                        start_offset = 1.0 if raw_duration > (target_duration + 2.0) else 0.0

                        loop_flags = []
                        if 0 < raw_duration < target_duration:
                            loop_flags = ["-stream_loop", "-1"]

                        # Normalization filter: Scale, Center-Crop to 1080x1920 9:16, 30fps
                        filter_graph = f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,crop={TARGET_WIDTH}:{TARGET_HEIGHT},setsar=1,fps={TARGET_FPS}"

                        cmd = [
                            "ffmpeg", "-y"
                        ] + loop_flags + [
                            "-ss", f"{start_offset:.2f}",
                            "-t", f"{target_duration:.3f}",
                            "-i", raw_temp,
                            "-vf", filter_graph,
                            "-an",
                            "-c:v", "libx264",
                            "-preset", "veryfast",
                            "-crf", "20",
                            "-pix_fmt", "yuv420p",
                            output_path
                        ]
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=40, check=True)
                        if os.path.exists(raw_temp):
                            os.remove(raw_temp)
                        log("VIDEO-PEXELS", f"Successfully formatted dynamic video clip for Scene {scene.scene_number}: {output_path}")
                        return True
        except Exception as e:
            log("VIDEO-PEXELS", f"Query '{q}' failed ({e}). Trying next query...")

    return False

def generate_scene_visual(
    scene: SceneStoryboard,
    recipe_title: str,
    output_path: str,
    hf_token: str,
    pexels_key: str
) -> str:
    """Executes visual generation waterfall: Pollinations.ai -> FLUX.1-schnell -> Pexels -> Procedural."""
    # Step 1: Pollinations.ai (Free, prompt-matched photorealistic visual)
    prompt = build_aesthetic_food_prompt(scene.primary_visual_prompt or scene.flux_prompt, recipe_title, scene.scene_name)
    if generate_image_pollinations(prompt, output_path):
        return output_path

    # Step 2: Hugging Face Flux.1-Schnell
    if hf_token and generate_image_flux(scene.flux_prompt, output_path, hf_token):
        return output_path

    # Step 3: Pexels API
    query = f"{recipe_title} {scene.scene_name}"
    if pexels_key and generate_image_pexels(query, output_path, pexels_key):
        return output_path

    # Step 4: Procedural Fallback
    generate_image_procedural(scene, recipe_title, output_path)
    return output_path


# ==============================================================================
# AUDIO & VOICEOVER GENERATION (EDGE-TTS + WORD BOUNDARIES)
# ==============================================================================

@dataclass
class WordTiming:
    word: str
    start: float
    end: float

async def synthesize_scene_audio(
    text: str,
    output_audio_path: str
) -> Tuple[float, List[WordTiming]]:
    """
    Synthesizes speech using Edge-TTS en-US-AvaNeural at speed -4%.
    Captures exact word boundaries for subtitle synchronization.
    Returns (audio_duration_seconds, list_of_word_timings).
    """
    import edge_tts

    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE_NAME,
        rate=VOICE_RATE
    )

    audio_bytes = bytearray()
    word_timings: List[WordTiming] = []

    async for chunk in communicate.stream():
        chunk_type = chunk.get("type")
        if chunk_type == "audio":
            audio_bytes.extend(chunk.get("data", b""))
        elif chunk_type == "WordBoundary":
            # offset and duration in 100-nanosecond ticks (1 tick = 1e-7 s)
            offset = chunk.get("offset", 0) / 10_000_000.0
            duration = chunk.get("duration", 0) / 10_000_000.0
            word_str = chunk.get("text", "").strip()
            if word_str:
                word_timings.append(WordTiming(
                    word=word_str,
                    start=offset,
                    end=offset + duration
                ))

    with open(output_audio_path, "wb") as f:
        f.write(audio_bytes)

    # Determine audio duration via ffprobe
    duration = get_media_duration(output_audio_path)
    if duration <= 0.0 and word_timings:
        duration = word_timings[-1].end + 0.3
    elif duration <= 0.0:
        duration = 7.5

    # Fallback word timings if Edge-TTS did not stream WordBoundary events
    if not word_timings and text:
        words = text.split()
        if words and duration > 0:
            word_dur = duration / len(words)
            for i, w in enumerate(words):
                word_timings.append(WordTiming(
                    word=w,
                    start=i * word_dur,
                    end=(i + 1) * word_dur
                ))

    return duration, word_timings


def get_media_duration(file_path: str) -> float:
    """Measures precise duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        log("FFPROBE", f"Could not measure duration of {file_path} via ffprobe: {e}")
        return 0.0


# ==============================================================================
# SUBTITLE GENERATION (.ASS FORMAT - LOWER-THIRD SAFE ZONE)
# ==============================================================================

def build_ass_subtitles(
    all_word_timings: List[WordTiming],
    output_ass_path: str
) -> None:
    """
    Generates clean .ass subtitles centered in the lower-third safe zone.
    Styles: Clean sans-serif (Arial), crisp white text with soft dark drop shadow (&H60000000).
    Chunks text into 2-3 words per line for TikTok-optimized retention.
    """
    # Group words into 2-3 word clusters
    clusters: List[Tuple[float, float, str]] = []
    chunk_size = 2
    i = 0
    n = len(all_word_timings)

    while i < n:
        # Alternate between 2 and 3 words for natural cadence
        take = 3 if (i // 2) % 2 == 1 and (i + 3 <= n) else 2
        take = min(take, n - i)
        sub_words = all_word_timings[i : i + take]
        start_t = sub_words[0].start
        end_t = sub_words[-1].end

        # Extend slightly to next word start to avoid blinking/flicker
        if i + take < n:
            next_start = all_word_timings[i + take].start
            if next_start > end_t and (next_start - end_t) < 0.25:
                end_t = next_start - 0.04

        cluster_text = " ".join(w.word for w in sub_words)
        clusters.append((start_t, end_t, cluster_text))
        i += take

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {TARGET_WIDTH}
PlayResY: {TARGET_HEIGHT}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTokClean,Arial,60,&H00FFFFFF,&H000000FF,&H90000000,&H60000000,-1,0,0,0,100,100,0,0,1,3.5,2.5,2,90,90,320,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: List[str] = []
    for start_t, end_t, text in clusters:
        start_str = format_ass_timestamp(start_t)
        end_str = format_ass_timestamp(end_t)
        # Clean text
        clean_text = text.replace("{", "").replace("}", "")
        line = f"Dialogue: 0,{start_str},{end_str},TikTokClean,,0,0,0,,{clean_text}"
        events.append(line)

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header + "\n".join(events) + "\n")

    log("SUBTITLES", f"Generated {len(clusters)} subtitle clusters saved to {output_ass_path}.")


# ==============================================================================
# BACKGROUND MUSIC GENERATION / SOURCING
# ==============================================================================

def ensure_lofi_background_music(output_path: str) -> str:
    """
    Ensures a subtle acoustic/lofi loop is available.
    If no pre-existing asset exists, synthesizes a warm, soothing 16-bit acoustic lofi loop
    directly via Python's wave module (zero external dependencies).
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        log("BGM", f"Using existing background music asset at {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log("BGM", "Synthesizing autonomous warm acoustic lofi background loop...")

    # Synthesis parameters
    sample_rate = 44100
    duration_seconds = 24.0  # 24 second loop
    num_samples = int(sample_rate * duration_seconds)

    # Chord progression: Cmaj7 - Am7 - Dm7 - G7 (warm jazz/lofi acoustic voicing)
    chord_notes = [
        # Cmaj7 (C3, G3, B3, E4)
        [130.81, 196.00, 246.94, 329.63],
        # Am7 (A2, E3, G3, C4)
        [110.00, 164.81, 196.00, 261.63],
        # Dm7 (D3, A3, C4, F4)
        [146.83, 220.00, 261.63, 349.23],
        # G7 (G2, D3, F3, B3)
        [98.00, 146.83, 174.61, 246.94]
    ]

    chord_duration = duration_seconds / len(chord_notes)  # 6.0 seconds per chord
    samples = []

    for i in range(num_samples):
        t = i / sample_rate
        chord_idx = int((t % duration_seconds) // chord_duration) % len(chord_notes)
        chord = chord_notes[chord_idx]
        t_chord = t % chord_duration

        # Warm acoustic decaying envelope
        env = math.exp(-t_chord * 0.45) * 0.4 + 0.15

        # Sum fundamental + soft harmonics
        sample_val = 0.0
        for freq in chord:
            # Fundamental
            sample_val += math.sin(2.0 * math.pi * freq * t) * 0.35
            # Soft 2nd harmonic (warmth)
            sample_val += math.sin(2.0 * math.pi * freq * 2.0 * t) * 0.12
            # Very subtle 3rd harmonic
            sample_val += math.sin(2.0 * math.pi * freq * 3.0 * t) * 0.04

        # Very subtle warm vinyl / analog crackle noise
        crackle = ((math.sin(t * 1234.5) * 43758.5453) % 1.0 - 0.5) * 0.015

        val = (sample_val * env + crackle) * 0.35
        # Clamp
        val = max(-0.95, min(0.95, val))
        samples.append(int(val * 32767.0))

    # Save to WAV file
    wav_path = output_path if output_path.endswith(".wav") else output_path.replace(".mp3", ".wav")
    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        packed = struct.pack(f"<{len(samples)}h", *samples)
        wf.writeframes(packed)

    # Convert to MP3 via ffmpeg if desired
    if output_path.endswith(".mp3"):
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", wav_path,
                "-codec:a", "libmp3lame", "-qscale:a", "4",
                output_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            log("BGM", f"Converted synthesized lofi track to {output_path}")
            return output_path
        except Exception:
            return wav_path

    return wav_path


# ==============================================================================
# CINEMATIC MOTION & FFMPEG RENDERING ENGINE
# ==============================================================================

def render_ken_burns_shot(
    image_path: str,
    duration: float,
    motion_type: str,
    output_video_path: str
) -> None:
    """
    Renders a single image into a smooth Ken Burns motion video clip (1080x1920 @ 30fps)
    using specified camera panning motions.
    """
    total_frames = max(1, int(round(duration * TARGET_FPS)))

    if motion_type == "zoom_in":
        # Slow macro push-in towards center
        zoompan_expr = (
            f"zoompan=z='min(zoom+0.0008,1.22)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={TARGET_WIDTH}x{TARGET_HEIGHT}:fps={TARGET_FPS}"
        )
    elif motion_type == "pan_down":
        # Slow downward pan into bowl / pan
        zoompan_expr = (
            f"zoompan=z='1.12':x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*(on/{total_frames})':"
            f"d={total_frames}:s={TARGET_WIDTH}x{TARGET_HEIGHT}:fps={TARGET_FPS}"
        )
    elif motion_type == "pan_up":
        # Slow upward pan following rising crust / browning
        zoompan_expr = (
            f"zoompan=z='1.12':x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*(1-on/{total_frames})':"
            f"d={total_frames}:s={TARGET_WIDTH}x{TARGET_HEIGHT}:fps={TARGET_FPS}"
        )
    elif motion_type == "zoom_out":
        # Slow pull-back revealing plated dish
        zoompan_expr = (
            f"zoompan=z='max(1.20-0.0007*on,1.0)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={TARGET_WIDTH}x{TARGET_HEIGHT}:fps={TARGET_FPS}"
        )
    else:
        # Subtle diagonal drift
        zoompan_expr = (
            f"zoompan=z='min(zoom+0.0006,1.16)':x='(iw-iw/zoom)*(on/{total_frames})*0.5':y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={TARGET_WIDTH}x{TARGET_HEIGHT}:fps={TARGET_FPS}"
        )

    # Directly feed single image to zoompan (DO NOT use -loop 1, as zoompan expands single frame to d frames)
    filter_graph = f"{zoompan_expr},format=yuv420p"

    cmd = [
        "ffmpeg", "-y",
        "-i", image_path,
        "-vf", filter_graph,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-t", f"{duration:.3f}",
        output_video_path
    ]

    log("FFMPEG", f"Rendering Ken Burns shot '{motion_type}' ({duration:.2f}s)...")
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=40, check=True)
    except Exception as e:
        log("FFMPEG", f"Ken Burns zoompan encountered issue ({e}). Using ultra-fast fallback clip...")
        cmd_fallback = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", image_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-t", f"{duration:.3f}",
            output_video_path
        ]
        subprocess.run(cmd_fallback, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20, check=True)

def render_ken_burns_scene(
    image_path: str,
    duration: float,
    scene_number: int,
    output_video_path: str
) -> None:
    """Convenience wrapper for single-scene Ken Burns motion."""
    motions = {1: "zoom_in", 2: "pan_down", 3: "diagonal", 4: "pan_up", 5: "zoom_out"}
    render_ken_burns_shot(image_path, duration, motions.get(scene_number, "zoom_in"), output_video_path)


def concatenate_scenes(scene_video_paths: List[str], output_path: str, temp_dir: str) -> None:
    """Concatenates all individual scene video clips using ffmpeg concat demuxer with filter fallback."""
    concat_list_file = os.path.join(temp_dir, "concat_scenes.txt")
    with open(concat_list_file, "w", encoding="utf-8") as f:
        for vp in scene_video_paths:
            # Concat demuxer requires absolute path or path relative to concat_scenes.txt itself
            abs_path = os.path.abspath(vp).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_file,
        "-c", "copy",
        output_path
    ]
    log("FFMPEG", f"Concatenating {len(scene_video_paths)} scene clips...")
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=40, check=True)
    except Exception as e:
        log("FFMPEG", f"Concat copy encountered issue ({e}). Re-encoding with filter_complex concat...")
        inputs = []
        for vp in scene_video_paths:
            inputs.extend(["-i", vp])
        n = len(scene_video_paths)
        filter_str = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vout]"
        cmd_filter = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_str, "-map", "[vout]", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", output_path]
        subprocess.run(cmd_filter, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, check=True)


def concatenate_scene_audios(scene_audio_paths: List[str], output_path: str, temp_dir: str) -> None:
    """Concatenates all individual scene audio clips into master voiceover."""
    concat_list_file = os.path.join(temp_dir, "concat_audio.txt")
    with open(concat_list_file, "w", encoding="utf-8") as f:
        for ap in scene_audio_paths:
            abs_path = os.path.abspath(ap).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_file,
        "-c:a", "libmp3lame",
        "-q:a", "2",
        output_path
    ]
    log("FFMPEG", "Concatenating scene audio tracks...")
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30, check=True)
    except Exception as e:
        log("FFMPEG", f"Audio concat copy failed ({e}). Re-encoding with filter_complex concat...")
        inputs = []
        for ap in scene_audio_paths:
            inputs.extend(["-i", ap])
        n = len(scene_audio_paths)
        filter_str = "".join(f"[{i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1[aout]"
        cmd_filter = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_str, "-map", "[aout]", output_path]
        subprocess.run(cmd_filter, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, check=True)


def assemble_final_tiktok_video(
    visual_video_path: str,
    voiceover_path: str,
    bgm_path: str,
    ass_subtitles_path: str,
    final_output_path: str
) -> None:
    """
    Mixes voiceover narration with background music at -22dB beneath,
    burns in aesthetic .ass subtitles, and encodes to TikTok specifications:
    1080x1920, 30fps, H.264 (yuv420p), AAC 192k, trimmed with -shortest.
    """
    log("FFMPEG", f"Assembling final TikTok video to {final_output_path}...")

    # Format subtitles path for ffmpeg filter syntax
    escaped_sub_path = ass_subtitles_path.replace("\\", "/").replace(":", "\\:")
    subtitle_filter = f"subtitles='{escaped_sub_path}'"

    # Audio mixing filter graph:
    # [0:a] voiceover (volume 1.0)
    # [1:a] looped background music (volume -22dB)
    # amix combines both with duration=first so music stops when narration finishes.
    audio_filter = f"[1:a]volume=1.0[vox];[2:a]volume={BGM_VOLUME_DB}[bg];[vox][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"

    cmd = [
        "ffmpeg", "-y",
        "-i", visual_video_path,
        "-i", voiceover_path,
        "-stream_loop", "-1",
        "-i", bgm_path,
        "-filter_complex", f"[0:v]{subtitle_filter}[vout];{audio_filter}",
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.2",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-r", str(TARGET_FPS),
        "-shortest",
        final_output_path
    ]

    log("FFMPEG", "Executing final encode & subtitle burn-in...")
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        log("FFMPEG", "Final render timed out after 180s. Retrying without subtitle burn-in as fallback...")
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", visual_video_path,
            "-i", voiceover_path,
            "-stream_loop", "-1",
            "-i", bgm_path,
            "-filter_complex", audio_filter,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "44100",
            "-shortest",
            final_output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, check=True)
        return
    if result.returncode != 0:
        # Fallback if subtitle filter syntax had issue with fonts
        log("FFMPEG", f"Primary render returned non-zero ({result.stderr[:200]}). Retrying with alternate subtitle syntax...")
        cmd_fallback = [
            "ffmpeg", "-y",
            "-i", visual_video_path,
            "-i", voiceover_path,
            "-stream_loop", "-1",
            "-i", bgm_path,
            "-filter_complex", f"[0:v]ass='{escaped_sub_path}'[vout];{audio_filter}",
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "44100",
            "-shortest",
            final_output_path
        ]
        subprocess.run(cmd_fallback, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

    log("FFMPEG", f"Final video render complete: {final_output_path}")


# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================

async def run_pipeline(
    recipe_topic: Optional[str] = None,
    output_dir: str = "output",
    history_file: str = "history.json",
    temp_dir: str = "temp",
    bgm_path: str = "assets/audio/lofi_acoustic_loop.mp3"
) -> str:
    """Executes the complete autonomous video generation lifecycle."""
    start_time = time.time()
    log("PIPELINE", "=== STARTING AUTONOMOUS TIKTOK VIDEO PIPELINE ===")

    # Prepare directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(bgm_path), exist_ok=True)

    # Resolve API keys from environment
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    pollinations_key = os.environ.get("POLLINATIONS_API_KEY", "").strip()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    pexels_key = os.environ.get("PEXELS_API_KEY", "").strip()

    if not gemini_key:
        raise ValueError("GEMINI_API_KEY environment variable is required.")

    # Step 1: History Management & Storyboard Brainstorming
    history_titles = load_history(history_file)
    project = brainstorm_storyboard(
        gemini_api_key=gemini_key,
        history_titles=history_titles,
        user_topic=recipe_topic
    )

    # Step 2: Ensure Background Music
    actual_bgm_path = ensure_lofi_background_music(bgm_path)

    # Step 3: Scene-by-Scene Audio & Visual Generation
    all_word_timings: List[WordTiming] = []
    cumulative_time = 0.0
    scene_video_paths: List[str] = []
    scene_audio_paths: List[str] = []

    for idx, scene in enumerate(project.scenes):
        log("PIPELINE", f"--- Processing Scene {scene.scene_number}: {scene.scene_name} ---")

        # Synthesize audio with Edge-TTS
        audio_filename = os.path.join(temp_dir, f"scene_{scene.scene_number}.mp3")
        duration, word_timings = await synthesize_scene_audio(scene.narration, audio_filename)
        scene.duration = max(3.5, duration)
        scene.audio_path = audio_filename
        scene_audio_paths.append(audio_filename)

        # Offset word boundaries by cumulative time
        for wt in word_timings:
            all_word_timings.append(WordTiming(
                word=wt.word,
                start=wt.start + cumulative_time,
                end=wt.end + cumulative_time
            ))
        cumulative_time += scene.duration

        # Generate Dynamic Scene Video Clip via Pollinations.ai Engine
        video_filename = os.path.join(temp_dir, f"scene_{scene.scene_number}.mp4")
        clip_created = generate_scene_clip_from_pollinations(
            scene=scene,
            recipe_title=project.recipe_title,
            output_path=video_filename,
            target_duration=scene.duration,
            temp_dir=temp_dir,
            pollinations_api_key=pollinations_key
        )

        if not clip_created:
            log("PIPELINE", f"Pollinations clip generation failed for Scene {scene.scene_number}. Checking secondary fallbacks...")
            if pexels_key and fetch_scene_video_clip(scene, project.recipe_title, video_filename, scene.duration, pexels_key):
                clip_created = True

        if not clip_created:
            log("PIPELINE", f"Falling back to visual waterfall + Ken Burns motion for Scene {scene.scene_number}...")
            image_filename = os.path.join(temp_dir, f"scene_{scene.scene_number}.png")
            generate_scene_visual(scene, project.recipe_title, image_filename, hf_token, pexels_key)
            scene.image_path = image_filename
            render_ken_burns_scene(image_filename, scene.duration, scene.scene_number, video_filename)

        scene.video_path = video_filename
        scene_video_paths.append(video_filename)

    project.total_duration = cumulative_time
    log("PIPELINE", f"Total scenes audio duration: {cumulative_time:.2f} seconds.")

    # Step 4: Subtitles (.ass)
    ass_path = os.path.join(temp_dir, "subtitles.ass")
    build_ass_subtitles(all_word_timings, ass_path)

    # Step 5: Video & Audio Assembly
    combined_video_path = os.path.join(temp_dir, "scenes_combined.mp4")
    concatenate_scenes(scene_video_paths, combined_video_path, temp_dir)

    combined_audio_path = os.path.join(temp_dir, "voiceover_combined.mp3")
    concatenate_scene_audios(scene_audio_paths, combined_audio_path, temp_dir)

    # Final Output File
    final_video_name = f"recipe-{project.recipe_slug}.mp4"
    final_video_path = os.path.join(output_dir, final_video_name)
    project.final_video_path = final_video_path

    assemble_final_tiktok_video(
        visual_video_path=combined_video_path,
        voiceover_path=combined_audio_path,
        bgm_path=actual_bgm_path,
        ass_subtitles_path=ass_path,
        final_output_path=final_video_path
    )

    # Step 6: Record History
    record_history(history_file, project.recipe_title, project.recipe_slug)

    # Step 7: Generate Execution Summary Report
    elapsed = time.time() - start_time
    summary_path = os.path.join(output_dir, "generation_summary.md")
    write_summary_report(project, elapsed, summary_path)

    log("PIPELINE", f"=== PIPELINE COMPLETE: {final_video_path} (Took {elapsed:.1f}s) ===")
    return final_video_path


def write_summary_report(project: VideoProject, elapsed_seconds: float, output_path: str) -> None:
    """Generates an aesthetic Markdown report for GitHub Actions step summary."""
    file_size_mb = 0.0
    if os.path.exists(project.final_video_path):
        file_size_mb = os.path.getsize(project.final_video_path) / (1024 * 1024)

    lines = [
        f"## 🧁 Recipe: {project.recipe_title}",
        f"- **Slug / Filename**: `{os.path.basename(project.final_video_path)}`",
        f"- **Total Duration**: {project.total_duration:.1f} seconds (Target: ~38–40s)",
        f"- **Total Narration Word Count**: {project.total_words} words (Target: 85–95 words)",
        f"- **Output Size**: {file_size_mb:.2f} MB",
        f"- **Resolution & Specs**: 1080x1920 (9:16 vertical), 30 fps, H.264 / AAC",
        f"- **Pipeline Execution Time**: {elapsed_seconds:.1f} seconds",
        "",
        "### 📜 Storyboard Breakdown",
        "| Scene | Concept | Narration Script | Visual Shots (Pollinations.ai Engine) |",
        "|---|---|---|---|"
    ]

    for s in project.scenes:
        clean_narr = s.narration.replace("|", "/")
        primary_shot = s.primary_visual_prompt.split(",")[0].replace("|", "/")
        secondary_shot = s.secondary_visual_prompt.split(",")[0].replace("|", "/")
        shots_desc = f"Shot 1: {primary_shot}<br>Shot 2: {secondary_shot}"
        lines.append(f"| **{s.scene_number}. {s.scene_name}** | {s.scene_name} | {clean_narr} | {shots_desc} |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Autonomous TikTok Recipe Video Generation Engine")
    parser.add_argument("--topic", type=str, default="", help="Optional manual recipe topic. Leave empty to auto-pick.")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for final video artifacts.")
    parser.add_argument("--history-file", type=str, default="history.json", help="Path to history tracking JSON file.")
    parser.add_argument("--temp-dir", type=str, default="temp", help="Directory for temporary video/audio assets.")
    parser.add_argument("--bgm", type=str, default="assets/audio/lofi_acoustic_loop.mp3", help="Background music path.")

    args = parser.parse_args()

    asyncio.run(run_pipeline(
        recipe_topic=args.topic,
        output_dir=args.output_dir,
        history_file=args.history_file,
        temp_dir=args.temp_dir,
        bgm_path=args.bgm
    ))

if __name__ == "__main__":
    main()
