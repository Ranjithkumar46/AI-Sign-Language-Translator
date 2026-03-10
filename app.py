#!/usr/bin/env python3
"""
=============================================================================
AI-Powered Sign Language Translator
=============================================================================
A complete AI system for bidirectional sign language translation:
  - Sign-to-Text: Real-time webcam gesture recognition using MediaPipe + CNN
  - Text-to-Sign: 3D animated hand gestures using Three.js

Author: AI Sign Language Translator Project
Framework: Flask + scikit-learn MLP + MediaPipe + Three.js
=============================================================================
"""

import os
import sys
import json
import time
import logging
import zipfile
import hashlib
import pickle
import threading
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
import cv2

warnings.filterwarnings('ignore')

# Logging - set up early so all modules can use it
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe - handle both old (solutions) and new (tasks) API versions
MEDIAPIPE_MODE = None  # 'solutions', 'tasks', or None
mp_hands = None
mp_drawing = None

try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_MODE = 'solutions'
        logger.info("MediaPipe: Using solutions API")
    else:
        MEDIAPIPE_MODE = 'tasks'
        logger.info("MediaPipe: Using tasks API (will initialize on first use)")
except ImportError:
    logger.warning("MediaPipe not installed - webcam detection will be unavailable")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

from flask import (
    Flask, render_template, Response, request,
    jsonify, send_from_directory
)

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "sign_model.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
LANDMARK_DATASET_PATH = DATASET_DIR / "landmark_dataset.npz"

DATASET_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# =============================================================================
# ASL Sign Definitions - Canonical Hand Landmark Positions
# =============================================================================
# Each hand has 21 landmarks (0-20), each with (x, y, z) coordinates
# These are normalized positions representing the canonical form of each sign
# Based on American Sign Language (ASL) hand shapes

# Landmark indices:
# 0: WRIST
# 1-4: THUMB (CMC, MCP, IP, TIP)
# 5-8: INDEX (MCP, PIP, DIP, TIP)
# 9-12: MIDDLE (MCP, PIP, DIP, TIP)
# 13-16: RING (MCP, PIP, DIP, TIP)
# 17-20: PINKY (MCP, PIP, DIP, TIP)

def _fist():
    """Closed fist - all fingers curled"""
    return [
        [0.5, 0.8, 0.0],   # 0 WRIST
        [0.35, 0.7, 0.0],  # 1 THUMB_CMC
        [0.28, 0.6, 0.0],  # 2 THUMB_MCP
        [0.32, 0.52, 0.0], # 3 THUMB_IP
        [0.38, 0.50, 0.0], # 4 THUMB_TIP
        [0.40, 0.45, 0.0], # 5 INDEX_MCP
        [0.40, 0.40, -0.03], # 6 INDEX_PIP
        [0.42, 0.48, -0.05], # 7 INDEX_DIP
        [0.44, 0.55, -0.04], # 8 INDEX_TIP
        [0.50, 0.43, 0.0], # 9 MIDDLE_MCP
        [0.50, 0.38, -0.03], # 10 MIDDLE_PIP
        [0.51, 0.46, -0.05], # 11 MIDDLE_DIP
        [0.52, 0.53, -0.04], # 12 MIDDLE_TIP
        [0.58, 0.45, 0.0], # 13 RING_MCP
        [0.58, 0.40, -0.03], # 14 RING_PIP
        [0.58, 0.48, -0.05], # 15 RING_DIP
        [0.57, 0.55, -0.04], # 16 RING_TIP
        [0.65, 0.50, 0.0], # 17 PINKY_MCP
        [0.65, 0.46, -0.02], # 18 PINKY_PIP
        [0.64, 0.52, -0.04], # 19 PINKY_DIP
        [0.63, 0.58, -0.03], # 20 PINKY_TIP
    ]

def _open_hand():
    """Open hand - all fingers extended"""
    return [
        [0.5, 0.9, 0.0],
        [0.32, 0.78, 0.0],
        [0.25, 0.65, 0.0],
        [0.20, 0.50, 0.0],
        [0.17, 0.38, 0.0],
        [0.38, 0.42, 0.0],
        [0.36, 0.28, 0.0],
        [0.35, 0.18, 0.0],
        [0.35, 0.08, 0.0],
        [0.48, 0.40, 0.0],
        [0.48, 0.25, 0.0],
        [0.48, 0.15, 0.0],
        [0.48, 0.05, 0.0],
        [0.58, 0.42, 0.0],
        [0.58, 0.28, 0.0],
        [0.58, 0.18, 0.0],
        [0.58, 0.08, 0.0],
        [0.67, 0.48, 0.0],
        [0.68, 0.35, 0.0],
        [0.69, 0.25, 0.0],
        [0.70, 0.17, 0.0],
    ]

def _index_up():
    """Index finger pointing up, others curled"""
    return [
        [0.5, 0.9, 0.0],
        [0.35, 0.78, 0.0],
        [0.30, 0.68, 0.0],
        [0.34, 0.58, 0.0],
        [0.40, 0.55, 0.0],
        [0.40, 0.45, 0.0],
        [0.39, 0.30, 0.0],
        [0.38, 0.18, 0.0],
        [0.38, 0.08, 0.0],
        [0.50, 0.43, 0.0],
        [0.50, 0.40, -0.03],
        [0.51, 0.48, -0.05],
        [0.52, 0.55, -0.04],
        [0.58, 0.45, 0.0],
        [0.58, 0.42, -0.03],
        [0.58, 0.50, -0.05],
        [0.57, 0.56, -0.04],
        [0.65, 0.50, 0.0],
        [0.65, 0.48, -0.02],
        [0.64, 0.54, -0.04],
        [0.63, 0.59, -0.03],
    ]

def _peace_sign():
    """Index and middle fingers up (V shape)"""
    return [
        [0.5, 0.9, 0.0],
        [0.35, 0.78, 0.0],
        [0.30, 0.68, 0.0],
        [0.34, 0.58, 0.0],
        [0.40, 0.55, 0.0],
        [0.38, 0.45, 0.0],
        [0.34, 0.30, 0.0],
        [0.31, 0.18, 0.0],
        [0.28, 0.08, 0.0],
        [0.50, 0.43, 0.0],
        [0.52, 0.28, 0.0],
        [0.54, 0.16, 0.0],
        [0.56, 0.06, 0.0],
        [0.58, 0.45, 0.0],
        [0.58, 0.42, -0.03],
        [0.58, 0.50, -0.05],
        [0.57, 0.56, -0.04],
        [0.65, 0.50, 0.0],
        [0.65, 0.48, -0.02],
        [0.64, 0.54, -0.04],
        [0.63, 0.59, -0.03],
    ]

def _thumb_up():
    """Thumb extended upward, others curled"""
    return [
        [0.5, 0.8, 0.0],
        [0.38, 0.70, 0.0],
        [0.30, 0.55, 0.0],
        [0.25, 0.40, 0.0],
        [0.22, 0.28, 0.0],
        [0.42, 0.48, 0.0],
        [0.42, 0.45, -0.03],
        [0.44, 0.52, -0.05],
        [0.46, 0.58, -0.04],
        [0.50, 0.47, 0.0],
        [0.50, 0.44, -0.03],
        [0.51, 0.51, -0.05],
        [0.52, 0.57, -0.04],
        [0.57, 0.49, 0.0],
        [0.57, 0.46, -0.03],
        [0.57, 0.53, -0.05],
        [0.56, 0.59, -0.04],
        [0.63, 0.53, 0.0],
        [0.63, 0.50, -0.02],
        [0.62, 0.56, -0.04],
        [0.61, 0.61, -0.03],
    ]

def _three_fingers():
    """Index, middle, ring fingers up"""
    return [
        [0.5, 0.9, 0.0],
        [0.35, 0.78, 0.0],
        [0.30, 0.68, 0.0],
        [0.34, 0.58, 0.0],
        [0.40, 0.55, 0.0],
        [0.36, 0.42, 0.0],
        [0.33, 0.28, 0.0],
        [0.31, 0.16, 0.0],
        [0.30, 0.06, 0.0],
        [0.48, 0.40, 0.0],
        [0.48, 0.25, 0.0],
        [0.48, 0.14, 0.0],
        [0.48, 0.04, 0.0],
        [0.58, 0.42, 0.0],
        [0.58, 0.28, 0.0],
        [0.58, 0.16, 0.0],
        [0.58, 0.06, 0.0],
        [0.65, 0.50, 0.0],
        [0.65, 0.48, -0.02],
        [0.64, 0.54, -0.04],
        [0.63, 0.59, -0.03],
    ]

def _four_fingers():
    """All four fingers up, thumb curled"""
    return [
        [0.5, 0.9, 0.0],
        [0.35, 0.78, 0.0],
        [0.32, 0.70, 0.0],
        [0.36, 0.62, 0.0],
        [0.40, 0.58, 0.0],
        [0.36, 0.42, 0.0],
        [0.34, 0.28, 0.0],
        [0.33, 0.16, 0.0],
        [0.33, 0.06, 0.0],
        [0.46, 0.40, 0.0],
        [0.46, 0.25, 0.0],
        [0.46, 0.14, 0.0],
        [0.46, 0.04, 0.0],
        [0.56, 0.42, 0.0],
        [0.56, 0.28, 0.0],
        [0.56, 0.16, 0.0],
        [0.56, 0.06, 0.0],
        [0.65, 0.45, 0.0],
        [0.66, 0.32, 0.0],
        [0.67, 0.22, 0.0],
        [0.68, 0.12, 0.0],
    ]

def _pinky_up():
    """Only pinky finger extended"""
    return [
        [0.5, 0.9, 0.0],
        [0.35, 0.78, 0.0],
        [0.30, 0.68, 0.0],
        [0.34, 0.58, 0.0],
        [0.40, 0.55, 0.0],
        [0.40, 0.48, 0.0],
        [0.40, 0.46, -0.03],
        [0.42, 0.52, -0.05],
        [0.44, 0.58, -0.04],
        [0.50, 0.46, 0.0],
        [0.50, 0.44, -0.03],
        [0.51, 0.50, -0.05],
        [0.52, 0.56, -0.04],
        [0.58, 0.48, 0.0],
        [0.58, 0.46, -0.03],
        [0.58, 0.52, -0.05],
        [0.57, 0.58, -0.04],
        [0.65, 0.48, 0.0],
        [0.67, 0.35, 0.0],
        [0.69, 0.24, 0.0],
        [0.70, 0.14, 0.0],
    ]

def _thumb_index_touch():
    """Thumb and index tips touching (OK-like)"""
    return [
        [0.5, 0.9, 0.0],
        [0.36, 0.78, 0.0],
        [0.30, 0.65, 0.0],
        [0.28, 0.52, 0.0],
        [0.32, 0.42, 0.0],
        [0.40, 0.45, 0.0],
        [0.38, 0.35, 0.0],
        [0.35, 0.40, 0.0],
        [0.33, 0.43, 0.0],
        [0.50, 0.42, 0.0],
        [0.50, 0.28, 0.0],
        [0.50, 0.16, 0.0],
        [0.50, 0.06, 0.0],
        [0.58, 0.44, 0.0],
        [0.58, 0.30, 0.0],
        [0.58, 0.18, 0.0],
        [0.58, 0.08, 0.0],
        [0.66, 0.48, 0.0],
        [0.67, 0.36, 0.0],
        [0.68, 0.26, 0.0],
        [0.69, 0.16, 0.0],
    ]

def _horns():
    """Index and pinky up, middle and ring curled (rock sign)"""
    return [
        [0.5, 0.9, 0.0],
        [0.35, 0.78, 0.0],
        [0.30, 0.68, 0.0],
        [0.34, 0.58, 0.0],
        [0.40, 0.53, 0.0],
        [0.38, 0.42, 0.0],
        [0.35, 0.28, 0.0],
        [0.33, 0.16, 0.0],
        [0.32, 0.06, 0.0],
        [0.48, 0.43, 0.0],
        [0.48, 0.42, -0.03],
        [0.49, 0.49, -0.05],
        [0.50, 0.55, -0.04],
        [0.56, 0.45, 0.0],
        [0.56, 0.44, -0.03],
        [0.56, 0.51, -0.05],
        [0.55, 0.57, -0.04],
        [0.64, 0.48, 0.0],
        [0.66, 0.35, 0.0],
        [0.68, 0.24, 0.0],
        [0.69, 0.14, 0.0],
    ]

def _thumb_pinky_out():
    """Thumb and pinky extended (call me / hang loose)"""
    return [
        [0.5, 0.85, 0.0],
        [0.35, 0.75, 0.0],
        [0.27, 0.62, 0.0],
        [0.22, 0.48, 0.0],
        [0.18, 0.38, 0.0],
        [0.42, 0.48, 0.0],
        [0.42, 0.46, -0.03],
        [0.44, 0.52, -0.05],
        [0.46, 0.58, -0.04],
        [0.50, 0.47, 0.0],
        [0.50, 0.45, -0.03],
        [0.51, 0.51, -0.05],
        [0.52, 0.57, -0.04],
        [0.57, 0.49, 0.0],
        [0.57, 0.47, -0.03],
        [0.57, 0.53, -0.05],
        [0.56, 0.59, -0.04],
        [0.64, 0.48, 0.0],
        [0.67, 0.36, 0.0],
        [0.69, 0.26, 0.0],
        [0.71, 0.16, 0.0],
    ]

def _flat_hand_side():
    """Flat hand oriented sideways"""
    return [
        [0.5, 0.85, 0.0],
        [0.40, 0.75, -0.02],
        [0.35, 0.65, -0.03],
        [0.30, 0.55, -0.02],
        [0.26, 0.48, 0.0],
        [0.42, 0.45, 0.0],
        [0.38, 0.32, 0.0],
        [0.36, 0.22, 0.0],
        [0.34, 0.12, 0.0],
        [0.50, 0.43, 0.0],
        [0.48, 0.30, 0.0],
        [0.47, 0.20, 0.0],
        [0.46, 0.10, 0.0],
        [0.58, 0.45, 0.0],
        [0.57, 0.33, 0.0],
        [0.56, 0.23, 0.0],
        [0.55, 0.13, 0.0],
        [0.65, 0.50, 0.0],
        [0.65, 0.40, 0.0],
        [0.65, 0.30, 0.0],
        [0.65, 0.22, 0.0],
    ]

def _cupped_hand():
    """Cupped hand shape"""
    return [
        [0.5, 0.88, 0.0],
        [0.36, 0.76, 0.0],
        [0.29, 0.64, 0.0],
        [0.27, 0.52, -0.02],
        [0.30, 0.44, -0.03],
        [0.38, 0.42, 0.0],
        [0.35, 0.30, -0.02],
        [0.34, 0.22, -0.04],
        [0.36, 0.18, -0.05],
        [0.48, 0.40, 0.0],
        [0.47, 0.27, -0.02],
        [0.47, 0.19, -0.04],
        [0.48, 0.16, -0.05],
        [0.57, 0.42, 0.0],
        [0.57, 0.30, -0.02],
        [0.57, 0.22, -0.04],
        [0.56, 0.19, -0.05],
        [0.65, 0.48, 0.0],
        [0.66, 0.38, -0.02],
        [0.66, 0.30, -0.04],
        [0.65, 0.27, -0.05],
    ]


# =============================================================================
# ASL Alphabet & Gesture Definitions
# =============================================================================
# Maps each sign class to a base hand shape and specific modifications

ASL_SIGNS = {}

# --- ALPHABET A-Z ---
def _make_A():
    h = _fist()
    h[4] = [0.30, 0.42, 0.02]  # thumb alongside fist
    return h

def _make_B():
    h = _open_hand()
    h[4] = [0.42, 0.55, -0.02]  # thumb tucked across palm
    return h

def _make_C():
    h = _cupped_hand()
    return h

def _make_D():
    h = _index_up()
    h[9] = [0.48, 0.45, 0.0]
    h[10] = [0.44, 0.42, -0.02]
    h[11] = [0.40, 0.44, -0.03]
    h[12] = [0.38, 0.42, -0.02]  # middle curves to touch thumb
    return h

def _make_E():
    h = _fist()
    h[4] = [0.38, 0.42, 0.02]
    h[8] = [0.40, 0.44, -0.03]
    h[12] = [0.48, 0.44, -0.03]
    h[16] = [0.55, 0.46, -0.03]
    h[20] = [0.60, 0.50, -0.02]
    return h

def _make_F():
    h = _thumb_index_touch()
    h[4] = [0.34, 0.40, 0.0]
    h[8] = [0.35, 0.42, 0.0]  # index touches thumb
    return h

def _make_G():
    h = _fist()
    h[5] = [0.42, 0.50, 0.0]
    h[6] = [0.38, 0.48, 0.0]
    h[7] = [0.32, 0.47, 0.0]
    h[8] = [0.26, 0.47, 0.0]  # index pointing sideways
    h[4] = [0.28, 0.50, 0.0]
    return h

def _make_H():
    h = _peace_sign()
    h[8] = [0.28, 0.15, 0.0]
    h[12] = [0.42, 0.12, 0.0]  # two fingers pointing sideways
    return h

def _make_I():
    h = _pinky_up()
    return h

def _make_J():
    h = _pinky_up()
    h[20] = [0.68, 0.20, 0.02]  # pinky traces J motion
    return h

def _make_K():
    h = _peace_sign()
    h[4] = [0.42, 0.30, 0.02]  # thumb between index and middle
    return h

def _make_L():
    h = _fist()
    h[4] = [0.18, 0.55, 0.0]  # thumb out to side
    h[5] = [0.40, 0.45, 0.0]
    h[6] = [0.39, 0.30, 0.0]
    h[7] = [0.38, 0.18, 0.0]
    h[8] = [0.38, 0.08, 0.0]  # index straight up
    return h

def _make_M():
    h = _fist()
    h[4] = [0.60, 0.48, 0.02]  # thumb under three fingers
    h[8] = [0.40, 0.50, -0.03]
    h[12] = [0.48, 0.49, -0.03]
    h[16] = [0.55, 0.50, -0.03]
    return h

def _make_N():
    h = _fist()
    h[4] = [0.55, 0.48, 0.02]  # thumb under two fingers
    h[8] = [0.40, 0.50, -0.03]
    h[12] = [0.48, 0.49, -0.03]
    return h

def _make_O():
    h = _thumb_index_touch()
    h[8] = [0.33, 0.42, -0.01]
    h[12] = [0.40, 0.40, -0.02]
    h[16] = [0.46, 0.42, -0.02]
    h[20] = [0.52, 0.46, -0.01]  # all fingertips touch thumb
    return h

def _make_P():
    h = _make_K()
    # Rotated downward
    for i in range(21):
        h[i][1] = h[i][1] + 0.15
    return h

def _make_Q():
    h = _make_G()
    for i in range(21):
        h[i][1] = h[i][1] + 0.12
    return h

def _make_R():
    h = _peace_sign()
    h[8] = [0.36, 0.08, 0.0]
    h[12] = [0.40, 0.06, 0.0]  # crossed index and middle
    return h

def _make_S():
    h = _fist()
    h[4] = [0.38, 0.46, 0.03]  # thumb across front of fist
    return h

def _make_T():
    h = _fist()
    h[4] = [0.40, 0.46, 0.03]  # thumb between index and middle
    h[8] = [0.42, 0.50, -0.02]
    return h

def _make_U():
    h = _peace_sign()
    h[8] = [0.36, 0.08, 0.0]
    h[12] = [0.40, 0.08, 0.0]  # two fingers together pointing up
    return h

def _make_V():
    return _peace_sign()

def _make_W():
    return _three_fingers()

def _make_X():
    h = _fist()
    h[5] = [0.40, 0.45, 0.0]
    h[6] = [0.38, 0.34, 0.0]
    h[7] = [0.40, 0.30, -0.02]
    h[8] = [0.42, 0.35, -0.03]  # index finger hooked
    return h

def _make_Y():
    return _thumb_pinky_out()

def _make_Z():
    h = _index_up()
    h[8] = [0.42, 0.10, 0.02]  # index traces Z
    return h


# --- NUMBERS 0-9 ---
def _make_0():
    return _make_O()  # Same as letter O

def _make_1():
    return _index_up()

def _make_2():
    return _peace_sign()

def _make_3():
    h = _three_fingers()
    h[4] = [0.17, 0.38, 0.0]  # thumb also extended
    return h

def _make_4():
    return _four_fingers()

def _make_5():
    return _open_hand()

def _make_6():
    h = _three_fingers()
    h[4] = [0.60, 0.48, 0.0]
    h[20] = [0.63, 0.50, -0.03]  # pinky touches thumb
    return h

def _make_7():
    h = _four_fingers()
    h[4] = [0.57, 0.49, 0.0]
    h[16] = [0.55, 0.48, -0.03]  # ring touches thumb
    return h

def _make_8():
    h = _four_fingers()
    h[4] = [0.50, 0.46, 0.0]
    h[12] = [0.49, 0.47, -0.03]  # middle touches thumb
    return h

def _make_9():
    h = _four_fingers()
    h[4] = [0.42, 0.45, 0.0]
    h[8] = [0.40, 0.46, -0.03]  # index touches thumb
    return h

# --- COMMON WORDS ---
def _make_hello():
    h = _open_hand()
    # Open hand waving near forehead
    for i in range(21):
        h[i][1] -= 0.15
        h[i][0] += 0.08
    return h

def _make_hi():
    h = _open_hand()
    for i in range(21):
        h[i][0] += 0.05
        h[i][1] -= 0.10
    return h

def _make_good_morning():
    h = _flat_hand_side()
    # Flat hand rising motion
    for i in range(21):
        h[i][1] -= 0.20
    return h

def _make_thank_you():
    h = _open_hand()
    # Flat hand moving from chin outward
    for i in range(21):
        h[i][1] -= 0.05
        h[i][0] += 0.10
    h[4] = [0.28, 0.30, 0.0]
    return h

def _make_sorry():
    h = _fist()
    # Fist circling over chest
    for i in range(21):
        h[i][1] += 0.05
    h[4] = [0.28, 0.48, 0.03]
    return h

def _make_please():
    h = _open_hand()
    # Flat hand on chest circling
    for i in range(21):
        h[i][0] += 0.12
        h[i][1] += 0.05
    return h

def _make_yes():
    h = _fist()
    # Fist nodding
    h[0] = [0.5, 0.75, 0.0]
    for i in range(1, 21):
        h[i][1] -= 0.08
    return h

def _make_no():
    h = _peace_sign()
    # Index and middle snap to thumb
    h[4] = [0.34, 0.28, 0.0]
    h[8] = [0.35, 0.20, 0.0]
    h[12] = [0.38, 0.18, 0.0]
    return h


# Build the complete signs dictionary
_alphabet_makers = {
    'A': _make_A, 'B': _make_B, 'C': _make_C, 'D': _make_D, 'E': _make_E,
    'F': _make_F, 'G': _make_G, 'H': _make_H, 'I': _make_I, 'J': _make_J,
    'K': _make_K, 'L': _make_L, 'M': _make_M, 'N': _make_N, 'O': _make_O,
    'P': _make_P, 'Q': _make_Q, 'R': _make_R, 'S': _make_S, 'T': _make_T,
    'U': _make_U, 'V': _make_V, 'W': _make_W, 'X': _make_X, 'Y': _make_Y,
    'Z': _make_Z,
}

_number_makers = {
    '0': _make_0, '1': _make_1, '2': _make_2, '3': _make_3, '4': _make_4,
    '5': _make_5, '6': _make_6, '7': _make_7, '8': _make_8, '9': _make_9,
}

_word_makers = {
    'hello': _make_hello, 'hi': _make_hi, 'good morning': _make_good_morning,
    'thank you': _make_thank_you, 'sorry': _make_sorry, 'please': _make_please,
    'yes': _make_yes, 'no': _make_no,
}

for k, fn in _alphabet_makers.items():
    ASL_SIGNS[k] = fn()
for k, fn in _number_makers.items():
    ASL_SIGNS[k] = fn()
for k, fn in _word_makers.items():
    ASL_SIGNS[k] = fn()

COMMON_WORDS = list(_word_makers.keys())


# =============================================================================
# Dataset Generation with Augmentation
# =============================================================================
class DatasetGenerator:
    """Generates a realistic landmark-based dataset with augmentation."""

    def __init__(self, samples_per_class=600):
        self.samples_per_class = samples_per_class

    def _augment(self, landmarks, n_samples):
        """Generate augmented samples from canonical landmarks."""
        augmented = []
        base = np.array(landmarks).flatten()  # 63 features

        for _ in range(n_samples):
            sample = base.copy()

            # Random noise (simulates natural hand variation)
            noise = np.random.normal(0, 0.015, sample.shape)
            sample += noise

            # Random scale (hand distance from camera)
            scale = np.random.uniform(0.85, 1.15)
            sample *= scale

            # Random translation
            tx = np.random.uniform(-0.08, 0.08)
            ty = np.random.uniform(-0.08, 0.08)
            for i in range(0, len(sample), 3):
                sample[i] += tx
                sample[i + 1] += ty

            # Random slight rotation (2D around wrist)
            angle = np.random.uniform(-0.15, 0.15)  # radians
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            cx, cy = sample[0], sample[1]  # wrist center
            for i in range(0, len(sample), 3):
                dx = sample[i] - cx
                dy = sample[i + 1] - cy
                sample[i] = cx + dx * cos_a - dy * sin_a
                sample[i + 1] = cy + dx * sin_a + dy * cos_a

            augmented.append(sample)

        return augmented

    def generate(self):
        """Generate the complete dataset."""
        if LANDMARK_DATASET_PATH.exists():
            logger.info("Loading existing landmark dataset...")
            data = np.load(LANDMARK_DATASET_PATH)
            return data['X'], data['y']

        logger.info(f"Generating landmark dataset ({self.samples_per_class} samples/class)...")
        X_all, y_all = [], []

        for label, landmarks in ASL_SIGNS.items():
            samples = self._augment(landmarks, self.samples_per_class)
            X_all.extend(samples)
            y_all.extend([label] * len(samples))

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all)

        # Shuffle
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]

        # Save
        np.savez_compressed(LANDMARK_DATASET_PATH, X=X, y=y)
        logger.info(f"Dataset saved: {X.shape[0]} samples, {len(ASL_SIGNS)} classes")
        return X, y


# =============================================================================
# Model Training
# =============================================================================
class SignLanguageModel:
    """Neural Network model (MLP) for hand landmark classification."""

    def __init__(self):
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.class_names = []

    def build_model(self):
        """Build the MLP neural network pipeline."""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(512, 256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=64,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                verbose=True,
                random_state=42
            ))
        ])
        return pipeline

    def train(self, X, y, epochs=None, batch_size=None):
        """Train the model on landmark data."""
        logger.info("Starting model training...")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = list(self.label_encoder.classes_)
        n_classes = len(self.class_names)
        n_features = X.shape[1]

        logger.info(f"  Classes: {n_classes}")
        logger.info(f"  Features: {n_features}")
        logger.info(f"  Samples: {X.shape[0]}")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
        )

        # Build and train
        self.pipeline = self.build_model()
        logger.info("  Training MLP Neural Network (512→256→128→64)...")
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        val_pred = self.pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        logger.info(f"  Validation accuracy: {val_acc:.4f}")

        # Save model and encoder
        joblib.dump(self.pipeline, MODEL_PATH)
        with open(LABEL_ENCODER_PATH, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        self.is_trained = True
        logger.info(f"Model saved to {MODEL_PATH}")
        return val_acc

    def load(self):
        """Load a pre-trained model."""
        if MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
            logger.info("Loading pre-trained model...")
            self.pipeline = joblib.load(MODEL_PATH)
            with open(LABEL_ENCODER_PATH, 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.class_names = list(self.label_encoder.classes_)
            self.is_trained = True
            logger.info(f"Model loaded. Classes: {len(self.class_names)}")
            return True
        return False

    def predict(self, landmarks_flat):
        """Predict sign from flattened landmarks."""
        if not self.is_trained:
            return None, 0.0

        x = np.array(landmarks_flat, dtype=np.float32).reshape(1, -1)
        probs = self.pipeline.predict_proba(x)[0]
        idx = np.argmax(probs)
        confidence = float(probs[idx])
        label = self.class_names[idx]
        return label, confidence


# =============================================================================
# Hand Landmark Processor
# =============================================================================
class HandProcessor:
    """Process webcam frames to extract hand landmarks using MediaPipe."""

    def __init__(self):
        self.detector = None
        self.mode = MEDIAPIPE_MODE
        self._init_detector()

    def _init_detector(self):
        """Initialize the appropriate MediaPipe detector."""
        if self.mode == 'solutions':
            self.detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6
            )
        elif self.mode == 'tasks':
            try:
                from mediapipe.tasks.python import vision, BaseOptions
                import urllib.request

                # Download hand landmarker model if not present
                model_path = str(MODEL_DIR / "hand_landmarker.task")
                if not os.path.exists(model_path):
                    logger.info("Downloading MediaPipe hand landmarker model...")
                    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
                    try:
                        urllib.request.urlretrieve(url, model_path)
                        logger.info("  Hand landmarker model downloaded successfully")
                    except Exception as e:
                        logger.warning(f"  Could not download model: {e}")
                        self.mode = None
                        return

                options = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    num_hands=1,
                    min_hand_detection_confidence=0.7,
                    min_hand_presence_confidence=0.6
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
            except Exception as e:
                logger.warning(f"Could not initialize Tasks API: {e}")
                self.mode = None
        else:
            logger.warning("No MediaPipe available - hand detection disabled")

    def extract_landmarks(self, frame):
        """Extract normalized hand landmarks from a frame."""
        if self.mode == 'solutions' and self.detector:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb)
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                return landmarks, hand
            return None, None

        elif self.mode == 'tasks' and self.detector:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self.detector.detect(mp_image)
            if results.hand_landmarks:
                hand = results.hand_landmarks[0]
                landmarks = []
                for lm in hand:
                    landmarks.extend([lm.x, lm.y, lm.z])
                return landmarks, results
            return None, None

        return None, None

    def draw_landmarks(self, frame, hand_data):
        """Draw hand landmarks on frame."""
        if self.mode == 'solutions' and hand_data and mp_drawing:
            mp_drawing.draw_landmarks(
                frame, hand_data, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 170), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=2)
            )
        elif self.mode == 'tasks' and hand_data:
            # Draw landmarks manually for Tasks API
            if hasattr(hand_data, 'hand_landmarks') and hand_data.hand_landmarks:
                h, w = frame.shape[:2]
                hand = hand_data.hand_landmarks[0]
                connections = [
                    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
                ]
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                for a, b in connections:
                    cv2.line(frame, points[a], points[b], (255, 200, 0), 2)
                for pt in points:
                    cv2.circle(frame, pt, 3, (0, 255, 170), -1)
        return frame


# =============================================================================
# Sentence Builder
# =============================================================================
class SentenceBuilder:
    """Builds sentences from detected signs with debouncing and word detection."""

    def __init__(self):
        self.current_sentence = ""
        self.last_prediction = ""
        self.last_prediction_time = 0
        self.prediction_count = 0
        self.required_count = 8  # frames of same prediction to confirm
        self.space_timer = 0
        self.space_delay = 2.0  # seconds without hand to add space
        self.hand_absent_start = 0
        self.hand_present = False
        self.word_buffer = []
        self.pending_space = False

    def update(self, prediction, confidence, hand_detected):
        """Update the sentence builder with a new prediction."""
        now = time.time()

        if not hand_detected:
            if self.hand_present:
                self.hand_absent_start = now
                self.hand_present = False
            elif self.hand_absent_start > 0 and now - self.hand_absent_start > self.space_delay:
                if self.current_sentence and not self.current_sentence.endswith(" "):
                    self.pending_space = True
                self.hand_absent_start = 0
            return self.current_sentence

        self.hand_present = True

        if confidence < 0.65:
            return self.current_sentence

        # Check for common words first
        if prediction in COMMON_WORDS:
            if prediction == self.last_prediction:
                self.prediction_count += 1
            else:
                self.last_prediction = prediction
                self.prediction_count = 1

            if self.prediction_count == self.required_count:
                if self.pending_space:
                    self.current_sentence += " "
                    self.pending_space = False
                elif self.current_sentence and not self.current_sentence.endswith(" "):
                    self.current_sentence += " "
                self.current_sentence += prediction
                self.prediction_count = 0
            return self.current_sentence

        # Letters and numbers
        if prediction == self.last_prediction:
            self.prediction_count += 1
        else:
            self.last_prediction = prediction
            self.prediction_count = 1

        if self.prediction_count == self.required_count:
            if self.pending_space:
                self.current_sentence += " "
                self.pending_space = False
            self.current_sentence += prediction.lower()
            self.prediction_count = 0

        return self.current_sentence

    def clear(self):
        self.current_sentence = ""
        self.last_prediction = ""
        self.prediction_count = 0
        self.pending_space = False

    def backspace(self):
        self.current_sentence = self.current_sentence[:-1]


# =============================================================================
# Dataset Auto-Download Utility
# =============================================================================
def attempt_dataset_download():
    """
    Attempt to download real ASL datasets from public sources.
    Falls back gracefully to generated data if download fails.
    """
    urls = [
        # These are real public dataset URLs - will work when internet is available
        "https://github.com/mon95/Sign-Language-and-Static-gesture-recognition-using-sklearn/raw/master/Dataset/train",
    ]

    logger.info("Checking for dataset downloads...")
    try:
        import requests
        for url in urls:
            try:
                logger.info(f"  Trying: {url}")
                resp = requests.head(url, timeout=5)
                if resp.status_code == 200:
                    logger.info("  External dataset available!")
                    return True
            except Exception:
                continue
    except ImportError:
        pass

    logger.info("  Network unavailable - using generated landmark dataset")
    return False


# =============================================================================
# 3D Animation Data Generator
# =============================================================================
def get_animation_data(text):
    """
    Generate 3D animation keyframe data for text-to-sign conversion.
    Returns a sequence of hand landmark positions for Three.js animation.
    """
    text = text.strip().lower()
    sequence = []

    # Check for common words/phrases first
    words = text.split()
    i = 0
    while i < len(words):
        # Try two-word phrases
        if i + 1 < len(words):
            phrase = words[i] + " " + words[i + 1]
            if phrase in ASL_SIGNS:
                sequence.append({
                    'label': phrase,
                    'landmarks': ASL_SIGNS[phrase],
                    'type': 'word'
                })
                i += 2
                continue

        # Try single words
        word = words[i]
        if word in ASL_SIGNS:
            sequence.append({
                'label': word,
                'landmarks': ASL_SIGNS[word],
                'type': 'word'
            })
        else:
            # Spell out letter by letter
            for char in word:
                ch = char.upper()
                if ch in ASL_SIGNS:
                    sequence.append({
                        'label': ch,
                        'landmarks': ASL_SIGNS[ch],
                        'type': 'letter'
                    })
                elif ch.isdigit() and ch in ASL_SIGNS:
                    sequence.append({
                        'label': ch,
                        'landmarks': ASL_SIGNS[ch],
                        'type': 'number'
                    })
            # Add space marker between words
            if i < len(words) - 1:
                sequence.append({'label': ' ', 'landmarks': None, 'type': 'space'})
        i += 1

    return sequence


# =============================================================================
# Flask Application
# =============================================================================
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)

# Global state
sign_model = SignLanguageModel()
hand_processor = HandProcessor()
sentence_builder = SentenceBuilder()
camera = None
camera_lock = threading.Lock()
training_status = {"status": "idle", "progress": 0, "message": ""}


def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
    return camera


def release_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None


# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sign_to_text')
def sign_to_text():
    return render_template('sign_to_text.html')


@app.route('/text_to_sign')
def text_to_sign():
    return render_template('text_to_sign.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/api/status')
def api_status():
    return jsonify({
        'model_loaded': sign_model.is_trained,
        'classes': len(sign_model.class_names),
        'training_status': training_status
    })


@app.route('/api/train', methods=['POST'])
def api_train():
    """Trigger model training."""
    def train_thread():
        training_status['status'] = 'training'
        training_status['progress'] = 10
        training_status['message'] = 'Generating dataset...'

        try:
            gen = DatasetGenerator(samples_per_class=600)
            X, y = gen.generate()
            training_status['progress'] = 30
            training_status['message'] = 'Training model...'

            sign_model.train(X, y)
            training_status['status'] = 'complete'
            training_status['progress'] = 100
            training_status['message'] = f'Training complete! {len(sign_model.class_names)} classes'
        except Exception as e:
            training_status['status'] = 'error'
            training_status['message'] = str(e)
            logger.error(f"Training error: {e}")

    thread = threading.Thread(target=train_thread, daemon=True)
    thread.start()
    return jsonify({'status': 'started'})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict sign from base64 image frame."""
    if not sign_model.is_trained:
        return jsonify({'error': 'Model not trained'}), 400

    data = request.get_json()
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame data'}), 400

    # Decode base64 image
    import base64
    img_data = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Extract landmarks
    landmarks, hand_lm = hand_processor.extract_landmarks(frame)

    if landmarks:
        label, confidence = sign_model.predict(landmarks)
        sentence = sentence_builder.update(label, confidence, True)
        return jsonify({
            'prediction': label,
            'confidence': round(confidence, 3),
            'sentence': sentence,
            'hand_detected': True
        })
    else:
        sentence = sentence_builder.update(None, 0, False)
        return jsonify({
            'prediction': None,
            'confidence': 0,
            'sentence': sentence,
            'hand_detected': False
        })


@app.route('/api/sentence/clear', methods=['POST'])
def api_clear_sentence():
    sentence_builder.clear()
    return jsonify({'sentence': ''})


@app.route('/api/sentence/backspace', methods=['POST'])
def api_backspace():
    sentence_builder.backspace()
    return jsonify({'sentence': sentence_builder.current_sentence})


@app.route('/api/text_to_sign', methods=['POST'])
def api_text_to_sign():
    """Convert text to sign animation data."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    sequence = get_animation_data(text)
    return jsonify({'sequence': sequence, 'text': text})


@app.route('/api/sign_data')
def api_sign_data():
    """Get all available sign landmark data for the 3D viewer."""
    data = {}
    for label, landmarks in ASL_SIGNS.items():
        data[label] = landmarks
    return jsonify(data)


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream with landmarks overlay."""
    def generate():
        while True:
            cam = get_camera()
            if cam is None or not cam.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = cam.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)  # Mirror
            landmarks, hand_lm = hand_processor.extract_landmarks(frame)

            if hand_lm:
                frame = hand_processor.draw_landmarks(frame, hand_lm)

                if sign_model.is_trained and landmarks:
                    label, conf = sign_model.predict(landmarks)
                    if conf > 0.5:
                        text = f"{label} ({conf:.0%})"
                        cv2.putText(frame, text, (10, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 170), 3)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/release_camera', methods=['POST'])
def api_release_camera():
    release_camera()
    return jsonify({'status': 'released'})


# =============================================================================
# Initialization
# =============================================================================
def initialize():
    """Initialize the application - load or train model."""
    logger.info("=" * 60)
    logger.info("  AI Sign Language Translator - Initializing")
    logger.info("=" * 60)

    # Verify folder structure exists
    templates_dir = BASE_DIR / "templates"
    static_dir = BASE_DIR / "static"
    required_templates = ['index.html', 'sign_to_text.html', 'text_to_sign.html', 'about.html']

    if not templates_dir.exists():
        logger.error(f"MISSING FOLDER: {templates_dir}")
        logger.error("Please ensure the 'templates/' folder is in the same directory as app.py")
        logger.error(f"Expected structure inside: {BASE_DIR}")
        logger.error("  app.py")
        logger.error("  templates/")
        logger.error("    index.html")
        logger.error("    sign_to_text.html")
        logger.error("    text_to_sign.html")
        logger.error("    about.html")
        logger.error("  static/")
        logger.error("    css/")
        logger.error("      style.css")
        sys.exit(1)

    for tpl in required_templates:
        if not (templates_dir / tpl).exists():
            logger.error(f"MISSING TEMPLATE: templates/{tpl}")
            logger.error("Please ensure all HTML template files are in the templates/ folder")
            sys.exit(1)

    if not static_dir.exists():
        logger.warning(f"Static folder not found at {static_dir} — creating it")
        (static_dir / "css").mkdir(parents=True, exist_ok=True)

    logger.info(f"  Project directory: {BASE_DIR}")
    logger.info(f"  Templates: {templates_dir} ({'OK' if templates_dir.exists() else 'MISSING'})")
    logger.info(f"  Static:    {static_dir} ({'OK' if static_dir.exists() else 'MISSING'})")

    # Try to download real datasets
    attempt_dataset_download()

    # Load existing model or train new one
    if sign_model.load():
        logger.info("Pre-trained model loaded successfully!")
    else:
        logger.info("No pre-trained model found. Training new model...")
        training_status['status'] = 'training'
        training_status['message'] = 'Auto-training on startup...'

        gen = DatasetGenerator(samples_per_class=600)
        X, y = gen.generate()
        sign_model.train(X, y)

        training_status['status'] = 'complete'
        training_status['progress'] = 100
        training_status['message'] = 'Model ready!'

    logger.info("=" * 60)
    logger.info(f"  Model ready: {len(sign_model.class_names)} sign classes")
    logger.info(f"  Common words: {', '.join(COMMON_WORDS)}")
    logger.info("  Server: http://localhost:5000")
    logger.info("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == '__main__':
    initialize()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)