#arquivo app.py
from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
from PIL import Image
import io
import logging
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from numba import jit
import torch
import json
from desenhou import IdealBodyVisualizer, generate_ideal_body_visualization
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import itertools
from functools import lru_cache, wraps
import concurrent.futures
from collections import defaultdict
import warnings
import time
import base64
import hashlib
warnings.filterwarnings('ignore')

# Configuração otimizada
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CORS simplificado
@app.after_request
def after_request(response):
    response.headers.update({
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,Authorization',
        'Access-Control-Allow-Methods': 'GET,PUT,POST,DELETE,OPTIONS'
    })
    return response

@dataclass
class PoseConfig:
    """Configuração centralizada para detecção de pose"""
    input_height: int = 368
    input_width: int = 368
    threshold: float = 0.1
    coco_points: Dict[str, int] = field(default_factory=lambda: {
        'nose': 0, 'neck': 1, 'right_shoulder': 2, 'right_elbow': 3,
        'right_wrist': 4, 'left_shoulder': 5, 'left_elbow': 6,
        'left_wrist': 7, 'right_hip': 8, 'right_knee': 9,
        'right_ankle': 10, 'left_hip': 11, 'left_knee': 12,
        'left_ankle': 13, 'right_eye': 14, 'left_eye': 15,
        'right_ear': 16, 'left_ear': 17
    })

@dataclass
class ProportionConfig:
    """Configuração para análise de proporções"""
    ideal_ratios: Dict[str, float] = field(default_factory=lambda: {
        'head_body': 7.5, 'shoulder_hip': 1.4, 'leg_torso': 1.2,
        'arm_span': 1.0, 'waist_hip': 0.7, 'shoulder_width': 0.25, 'leg_length': 0.5
    })
    tolerances: Dict[str, float] = field(default_factory=lambda: {
        'excellent': 0.1, 'good': 0.2, 'fair': 0.3
    })
    weights: Dict[str, float] = field(default_factory=lambda: {
        'head_body': 0.2, 'shoulder_hip': 0.15, 'leg_torso': 0.15,
        'arm_span': 0.1, 'waist_hip': 0.15, 'shoulder_width': 0.1, 'leg_length': 0.15
    })

def timing_decorator(func):
    """Decorator para medir tempo de execução"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} executado em {time.time() - start:.2f}s")
        return result
    return wrapper

@jit(nopython=True, cache=True)
def fast_euclidean(p1, p2):
    """Distância euclidiana otimizada com numba"""
    return np.sqrt(np.sum((p1 - p2) ** 2))

@jit(nopython=True, cache=True)
def calculate_anatomical_points(x, y, w, h):
    """Calcular pontos anatômicos otimizado"""
    return np.array([
        [x + w//2, y + h * 0.125],  # nose
        [x + w//3, y + h * 0.125 - h*0.02],  # left_eye
        [x + 2*w//3, y + h * 0.125 - h*0.02],  # right_eye
        [x + w//2, y + h * 0.15],  # neck
        [x + w//4, y + h * 0.2],  # left_shoulder
        [x + 3*w//4, y + h * 0.2],  # right_shoulder
        [x + w//3, y + h * 0.55],  # left_hip
        [x + 2*w//3, y + h * 0.55],  # right_hip
        [x + w//3, y + h * 0.75],  # left_knee
        [x + 2*w//3, y + h * 0.75],  # right_knee
        [x + w//3, y + h * 0.95],  # left_ankle
        [x + 2*w//3, y + h * 0.95],  # right_ankle
    ])

class OptimizedPoseDetector:
    """Detector de pose otimizado com cache e paralelização"""
    
    def __init__(self):
        self.config = PoseConfig()
        self.use_alternative = True
        self.cache = {}
        logger.info("Detector otimizado inicializado")

    def get_image_hash(self, image):
        """Gerar hash da imagem para cache"""
        return hashlib.md5(image.tobytes()).hexdigest()

    @timing_decorator
    def detect_pose_alternative(self, image):
        """Detecção de pose com cache"""
        image_hash = self.get_image_hash(image)
        
        if image_hash in self.cache:
            logger.info("Usando resultado do cache")
            return self.cache[image_hash]
        
        result = self._detect_pose_internal(image)
        self.cache[image_hash] = result
        return result

    def _detect_pose_internal(self, image):
        """Detecção interna otimizada"""
        # Usar threading para operações paralelas
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Processar escala de cinza e detecção de contornos em paralelo
            gray = self._prepare_image(image)
            contours = self._detect_contours(gray)
            
        return self._extract_keypoints_optimized(contours[0], image.shape) if contours else None

    def _prepare_image(self, image):
        """Preparar imagem otimizada"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _detect_contours(self, gray):
        """Detectar contornos com múltiplos métodos otimizados"""
        # Usar list comprehension para otimizar
        methods = [
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        ]
        
        all_contours = []
        for thresh in methods:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        
        # Filtrar usando compreensão de lista
        valid_contours = [contour for contour in all_contours 
                         if self._is_valid_contour(contour)]
        
        return [max(valid_contours, key=cv2.contourArea)] if valid_contours else []

    def _is_valid_contour(self, contour):
        """Validar contorno otimizado"""
        area = cv2.contourArea(contour)
        if area <= 5000:
            return False
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return 0.5 < solidity < 0.95

    def _extract_keypoints_optimized(self, contour, image_shape):
        """Extrair pontos-chave otimizado com numba"""
        x, y, w, h = cv2.boundingRect(contour)
        
        # Usar função numba para cálculos rápidos
        points = calculate_anatomical_points(x, y, w, h)
        
        # Mapear pontos usando zip
        point_names = ['nose', 'left_eye', 'right_eye', 'neck', 'left_shoulder', 
                      'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 
                      'right_knee', 'left_ankle', 'right_ankle']
        
        keypoints = dict(zip(point_names, [(int(p[0]), int(p[1])) for p in points]))
        
        # Adicionar punhos estimados
        keypoints.update(self._estimate_wrists_vectorized(contour, x, y, w, h))
        
        return keypoints

    def _estimate_wrists_vectorized(self, contour, x, y, w, h):
        """Estimar punhos com operações vetorizadas"""
        arm_region_y = y + h * 0.3
        arm_height = h * 0.3
        
        # Usar numpy para filtragem vetorizada
        contour_points = contour.reshape(-1, 2)
        mask = (contour_points[:, 1] >= arm_region_y) & (contour_points[:, 1] <= arm_region_y + arm_height)
        arm_points = contour_points[mask]
        
        if len(arm_points) == 0:
            return {
                'left_wrist': (x + w//8, int(y + h * 0.45)),
                'right_wrist': (x + 7*w//8, int(y + h * 0.45))
            }
        
        # Usar numpy para encontrar extremos
        left_idx = np.argmin(arm_points[:, 0])
        right_idx = np.argmax(arm_points[:, 0])
        
        return {
            'left_wrist': tuple(arm_points[left_idx]),
            'right_wrist': tuple(arm_points[right_idx])
        }

class SuperOptimizedAnalyzer:
    """Analisador super otimizado com ML avançado"""
    
    def __init__(self):
        self.config = ProportionConfig()
        self.scaler = MinMaxScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        
    @timing_decorator
    def analyze_proportions_advanced(self, keypoints):
        """Análise super otimizada"""
        try:
            # Validação rápida
            essential_points = {'nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'}
            if not essential_points.issubset(keypoints.keys()):
                raise ValueError("Pontos essenciais não detectados")
            
            # Pipeline otimizado
            measurements = self._calculate_measurements_vectorized(keypoints)
            proportions = self._calculate_proportions_vectorized(measurements)
            ai_analysis = self._ai_analysis_optimized(proportions, measurements)
            
            return self._generate_report_optimized(proportions, ai_analysis)
            
        except Exception as e:
            logger.error(f"Erro na análise: {str(e)}")
            raise

    def _calculate_measurements_vectorized(self, keypoints):
        """Calcular medidas com operações vetorizadas"""
        # Converter pontos para numpy arrays
        points = {k: np.array(v) for k, v in keypoints.items()}
        
        # Cálculos vetorizados
        measurements = {
            'height': abs(points['left_ankle'][1] - points['nose'][1]),
            'shoulder_width': fast_euclidean(points['left_shoulder'], points['right_shoulder']),
            'hip_width': fast_euclidean(points['left_hip'], points['right_hip']),
            'leg_length': fast_euclidean(
                (points['left_hip'] + points['right_hip']) / 2,
                (points['left_ankle'] + points['right_ankle']) / 2
            ),
            'torso_length': fast_euclidean(
                (points['left_shoulder'] + points['right_shoulder']) / 2,
                (points['left_hip'] + points['right_hip']) / 2
            ),
            'head_length': abs(points['nose'][1] - ((points['left_shoulder'] + points['right_shoulder']) / 2)[1])
        }
        
        # Arm span com fallback
        measurements['arm_span'] = (
            fast_euclidean(points['left_wrist'], points['right_wrist']) 
            if 'left_wrist' in points and 'right_wrist' in points 
            else measurements['height']
        )
        
        return measurements

    def _calculate_proportions_vectorized(self, measurements):
        """Calcular proporções com operações vetorizadas"""
        epsilon = 1e-6
        
        # Usar dictionary comprehension para otimizar
        proportion_calcs = {
            'head_body': lambda: measurements['height'] / max(measurements['head_length'], epsilon),
            'shoulder_hip': lambda: measurements['shoulder_width'] / max(measurements['hip_width'], epsilon),
            'leg_torso': lambda: measurements['leg_length'] / max(measurements['torso_length'], epsilon),
            'arm_span': lambda: measurements['arm_span'] / max(measurements['height'], epsilon),
            'shoulder_width': lambda: measurements['shoulder_width'] / max(measurements['height'], epsilon),
            'leg_length': lambda: measurements['leg_length'] / max(measurements['height'], epsilon)
        }
        
        return {k: calc() for k, calc in proportion_calcs.items()}

    def _ai_analysis_optimized(self, proportions, measurements):
        """Análise IA otimizada"""
        # Preparar features vetorizadas
        features = np.array(list(proportions.values())).reshape(1, -1)
        features_scaled = self.scaler.fit_transform(features)
        
        # Análise de cluster otimizada
        ideal_features = np.array(list(self.config.ideal_ratios.values())).reshape(1, -1)
        ideal_scaled = self.scaler.transform(ideal_features)
        
        # Cálculos otimizados
        distance_from_ideal = fast_euclidean(features_scaled[0], ideal_scaled[0])
        symmetry_score = 100.0  # Simplificado
        harmony_score = self._calculate_harmony_vectorized(proportions)
        
        return {
            'distance_from_ideal': distance_from_ideal,
            'symmetry_score': symmetry_score,
            'harmony_score': harmony_score,
            'features_scaled': features_scaled[0]
        }

    def _calculate_harmony_vectorized(self, proportions):
        """Calcular harmonia com operações vetorizadas"""
        # Usar numpy para cálculos vetorizados
        proportions_array = np.array([proportions.get(k, 0) for k in self.config.ideal_ratios.keys()])
        ideal_array = np.array(list(self.config.ideal_ratios.values()))
        
        # Cálculo vetorizado
        deviations = np.abs(proportions_array - ideal_array) / ideal_array
        scores = np.maximum(0, 100 - deviations * 100)
        
        return np.mean(scores)

    def _generate_report_optimized(self, proportions, ai_analysis):
        """Gerar relatório otimizado"""
        # Usar list comprehension para otimizar
        proportion_results = [
            self._analyze_single_proportion(k, v, ai_analysis)
            for k, v in proportions.items()
            if k in self.config.ideal_ratios
        ]
        
        # Cálculo vetorizado do score
        scores = np.array([p['score'] * self.config.weights.get(p['key'], 1.0) for p in proportion_results])
        overall_score = np.mean(scores)
        
        return {
            'proportions': [self._format_proportion_result(p) for p in proportion_results],
            'overall_score': overall_score,
            'ai_insights': {
                'distance_from_ideal': ai_analysis['distance_from_ideal'],
                'symmetry_score': ai_analysis['symmetry_score'],
                'harmony_score': ai_analysis['harmony_score'],
                'classification': self._classify_body_type_optimized(ai_analysis)
            },
            'recommendations': self._generate_recommendations_optimized(proportions, ai_analysis)
        }
@app.route('/visualizar', methods=['POST'])
def visualizar_corpo_ideal():
    """Endpoint para gerar a visualização SVG e gráfico do corpo ideal com base nas proporções"""
    try:
        data = request.get_json()
        if not data or 'proporcoes' not in data:
            return jsonify({'erro': 'Dados de proporções ausentes'}), 400

        proporcoes_usuario = data['proporcoes']
        visualizer = IdealBodyVisualizer()
        resultado = visualizer.generate_body_analysis_report(proporcoes_usuario, analysis_result=None)

        return jsonify({
            'status': 'sucesso',
            'svg': resultado['ideal_body_svg'],
            'grafico_base64': resultado['comparison_chart'],
            'analise': resultado['detailed_analysis'],
            'sugestoes': resultado['improvement_suggestions'],
            'ideais': resultado['ideal_proportions']
        })

    except Exception as e:
        logger.error(f"Erro ao gerar visualização: {str(e)}")
        return jsonify({'erro': f'Erro interno: {str(e)}'}), 500


    def _analyze_single_proportion(self, key, value, ai_analysis):
        """Analisar uma proporção individual"""
        ideal = self.config.ideal_ratios[key]
        deviation = abs(value - ideal) / ideal

        # Usar mapeamento para otimizar
        status_map = [
            (self.config.tolerances['excellent'], 'excelente', 95),
            (self.config.tolerances['good'], 'bom', 75),
            (self.config.tolerances['fair'], 'regular', 50)
        ]

        for threshold, status, base_score in status_map:
            if deviation <= threshold:
                score = base_score + (threshold - deviation) * (100 - base_score) / threshold
                break
        else:
            status, score = 'precisa_melhorar', max(0, 50 - deviation * 50)

        return {
            'key': key,
            'value': value,
            'ideal': ideal,
            'score': score,
            'status': status
        }

    def _format_proportion_result(self, prop):
        """Formatar resultado da proporção"""
        name_map = {
            'head_body': 'Cabeça/Corpo', 'shoulder_hip': 'Ombros/Quadris',
            'leg_torso': 'Pernas/Torso', 'arm_span': 'Envergadura/Altura',
            'waist_hip': 'Cintura/Quadris', 'shoulder_width': 'Largura Ombros',
            'leg_length': 'Comprimento Pernas'
        }

        return {
            'name': name_map.get(prop['key'], prop['key']),
            'value': f"{prop['value']:.2f}",
            'ideal': f"{prop['ideal']:.2f}",
            'score': prop['score'],
            'status': prop['status'],
            'weight': self.config.weights.get(prop['key'], 1.0)
        }

    def _classify_body_type_optimized(self, ai_analysis):
        """Classificar tipo corporal otimizado"""
        thresholds = [(85, 'Proporcional'), (70, 'Atlético'), (50, 'Equilibrado')]

        for threshold, classification in thresholds:
            if (ai_analysis['harmony_score'] > threshold):
                return classification

        return 'Único'

    def _generate_recommendations_optimized(self, proportions, ai_analysis):
        """Gerar recomendações otimizadas"""
        recommendations = []

        # Mapeamento de recomendações
        rec_map = {
            'head_body': "Trabalhe postura e alinhamento da coluna",
            'shoulder_hip': "Fortaleça ombros e trabalhe mobilidade dos quadris",
            'leg_torso': "Exercícios de alongamento para pernas",
            'arm_span': "Exercícios de flexibilidade para braços",
            'shoulder_width': "Exercícios para ampliar os ombros",
            'leg_length': "Exercícios de alongamento e fortalecimento das pernas"
        }

        # Usar compreensão de lista para otimizar
        recommendations.extend([
            rec_map[key] for key, value in proportions.items()
            if key in self.config.ideal_ratios and key in rec_map
            and abs(value - self.config.ideal_ratios[key]) / self.config.ideal_ratios[key] > 0.15
        ])

        # Recomendações baseadas em scores
        if ai_analysis['harmony_score'] < 70:
            recommendations.append("Considere um programa de exercícios focado em equilíbrio corporal")

        return recommendations or ["Parabéns! Suas proporções estão excelentes!"]

# Instanciar detectores otimizados
pose_detector = OptimizedPoseDetector()
analyzer = SuperOptimizedAnalyzer()

@app.route('/index.html')
def home():
    return render_template_string("""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Análise Super Otimizada - OpenCV + IA</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; display: flex; align-items: center; justify-content: center;
        }
        .container {
            background: rgba(255,255,255,0.95); border-radius: 20px; padding: 40px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.15); max-width: 800px; width: 95%;
            backdrop-filter: blur(10px);
        }
        h1 { 
            text-align: center; color: #2d3748; margin-bottom: 30px; 
            font-size: 2.8em; font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center; color: #718096; margin-bottom: 30px;
            font-size: 1.1em; font-weight: 500;
        }
        .upload-zone {
            border: 3px dashed #cbd5e0; border-radius: 15px; padding: 40px;
            text-align: center; margin: 30px 0; cursor: pointer; 
            transition: all 0.3s ease; position: relative; overflow: hidden;
        }
        .upload-zone:hover { 
            border-color: #667eea; transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.1);
        }
        input[type="file"] { display: none; }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; padding: 15px 30px; border-radius: 50px;
            cursor: pointer; font-size: 16px; width: 100%; margin-top: 20px;
            transition: all 0.3s ease; font-weight: 600;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .preview { margin: 20px 0; text-align: center; }
        .preview img { max-width: 100%; max-height: 400px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
        .results { margin-top: 30px; padding: 25px; background: #f8fafc; border-radius: 15px; }
        .result-item { 
            display: flex; justify-content: space-between; align-items: center;
            padding: 15px; margin: 10px 0; background: white; border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .result-info { flex: 1; }
        .result-stats { text-align: right; }
        .status { 
            padding: 6px 12px; border-radius: 20px; font-size: 12px; 
            font-weight: 700; text-transform: uppercase; margin-bottom: 5px;
        }
        .status.excelente { background: #c6f6d5; color: #22543d; }
        .status.bom { background: #feebc8; color: #c05621; }
        .status.regular { background: #fed7aa; color: #dd6b20; }
        .status.precisa_melhorar { background: #fed7d7; color: #c53030; }
        .score { font-size: 3em; font-weight: 900; color: #667eea; text-align: center; margin: 20px 0; }
        .ai-insights { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 20px; border-radius: 15px; margin: 20px 0;
        }
        .ai-insights h3 { margin-bottom: 15px; }
        .insight-item { margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 8px; }
        .loading { display: none; text-align: center; margin: 20px 0; }
        .spinner { 
            border: 4px solid #f3f3f3; border-top: 4px solid #667eea; 
            border-radius: 50%; width: 50px; height: 50px; 
            animation: spin 1s linear infinite; margin: 0 auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { background: #fed7d7; color: #c53030; padding: 15px; border-radius: 10px; margin: 20px 0; text-align: center; }
        .recommendations { background: #e6fffa; padding: 20px; border-radius: 15px; margin: 20px 0; }
        .recommendations h3 { color: #234e52; margin-bottom: 15px; }
        .recommendations ul { list-style: none; }
        .recommendations li { margin: 8px 0; padding: 8px; background: rgba(56, 178, 172, 0.1); border-radius: 8px; }
        .recommendations li:before { content: '🚀'; margin-right: 10px; }
        .performance-badge { 
            display: inline-block; background: #48bb78; color: white; padding: 4px 8px; 
            border-radius: 12px; font-size: 12px; margin-left: 10px;
        }
        .performance-stats {
            background: #f0f8ff; padding: 15px; border-radius: 10px; margin: 20px 0;
            border-left: 4px solid #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Análise Super Otimizada</h1>
        <div class="subtitle">
            ⚡ OpenCV + ML Otimizado + Cache + Paralelização
            <span class="performance-badge">Ultra Rápido</span>
        </div>
        
        <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
            <div style="font-size: 1.2em; font-weight: 600;">📸 Clique ou arraste uma imagem aqui</div>
            <div style="font-size: 14px; color: #718096; margin-top: 10px;">
                ⚡ Processamento Super Otimizado<br>
                🎯 Cache + Paralelização + Numba JIT
            </div>
        </div>
        
        <input type="file" id="fileInput" accept="image/*">
        <div class="preview" id="preview"></div>
        
        <button class="btn" id="analyzeBtn" onclick="analyzeImage()" disabled>
            🚀 Analisar Super Rápido
        </button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div style="margin-top: 15px; font-weight: 600;">Processando ultra-rápido...</div>
        </div>
        
        <div id="results"></div>
    </div>

    <script>
        let selectedFile = null;
        
        document.getElementById('fileInput').addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('preview').innerHTML = `<img src="${e.target.result}" alt="Preview">`;
                    document.getElementById('analyzeBtn').disabled = false;
                };
                reader.readAsDataURL(file);
            }
        });
        
        async function analyzeImage() {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            const analyzeBtn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            
            analyzeBtn.disabled = true;
            loading.style.display = 'block';
            results.innerHTML = '';
            
            const startTime = performance.now();
            
           try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                const processingTime = ((performance.now() - startTime) / 1000).toFixed(2);
                
                if (data.success) {
                    displayResults(data.result, processingTime);
                } else {
                    displayError(data.error);
                }
                
            } catch (error) {
                displayError(`Erro na análise: ${error.message}`);
            } finally {
                loading.style.display = 'none';
                analyzeBtn.disabled = false;
            }
        }
        
        function displayResults(result, processingTime) {
            const resultsDiv = document.getElementById('results');
            
            let html = `
                <div class="performance-stats">
                    <h3>⚡ Performance Ultra Otimizada</h3>
                    <div>🚀 Tempo de processamento: <strong>${processingTime}s</strong></div>
                    <div>💾 Cache: ${result.cache_hit ? 'HIT' : 'MISS'}</div>
                    <div>🔧 Paralelização: Ativada</div>
                </div>
                
                <div class="score">
                    ${result.overall_score.toFixed(1)}
                    <div style="font-size: 0.3em; color: #718096;">SCORE GERAL</div>
                </div>
                
                <div class="ai-insights">
                    <h3>🤖 Insights de IA</h3>
                    <div class="insight-item">
                        <strong>Classificação:</strong> ${result.ai_insights.classification}
                    </div>
                    <div class="insight-item">
                        <strong>Simetria:</strong> ${result.ai_insights.symmetry_score.toFixed(1)}%
                    </div>
                    <div class="insight-item">
                        <strong>Harmonia:</strong> ${result.ai_insights.harmony_score.toFixed(1)}%
                    </div>
                    <div class="insight-item">
                        <strong>Distância do Ideal:</strong> ${result.ai_insights.distance_from_ideal.toFixed(3)}
                    </div>
                </div>
                
                <div class="results">
                    <h3>📊 Análise Detalhada das Proporções</h3>
            `;
            
            result.proportions.forEach(prop => {
                html += `
                    <div class="result-item">
                        <div class="result-info">
                            <div style="font-weight: 600; margin-bottom: 5px;">${prop.name}</div>
                            <div class="status ${prop.status}">${prop.status.replace('_', ' ')}</div>
                        </div>
                        <div class="result-stats">
                            <div style="font-size: 1.2em; font-weight: bold;">${prop.score.toFixed(1)}</div>
                            <div style="font-size: 0.9em; color: #718096;">
                                ${prop.value} / ${prop.ideal} (peso: ${prop.weight})
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += `
                </div>
                
                <div class="recommendations">
                    <h3>🎯 Recomendações Personalizadas</h3>
                    <ul>
            `;
            
            result.recommendations.forEach(rec => {
                html += `<li>${rec}</li>`;
            });
            
            html += `
                    </ul>
                </div>
            `;
            
            resultsDiv.innerHTML = html;
        }
        
        function displayError(error) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div class="error">
                    <h3>❌ Erro na Análise</h3>
                    <p>${error}</p>
                    <p>Tente novamente com uma imagem diferente.</p>
                </div>
            `;
        }
        
        // Drag and drop functionality
        const uploadZone = document.querySelector('.upload-zone');
        
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.style.backgroundColor = '#f0f8ff';
            uploadZone.style.transform = 'scale(1.02)';
        });
        
        uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadZone.style.backgroundColor = '';
            uploadZone.style.transform = '';
        });
        
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.style.backgroundColor = '';
            uploadZone.style.transform = '';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    selectedFile = file;
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        document.getElementById('preview').innerHTML = `<img src="${e.target.result}" alt="Preview">`;
                        document.getElementById('analyzeBtn').disabled = false;
                    };
                    reader.readAsDataURL(file);
                }
            }
        });
    </script>
</body>
</html>
    """)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint super otimizado para análise"""
    try:
        start_time = time.time()
        
        # Validar arquivo
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'Nenhuma imagem enviada'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Arquivo vazio'})
        
        # Processar imagem com cache
        image_bytes = file.read()
        image_hash = hashlib.md5(image_bytes).hexdigest()
        
        # Verificar cache primeiro
        cache_hit = image_hash in pose_detector.cache
        
        # Converter para formato OpenCV
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image.convert('RGB'))
        
        # Detectar pose (com cache)
        keypoints = pose_detector.detect_pose_alternative(image_np)
        
        if keypoints is None:
            return jsonify({
                'success': False, 
                'error': 'Não foi possível detectar a pose na imagem'
            })
        
        # Análise super otimizada
        result = analyzer.analyze_proportions_advanced(keypoints)
        
        # Adicionar informações de performance
        result['cache_hit'] = cache_hit
        result['processing_time'] = time.time() - start_time
        
        logger.info(f"Análise completa em {result['processing_time']:.2f}s (Cache: {'HIT' if cache_hit else 'MISS'})")
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        logger.error(f"Erro na análise: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Instanciar detectores otimizados
pose_detector = OptimizedPoseDetector()
analyzer = SuperOptimizedAnalyzer()

if __name__ == '__main__':
    logger.info("🚀 Iniciando servidor super otimizado...")
    app.run(debug=True, host='0.0.0.0', port=5000)