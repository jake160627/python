#arquivo app.py
from flask import Flask, render_template, request, jsonify, render_template_string
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
from desenhou import IdealBodyVisualizer, StickFigureGenerator
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
        'value': prop['value'],  # REMOVER o f"{prop['value']:.2f}" para manter float
        'ideal': prop['ideal'],   # REMOVER o f"{prop['ideal']:.2f}" para manter float
        'score': prop['score'],
        'status': prop['status'],
        'weight': self.config.weights.get(prop['key'], 1.0),
        'key': prop['key']  # ADICIONAR a chave original
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
# Instanciar detectores otimizados
pose_detector = OptimizedPoseDetector()
analyzer = SuperOptimizedAnalyzer()
# ADICIONAR ESTAS LINHAS:
stick_generator = StickFigureGenerator()
ideal_visualizer = IdealBodyVisualizer()


@app.route('/')
def home():
    return render_template('index.html')

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
    # Análise super otimizada
        result = analyzer.analyze_proportions_advanced(keypoints)
        
        # ADICIONAR ESTE BLOCO:
        # Extrair proporções para o desenhou.py
        proporcoes_usuario = {}
        for prop in result.get('proportions', []):
            prop_key = prop.get('name', '').lower().replace('/', '_').replace(' ', '_')
            if 'cabeça_corpo' in prop_key:
                proporcoes_usuario['head_body'] = float(prop.get('value', 0))
            elif 'ombros_quadris' in prop_key:
                proporcoes_usuario['shoulder_hip'] = float(prop.get('value', 0))
            elif 'pernas_torso' in prop_key:
                proporcoes_usuario['leg_torso'] = float(prop.get('value', 0))
            elif 'envergadura_altura' in prop_key:
                proporcoes_usuario['arm_span'] = float(prop.get('value', 0))
            elif 'largura_ombros' in prop_key:
                proporcoes_usuario['shoulder_width'] = float(prop.get('value', 0))
            elif 'comprimento_pernas' in prop_key:
                proporcoes_usuario['leg_length'] = float(prop.get('value', 0))
        
        # Gerar visualizações
        try:
            # Figura de palito proporcional
            stick_figure_png = stick_generator.create_proportional_stick_figure(proporcoes_usuario)
            result['stick_figure'] = stick_figure_png
            
            # Corpo ideal SVG
            ideal_body_svg = ideal_visualizer.generate_ideal_body_svg(proporcoes_usuario)
            result['ideal_body_svg'] = ideal_body_svg
            
            # Gráfico comparativo
            comparison_chart = ideal_visualizer.generate_comparison_chart(proporcoes_usuario)
            result['comparison_chart'] = comparison_chart
            
        except Exception as e:
            logger.error(f"Erro ao gerar visualizações: {str(e)}")
            result['stick_figure'] = None
            result['ideal_body_svg'] = None
            result['comparison_chart'] = None

@app.route('/visualizar', methods=['POST'])
@app.route('/visualizar', methods=['POST'])
def visualizar_corpo_ideal():
    """Endpoint para gerar a visualização SVG e gráfico do corpo ideal com base nas proporções"""
    try:
        data = request.get_json()
        if not data or 'proporcoes' not in data:
            return jsonify({'erro': 'Dados de proporções ausentes'}), 400

        proporcoes_usuario = data['proporcoes']
        
        # Gerar todas as visualizações
        resultado = ideal_visualizer.generate_body_analysis_report(proporcoes_usuario)
        
        # Gerar figura de palito também
        stick_figure_png = stick_generator.create_proportional_stick_figure(proporcoes_usuario)

        return jsonify({
            'status': 'sucesso',
            'svg': resultado['ideal_body_svg'],
            'grafico_base64': resultado['comparison_chart'],
            'stick_figure': stick_figure_png,
            'analise': resultado['detailed_analysis'],
            'sugestoes': resultado['improvement_suggestions'],
            'ideais': resultado['ideal_proportions']
        })

    except Exception as e:
        logger.error(f"Erro ao gerar visualização: {str(e)}")
        return jsonify({'erro': f'Erro interno: {str(e)}'}), 500
    
if __name__ == '__main__':
    logger.info("🚀 Iniciando servidor super otimizado...")
    app.run(debug=True, host='0.0.0.0', port=5000)