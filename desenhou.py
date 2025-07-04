#arquivo desenhou.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, Circle, Rectangle
import io
import base64
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class IdealBodyVisualizer:
    """Gerador de visualização corporal ideal com proporções perfeitas"""
    
    def __init__(self):
        self.ideal_proportions = {
            'head_body': 7.5,
            'shoulder_hip': 1.4,
            'leg_torso': 1.2,
            'arm_span': 1.0,
            'waist_hip': 0.7,
            'shoulder_width': 0.25,
            'leg_length': 0.5
        }
        
        # Configurações de desenho
        self.body_height = 100
        self.body_width = 25
        self.colors = {
            'body': '#FFE4C4',
            'outline': '#8B4513',
            'muscle': '#DEB887',
            'joints': '#CD853F',
            'ideal_lines': '#FF4500',
            'measurements': '#0000FF'
        }
    
    def generate_ideal_body_svg(self, user_proportions: Dict = None, gender: str = 'unisex') -> str:
        """Gera SVG do corpo ideal com proporções perfeitas"""
        
        # Calcular proporções ideais
        head_height = self.body_height / self.ideal_proportions['head_body']
        shoulder_width = self.body_width * self.ideal_proportions['shoulder_width'] * 4
        hip_width = shoulder_width / self.ideal_proportions['shoulder_hip']
        leg_length = self.body_height * self.ideal_proportions['leg_length']
        torso_length = self.body_height - head_height - leg_length
        
        # Definir pontos anatômicos
        points = self._calculate_ideal_points(head_height, shoulder_width, hip_width, leg_length, torso_length)
        
        # Gerar SVG
        svg = self._create_svg_body(points, gender)
        
        # Adicionar medidas e proporções
        if user_proportions:
            svg += self._add_comparison_lines(points, user_proportions)
        
        return self._wrap_svg(svg)
    
    def _calculate_ideal_points(self, head_height, shoulder_width, hip_width, leg_length, torso_length):
        """Calcula pontos anatômicos ideais"""
        center_x = self.body_width / 2
        
        # Cabeça
        head_top = 5
        head_bottom = head_top + head_height
        head_center = (head_top + head_bottom) / 2
        
        # Pescoço
        neck_y = head_bottom + 2
        
        # Ombros
        shoulder_y = neck_y + 3
        left_shoulder = (center_x - shoulder_width/2, shoulder_y)
        right_shoulder = (center_x + shoulder_width/2, shoulder_y)
        
        # Torso
        waist_y = shoulder_y + torso_length * 0.6
        hip_y = shoulder_y + torso_length
        
        # Quadris
        left_hip = (center_x - hip_width/2, hip_y)
        right_hip = (center_x + hip_width/2, hip_y)
        
        # Pernas
        knee_y = hip_y + leg_length * 0.5
        ankle_y = hip_y + leg_length
        
        # Braços
        elbow_y = shoulder_y + torso_length * 0.4
        wrist_y = hip_y
        
        return {
            'head': (center_x, head_center, head_height/2),
            'neck': (center_x, neck_y),
            'shoulders': (left_shoulder, right_shoulder),
            'waist': (center_x, waist_y, hip_width * 0.7),
            'hips': (left_hip, right_hip),
            'knees': ((left_hip[0], knee_y), (right_hip[0], knee_y)),
            'ankles': ((left_hip[0], ankle_y), (right_hip[0], ankle_y)),
            'elbows': ((left_shoulder[0], elbow_y), (right_shoulder[0], elbow_y)),
            'wrists': ((left_shoulder[0], wrist_y), (right_shoulder[0], wrist_y)),
            'dimensions': {
                'head_height': head_height,
                'shoulder_width': shoulder_width,
                'hip_width': hip_width,
                'leg_length': leg_length,
                'torso_length': torso_length,
                'total_height': self.body_height
            }
        }
    
    def _create_svg_body(self, points, gender):
        """Cria o corpo em SVG"""
        svg = []
        
        # Cabeça
        head_x, head_y, head_radius = points['head']
        svg.append(f'<circle cx="{head_x}" cy="{head_y}" r="{head_radius}" fill="{self.colors["body"]}" stroke="{self.colors["outline"]}" stroke-width="2"/>')
        
        # Rosto básico
        eye_offset = head_radius * 0.3
        svg.append(f'<circle cx="{head_x - eye_offset}" cy="{head_y - head_radius*0.2}" r="1.5" fill="black"/>')
        svg.append(f'<circle cx="{head_x + eye_offset}" cy="{head_y - head_radius*0.2}" r="1.5" fill="black"/>')
        svg.append(f'<path d="M {head_x - eye_offset*0.5} {head_y + head_radius*0.2} Q {head_x} {head_y + head_radius*0.4} {head_x + eye_offset*0.5} {head_y + head_radius*0.2}" stroke="black" fill="none" stroke-width="1"/>')
        
        # Pescoço
        neck_x, neck_y = points['neck']
        svg.append(f'<rect x="{neck_x-2}" y="{head_y + head_radius}" width="4" height="{neck_y - (head_y + head_radius)}" fill="{self.colors["body"]}" stroke="{self.colors["outline"]}" stroke-width="1"/>')
        
        # Torso
        left_shoulder, right_shoulder = points['shoulders']
        left_hip, right_hip = points['hips']
        waist_x, waist_y, waist_width = points['waist']
        
        # Criar torso como polígono suave
        torso_points = [
            left_shoulder,
            (left_shoulder[0] + 2, left_shoulder[1] + 5),  # Peito
            (waist_x - waist_width/2, waist_y),  # Cintura esquerda
            left_hip,
            right_hip,
            (waist_x + waist_width/2, waist_y),  # Cintura direita
            (right_shoulder[0] - 2, right_shoulder[1] + 5),  # Peito
            right_shoulder
        ]
        
        torso_path = "M " + " L ".join([f"{p[0]},{p[1]}" for p in torso_points]) + " Z"
        svg.append(f'<path d="{torso_path}" fill="{self.colors["body"]}" stroke="{self.colors["outline"]}" stroke-width="2"/>')
        
        # Braços
        left_elbow, right_elbow = points['elbows']
        left_wrist, right_wrist = points['wrists']
        
        # Braço esquerdo
        svg.append(f'<line x1="{left_shoulder[0]}" y1="{left_shoulder[1]}" x2="{left_elbow[0]}" y2="{left_elbow[1]}" stroke="{self.colors["outline"]}" stroke-width="6" stroke-linecap="round"/>')
        svg.append(f'<line x1="{left_elbow[0]}" y1="{left_elbow[1]}" x2="{left_wrist[0]}" y2="{left_wrist[1]}" stroke="{self.colors["outline"]}" stroke-width="5" stroke-linecap="round"/>')
        
        # Braço direito
        svg.append(f'<line x1="{right_shoulder[0]}" y1="{right_shoulder[1]}" x2="{right_elbow[0]}" y2="{right_elbow[1]}" stroke="{self.colors["outline"]}" stroke-width="6" stroke-linecap="round"/>')
        svg.append(f'<line x1="{right_elbow[0]}" y1="{right_elbow[1]}" x2="{right_wrist[0]}" y2="{right_wrist[1]}" stroke="{self.colors["outline"]}" stroke-width="5" stroke-linecap="round"/>')
        
        # Pernas
        left_knee, right_knee = points['knees']
        left_ankle, right_ankle = points['ankles']
        
        # Perna esquerda
        svg.append(f'<line x1="{left_hip[0]}" y1="{left_hip[1]}" x2="{left_knee[0]}" y2="{left_knee[1]}" stroke="{self.colors["outline"]}" stroke-width="8" stroke-linecap="round"/>')
        svg.append(f'<line x1="{left_knee[0]}" y1="{left_knee[1]}" x2="{left_ankle[0]}" y2="{left_ankle[1]}" stroke="{self.colors["outline"]}" stroke-width="6" stroke-linecap="round"/>')
        
        # Perna direita
        svg.append(f'<line x1="{right_hip[0]}" y1="{right_hip[1]}" x2="{right_knee[0]}" y2="{right_knee[1]}" stroke="{self.colors["outline"]}" stroke-width="8" stroke-linecap="round"/>')
        svg.append(f'<line x1="{right_knee[0]}" y1="{right_knee[1]}" x2="{right_ankle[0]}" y2="{right_ankle[1]}" stroke="{self.colors["outline"]}" stroke-width="6" stroke-linecap="round"/>')
        
        # Articulações
        joints = [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
                 left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
        
        for joint in joints:
            svg.append(f'<circle cx="{joint[0]}" cy="{joint[1]}" r="2" fill="{self.colors["joints"]}" stroke="{self.colors["outline"]}" stroke-width="1"/>')
        
        # Mãos
        svg.append(f'<circle cx="{left_wrist[0]}" cy="{left_wrist[1]}" r="3" fill="{self.colors["body"]}" stroke="{self.colors["outline"]}" stroke-width="1"/>')
        svg.append(f'<circle cx="{right_wrist[0]}" cy="{right_wrist[1]}" r="3" fill="{self.colors["body"]}" stroke="{self.colors["outline"]}" stroke-width="1"/>')
        
        # Pés
        svg.append(f'<ellipse cx="{left_ankle[0]}" cy="{left_ankle[1]}" rx="4" ry="2" fill="{self.colors["body"]}" stroke="{self.colors["outline"]}" stroke-width="1"/>')
        svg.append(f'<ellipse cx="{right_ankle[0]}" cy="{right_ankle[1]}" rx="4" ry="2" fill="{self.colors["body"]}" stroke="{self.colors["outline"]}" stroke-width="1"/>')
        
        return "\n".join(svg)
    
    def _add_comparison_lines(self, points, user_proportions):
        """Adiciona linhas de comparação com proporções do usuário"""
        svg = []
        
        # Linhas de medição
        dimensions = points['dimensions']
        
        # Linha da altura total
        svg.append(f'<line x1="{self.body_width + 5}" y1="5" x2="{self.body_width + 5}" y2="{5 + dimensions["total_height"]}" stroke="{self.colors["measurements"]}" stroke-width="1" stroke-dasharray="5,5"/>')
        svg.append(f'<text x="{self.body_width + 8}" y="{5 + dimensions["total_height"]/2}" fill="{self.colors["measurements"]}" font-size="8" transform="rotate(-90 {self.body_width + 8} {5 + dimensions["total_height"]/2})">Altura Total</text>')
        
        # Linha da largura dos ombros
        left_shoulder, right_shoulder = points['shoulders']
        svg.append(f'<line x1="{left_shoulder[0]}" y1="{left_shoulder[1] - 5}" x2="{right_shoulder[0]}" y2="{right_shoulder[1] - 5}" stroke="{self.colors["measurements"]}" stroke-width="1" stroke-dasharray="5,5"/>')
        svg.append(f'<text x="{(left_shoulder[0] + right_shoulder[0])/2}" y="{left_shoulder[1] - 8}" fill="{self.colors["measurements"]}" font-size="8" text-anchor="middle">Largura Ombros</text>')
        
        # Adicionar indicadores de desvio
        if user_proportions:
            svg.extend(self._add_deviation_indicators(points, user_proportions))
        
        return "\n".join(svg)
    
    def _add_deviation_indicators(self, points, user_proportions):
        """Adiciona indicadores visuais de desvio das proporções ideais"""
        svg = []
        
        # Comparar cada proporção
        for prop_name, user_value in user_proportions.items():
            if prop_name in self.ideal_proportions:
                ideal_value = self.ideal_proportions[prop_name]
                deviation = abs(user_value - ideal_value) / ideal_value
                
                # Cor baseada no desvio
                if deviation <= 0.1:
                    color = "#00FF00"  # Verde - Excelente
                elif deviation <= 0.2:
                    color = "#FFFF00"  # Amarelo - Bom
                elif deviation <= 0.3:
                    color = "#FF8000"  # Laranja - Regular
                else:
                    color = "#FF0000"  # Vermelho - Precisa melhorar
                
                # Adicionar indicador visual específico para cada proporção
                indicator = self._create_proportion_indicator(prop_name, points, deviation, color)
                if indicator:
                    svg.append(indicator)
        
        return svg
    
    def _create_proportion_indicator(self, prop_name, points, deviation, color):
        """Cria indicador visual específico para cada proporção"""
        if prop_name == 'head_body':
            # Indicador na cabeça
            head_x, head_y, head_radius = points['head']
            return f'<circle cx="{head_x + head_radius + 5}" cy="{head_y}" r="3" fill="{color}" stroke="black" stroke-width="1"/>'
        
        elif prop_name == 'shoulder_hip':
            # Indicador nos ombros
            left_shoulder, right_shoulder = points['shoulders']
            return f'<rect x="{right_shoulder[0] + 3}" y="{right_shoulder[1] - 2}" width="6" height="4" fill="{color}" stroke="black" stroke-width="1"/>'
        
        elif prop_name == 'leg_torso':
            # Indicador nas pernas
            left_knee, right_knee = points['knees']
            return f'<circle cx="{right_knee[0] + 5}" cy="{right_knee[1]}" r="3" fill="{color}" stroke="black" stroke-width="1"/>'
        
        return None
    
    def _wrap_svg(self, svg_content):
        """Envolve o conteúdo SVG com tags apropriadas"""
        return f'''<svg width="200" height="120" viewBox="0 0 50 120" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="white"/>
            {svg_content}
            
            <!-- Legenda -->
            <g transform="translate(5, 105)">
                <text x="0" y="0" font-size="6" fill="black" font-weight="bold">Corpo Ideal</text>
                <circle cx="0" cy="5" r="2" fill="#00FF00"/>
                <text x="5" y="8" font-size="4" fill="black">Excelente</text>
                <circle cx="0" cy="10" r="2" fill="#FFFF00"/>
                <text x="5" y="13" font-size="4" fill="black">Bom</text>
                <circle cx="25" cy="5" r="2" fill="#FF8000"/>
                <text x="30" y="8" font-size="4" fill="black">Regular</text>
                <circle cx="25" cy="10" r="2" fill="#FF0000"/>
                <text x="30" y="13" font-size="4" fill="black">Melhorar</text>
            </g>
        </svg>'''
    
    def generate_comparison_chart(self, user_proportions):
        """Gera gráfico de comparação das proporções"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Dados para o gráfico
        categories = []
        ideal_values = []
        user_values = []
        
        for prop_name in self.ideal_proportions:
            if prop_name in user_proportions:
                categories.append(prop_name.replace('_', ' ').title())
                ideal_values.append(self.ideal_proportions[prop_name])
                user_values.append(user_proportions[prop_name])
        
        x = np.arange(len(categories))
        width = 0.35
        
        # Barras
        bars1 = ax.bar(x - width/2, ideal_values, width, label='Ideal', color='#4CAF50', alpha=0.8)
        bars2 = ax.bar(x + width/2, user_values, width, label='Usuário', color='#2196F3', alpha=0.8)
        
        # Personalização
        ax.set_xlabel('Proporções')
        ax.set_ylabel('Valores')
        ax.set_title('Comparação: Proporções Ideais vs. Usuário')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Converter para base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return chart_base64
    
    def generate_body_analysis_report(self, user_proportions, analysis_result):
        """Gera relatório completo com visualizações"""
        report = {
            'ideal_body_svg': self.generate_ideal_body_svg(user_proportions),
            'comparison_chart': self.generate_comparison_chart(user_proportions),
            'detailed_analysis': self._generate_detailed_analysis(user_proportions, analysis_result),
            'improvement_suggestions': self._generate_improvement_suggestions(user_proportions),
            'ideal_proportions': self.ideal_proportions
        }
        
        return report
    
    def _generate_detailed_analysis(self, user_proportions, analysis_result):
        """Gera análise detalhada textual"""
        analysis = []
        
        for prop_name, user_value in user_proportions.items():
            if prop_name in self.ideal_proportions:
                ideal_value = self.ideal_proportions[prop_name]
                deviation = abs(user_value - ideal_value) / ideal_value
                
                status = "excelente" if deviation <= 0.1 else \
                        "bom" if deviation <= 0.2 else \
                        "regular" if deviation <= 0.3 else "precisa melhorar"
                
                analysis.append({
                    'proportion': prop_name.replace('_', ' ').title(),
                    'user_value': user_value,
                    'ideal_value': ideal_value,
                    'deviation_percent': deviation * 100,
                    'status': status,
                    'description': self._get_proportion_description(prop_name)
                })
        
        return analysis
    
    def _get_proportion_description(self, prop_name):
        """Retorna descrição da proporção"""
        descriptions = {
            'head_body': 'Relação entre a altura da cabeça e o corpo total',
            'shoulder_hip': 'Relação entre largura dos ombros e quadris',
            'leg_torso': 'Relação entre comprimento das pernas e torso',
            'arm_span': 'Relação entre envergadura dos braços e altura',
            'waist_hip': 'Relação entre cintura e quadris',
            'shoulder_width': 'Largura dos ombros em relação à altura',
            'leg_length': 'Comprimento das pernas em relação à altura total'
        }
        return descriptions.get(prop_name, 'Proporção anatômica')
    
    def _generate_improvement_suggestions(self, user_proportions):
        """Gera sugestões específicas de melhoria"""
        suggestions = []
        
        for prop_name, user_value in user_proportions.items():
            if prop_name in self.ideal_proportions:
                ideal_value = self.ideal_proportions[prop_name]
                deviation = abs(user_value - ideal_value) / ideal_value
                
                if deviation > 0.15:  # Só sugerir se o desvio for significativo
                    suggestion = self._get_specific_suggestion(prop_name, user_value, ideal_value)
                    if suggestion:
                        suggestions.append(suggestion)
        
        return suggestions
    
    def _get_specific_suggestion(self, prop_name, user_value, ideal_value):
        """Retorna sugestão específica para cada proporção"""
        suggestions_map = {
            'head_body': {
                'title': 'Proporção Cabeça/Corpo',
                'exercise': 'Exercícios de postura e alongamento da coluna',
                'tip': 'Manter postura ereta ajuda a otimizar esta proporção'
            },
            'shoulder_hip': {
                'title': 'Proporção Ombros/Quadris',
                'exercise': 'Exercícios para ombros (elevação lateral) e quadris (agachamentos)',
                'tip': 'Fortalecer ombros e trabalhar mobilidade dos quadris'
            },
            'leg_torso': {
                'title': 'Proporção Pernas/Torso',
                'exercise': 'Alongamento de pernas e fortalecimento do core',
                'tip': 'Exercícios de flexibilidade podem melhorar esta proporção'
            },
            'arm_span': {
                'title': 'Envergadura/Altura',
                'exercise': 'Alongamento de braços e exercícios de flexibilidade',
                'tip': 'Exercícios de natação podem ajudar'
            },
            'shoulder_width': {
                'title': 'Largura dos Ombros',
                'exercise': 'Exercícios de fortalecimento dos deltoides',
                'tip': 'Desenvolvimento muscular dos ombros'
            },
            'leg_length': {
                'title': 'Comprimento das Pernas',
                'exercise': 'Exercícios de alongamento e fortalecimento',
                'tip': 'Pilates e yoga podem ajudar na percepção desta proporção'
            }
        }
        
        return suggestions_map.get(prop_name)

# Função principal para integração
def generate_ideal_body_visualization(user_proportions, analysis_result=None):
    """Função principal para gerar visualização do corpo ideal"""
    visualizer = IdealBodyVisualizer()
    return visualizer.generate_body_analysis_report(user_proportions, analysis_result)

# Exemplo de uso
if __name__ == "__main__":
    # Exemplo de proporções do usuário
    user_props = {
        'head_body': 7.2,
        'shoulder_hip': 1.3,
        'leg_torso': 1.1,
        'arm_span': 0.98,
        'waist_hip': 0.75,
        'shoulder_width': 0.23,
        'leg_length': 0.48
    }
    
    # Gerar visualização
    result = generate_ideal_body_visualization(user_props)
    
    # Salvar SVG
    with open('ideal_body.svg', 'w') as f:
        f.write(result['ideal_body_svg'])
    
    print("Visualização gerada com sucesso!")
    print(f"Relatório completo: {len(result['detailed_analysis'])} análises detalhadas")
    print(f"Sugestões de melhoria: {len(result['improvement_suggestions'])}")