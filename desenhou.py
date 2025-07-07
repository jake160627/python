# arquivo desenhou.py - Versão Otimizada
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import io
import base64
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

class IdealBodyVisualizer:
    """Gerador de visualização corporal ideal - Versão Otimizada"""
    
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
        
        self.body_height = 100
        self.body_width = 25
        self.colors = {
            'body': '#FFE4C4',
            'outline': '#8B4513',
            'joints': '#CD853F',
            'ideal_lines': '#FF4500',
            'measurements': '#0000FF'
        }
    
    def generate_ideal_body_svg(self, user_proportions: Dict = None) -> str:
        """Gera SVG do corpo ideal com proporções perfeitas"""
        
        head_height = self.body_height / self.ideal_proportions['head_body']
        shoulder_width = self.body_width * self.ideal_proportions['shoulder_width'] * 4
        hip_width = shoulder_width / self.ideal_proportions['shoulder_hip']
        leg_length = self.body_height * self.ideal_proportions['leg_length']
        torso_length = self.body_height - head_height - leg_length
        
        points = self._calculate_ideal_points(head_height, shoulder_width, hip_width, leg_length, torso_length)
        svg = self._create_svg_body(points)
        
        if user_proportions:
            svg += self._add_comparison_lines(points, user_proportions)
        
        return self._wrap_svg(svg)
    
    def _calculate_ideal_points(self, head_height, shoulder_width, hip_width, leg_length, torso_length):
        """Calcula pontos anatômicos ideais"""
        center_x = self.body_width / 2
        
        head_top = 5
        head_bottom = head_top + head_height
        head_center = (head_top + head_bottom) / 2
        
        neck_y = head_bottom + 2
        shoulder_y = neck_y + 3
        left_shoulder = (center_x - shoulder_width/2, shoulder_y)
        right_shoulder = (center_x + shoulder_width/2, shoulder_y)
        
        waist_y = shoulder_y + torso_length * 0.6
        hip_y = shoulder_y + torso_length
        
        left_hip = (center_x - hip_width/2, hip_y)
        right_hip = (center_x + hip_width/2, hip_y)
        
        knee_y = hip_y + leg_length * 0.5
        ankle_y = hip_y + leg_length
        
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
            'wrists': ((left_shoulder[0], wrist_y), (right_shoulder[0], wrist_y))
        }
    
    def _create_svg_body(self, points):
        """Cria o corpo em SVG"""
        svg = []
        
        # Cabeça
        head_x, head_y, head_radius = points['head']
        svg.append(f'<circle cx="{head_x}" cy="{head_y}" r="{head_radius}" fill="{self.colors["body"]}" stroke="{self.colors["outline"]}" stroke-width="2"/>')
        
        # Olhos e boca
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
        
        torso_points = [
            left_shoulder,
            (left_shoulder[0] + 2, left_shoulder[1] + 5),
            (waist_x - waist_width/2, waist_y),
            left_hip,
            right_hip,
            (waist_x + waist_width/2, waist_y),
            (right_shoulder[0] - 2, right_shoulder[1] + 5),
            right_shoulder
        ]
        
        torso_path = "M " + " L ".join([f"{p[0]},{p[1]}" for p in torso_points]) + " Z"
        svg.append(f'<path d="{torso_path}" fill="{self.colors["body"]}" stroke="{self.colors["outline"]}" stroke-width="2"/>')
        
        # Braços
        left_elbow, right_elbow = points['elbows']
        left_wrist, right_wrist = points['wrists']
        
        svg.append(f'<line x1="{left_shoulder[0]}" y1="{left_shoulder[1]}" x2="{left_elbow[0]}" y2="{left_elbow[1]}" stroke="{self.colors["outline"]}" stroke-width="6" stroke-linecap="round"/>')
        svg.append(f'<line x1="{left_elbow[0]}" y1="{left_elbow[1]}" x2="{left_wrist[0]}" y2="{left_wrist[1]}" stroke="{self.colors["outline"]}" stroke-width="5" stroke-linecap="round"/>')
        svg.append(f'<line x1="{right_shoulder[0]}" y1="{right_shoulder[1]}" x2="{right_elbow[0]}" y2="{right_elbow[1]}" stroke="{self.colors["outline"]}" stroke-width="6" stroke-linecap="round"/>')
        svg.append(f'<line x1="{right_elbow[0]}" y1="{right_elbow[1]}" x2="{right_wrist[0]}" y2="{right_wrist[1]}" stroke="{self.colors["outline"]}" stroke-width="5" stroke-linecap="round"/>')
        
        # Pernas
        left_knee, right_knee = points['knees']
        left_ankle, right_ankle = points['ankles']
        
        svg.append(f'<line x1="{left_hip[0]}" y1="{left_hip[1]}" x2="{left_knee[0]}" y2="{left_knee[1]}" stroke="{self.colors["outline"]}" stroke-width="8" stroke-linecap="round"/>')
        svg.append(f'<line x1="{left_knee[0]}" y1="{left_knee[1]}" x2="{left_ankle[0]}" y2="{left_ankle[1]}" stroke="{self.colors["outline"]}" stroke-width="6" stroke-linecap="round"/>')
        svg.append(f'<line x1="{right_hip[0]}" y1="{right_hip[1]}" x2="{right_knee[0]}" y2="{right_knee[1]}" stroke="{self.colors["outline"]}" stroke-width="8" stroke-linecap="round"/>')
        svg.append(f'<line x1="{right_knee[0]}" y1="{right_knee[1]}" x2="{right_ankle[0]}" y2="{right_ankle[1]}" stroke="{self.colors["outline"]}" stroke-width="6" stroke-linecap="round"/>')
        
        # Articulações
        joints = [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
                 left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
        
        for joint in joints:
            svg.append(f'<circle cx="{joint[0]}" cy="{joint[1]}" r="2" fill="{self.colors["joints"]}" stroke="{self.colors["outline"]}" stroke-width="1"/>')
        
        # Mãos e pés
        svg.append(f'<circle cx="{left_wrist[0]}" cy="{left_wrist[1]}" r="3" fill="{self.colors["body"]}" stroke="{self.colors["outline"]}" stroke-width="1"/>')
        svg.append(f'<circle cx="{right_wrist[0]}" cy="{right_wrist[1]}" r="3" fill="{self.colors["body"]}" stroke="{self.colors["outline"]}" stroke-width="1"/>')
        svg.append(f'<ellipse cx="{left_ankle[0]}" cy="{left_ankle[1]}" rx="4" ry="2" fill="{self.colors["body"]}" stroke="{self.colors["outline"]}" stroke-width="1"/>')
        svg.append(f'<ellipse cx="{right_ankle[0]}" cy="{right_ankle[1]}" rx="4" ry="2" fill="{self.colors["body"]}" stroke="{self.colors["outline"]}" stroke-width="1"/>')
        
        return "\n".join(svg)
    
    def _add_comparison_lines(self, points, user_proportions):
        """Adiciona indicadores de comparação"""
        svg = []
        
        for prop_name, user_value in user_proportions.items():
            if prop_name in self.ideal_proportions:
                ideal_value = self.ideal_proportions[prop_name]
                deviation = abs(user_value - ideal_value) / ideal_value
                
                if deviation <= 0.1:
                    color = "#00FF00"  # Verde - Excelente
                elif deviation <= 0.2:
                    color = "#FFFF00"  # Amarelo - Bom
                elif deviation <= 0.3:
                    color = "#FF8000"  # Laranja - Regular
                else:
                    color = "#FF0000"  # Vermelho - Precisa melhorar
                
                indicator = self._create_proportion_indicator(prop_name, points, color)
                if indicator:
                    svg.append(indicator)
        
        return "\n".join(svg)
    
    def _create_proportion_indicator(self, prop_name, points, color):
        """Cria indicador visual específico"""
        if prop_name == 'head_body':
            head_x, head_y, head_radius = points['head']
            return f'<circle cx="{head_x + head_radius + 5}" cy="{head_y}" r="3" fill="{color}" stroke="black" stroke-width="1"/>'
        elif prop_name == 'shoulder_hip':
            left_shoulder, right_shoulder = points['shoulders']
            return f'<rect x="{right_shoulder[0] + 3}" y="{right_shoulder[1] - 2}" width="6" height="4" fill="{color}" stroke="black" stroke-width="1"/>'
        elif prop_name == 'leg_torso':
            left_knee, right_knee = points['knees']
            return f'<circle cx="{right_knee[0] + 5}" cy="{right_knee[1]}" r="3" fill="{color}" stroke="black" stroke-width="1"/>'
        return None
    
    def _wrap_svg(self, svg_content):
        """Envolve o conteúdo SVG"""
        return f'''<svg width="200" height="120" viewBox="0 0 50 120" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="white"/>
            {svg_content}
        </svg>'''
    
    def generate_comparison_chart(self, user_proportions):
        """Gera gráfico de comparação PNG/JPEG"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
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
        
        ax.bar(x - width/2, ideal_values, width, label='Ideal', color='#4CAF50', alpha=0.8)
        ax.bar(x + width/2, user_values, width, label='Usuário', color='#2196F3', alpha=0.8)
        
        ax.set_xlabel('Proporções')
        ax.set_ylabel('Valores')
        ax.set_title('Comparação: Proporções Ideais vs. Usuário')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Converter para base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return chart_base64
    
    def generate_body_analysis_report(self, user_proportions, analysis_result=None):
        """Gera relatório completo"""
        return {
            'ideal_body_svg': self.generate_ideal_body_svg(user_proportions),
            'comparison_chart': self.generate_comparison_chart(user_proportions),
            'detailed_analysis': self._generate_detailed_analysis(user_proportions),
            'improvement_suggestions': self._generate_improvement_suggestions(user_proportions),
            'ideal_proportions': self.ideal_proportions
        }
    
    def _generate_detailed_analysis(self, user_proportions):
        """Gera análise detalhada"""
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
                    'status': status
                })
        
        return analysis
    
    def _generate_improvement_suggestions(self, user_proportions):
        """Gera sugestões de melhoria"""
        suggestions = []
        
        suggestions_map = {
            'head_body': 'Exercícios de postura e alongamento da coluna',
            'shoulder_hip': 'Exercícios para ombros e quadris',
            'leg_torso': 'Alongamento de pernas e fortalecimento do core',
            'arm_span': 'Exercícios de flexibilidade para braços',
            'shoulder_width': 'Exercícios de fortalecimento dos deltoides',
            'leg_length': 'Exercícios de alongamento e pilates'
        }
        
        for prop_name, user_value in user_proportions.items():
            if prop_name in self.ideal_proportions:
                ideal_value = self.ideal_proportions[prop_name]
                deviation = abs(user_value - ideal_value) / ideal_value
                
                if deviation > 0.15 and prop_name in suggestions_map:
                    suggestions.append(suggestions_map[prop_name])
        
        return suggestions if suggestions else ["Parabéns! Suas proporções estão excelentes!"]

# Classe StickFigureGenerator simplificada
class StickFigureGenerator:
    def __init__(self):
        self.fig_size = (8, 10)
        self.colors = {
            'body': '#2E8B57',
            'joints': '#FF6B6B'
        }

    def create_proportional_stick_figure(self, proportions, action='standing'):
        """Cria figura de palito proporcional"""
        fig, ax = plt.subplots(figsize=self.fig_size)
        points = self._calculate_proportional_points(proportions)
        self._draw_stick_figure(ax, points)
        return self._convert_to_png(fig)

    def _calculate_proportional_points(self, proportions):
        """Calcula pontos com base nas proporções"""
        base_height = 100
        head_size = base_height / proportions.get('head_body', 7.5)
        shoulder_width = base_height * proportions.get('shoulder_width', 0.25)
        leg_length = base_height * proportions.get('leg_length', 0.5)

        return {
            'head': (50, 90),
            'neck': (50, 85),
            'shoulders': [(50 - shoulder_width / 2, 80), (50 + shoulder_width / 2, 80)],
            'hips': [(45, 50), (55, 50)],
            'knees': [(45, 25), (55, 25)],
            'ankles': [(45, 5), (55, 5)],
            'elbows': [(40, 65), (60, 65)],
            'wrists': [(35, 45), (65, 45)]
        }

    def _draw_stick_figure(self, ax, points):
        """Desenha a figura de palito"""
        # Cabeça
        head = Circle(points['head'], 5, color=self.colors['body'], linewidth=2)
        ax.add_patch(head)

        # Pescoço
        ax.plot([points['head'][0], points['neck'][0]],
                [points['head'][1] - 5, points['neck'][1]],
                color=self.colors['body'], linewidth=3)

        # Torso
        ax.plot([points['neck'][0], (points['hips'][0][0] + points['hips'][1][0]) / 2],
                [points['neck'][1], (points['hips'][0][1] + points['hips'][1][1]) / 2],
                color=self.colors['body'], linewidth=4)

        # Braços e pernas
        for i in range(2):
            # Braços
            ax.plot([points['shoulders'][i][0], points['elbows'][i][0]],
                    [points['shoulders'][i][1], points['elbows'][i][1]],
                    color=self.colors['body'], linewidth=3)
            ax.plot([points['elbows'][i][0], points['wrists'][i][0]],
                    [points['elbows'][i][1], points['wrists'][i][1]],
                    color=self.colors['body'], linewidth=3)
            
            # Pernas
            ax.plot([points['hips'][i][0], points['knees'][i][0]],
                    [points['hips'][i][1], points['knees'][i][1]],
                    color=self.colors['body'], linewidth=4)
            ax.plot([points['knees'][i][0], points['ankles'][i][0]],
                    [points['knees'][i][1], points['ankles'][i][1]],
                    color=self.colors['body'], linewidth=4)

        # Articulações
        for joint_group in ['shoulders', 'hips', 'knees', 'ankles', 'elbows', 'wrists']:
            for x, y in points[joint_group]:
                joint = Circle((x, y), 1.5, color=self.colors['joints'])
                ax.add_patch(joint)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')

    def _convert_to_png(self, fig):
        """Converte para PNG base64"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                 facecolor='white', edgecolor='none')
        buffer.seek(0)
        png_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{png_base64}"