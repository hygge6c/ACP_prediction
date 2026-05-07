import re
import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json


def validate_sequence(sequence: str) -> Tuple[bool, str]:
    """验证肽序列是否有效"""
    sequence = sequence.strip().upper()

    if not sequence:
        return False, "序列不能为空"

    if len(sequence) < 5:
        return False, "序列太短（至少5个氨基酸）"

    if len(sequence) > 100:
        return False, "序列太长（最多100个氨基酸）"

    # 检查是否只包含标准氨基酸
    valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
    invalid_chars = set(sequence) - valid_chars

    if invalid_chars:
        return False, f"发现无效字符: {''.join(invalid_chars)}"

    return True, sequence


def encode_sequence(sequence: str, vocab: Dict, max_len: int = 30) -> torch.Tensor:
    """编码肽序列为张量"""
    encoded = [vocab.get(aa, 0) for aa in sequence]

    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]

    return torch.tensor([encoded], dtype=torch.long)


def calculate_amino_acid_composition(sequence: str) -> Dict:
    """计算氨基酸组成"""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    composition = {}
    total = len(sequence)

    for aa in amino_acids:
        count = sequence.count(aa)
        composition[aa] = {
            'count': count,
            'percentage': round(count / total * 100, 2) if total > 0 else 0
        }

    return composition


def get_amino_acid_properties() -> Dict[str, Dict[str, str]]:
    """获取氨基酸性质"""
    return {
        'A': {'name': 'Alanine', 'type': 'Hydrophobic', 'color': '#FF6B6B'},
        'C': {'name': 'Cysteine', 'type': 'Polar', 'color': '#4ECDC4'},
        'D': {'name': 'Aspartic Acid', 'type': 'Acidic', 'color': '#FFD166'},
        'E': {'name': 'Glutamic Acid', 'type': 'Acidic', 'color': '#FFD166'},
        'F': {'name': 'Phenylalanine', 'type': 'Hydrophobic', 'color': '#FF6B6B'},
        'G': {'name': 'Glycine', 'type': 'Special', 'color': '#06D6A0'},
        'H': {'name': 'Histidine', 'type': 'Basic', 'color': '#118AB2'},
        'I': {'name': 'Isoleucine', 'type': 'Hydrophobic', 'color': '#FF6B6B'},
        'K': {'name': 'Lysine', 'type': 'Basic', 'color': '#118AB2'},
        'L': {'name': 'Leucine', 'type': 'Hydrophobic', 'color': '#FF6B6B'},
        'M': {'name': 'Methionine', 'type': 'Hydrophobic', 'color': '#FF6B6B'},
        'N': {'name': 'Asparagine', 'type': 'Polar', 'color': '#4ECDC4'},
        'P': {'name': 'Proline', 'type': 'Special', 'color': '#06D6A0'},
        'Q': {'name': 'Glutamine', 'type': 'Polar', 'color': '#4ECDC4'},
        'R': {'name': 'Arginine', 'type': 'Basic', 'color': '#118AB2'},
        'S': {'name': 'Serine', 'type': 'Polar', 'color': '#4ECDC4'},
        'T': {'name': 'Threonine', 'type': 'Polar', 'color': '#4ECDC4'},
        'V': {'name': 'Valine', 'type': 'Hydrophobic', 'color': '#FF6B6B'},
        'W': {'name': 'Tryptophan', 'type': 'Hydrophobic', 'color': '#FF6B6B'},
        'Y': {'name': 'Tyrosine', 'type': 'Polar', 'color': '#4ECDC4'}
    }


def create_attention_heatmap(sequence: str, attention_weights: Any,
                             head_idx: int = 0) -> str:
    """创建注意力热力图"""
    seq_len = len(sequence)

    # 获取指定头的注意力权重
    if attention_weights is None:
        # 如果没有注意力权重，创建模拟的注意力权重
        # 创建一个基于序列相似性的简单注意力模式
        attn = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                # 模拟自注意力：对角线上有较高权重
                if i == j:
                    attn[i, j] = 0.3
                # 相邻位置有中等权重
                elif abs(i - j) == 1:
                    attn[i, j] = 0.2
                # 其他位置有较低权重
                else:
                    attn[i, j] = 0.5 / (abs(i - j) + 1)

        # 归一化每一行
        attn = attn / attn.sum(axis=1, keepdims=True)
    else:
        # 如果提供了注意力权重
        if isinstance(attention_weights, torch.Tensor):
            if attention_weights.dim() == 4:
                attn = attention_weights[0, head_idx, :seq_len, :seq_len].cpu().numpy()
            else:
                attn = attention_weights[0, :seq_len, :seq_len].cpu().numpy()
        else:
            attn = attention_weights

    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=attn,
        x=list(sequence),
        y=list(sequence),
        colorscale='Viridis',
        colorbar=dict(title="Attention Weight")
    ))

    fig.update_layout(
        title=f"Attention Heatmap (Head {head_idx})",
        xaxis_title="Target Amino Acid",
        yaxis_title="Source Amino Acid",
        height=500,
        width=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig.to_json()


def create_feature_importance_plot(sequence: str, importance_scores: np.ndarray) -> str:
    """创建特征重要性图"""
    seq_len = len(sequence)
    importance = importance_scores[:seq_len]

    # 创建条形图
    fig = go.Figure(data=go.Bar(
        x=list(range(1, seq_len + 1)),
        y=importance,
        marker_color=['#FF6B6B' if score > np.mean(importance) else '#4ECDC4'
                      for score in importance],
        text=[f"{score:.4f}" for score in importance],
        textposition='auto',
    ))

    fig.update_layout(
        title="Feature Importance Scores",
        xaxis_title="Amino Acid Position",
        yaxis_title="Importance Score",
        height=400,
        width=800,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # 在x轴上添加氨基酸标签
    fig.update_xaxes(
        ticktext=list(sequence),
        tickvals=list(range(1, seq_len + 1)),
        tickmode='array'
    )

    return fig.to_json()


def create_amino_acid_composition_plot(composition: Dict) -> str:
    """创建氨基酸组成图"""
    amino_acids = list(composition.keys())
    counts = [composition[aa]['count'] for aa in amino_acids]
    percentages = [composition[aa]['percentage'] for aa in amino_acids]

    # 获取颜色
    properties = get_amino_acid_properties()
    colors = [properties[aa]['color'] for aa in amino_acids]

    # 创建子图
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Counts', 'Percentages'),
        specs=[[{'type': 'bar'}, {'type': 'pie'}]]
    )

    # 条形图
    fig.add_trace(
        go.Bar(x=amino_acids, y=counts, marker_color=colors, name='Count',
               text=counts, textposition='auto'),
        row=1, col=1
    )

    # 饼图
    fig.add_trace(
        go.Pie(labels=amino_acids, values=percentages,
               marker=dict(colors=colors), hole=0.3),
        row=1, col=2
    )

    fig.update_layout(
        title="Amino Acid Composition",
        height=500,
        width=900,
        showlegend=False,
        margin=dict(l=50, r=50, t=100, b=50)
    )

    # 更新x轴标题
    fig.update_xaxes(title_text="Amino Acid", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    return fig.to_json()


def create_performance_radar_chart(metrics: Dict) -> str:
    """创建性能雷达图"""
    categories = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'MCC']
    values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['specificity'],
        metrics['f1'],
        metrics['mcc']
    ]

    # 确保所有值都在0-1之间
    values = [max(0, min(1, v)) for v in values]

    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],  # 闭合图形
        theta=categories + [categories[0]],
        fill='toself',
        name='Model Performance',
        line=dict(color='#118AB2', width=2)
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='lightgray',
                linecolor='gray',
                showticklabels=True
            ),
            angularaxis=dict(
                gridcolor='lightgray',
                linecolor='gray'
            )
        ),
        showlegend=False,
        title="Model Performance Metrics",
        height=400,
        width=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig.to_json()


def create_batch_statistics_plot(stats: Dict) -> str:
    """创建批量统计图"""
    labels = ['Anticancer', 'Non-Anticancer']
    values = [stats['anticancer_count'], stats['non_anticancer_count']]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker_colors=['#27ae60', '#e74c3c']
    )])

    fig.update_layout(
        title=f"Batch Prediction Results (Total: {stats['total_sequences']})",
        height=400,
        width=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig.to_json()


def calculate_sequence_properties(sequence: str) -> Dict:
    """计算序列基本属性"""
    # 计算分子量（近似值）
    aa_weights = {
        'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13,
        'F': 165.19, 'G': 75.07, 'H': 155.16, 'I': 131.18,
        'K': 146.19, 'L': 131.18, 'M': 149.21, 'N': 132.12,
        'P': 115.13, 'Q': 146.15, 'R': 174.20, 'S': 105.09,
        'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
    }

    mol_weight = sum(aa_weights.get(aa, 110) for aa in sequence)

    # 计算净电荷（在pH 7.0时）
    # 酸性氨基酸 (D, E): 带负电
    # 碱性氨基酸 (K, R, H): 带正电
    acidic = sequence.count('D') + sequence.count('E')
    basic = sequence.count('K') + sequence.count('R') + sequence.count('H')
    net_charge = basic - acidic

    # 疏水性分析（基于GRAVY评分）
    hydropathy = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5,
        'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
        'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8,
        'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }

    gravy_score = sum(hydropathy.get(aa, 0) for aa in sequence) / len(sequence)

    return {
        'molecular_weight': round(mol_weight, 2),
        'length': len(sequence),
        'net_charge': net_charge,
        'gravy_score': round(gravy_score, 3),
        'acidic_count': acidic,
        'basic_count': basic,
        'hydrophobic_count': sum(1 for aa in sequence if aa in 'AILMFWV'),
        'aromatic_count': sum(1 for aa in sequence if aa in 'FYW')
    }