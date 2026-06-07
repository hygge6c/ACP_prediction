import re
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple, List

# ----------------------------------------------------------------------
# Sequence validation and encoding
# ----------------------------------------------------------------------
def validate_sequence(seq: str) -> Tuple[bool, str]:
    seq = seq.strip().upper()
    if not seq:
        return False, "Sequence cannot be empty"
    if len(seq) < 5:
        return False, "Sequence too short (min 5 amino acids)"
    if len(seq) > 100:
        return False, "Sequence too long (max 100 amino acids)"
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    invalid = set(seq) - valid
    if invalid:
        return False, f"Invalid characters: {''.join(invalid)}"
    return True, seq

def encode_sequence(seq: str, vocab: Dict, max_len: int = 30) -> torch.Tensor:
    encoded = [vocab.get(aa, 0) for aa in seq]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return torch.tensor([encoded], dtype=torch.long)

# ----------------------------------------------------------------------
# Sequence property calculations
# ----------------------------------------------------------------------
def calculate_amino_acid_composition(seq: str) -> Dict[str, Dict]:
    composition = {}
    total = len(seq)
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        cnt = seq.count(aa)
        composition[aa] = {'count': cnt, 'percentage': round(cnt/total*100, 2) if total else 0}
    return composition

def calculate_sequence_properties(seq: str) -> Dict:
    aa_weights = {'A':89.09,'C':121.16,'D':133.10,'E':147.13,'F':165.19,'G':75.07,
                  'H':155.16,'I':131.18,'K':146.19,'L':131.18,'M':149.21,'N':132.12,
                  'P':115.13,'Q':146.15,'R':174.20,'S':105.09,'T':119.12,'V':117.15,
                  'W':204.23,'Y':181.19}
    hydropathy = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,
                  'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,
                  'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
    mol_weight = sum(aa_weights.get(aa, 110) for aa in seq)
    acidic = seq.count('D') + seq.count('E')
    basic = seq.count('K') + seq.count('R') + seq.count('H')
    gravy = sum(hydropathy.get(aa, 0) for aa in seq) / len(seq)
    return {
        'molecular_weight': round(mol_weight, 2),
        'length': len(seq),
        'net_charge': basic - acidic,
        'gravy_score': round(gravy, 3),
        'acidic_count': acidic,
        'basic_count': basic,
        'hydrophobic_count': sum(1 for aa in seq if aa in 'AILMFWV'),
        'aromatic_count': sum(1 for aa in seq if aa in 'FYW')
    }

def get_amino_acid_properties() -> Dict:
    return {
        'A': {'name':'Alanine','type':'Hydrophobic','color':'#FF6B6B'},
        'C': {'name':'Cysteine','type':'Polar','color':'#4ECDC4'},
        'D': {'name':'Aspartic Acid','type':'Acidic','color':'#FFD166'},
        'E': {'name':'Glutamic Acid','type':'Acidic','color':'#FFD166'},
        'F': {'name':'Phenylalanine','type':'Hydrophobic','color':'#FF6B6B'},
        'G': {'name':'Glycine','type':'Special','color':'#06D6A0'},
        'H': {'name':'Histidine','type':'Basic','color':'#118AB2'},
        'I': {'name':'Isoleucine','type':'Hydrophobic','color':'#FF6B6B'},
        'K': {'name':'Lysine','type':'Basic','color':'#118AB2'},
        'L': {'name':'Leucine','type':'Hydrophobic','color':'#FF6B6B'},
        'M': {'name':'Methionine','type':'Hydrophobic','color':'#FF6B6B'},
        'N': {'name':'Asparagine','type':'Polar','color':'#4ECDC4'},
        'P': {'name':'Proline','type':'Special','color':'#06D6A0'},
        'Q': {'name':'Glutamine','type':'Polar','color':'#4ECDC4'},
        'R': {'name':'Arginine','type':'Basic','color':'#118AB2'},
        'S': {'name':'Serine','type':'Polar','color':'#4ECDC4'},
        'T': {'name':'Threonine','type':'Polar','color':'#4ECDC4'},
        'V': {'name':'Valine','type':'Hydrophobic','color':'#FF6B6B'},
        'W': {'name':'Tryptophan','type':'Hydrophobic','color':'#FF6B6B'},
        'Y': {'name':'Tyrosine','type':'Polar','color':'#4ECDC4'}
    }

# ----------------------------------------------------------------------
# Plot generators (return Plotly JSON strings)
# ----------------------------------------------------------------------
def create_attention_heatmap(seq: str, attn_weights: np.ndarray = None) -> str:
    L = len(seq)
    if attn_weights is None:
        # Simulate attention for demo
        attn = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                attn[i, j] = 0.3 if i==j else 0.2/(abs(i-j)+1)
        attn = attn / attn.sum(axis=1, keepdims=True)
    else:
        attn = attn_weights[:L, :L]
    fig = go.Figure(data=go.Heatmap(z=attn, x=list(seq), y=list(seq), colorscale='Viridis'))
    fig.update_layout(title="Attention Heatmap", xaxis_title="Target", yaxis_title="Source", height=400)
    return fig.to_json()

def create_feature_importance_plot(seq: str, importance: np.ndarray) -> str:
    L = len(seq)
    scores = importance[:L]
    fig = go.Figure(data=go.Bar(x=list(range(1, L+1)), y=scores,
                                marker_color=['#FF6B6B' if s>np.mean(scores) else '#4ECDC4' for s in scores]))
    fig.update_layout(title="Feature Importance", xaxis_title="Position", yaxis_title="Score")
    fig.update_xaxes(tickvals=list(range(1, L+1)), ticktext=list(seq))
    return fig.to_json()

def create_amino_acid_composition_plot(composition: Dict) -> str:
    aa = list(composition.keys())
    counts = [composition[a]['count'] for a in aa]
    percs = [composition[a]['percentage'] for a in aa]
    colors = [get_amino_acid_properties()[a]['color'] for a in aa]
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Counts', 'Percentages'))
    fig.add_trace(go.Bar(x=aa, y=counts, marker_color=colors), row=1, col=1)
    fig.add_trace(go.Pie(labels=aa, values=percs, marker_colors=colors), row=1, col=2)
    fig.update_layout(title="Amino Acid Composition", height=450)
    return fig.to_json()

def create_batch_statistics_plot(stats: Dict) -> str:
    fig = go.Figure(data=go.Pie(labels=['Anticancer','Non-Anticancer'],
                                values=[stats['anticancer_count'], stats['non_anticancer_count']],
                                hole=0.3, marker_colors=['#27ae60','#e74c3c']))
    fig.update_layout(title=f"Batch Results (Total: {stats['total_sequences']})", height=400)
    return fig.to_json()