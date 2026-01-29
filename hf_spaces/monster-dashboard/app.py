import gradio as gr
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
import plotly.graph_objects as go
from datetime import datetime
import json

# Configuration
REPOS = [
    "introspector/data-moonshine",
    "meta-introspector/monster-perf-proofs"
]

PERSONAS = ["Knuth", "ITIL", "ISO9001", "GMP", "SixSigma", 
            "RustEnforcer", "FakeDetector", "SecurityAuditor", "MathProfessor"]

STACK_V2_REPO = "bigcode/the-stack-v2"

def load_stack_comparison():
    """Load Stack v2 comparison data."""
    try:
        files = list_repo_files(REPOS[0], repo_type="dataset")
        stack_files = [f for f in files if 'stack' in f and f.endswith('.parquet')]
        
        if not stack_files:
            return pd.DataFrame()
        
        latest = sorted([f for f in stack_files if 'comparison' in f])[-1]
        path = hf_hub_download(REPOS[0], latest, repo_type="dataset")
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def create_stack_comparison_chart(df_our, df_stack):
    """Create comparison chart with The Stack v2."""
    if df_our.empty or df_stack.empty:
        return None
    
    fig = go.Figure(data=[
        go.Bar(name='Monster Project', x=['Avg Size', 'Avg Lines'], 
               y=[df_our['size'].mean(), df_our['lines'].mean()]),
        go.Bar(name='The Stack v2', x=['Avg Size', 'Avg Lines'],
               y=[df_stack['size'].mean(), df_stack.get('lines', [0]).mean()])
    ])
    
    fig.update_layout(
        title="Code Comparison: Monster vs The Stack v2",
        barmode='group'
    )
    
    return fig

def load_latest_reviews():
    """Load latest review parquet from HuggingFace datasets."""
    try:
        files = list_repo_files(REPOS[0], repo_type="dataset")
        parquet_files = [f for f in files if f.endswith('.parquet') and 'review' in f]
        
        if not parquet_files:
            return pd.DataFrame()
        
        latest = sorted(parquet_files)[-1]
        path = hf_hub_download(REPOS[0], latest, repo_type="dataset")
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def create_score_chart(df):
    """Create bar chart of persona scores."""
    if df.empty or 'reviewer' not in df.columns:
        return None
    
    fig = go.Figure(data=[
        go.Bar(x=df['reviewer'], y=df['score'], 
               marker_color='rgb(55, 83, 109)')
    ])
    
    fig.update_layout(
        title="Review Scores by Persona",
        xaxis_title="Reviewer",
        yaxis_title="Score (0-10)",
        yaxis_range=[0, 10]
    )
    
    return fig

def create_metrics_summary(df):
    """Create summary metrics."""
    if df.empty or 'score' not in df.columns:
        return "No data available"
    
    total = df['score'].sum()
    avg = df['score'].mean()
    max_score = len(df) * 10
    
    return f"""
    ### Review Summary
    - **Total Score**: {total}/{max_score} ({total/max_score*100:.1f}%)
    - **Average Score**: {avg:.1f}/10
    - **Personas**: {len(df)}
    - **Status**: {'✅ APPROVED' if total >= 70 else '⚠️ NEEDS WORK'}
    """

def refresh_dashboard():
    """Refresh all dashboard components."""
    df = load_latest_reviews()
    
    if df.empty:
        return "No reviews found", None, pd.DataFrame()
    
    summary = create_metrics_summary(df)
    chart = create_score_chart(df)
    
    return summary, chart, df

# Gradio Interface
with gr.Blocks(title="Monster Review Dashboard") as demo:
    gr.Markdown("# 👹 Monster Review Dashboard")
    gr.Markdown("Real-time tracking of 9-persona review team performance")
    
    with gr.Tabs():
        with gr.Tab("Reviews"):
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", variant="primary")
            
            with gr.Row():
                summary_md = gr.Markdown()
            
            with gr.Row():
                chart_plot = gr.Plot()
            
            with gr.Row():
                data_table = gr.Dataframe(label="Review Data")
        
        with gr.Tab("Stack v2 Comparison"):
            with gr.Row():
                stack_refresh_btn = gr.Button("🔄 Refresh Stack Data", variant="primary")
            
            with gr.Row():
                stack_chart = gr.Plot()
            
            with gr.Row():
                stack_summary = gr.Markdown()
    
    # Auto-refresh on load
    demo.load(refresh_dashboard, outputs=[summary_md, chart_plot, data_table])
    
    # Manual refresh
    refresh_btn.click(refresh_dashboard, outputs=[summary_md, chart_plot, data_table])
    
    def refresh_stack():
        df_stack = load_stack_comparison()
        if df_stack.empty:
            return None, "No Stack comparison data available"
        
        summary = f"""
        ### The Stack v2 Comparison
        - **Samples**: {len(df_stack)}
        - **Avg Size**: {df_stack['size'].mean():.0f} bytes
        - **Languages**: {', '.join(df_stack.get('language', ['N/A']).unique())}
        """
        
        return None, summary
    
    stack_refresh_btn.click(refresh_stack, outputs=[stack_chart, stack_summary])

if __name__ == "__main__":
    demo.launch()
