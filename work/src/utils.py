"""Something."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_trajectory(trajectory, hamiltonian, name, skip=25, **kwargs):
    """Something."""
    # Make Subplots
    fig = make_subplots(rows=2, cols=2, specs=[[{'rowspan': 2}, {}], [None, {}]],
                        column_widths=[0.625, 0.375], shared_xaxes=True)
    # Plot Trajectory
    for body in range(trajectory.q.shape[1]):
        fig.add_trace(go.Scatter(x=trajectory[::skip, 0, body, 0].detach().numpy(),
                                 y=trajectory[::skip, 0, body, 1].detach().numpy(),
                                 name='Body ' + str(body)), row=1, col=1)
    fig.update_xaxes(title='x', row=1, col=1)
    fig.update_yaxes(dict(scaleanchor='x', scaleratio=1), title='y', row=1, col=1)
    # Plot Energy
    fig.add_trace(go.Scatter(x=trajectory.t.detach().numpy()[::skip],
                             y=hamiltonian(trajectory, **kwargs).detach().numpy()[::skip],
                             name='Energy'),
                  row=1, col=2)
    fig.update_yaxes(title='Energy', row=1, col=2)
    # Plot Momentum
    fig.add_trace(go.Scatter(x=trajectory.t.detach().numpy()[::skip],
                             y=trajectory[:, 1, :, 0].sum(1).detach().numpy()[::skip],
                             name='Momentum'),
                  row=2, col=2)
    fig.update_xaxes(title='Time', row=2, col=2)
    fig.update_yaxes(title='Momentum', row=2, col=2)
    # Update Global Layout
    fig.update_layout(title_text=name, title_y=0.9, title_x=0.5, title_xanchor='center',
                      title_yanchor='top', title_font_size=24, showlegend=False)
    return fig