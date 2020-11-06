import numpy as np
import plotly.figure_factory as ff

def generate_plot(x=None):
    """
    Generates the histogram plot for the prediction.
    A html of the histogram is saved using plotly.
    PARAMS: None
    RETURNS: None
    """
    
    if type(x) != np.ndarray:
        np.random.seed(0)
        x = np.random.randn(1000)

    hist_data = [x]
    group_labels = ['Probability Distribution']
    colors = ['#A6ACEC']

    fig = ff.create_distplot(hist_data, group_labels=group_labels, colors=colors, show_curve=False,
                             bin_size=.05, show_rug=False)

    fig.update_layout(title_text='Uncertainty Plot')
    fig.write_html("src/static/prob_plot.html")

if __name__ == "__main__":
    generate_plot()