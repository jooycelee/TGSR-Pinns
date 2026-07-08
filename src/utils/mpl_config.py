DEFAULT_SANS_FONTS = [
    'Microsoft YaHei',
    'SimHei',
    'Noto Sans CJK SC',
    'Source Han Sans SC',
    'WenQuanYi Zen Hei',
    'DejaVu Sans',
]


def configure_matplotlib(plt):
    """Apply a consistent font/mathtype setup for plotting utilities."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = DEFAULT_SANS_FONTS
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    plt.rcParams['mathtext.default'] = 'regular'
