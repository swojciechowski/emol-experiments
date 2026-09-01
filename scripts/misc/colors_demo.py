import colorsys
import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import numpy as np

# Try importing the official glasbey library, with a fallback notification
try:
    import glasbey

    HAS_GLASBEY = True
except ImportError:
    HAS_GLASBEY = False

# ==========================================
# Color Generation Strategies
# ==========================================


def oklch_to_rgb(l, c, h_deg):
    """Converts OKLCH (Lightness 0-1, Chroma 0-0.37, Hue 0-360) to sRGB tuple."""
    h_rad = math.radians(h_deg)
    a, b = c * math.cos(h_rad), c * math.sin(h_rad)

    l_ = l + 0.3963377774 * a + 0.2158037573 * b
    m_ = l - 0.1055613458 * a - 0.0638541728 * b
    s_ = l - 0.0894841775 * a - 1.2914855480 * b

    l3, m3, s3 = l_**3, m_**3, s_**3

    r_lin = +4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
    g_lin = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
    b_lin = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3

    def gamma(x):
        x = max(0.0, min(1.0, x))
        return 12.92 * x if x <= 0.0031308 else 1.055 * (x ** (1.0 / 2.4)) - 0.055

    return (gamma(r_lin), gamma(g_lin), gamma(b_lin))


def generate_hsv(n, start_hue, s, v):
    """1. Standard HSV hue stepping."""
    offset = (start_hue % 360.0) / 360.0
    return [colorsys.hsv_to_rgb((offset + i / n) % 1.0, s, v) for i in range(n)]


def generate_oklch(n, start_hue, chroma, lightness):
    """2. Perceptually uniform OKLCH hue stepping."""
    c = chroma * 0.32  # Scale [0, 1] slider to valid OKLCH chroma range
    return [
        oklch_to_rgb(lightness, c, (start_hue + i * 360.0 / n) % 360.0)
        for i in range(n)
    ]


def generate_official_glasbey(n, chroma, lightness):
    """3. Official Glasbey Library palette generation (CAM02-UCS / JCh optimization)."""
    if not HAS_GLASBEY:
        # Fallback if pip install glasbey wasn't run
        return generate_hsv(n, 0.0, chroma, lightness)

    # Convert sliders [0, 1] to Glasbey lightness/chroma bounds
    c_max = max(10, int(chroma * 100))
    l_max = max(20, int(lightness * 100))

    raw_colors = glasbey.create_palette(
        palette_size=n,
        chroma_bounds=(0, c_max),
        lightness_bounds=(10, l_max),
        as_hex=False,
    )

    # Normalize to (R, G, B) float tuples [0.0, 1.0]
    return [mcolors.to_rgb(c) for c in raw_colors]


# ==========================================
# GUI Application Setup
# ==========================================

fig, ax = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(bottom=0.42, left=0.28)

# Initial Parameters
init_n = 14
init_hue = 0.0
init_p1 = 0.85
init_p2 = 0.85
current_method = "Official Glasbey" if HAS_GLASBEY else "HSV (Standard)"

# Initial rendering
if HAS_GLASBEY:
    colors_rgb = generate_official_glasbey(init_n, init_p1, init_p2)
else:
    colors_rgb = generate_hsv(init_n, init_hue, init_p1, init_p2)

img_data = np.array(colors_rgb)[np.newaxis, :, :]
img_obj = ax.imshow(
    img_data, extent=[0, 1, 0, 1], aspect="auto", origin="lower"
)
ax.set_title(f"Method: {current_method} | {init_n} Colors")
ax.set_xticks([])
ax.set_yticks([])

# Controls Axes
ax_radio = plt.axes([0.03, 0.45, 0.20, 0.35])
ax_n = plt.axes([0.33, 0.28, 0.58, 0.03])
ax_hue = plt.axes([0.33, 0.21, 0.58, 0.03])
ax_p1 = plt.axes([0.33, 0.14, 0.58, 0.03])
ax_p2 = plt.axes([0.33, 0.07, 0.58, 0.03])

# Interactive Widgets
active_idx = 2 if HAS_GLASBEY else 0
radio = RadioButtons(
    ax_radio,
    ("HSV (Standard)", "OKLCH (Uniform)", "Official Glasbey"),
    active=active_idx,
)
s_n = Slider(ax_n, "N Colors", 1, 40, valinit=init_n, valstep=1)
s_hue = Slider(ax_hue, "Start Hue (°)", 0, 360, valinit=init_hue, valstep=1)
s_p1 = Slider(
    ax_p1, "Sat / Max Chroma", 0.0, 1.0, valinit=init_p1, valstep=0.01
)
s_p2 = Slider(
    ax_p2, "Val / Max Lightness", 0.0, 1.0, valinit=init_p2, valstep=0.01
)


def update(val):
    method = radio.value_selected
    n = int(s_n.val)
    hue = s_hue.val
    p1 = s_p1.val
    p2 = s_p2.val

    if method == "HSV (Standard)":
        colors = generate_hsv(n, hue, p1, p2)
    elif method == "OKLCH (Uniform)":
        colors = generate_oklch(n, hue, p1, p2)
    elif method == "Official Glasbey":
        colors = generate_official_glasbey(n, p1, p2)

    new_img_data = np.array(colors)[np.newaxis, :, :]
    img_obj.set_data(new_img_data)
    ax.set_title(f"Method: {method} | {n} Colors")
    fig.canvas.draw_idle()


# Attach callbacks
s_n.on_changed(update)
s_hue.on_changed(update)
s_p1.on_changed(update)
s_p2.on_changed(update)
radio.on_clicked(update)

plt.show()