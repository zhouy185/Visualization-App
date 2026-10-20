import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="CSV Visualization Dashboard", layout="wide")
st.title("📊 CSV Visualization Dashboard")

NONE = "(none)"
MAX_LEVELS = 50  # warn when a categorical variable has more levels than this


# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read a CSV, trying a few common encodings; make column names unique strings."""
    import io

    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, sep=None, engine="python")
            break
        except UnicodeDecodeError as e:
            last_err = e
    else:
        raise last_err
    cols, seen = [], {}
    for c in df.columns:
        c = str(c).strip() or "unnamed"
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    df.columns = cols
    return df


def split_columns(df: pd.DataFrame):
    """Numeric = numeric non-bool dtype; everything else is treated as categorical/text."""
    num = [c for c in df.columns
           if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    cat = [c for c in df.columns if c not in num]
    return num, cat


def as_cat(s: pd.Series) -> pd.Series:
    """Make a column safe to use as a category (handles mixed types); keep NaN as NaN."""
    return s.astype("object").where(s.isna(), s.astype(str))


def opt(label, options, key, help=None):
    return st.sidebar.selectbox(label, [NONE] + list(options), key=key, help=help)


def range_control(label: str, series: pd.Series, key: str):
    """Checkbox + range slider initialised from the data. Returns [lo, hi] or None (auto)."""
    if not st.sidebar.checkbox(f"Set custom {label} range", key=f"{key}_chk"):
        return None
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        st.sidebar.warning(f"No valid values to set a {label} range.")
        return None
    lo, hi = float(s.min()), float(s.max())
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    pad = (hi - lo) * 0.05
    return list(st.sidebar.slider(f"{label} range", lo - pad, hi + pad, (lo, hi), key=f"{key}_rng"))


def apply_ranges(fig, xr=None, yr=None):
    if xr:
        fig.update_xaxes(range=xr)
    if yr:
        fig.update_yaxes(range=yr)


def check_levels(df, col):
    n = df[col].nunique()
    if n > MAX_LEVELS:
        st.warning(f"'{col}' has {n} distinct values; the chart may be cluttered.")


# ---------- Chart builders ----------
def histogram(df, num, cat):
    x = st.sidebar.selectbox("Numeric variable", num, key="h_x")
    bins = st.sidebar.slider("Number of bins", 2, 200, 30, key="h_bins")
    color = opt("Color / group (optional)", cat, "h_color")
    opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.7, 0.05, key="h_op")
    cols = [x] + ([color] if color != NONE else [])
    d = df[cols].copy()
    d[x] = d[x].replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[x])
    if d.empty:
        return None, "The selected variable has no valid values."
    if color != NONE:
        check_levels(df, color)
        d[color] = as_cat(d[color]).fillna("(missing)")
    xr = range_control("X", d[x], "h_x")
    max_count = int(np.histogram(d[x], bins=bins)[0].max())
    yr = None
    if st.sidebar.checkbox("Set custom Y range", key="h_y_chk"):
        yr = list(st.sidebar.slider("Y range (count)", 0, max(2, int(max_count * 1.5)),
                                    (0, max_count), key="h_y_rng"))
    fig = px.histogram(d, x=x, color=None if color == NONE else color, nbins=bins,
                       opacity=opacity, barmode="overlay", title=f"Distribution of {x}",
                       labels={x: x, "count": "Count"})
    fig.update_traces(hovertemplate=f"{x}=%{{x}}<br>Count=%{{y}}<extra>%{{fullData.name}}</extra>")
    fig.update_yaxes(title_text="Count")
    apply_ranges(fig, xr, yr)
    return fig, None


def bar(df, num, cat):
    x = st.sidebar.selectbox("Categorical variable (x-axis)", cat, key="b_x")
    options = ["Count of observations"] + num
    val = st.sidebar.selectbox("Value to plot", options, key="b_val")
    agg = "count"
    if val != options[0]:
        agg = st.sidebar.selectbox("Aggregation", ["mean", "median", "sum", "count"], key="b_agg")
    color = opt("Color / group (optional)", [c for c in cat if c != x], "b_color")
    mode = st.sidebar.radio("Bar mode", ["group", "stack"], horizontal=True, key="b_mode",
                            format_func=lambda m: "Grouped" if m == "group" else "Stacked")
    sort = st.sidebar.checkbox("Sort categories by plotted value", key="b_sort")

    cols = [x] + ([color] if color != NONE else []) + ([] if val == options[0] else [val])
    d = df[cols].copy()
    check_levels(d, x)
    d[x] = as_cat(d[x]).fillna("(missing)")
    keys = [x]
    if color != NONE:
        d[color] = as_cat(d[color]).fillna("(missing)")
        keys.append(color)
    if val == options[0]:
        g = d.groupby(keys, observed=True).size().reset_index(name="value")
        ylab = "Count of observations"
    else:
        d[val] = d[val].replace([np.inf, -np.inf], np.nan)
        g = d.groupby(keys, observed=True)[val].agg(agg).reset_index(name="value")
        ylab = f"{agg.capitalize()} of {val}"
        g = g.dropna(subset=["value"])
    if g.empty:
        return None, "No data left to plot after removing missing values."
    order = None
    if sort:
        order = g.groupby(x)["value"].sum().sort_values(ascending=False).index.tolist()
    fig = px.bar(g, x=x, y="value", color=None if color == NONE else color, barmode=mode,
                 category_orders={x: order} if order else None,
                 labels={"value": ylab}, title=f"{ylab} by {x}")
    fig.update_yaxes(title_text=ylab)
    return fig, None


def box(df, num, cat):
    y = st.sidebar.selectbox("Numeric variable", num, key="x_y")
    grp = opt("Compare groups by (optional)", cat, "x_grp")
    color = opt("Color by (optional)", cat, "x_color")
    pts = st.sidebar.radio("Show points", ["none", "outliers only", "all points"], key="x_pts")
    cols = list(dict.fromkeys([y] + [c for c in (grp, color) if c != NONE]))
    d = df[cols].copy()
    d[y] = d[y].replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[y])
    if d.empty:
        return None, "The selected variable has no valid values."
    for c in (grp, color):
        if c != NONE:
            check_levels(df, c)
            d[c] = as_cat(d[c]).fillna("(missing)")
    yr = range_control("Y", d[y], "x")
    fig = px.box(d, y=y, x=None if grp == NONE else grp, color=None if color == NONE else color,
                 points={"none": False, "outliers only": "outliers", "all points": "all"}[pts],
                 title=f"Distribution of {y}" + ("" if grp == NONE else f" by {grp}"))
    apply_ranges(fig, None, yr)
    return fig, None


def scatter(df, num, cat):
    if len(num) < 2:
        return None, "A scatter plot needs at least two numeric columns."
    x = st.sidebar.selectbox("X (numeric)", num, index=0, key="s_x")
    y = st.sidebar.selectbox("Y (numeric)", num, index=1, key="s_y")
    color = opt("Color (optional)", [c for c in df.columns if c not in (x, y)], "s_color")
    size = opt("Size (optional, numeric)", num, "s_size")
    symbol = opt("Symbol (optional)", cat, "s_symbol")
    hover = st.sidebar.multiselect("Extra hover variables", list(df.columns), key="s_hover")
    opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.7, 0.05, key="s_op")
    trend = st.sidebar.checkbox("Add trend line", key="s_trend")

    # Trend line is only valid for numeric x and y (guaranteed by the selectors; checked defensively)
    if trend and not (x in num and y in num):
        st.warning("Trend line requires numeric X and Y variables.")
        trend = False

    extras = [c for c in (color, size, symbol) if c != NONE]
    cols = list(dict.fromkeys([x, y] + extras + hover))
    d = df[cols].copy()
    for c in (x, y):
        d[c] = d[c].replace([np.inf, -np.inf], np.nan)
    required = [x, y] + ([size] if size != NONE else [])
    d = d.dropna(subset=required)
    if size != NONE:
        d[size] = d[size].replace([np.inf, -np.inf], np.nan)
        d = d.dropna(subset=[size])
        if (d[size] < 0).any():
            st.warning(f"'{size}' has negative values; those rows were dropped for point sizing.")
            d = d[d[size] >= 0]
    if d.empty:
        return None, "No complete rows remain for the selected variables."
    for c in [color, symbol] + hover:
        if c != NONE and c in d and c not in num:
            d[c] = as_cat(d[c]).fillna("(missing)")
    if color != NONE and color not in num:
        check_levels(d, color)
    if symbol != NONE:
        check_levels(d, symbol)
    xr = range_control("X", d[x], "s_x")
    yr = range_control("Y", d[y], "s_y")
    kw = dict(x=x, y=y, opacity=opacity, hover_data=[h for h in hover if h not in (x, y)] or None,
              title=f"{y} vs {x}")
    if color != NONE:
        kw["color"] = color
    if size != NONE:
        kw["size"] = size
    if symbol != NONE:
        kw["symbol"] = symbol
    if trend:
        if len(d) < 2:
            return None, "Not enough points for a trend line."
        kw["trendline"] = "ols"  # requires statsmodels
    try:
        fig = px.scatter(d, **kw)
    except ImportError:
        return None, "Trend lines require `statsmodels` (pip install statsmodels)."
    apply_ranges(fig, xr, yr)
    return fig, None


QUESTIONS = {
    "Histogram": ["What is the shape of the distribution?", "Are there outliers?",
                  "Is the distribution symmetric or skewed?"],
    "Box plot": ["Do the groups differ in median?", "Which groups show more variability?",
                 "Are there notable outliers?"],
    "Scatter plot": ["Is there an apparent relationship between X and Y?",
                     "Is the relationship linear?", "Are there clusters or unusual observations?",
                     "Do color or size reveal additional patterns?"],
    "Bar chart": ["Which categories are largest or smallest?",
                  "Are differences economically meaningful?",
                  "What business decision might this comparison inform?"],
}

# ---------- Upload and overview ----------
upload = st.file_uploader("Upload a CSV file", type="csv")
if upload is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

try:
    df = load_csv(upload.getvalue())
except Exception as e:
    st.error(f"Could not read this file as CSV: {e}")
    st.stop()

if df.empty or df.shape[1] == 0:
    st.error("The uploaded file contains no data.")
    st.stop()

num_cols, cat_cols = split_columns(df)
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
st.dataframe(df.head(10), width="stretch")
with st.expander("Column types"):
    st.write(f"**Numeric ({len(num_cols)}):** {', '.join(num_cols) or '—'}")
    st.write(f"**Categorical/text ({len(cat_cols)}):** {', '.join(cat_cols) or '—'}")

# ---------- Sidebar controls ----------
st.sidebar.header("Chart controls")
available = []
if num_cols:
    available += ["Histogram", "Box plot"]
if cat_cols:
    available.append("Bar chart")
if len(num_cols) >= 2:
    available.append("Scatter plot")
if not available:
    st.error("No usable columns found in this dataset.")
    st.stop()

chart = st.sidebar.selectbox("Chart type", available)
if not num_cols:
    st.sidebar.caption("Only categorical columns found: numeric charts are unavailable.")
elif not cat_cols:
    st.sidebar.caption("Only numeric columns found: bar chart is unavailable.")

builders = {"Histogram": histogram, "Bar chart": bar, "Box plot": box, "Scatter plot": scatter}
st.subheader(chart)
try:
    fig, problem = builders[chart](df, num_cols, cat_cols)
except Exception as e:
    fig, problem = None, f"Could not build this chart: {e}"

if problem:
    st.warning(problem)
elif fig is not None:
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Questions to consider"):
    for q in QUESTIONS[chart]:
        st.markdown(f"- {q}")
