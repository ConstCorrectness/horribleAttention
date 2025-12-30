import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def make_attention_matrices(tokens):
    n = len(tokens)
    # base: small random noise
    base = np.full((n, n), 0.02)

    # context variant: 'tired' -> 'it' (7) attends to 'animal' (1)
    tired = base.copy()
    tired[7] = 0.02
    tired[7, 1] = 0.75
    tired[7, 0] = 0.05
    tired[7, 5] = 0.05
    tired = tired / tired.sum(axis=1, keepdims=True)

    # context variant: 'wide' -> 'it' (7) attends to 'street' (5)
    wide = base.copy()
    wide[7] = 0.02
    wide[7, 5] = 0.75
    wide[7, 1] = 0.05
    wide[7, 0] = 0.05
    wide = wide / wide.sum(axis=1, keepdims=True)

    uniform = np.full((n, n), 1.0 / n)
    return {"Too Tired": tired, "Too Wide": wide, "Uniform": uniform}

def draw_token_arcs(tokens, attn_row):
    n = len(tokens)
    
    x = np.linspace(0, 1, n)
    y = np.zeros(n)

    fig = go.Figure()

    # draw token markers with labels
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers+text", text=tokens,
                             textposition="bottom center", marker=dict(size=18, color="#06B6D4"),
                             hoverinfo="text"))

    # draw arcs/lines from selected token to others
    src_idx = np.argmax(attn_row) if False else None

    # allow explicit src selection by passing a row that is the query
    # here attn_row is already the selected query's attention vector
    for i, w in enumerate(attn_row):
        if w <= 0.001:
            continue
        # straight line with slight vertical offset proportional to distance
        xi, xj = x[7], x[i]
        # create a curved path using quadratic Bezier via many segments
        mx = (xi + xj) / 2
        height = 0.2 * abs(i - 7)  # scale by distance
        xs = np.linspace(xi, xj, 30)
        ys = height * (4 * (xs - mx) ** 2) / ((xj - xi) ** 2 + 1e-9)  # inverted parabola
        ys = np.max(ys) - ys  # flip to create arch
        # shift so baseline at 0.08
        ys = ys * 0.5 + 0.08
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=1 + 8 * w, color="rgba(6,182,212,0.8)"), hoverinfo="none"))

    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=10, r=10, t=10, b=40), height=320,
                      plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(range=[-0.05, 0.5])
    return fig


def main():
    st.set_page_config(page_title="Attention Explorer", layout="wide")
    st.title("Visualizing Attention Mechanisms — Streamlit")
    st.markdown(
        """
        Explore a simplified, interactive view of attention. Select a context variant and a token to see where it attends.
        """
    )
    tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired", "."]
    matrices = make_attention_matrices(tokens)
    col1, col2 = st.columns([3, 2])
    with col1:
        ctx = st.radio("Context Variant", options=list(matrices.keys()), index=0)
        st.markdown("**Sentence**")
        st.write(" ")

        # Display tokens in a single line
        token_sel = st.selectbox("Select token (query)", options=[f"{i}: {t}" for i, t in enumerate(tokens)], index=7)
        sel_idx = int(token_sel.split(":")[0])

        attn = matrices[ctx]
        attn_row = attn[sel_idx]

        st.markdown("**Attention arcs**")
        fig = draw_token_arcs(tokens, attn_row)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Explanation**")
        if ctx == "Too Tired":
            st.info("When the sentence ends with 'tired', the model attends from 'it' to 'animal', so 'it' refers to the animal.")
        elif ctx == "Too Wide":
            st.info("In the alternate context, 'it' can refer to 'street' — attention shifts accordingly.")
        else:
            st.info("Uniform attention for demonstration.")

    with col2:
        st.markdown("**Attention Matrix (heatmap)**")
        hm = matrices[ctx]
        fig2 = px.imshow(hm, x=tokens, y=[f"q={i}" for i in range(len(tokens))], color_continuous_scale="Viridis")
        fig2.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Controls / Tweak**")
        st.write("Adjust the strength of the top attention link (live-preview)")
        strength = st.slider("Top-link strength", min_value=0.0, max_value=1.0, value=0.75)

        # apply tweak to the selected context dynamically
        if ctx in ["Too Tired", "Too Wide"]:
            m = matrices[ctx].copy()
            # normalize but keep relative distribution
            m[sel_idx] = np.maximum(m[sel_idx], 0.02)
            # find largest index (for demo we force same important idx)
            if ctx == "Too Tired":
                target = 1
            else:
                target = 5
            m[sel_idx, :] = 0.02
            m[sel_idx, target] = strength
            m[sel_idx] += 0.0001
            m[sel_idx] = m[sel_idx] / m[sel_idx].sum()
            st.markdown("**Updated attention row**")
            st.write({tokens[i]: float(f) for i, f in enumerate(m[sel_idx])})

    st.markdown("---")
    st.markdown("**Notes**: This is a simplified didactic visualization — real transformers use multi-head attention and learned Q/K/V projections.")

if __name__ == "__main__":
    main()
