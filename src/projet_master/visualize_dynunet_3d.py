"""
Interactive 3D visualization of the DynUNet architecture used in AVCSegModel.

Install:
    pip install plotly

Run:
    python visualize_dynunet_3d.py

Output:
    dynunet_3d_architecture.html

"""

import plotly.graph_objects as go


# ---------------------------------------------------------------------
# Architecture copied from AVCSegModel
# ---------------------------------------------------------------------
INPUT_SHAPE = (1, 20, 256, 256)  # (C, D, H, W)

FILTERS = [16, 32, 64, 128, 256]
STRIDES = [
    (1, 1, 1),
    (1, 2, 2),
    (1, 2, 2),
    (1, 2, 2),
    (1, 2, 2),
]
DEEP_SUPERVISION_HEADS = 3


def feature_shapes():
    """Return (D, H, W) at each encoder level."""
    d, h, w = INPUT_SHAPE[1:]
    shapes = []
    for i, channels in enumerate(FILTERS):
        if i > 0:
            sd, sh, sw = STRIDES[i]
            d, h, w = d // sd, h // sh, w // sw
        shapes.append((channels, d, h, w))
    return shapes


def cuboid_trace(x, y, z, dx, dy, dz, color, name, hover_text):
    """Create a cuboid Mesh3d."""
    vertices = [
        (x,      y,      z),
        (x + dx, y,      z),
        (x + dx, y + dy, z),
        (x,      y + dy, z),
        (x,      y,      z + dz),
        (x + dx, y,      z + dz),
        (x + dx, y + dy, z + dz),
        (x,      y + dy, z + dz),
    ]

    xs, ys, zs = zip(*vertices)

    # Two triangles per face
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]

    return go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=i, j=j, k=k,
        color=color,
        opacity=0.88,
        flatshading=True,
        name=name,
        hovertext=hover_text,
        hoverinfo="text",
        showlegend=False,
    )


def add_arrow(fig, start, end, color="#555555", width=5):
    """Add a 3D connector with an arrowhead."""
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.add_trace(go.Cone(
        x=[end[0]], y=[end[1]], z=[end[2]],
        u=[end[0] - start[0]],
        v=[end[1] - start[1]],
        w=[end[2] - start[2]],
        anchor="tip",
        sizemode="absolute",
        sizeref=0.22,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        hoverinfo="skip",
        showlegend=False,
    ))


def add_label(fig, x, y, z, text, color="#202020", size=11):
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode="text",
        text=[text],
        textfont=dict(size=size, color=color),
        hoverinfo="skip",
        showlegend=False,
    ))


def main():
    fig = go.Figure()
    shapes = feature_shapes()

    # Coordinate convention:
    # X = progression through the network
    # Y = spatial feature-map height/width (schematic square footprint)
    # Z = channel dimension (schematic thickness)
    #
    # D is retained in the labels because it remains 20 throughout this model.
    spatial_scale = 0.018
    channel_scale = 0.012
    block_length = 0.48

    encoder_positions = [0.0, 1.0, 2.0, 3.0, 4.0]
    decoder_positions = [6.0, 7.0, 8.0, 9.0]
    colors = [
        "#4C9BE8", "#3987D1", "#2875B9", "#1E639F", "#174F83"
    ]

    # Input volume
    input_side = 256 * spatial_scale
    input_thickness = 1.0
    fig.add_trace(cuboid_trace(
        -1.0, -input_side / 2, 0,
        0.35, input_side, input_thickness,
        "#AAB2BD", "Input volume",
        "Input: 1 × 20 × 256 × 256 (C × D × H × W)"
    ))
    add_label(fig, -0.82, -input_side / 2 - 0.35, 0.5,
              "Input\n1 × 20 × 256 × 256")

    # Encoder levels
    encoder_centers = []
    for level, (channels, d, h, w) in enumerate(shapes):
        x = encoder_positions[level]
        side = h * spatial_scale
        thickness = max(0.35, channels * channel_scale)
        y = -side / 2
        z = 0

        name = f"Encoder {level + 1}"
        fig.add_trace(cuboid_trace(
            x, y, z, block_length, side, thickness,
            colors[level], name,
            f"{name}<br>Channels: {channels}<br>"
            f"Spatial size (D×H×W): {d}×{h}×{w}"
        ))
        encoder_centers.append((x + block_length / 2, 0, thickness / 2))
        add_label(
            fig, x + block_length / 2, -side / 2 - 0.30,
            thickness + 0.08,
            f"E{level + 1}: {channels} ch<br>{d}×{h}×{w}",
            size=10,
        )

    # Bottleneck is the deepest encoder representation in this schematic.
    bottleneck = encoder_centers[-1]

    # Decoder levels: reverse the encoder resolutions, excluding deepest level.
    decoder_specs = [
        (128, 20, 32, 32),
        (64, 20, 64, 64),
        (32, 20, 128, 128),
        (16, 20, 256, 256),
    ]

    decoder_centers = []
    decoder_colors = ["#E5A13D", "#E8B45D", "#EBC77F", "#F0D69B"]

    for idx, (channels, d, h, w) in enumerate(decoder_specs):
        x = decoder_positions[idx]
        side = h * spatial_scale
        thickness = max(0.35, channels * channel_scale)
        y = -side / 2
        z = 0

        fig.add_trace(cuboid_trace(
            x, y, z, block_length, side, thickness,
            decoder_colors[idx],
            f"Decoder {idx + 1}",
            f"Decoder {idx + 1}<br>Channels: {channels}<br>"
            f"Output feature map: {d}×{h}×{w}<br>"
            "Includes upsampling and concatenation with encoder skip"
        ))
        center = (x + block_length / 2, 0, thickness / 2)
        decoder_centers.append(center)

        add_label(
            fig, center[0], -side / 2 - 0.30, thickness + 0.08,
            f"D{idx + 1}: {channels} ch<br>{d}×{h}×{w}",
            size=10,
        )

    # Input -> encoder and encoder progression
    add_arrow(fig, ( -0.65, 0, 0.5), encoder_centers[0])
    for a, b in zip(encoder_centers[:-1], encoder_centers[1:]):
        add_arrow(fig, a, b)

    # Bottleneck -> decoder -> output
    add_arrow(fig, encoder_centers[-1], decoder_centers[0])
    for a, b in zip(decoder_centers[:-1], decoder_centers[1:]):
        add_arrow(fig, a, b)

    # Skip connections from encoder levels to matching decoder levels.
    # Draw them as elevated arcs using line segments.
    skip_pairs = [
        (encoder_centers[3], decoder_centers[0]),
        (encoder_centers[2], decoder_centers[1]),
        (encoder_centers[1], decoder_centers[2]),
        (encoder_centers[0], decoder_centers[3]),
    ]

    for idx, (start, end) in enumerate(skip_pairs):
        z_arc = max(start[2], end[2]) + 1.2 + idx * 0.12
        x_mid = (start[0] + end[0]) / 2
        fig.add_trace(go.Scatter3d(
            x=[start[0], start[0], x_mid, end[0], end[0]],
            y=[start[1], 0, 0, 0, end[1]],
            z=[start[2], z_arc, z_arc, z_arc, end[2]],
            mode="lines",
            line=dict(color="#2E9D65", width=5, dash="dash"),
            name="Skip connection" if idx == 0 else None,
            hovertext="U-Net skip connection: encoder feature concatenated with decoder feature",
            hoverinfo="text",
            showlegend=(idx == 0),
        ))

    # Final 1-channel segmentation output
    output_x = 10.0
    output_side = 256 * spatial_scale
    fig.add_trace(cuboid_trace(
        output_x, -output_side / 2, 0,
        0.35, output_side, 0.30,
        "#C85B65", "Segmentation output",
        "Final output: 1 × 20 × 256 × 256<br>One-channel lesion segmentation logits"
    ))
    add_arrow(fig, decoder_centers[-1], (output_x, 0, 0.15))
    add_label(fig, output_x + 0.2, -output_side / 2 - 0.35, 0.4,
              "Output\n1 × 20 × 256 × 256", size=10)

    # Deep supervision heads: auxiliary predictions from intermediate decoder levels.
    ds_decoder_indices = [0, 1, 2]
    for head_idx, decoder_idx in enumerate(ds_decoder_indices, start=1):
        cx, cy, cz = decoder_centers[decoder_idx]
        ds_x = cx
        ds_y = 2.0
        ds_z = cz + 0.8

        fig.add_trace(go.Scatter3d(
            x=[cx, ds_x],
            y=[cy, ds_y],
            z=[cz, ds_z],
            mode="lines",
            line=dict(color="#9B59B6", width=3, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[ds_x], y=[ds_y], z=[ds_z],
            mode="markers+text",
            marker=dict(size=8, color="#9B59B6", symbol="diamond"),
            text=[f"DS head {head_idx}"],
            textposition="top center",
            textfont=dict(size=10, color="#7D3C98"),
            name="Deep supervision" if head_idx == 1 else None,
            hovertext=f"Auxiliary deep-supervision output {head_idx}",
            hoverinfo="text",
            showlegend=(head_idx == 1),
        ))

    fig.update_layout(
        title=dict(
            text=(
                "3D DynUNet architecture — AVCSegModel"
                "<br><sup>Input: 1×20×256×256 | Filters: 16–32–64–128–256 "
                "| Deep supervision: 3 auxiliary heads</sup>"
            ),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="Network progression", showbackground=True),
            yaxis=dict(title="Feature-map spatial extent (schematic)", showbackground=True),
            zaxis=dict(title="Channels (schematic thickness)", showbackground=True),
            aspectmode="manual",
            aspectratio=dict(x=2.4, y=1.25, z=0.9),
            camera=dict(eye=dict(x=1.7, y=1.6, z=1.3)),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=0, r=0, t=100, b=0),
        template="plotly_white",
    )

    output_file = "dynunet_3d_architecture.html"
    fig.write_html(output_file, include_plotlyjs=True)
    fig.show()


if __name__ == "__main__":
    main()
