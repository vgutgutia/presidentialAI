"""
Visualization utilities for marine debris detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import rasterio


def visualize_prediction(
    image: np.ndarray,
    probability_map: np.ndarray,
    mask: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    title: str = "Marine Debris Detection",
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Visualize prediction results.
    
    Args:
        image: Input image (C, H, W) - will use RGB bands if available
        probability_map: Debris probability map (H, W)
        mask: Ground truth mask (optional)
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_plots = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # Prepare RGB image for display
    if image.shape[0] >= 3:
        # Assume bands are in order: B2, B3, B4 (Blue, Green, Red)
        rgb = np.stack([image[2], image[1], image[0]], axis=-1)  # RGB
        rgb = np.clip(rgb * 3, 0, 1)  # Enhance contrast
    else:
        rgb = np.stack([image[0]] * 3, axis=-1)
    
    # Plot 1: Input image
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 Image")
    axes[0].axis("off")
    
    # Plot 2: Probability heatmap
    im = axes[1].imshow(probability_map, cmap="YlOrRd", vmin=0, vmax=1)
    axes[1].set_title("Debris Probability")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot 3: Ground truth (if available)
    if mask is not None:
        # Overlay
        overlay = rgb.copy()
        overlay[mask == 1] = [1, 0, 0]  # Red for debris
        axes[2].imshow(overlay)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")
        
        # Legend
        legend_elements = [
            Patch(facecolor="red", label="Marine Debris"),
        ]
        axes[2].legend(handles=legend_elements, loc="lower right")
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def visualize_hotspots(
    image: np.ndarray,
    hotspots: List[Dict],
    output_path: Optional[str] = None,
    title: str = "Detected Debris Hotspots",
) -> plt.Figure:
    """
    Visualize detected hotspots on image.
    
    Args:
        image: Input image
        hotspots: List of hotspot dictionaries with coordinates
        output_path: Path to save figure
        title: Figure title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Prepare RGB
    if image.shape[0] >= 3:
        rgb = np.stack([image[2], image[1], image[0]], axis=-1)
        rgb = np.clip(rgb * 3, 0, 1)
    else:
        rgb = np.stack([image[0]] * 3, axis=-1)
    
    ax.imshow(rgb)
    
    # Plot hotspots
    for i, hotspot in enumerate(hotspots[:20]):  # Top 20
        y, x = hotspot.get("centroid_row", 0), hotspot.get("centroid_col", 0)
        confidence = hotspot.get("confidence", 0)
        
        # Size based on area, color based on confidence
        size = np.sqrt(hotspot.get("area_pixels", 100)) * 2
        color = plt.cm.Reds(confidence)
        
        circle = plt.Circle(
            (x, y), size,
            fill=False, color=color, linewidth=2
        )
        ax.add_patch(circle)
        ax.annotate(
            f"{i+1}", (x, y),
            color="white", fontsize=8,
            ha="center", va="center",
            fontweight="bold",
        )
    
    ax.set_title(title)
    ax.axis("off")
    
    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax, label="Confidence", fraction=0.046, pad=0.04)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_training_history(
    history: Dict[str, List],
    output_path: Optional[str] = None,
    metrics: List[str] = None,
) -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        output_path: Path to save figure
        metrics: Metrics to plot
        
    Returns:
        Matplotlib figure
    """
    metrics = metrics or ["loss", "iou_debris", "dice_debris"]
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        train_values = [epoch.get(metric, 0) for epoch in history.get("train", [])]
        val_values = [epoch.get(metric, 0) for epoch in history.get("val", [])]
        
        epochs = range(1, len(train_values) + 1)
        
        ax.plot(epochs, train_values, "b-", label="Train")
        ax.plot(epochs, val_values, "r-", label="Val")
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def create_interactive_map(
    hotspots_gdf,
    output_path: str,
    tile_layer: str = "OpenStreetMap",
) -> str:
    """
    Create an interactive Folium map of hotspots.
    
    Args:
        hotspots_gdf: GeoDataFrame of hotspots
        output_path: Path to save HTML map
        tile_layer: Base map tile layer
        
    Returns:
        Path to HTML file
    """
    import folium
    
    # Get center point
    bounds = hotspots_gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Add hotspots
    for _, row in hotspots_gdf.iterrows():
        # Color based on confidence
        color = "red" if row["confidence"] > 0.7 else "orange" if row["confidence"] > 0.5 else "yellow"
        
        # Add polygon
        folium.GeoJson(
            row["geometry"],
            style_function=lambda x, color=color: {
                "fillColor": color,
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.5,
            },
            popup=folium.Popup(
                f"<b>Rank:</b> {row['rank']}<br>"
                f"<b>Confidence:</b> {row['confidence']:.2f}<br>"
                f"<b>Area:</b> {row['area_m2']/1000:.2f} km²",
                max_width=200,
            ),
        ).add_to(m)
        
        # Add marker at centroid
        folium.Marker(
            [row["centroid_lat"], row["centroid_lon"]],
            popup=f"Hotspot #{row['rank']}",
            icon=folium.Icon(color=color, icon="exclamation-triangle", prefix="fa"),
        ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray;">
        <b>Debris Confidence</b><br>
        <i style="background:red; width:12px; height:12px; display:inline-block;"></i> High (>0.7)<br>
        <i style="background:orange; width:12px; height:12px; display:inline-block;"></i> Medium (0.5-0.7)<br>
        <i style="background:yellow; width:12px; height:12px; display:inline-block;"></i> Low (<0.5)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save
    m.save(output_path)
    
    return output_path


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str] = None,
    output_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: Class names
        output_path: Path to save figure
        normalize: Whether to normalize
        
    Returns:
        Matplotlib figure
    """
    class_names = class_names or ["Background", "Marine Debris"]
    
    if normalize:
        cm = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1, keepdims=True)
    else:
        cm = confusion_matrix
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    im = ax.imshow(cm, cmap="Blues")
    
    # Labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]}"
            ax.text(j, i, text, ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig
