import os
import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_seepage_results(model, dataset, final_loss, save_dir):
    model.eval()
    model.cpu()

    v = dataset.vertices
    h_up, h_down = dataset.h_up, dataset.h_down
    x_min, y_min = v[0]
    x_max_dam = v[1][0]
    y_max = v[2][1]

    target_dtype = next(model.parameters()).dtype

    xx, yy = np.meshgrid(np.linspace(x_min, x_max_dam, 400), np.linspace(y_min, y_max, 400))
    xy_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=target_dtype)

    with torch.no_grad():
        H_pred = model.forward_head(xy_tensor).numpy().reshape(xx.shape)

    # Compute velocity field q = -k * grad(H) on the mesh
    k = getattr(dataset, 'k', 1.0)
    S = model.scale_global
    H_scale = model.scale_h
    xy_norm = (xy_tensor / S).clone().detach().requires_grad_(True)
    u = model.net(xy_norm)
    du = torch.autograd.grad(
        outputs=u, inputs=xy_norm, grad_outputs=torch.ones_like(u),
        create_graph=False, retain_graph=False
    )[0]
    dH_dx = (du[:, 0:1] * H_scale / S).detach().numpy().reshape(xx.shape)
    dH_dy = (du[:, 1:2] * H_scale / S).detach().numpy().reshape(xx.shape)
    qx = -k * dH_dx
    qy = -k * dH_dy

    # Masks
    slope_l = (v[2][0] - v[0][0]) / y_max
    slope_r = (v[3][0] - v[1][0]) / y_max
    x_left_bound = x_min + yy * slope_l
    x_right_bound = x_max_dam + yy * slope_r
    mask_outside = (xx < x_left_bound) | (xx > x_right_bound)

    fs_y_interp = np.interp(xx, dataset.fs_x, dataset.fs_y)
    mask_dry = yy > fs_y_interp

    H_pred[mask_outside | mask_dry] = np.nan
    H_pred = np.clip(H_pred, h_down, h_up)

    # Mask velocity field the same way
    qx[mask_outside | mask_dry] = np.nan
    qy[mask_outside | mask_dry] = np.nan

    os.makedirs(save_dir, exist_ok=True)
    data_save_path = os.path.join(save_dir, "seepage_plot_data.npz")
    np.savez(data_save_path,
             xx=xx, yy=yy, H_pred=H_pred,
             fs_x=dataset.fs_x, fs_y=dataset.fs_y,
             vertices=np.array(v),
             h_up=h_up, h_down=h_down,
             final_loss=final_loss,
             qx=qx, qy=qy)

    # Dynamically size figure to match data aspect ratio so equal-aspect axes fill the figure
    x_range = x_max_dam * 1.1 - (-x_max_dam * 0.1)
    y_range = y_max * 1.2 - (-y_max * 0.1)
    data_aspect = y_range / x_range
    base_width = 12
    fig_height = max(6, min(18, base_width * data_aspect))
    fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=150)
    ax.set_facecolor('#F0F0F0')

    levels = np.linspace(h_down, h_up, 16)
    contour = ax.contourf(xx, yy, H_pred, levels=levels, cmap='jet', alpha=0.85, vmin=h_down, vmax=h_up)
    ax.contour(xx, yy, H_pred, levels=levels, colors='black', linewidths=0.5, alpha=0.6)

    fig.colorbar(contour, ax=ax, label="Total Head H (m) in Saturated Zone")

    ax.plot(dataset.fs_x, dataset.fs_y, color='red', linestyle='dashed', linewidth=3, label='Phreatic Line (Free Surface)', zorder=15)

    ordered_v = [v[0], v[1], v[3], v[2]]
    dam_polygon = Polygon(ordered_v, closed=True, edgecolor='black', facecolor='none', linewidth=1.5, zorder=10)
    ax.add_patch(dam_polygon)

    x_up_intersect = x_min + h_up * slope_l
    x_down_intersect = x_max_dam + h_down * slope_r

    up_water = Polygon([[-x_max_dam*0.1, 0], [v[0][0], 0], [x_up_intersect, h_up], [-x_max_dam*0.1, h_up]],
                       closed=True, color='royalblue', alpha=0.4, zorder=5)
    ax.add_patch(up_water)
    ax.hlines(h_up, -x_max_dam*0.1, x_up_intersect, colors='blue', lw=2.5, zorder=6)
    ax.text(-x_max_dam*0.05, h_up + 0.5, f" {h_up}m", color='blue', fontweight='bold')

    if h_down > 0:
        down_water = Polygon([[x_max_dam, 0], [x_max_dam*1.1, 0], [x_max_dam*1.1, h_down], [x_down_intersect, h_down]],
                             closed=True, color='royalblue', alpha=0.4, zorder=5)
        ax.add_patch(down_water)
        ax.hlines(h_down, x_down_intersect, x_max_dam*1.1, colors='blue', lw=2.5, zorder=6)
        ax.text(x_max_dam*1.05, h_down + 0.5, f" {h_down}m", color='blue', fontweight='bold', ha='right')

    ax.set_aspect('equal')
    ax.set_xlim(-x_max_dam*0.1, x_max_dam*1.1)
    ax.set_ylim(-y_max*0.1, y_max*1.2)
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Laplace Equation Iterative PINN Seepage Analysis", fontsize=15, fontweight='bold')

    textstr = f'Final System Loss: {final_loss:.2e}'
    ax.text(0.03, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), zorder=20)

    leg = ax.legend(loc='upper right')
    leg.set_zorder(20)

    plot_path = os.path.join(save_dir, "pinn_seepage_result.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return plot_path, data_save_path
