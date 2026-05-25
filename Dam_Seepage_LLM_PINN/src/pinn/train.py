import os
import torch
import torch.optim as optim
import numpy as np
import warnings

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")

if torch.cuda.is_available():
    torch.cuda.init()

from src.pinn.dataset import SeepageDataset
from src.pinn.model import SeepagePINN
from src.pinn.loss import SeepageLoss
from src.utils.plot_utils import plot_seepage_results

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "data", "outputs", "pinn_domain_config.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "outputs")


class TrainingCancelled(Exception):
    """Raised when user cancels training."""
    pass


def train_pinn(config_dict, output_dir, progress_callback=None, cancel_check=None):
    """Programmatic PINN training entry point.

    Args:
        config_dict: Configuration dict with geometry/hydraulic/training keys.
        output_dir: Directory for saving model and plots.
        progress_callback: Optional callable(phase, iter, max_iter, loss) for progress reporting.
        cancel_check: Optional callable() -> bool. Returns True to stop training.

    Returns:
        dict with keys: final_loss, model_path, plot_path, npz_path.
    Raises:
        TrainingCancelled: If cancel_check returns True.
    """

    def _check_cancel():
        if cancel_check and cancel_check():
            raise TrainingCancelled("Training cancelled by user")
    dataset = SeepageDataset(config_dict=config_dict)
    h_up, h_down = dataset.h_up, dataset.h_down
    k = dataset.k

    model = SeepagePINN(
        scale_x=dataset.x_max, scale_y=dataset.y_max, scale_h=h_up, k=k
    ).to(dataset.device)
    loss_fn = SeepageLoss(model, k=k)

    max_outer_iters = config_dict.get("training", {}).get("outer_iters", 30)

    for outer_iter in range(max_outer_iters):
        _check_cancel()
        if progress_callback:
            progress_callback("outer_iter", outer_iter + 1, max_outer_iters, None)

        batch_data = dataset.generate_points(target_dtype=torch.float64)
        current_alpha = max(0.01, 0.2 * (0.8 ** outer_iter))

        _train_cfg = config_dict.get("training", {})
        _adam_epochs = _train_cfg.get("adam_epochs", 1000)
        _lbfgs_inner_iter = min(_train_cfg.get("lbfgs_max_iter", 200), 200)
        _lbfgs_report_iter = _train_cfg.get("lbfgs_max_iter", 5000)

        optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(_adam_epochs):
            optimizer_adam.zero_grad()
            loss = loss_fn(batch_data, h_up, h_down)
            loss.backward()
            optimizer_adam.step()
            if progress_callback and epoch % max(1, _adam_epochs // 2) == 0:
                _check_cancel()
                progress_callback("adam", epoch, _adam_epochs, loss.item())

        optimizer_lbfgs_inner = optim.LBFGS(
            model.parameters(), lr=0.5, max_iter=_lbfgs_inner_iter,
            tolerance_grad=1e-9, tolerance_change=1e-9,
            line_search_fn='strong_wolfe'
        )

        def closure_inner():
            optimizer_lbfgs_inner.zero_grad()
            loss = loss_fn(batch_data, h_up, h_down)
            loss.backward()
            return loss
        optimizer_lbfgs_inner.step(closure_inner)

        with torch.no_grad():
            fs_tensor = torch.tensor(
                np.stack([dataset.fs_x, dataset.fs_y], axis=1),
                dtype=torch.float64, device=dataset.device
            )
            H_pred = model.forward_head(fs_tensor).cpu().numpy().flatten()
            mse_error = np.mean((H_pred - dataset.fs_y)**2)

            if progress_callback:
                progress_callback("fs_update", outer_iter + 1, max_outer_iters, mse_error)

            if mse_error < 5e-4:
                break
            dataset.update_fs(H_pred, alpha_relax=current_alpha)

    _check_cancel()

    _lbfgs_final_max = _lbfgs_report_iter
    # Final L-BFGS
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), lr=0.5, max_iter=_lbfgs_final_max,
        tolerance_grad=1e-12, tolerance_change=1e-12,
        line_search_fn='strong_wolfe'
    )

    lbfgs_iter = 0

    def closure():
        nonlocal lbfgs_iter
        optimizer_lbfgs.zero_grad()
        loss = loss_fn(batch_data, h_up, h_down)
        loss.backward()
        if progress_callback and lbfgs_iter % max(1, _lbfgs_final_max // 10) == 0:
            _check_cancel()
            progress_callback("lbfgs_final", lbfgs_iter, _lbfgs_final_max, loss.item())
        lbfgs_iter += 1
        return loss

    try:
        optimizer_lbfgs.step(closure)
    except KeyboardInterrupt:
        pass

    model.eval()
    final_display_loss = loss_fn.evaluate_unweighted_loss(batch_data, h_up, h_down)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "pinn_model_final.pth")
    torch.save(model.state_dict(), model_path)

    plot_path, npz_path = plot_seepage_results(model, dataset, final_display_loss, output_dir)

    return {
        "final_loss": final_display_loss,
        "model_path": model_path,
        "plot_path": plot_path,
        "npz_path": npz_path,
    }


def train_iterative_pinn():
    dataset = SeepageDataset(CONFIG_PATH)
    h_up, h_down = dataset.h_up, dataset.h_down

    model = SeepagePINN(scale_x=dataset.x_max, scale_y=dataset.y_max, scale_h=h_up).to(dataset.device)
    loss_fn = SeepageLoss(model)

    max_outer_iters = 30

    for outer_iter in range(max_outer_iters):
        print(f"\n[{'='*10} outer iteration {outer_iter+1}/{max_outer_iters} {'='*10}]")
        batch_data = dataset.generate_points(target_dtype=torch.float64)

        current_alpha = max(0.01, 0.2 * (0.8 ** outer_iter))
        print(f"  Alpha: {current_alpha:.4f}")

        optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1000):
            optimizer_adam.zero_grad()
            loss = loss_fn(batch_data, h_up, h_down)
            loss.backward()
            optimizer_adam.step()
            if epoch > 0 and epoch % 500 == 0:
                print(f"  Adam Epoch {epoch:04d} | Weighted Loss: {loss.item():.4e}")

        optimizer_lbfgs_inner = optim.LBFGS(
            model.parameters(), lr=0.5,
            max_iter=200,
            tolerance_grad=1e-9,
            tolerance_change=1e-9,
            line_search_fn='strong_wolfe'
        )

        def closure_inner():
            optimizer_lbfgs_inner.zero_grad()
            loss = loss_fn(batch_data, h_up, h_down)
            loss.backward()
            return loss
        optimizer_lbfgs_inner.step(closure_inner)

        with torch.no_grad():
            fs_tensor = torch.tensor(np.stack([dataset.fs_x, dataset.fs_y], axis=1),
                                     dtype=torch.float64, device=dataset.device)
            H_pred = model.forward_head(fs_tensor).cpu().numpy().flatten()
            mse_error = np.mean((H_pred - dataset.fs_y)**2)
            print(f"  --> FS update MSE: {mse_error:.6f}")

            if mse_error < 5e-4:
                print("FS geometry converged, exiting outer iteration!")
                break

            dataset.update_fs(H_pred, alpha_relax=current_alpha)

    print("\nStarting global L-BFGS (Float64)...")
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), lr=0.5,
        max_iter=5000,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn='strong_wolfe'
    )

    lbfgs_iter = 0

    def closure():
        nonlocal lbfgs_iter
        optimizer_lbfgs.zero_grad()
        loss = loss_fn(batch_data, h_up, h_down)
        loss.backward()
        if lbfgs_iter % 100 == 0:
            print(f"  --> Final Press {lbfgs_iter:04d} | Loss: {loss.item():.4e}")
        lbfgs_iter += 1
        return loss

    try:
        optimizer_lbfgs.step(closure)
    except KeyboardInterrupt:
        print("\nInterrupted, extracting current best field...")

    model.eval()

    final_display_loss = loss_fn.evaluate_unweighted_loss(batch_data, h_up, h_down)
    print(f"\nFinal unweighted MSE: {final_display_loss:.4e}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_save_path = os.path.join(OUTPUT_DIR, "pinn_model_final.pth")
    torch.save(model.state_dict(), model_save_path)

    plot_seepage_results(model, dataset, final_display_loss, OUTPUT_DIR)

if __name__ == "__main__":
    train_iterative_pinn()
