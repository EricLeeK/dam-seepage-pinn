import json
import torch
import numpy as np

class SeepageDataset:
    def __init__(self, config_path=None, num_domain=10000, num_boundary=2000, config_dict=None):
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config_dict is not None:
            self._config = config_dict
        elif config_path is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        self._load_config()

        self.num_fs_points = 100
        self.x_up_intersect = self.x_min + self.h_up * self.slope_left
        initial_exit_y = self.h_down + (self.h_up - self.h_down) * 0.3
        x_down_intersect = self.vertices[1][0] + initial_exit_y * self.slope_right

        self.fs_x = np.linspace(self.x_up_intersect, x_down_intersect, self.num_fs_points)
        a_param = (initial_exit_y - self.h_up) / ((x_down_intersect - self.x_up_intersect)**2)
        self.fs_y = a_param * (self.fs_x - self.x_up_intersect)**2 + self.h_up

    def _load_config(self):
        data = self._config

        # Support new spec format: { geometry: {...}, hydraulic: {...}, training: {...} }
        if "geometry" in data and "hydraulic" in data:
            geom = data["geometry"]
            hyd = data["hydraulic"]
            self.vertices = geom["vertices"]
            self.h_up = float(hyd["upstream_head"])
            self.h_down = float(hyd["downstream_head"])
            self.k = float(hyd.get("permeability_k", 1.0))
            if "training" in data:
                self.num_domain = int(data["training"].get("num_domain", self.num_domain))
                self.num_boundary = int(data["training"].get("num_boundary", self.num_boundary))
        # Support legacy format: { pinn_domain: {...} }
        elif "pinn_domain" in data:
            domain = data["pinn_domain"]
            self.vertices = domain["vertices"]
            self.h_up = float(domain["upstream_head"])
            self.h_down = float(domain["downstream_head"])
            self.k = float(domain.get("permeability_k", 1.0))
        else:
            raise ValueError("Config must contain either 'geometry'+'hydraulic' or 'pinn_domain' keys")

        self.x_min, self.y_min = self.vertices[0]
        self.x_max = max(v[0] for v in self.vertices)
        self.y_max = max(v[1] for v in self.vertices)
        self.slope_left = (self.vertices[2][0] - self.vertices[0][0]) / self.y_max
        self.slope_right = (self.vertices[3][0] - self.vertices[1][0]) / self.y_max

    def is_inside_dam(self, x, y):
        x_left_bound = self.x_min + y * self.slope_left
        x_right_bound = self.vertices[1][0] + y * self.slope_right
        return (x >= x_left_bound) & (x <= x_right_bound)

    def get_fs_normals(self):
        dx = np.gradient(self.fs_x)
        dy = np.gradient(self.fs_y)
        norm = np.sqrt(dx**2 + dy**2)
        nx = -dy / norm
        ny = dx / norm
        return np.stack([nx, ny], axis=1)

    def update_fs(self, H_pred, alpha_relax=0.2):
        y_raw = (1.0 - alpha_relax) * self.fs_y + alpha_relax * H_pred
        y_raw = np.clip(y_raw, self.h_down, self.h_up)

        y_smooth = np.copy(y_raw)
        for _ in range(5):
            y_smooth[1:-1] = 0.25 * y_smooth[:-2] + 0.5 * y_smooth[1:-1] + 0.25 * y_smooth[2:]

        y_smooth[0] = self.h_up
        for i in range(1, len(y_smooth)):
            if y_smooth[i] > y_smooth[i-1]:
                y_smooth[i] = y_smooth[i-1]

        y_smooth[-1] = max(y_smooth[-1], self.h_down)
        self.fs_y = y_smooth

        x_exit_new = self.vertices[1][0] + self.fs_y[-1] * self.slope_right
        x_new = np.linspace(self.x_up_intersect, x_exit_new, self.num_fs_points)
        self.fs_y = np.interp(x_new, self.fs_x, self.fs_y)
        self.fs_x = x_new

    def generate_points(self, target_dtype=torch.float32):
        x_dom_raw = np.random.uniform(self.x_min, self.x_max, self.num_domain * 3)
        y_dom_raw = np.random.uniform(self.y_min, self.y_max, self.num_domain * 3)
        fs_y_interp = np.interp(x_dom_raw, self.fs_x, self.fs_y)
        valid_mask = self.is_inside_dam(x_dom_raw, y_dom_raw) & (y_dom_raw <= fs_y_interp)
        x_dom, y_dom = x_dom_raw[valid_mask][:self.num_domain], y_dom_raw[valid_mask][:self.num_domain]

        y_up = np.random.uniform(self.y_min, self.h_up, self.num_boundary)
        x_up = self.x_min + y_up * self.slope_left
        y_down = np.random.uniform(self.y_min, self.h_down, self.num_boundary)
        x_down = self.vertices[1][0] + y_down * self.slope_right

        safe_fs_y_end = max(self.fs_y[-1], self.h_down + 0.05)
        y_seepage = np.random.uniform(self.h_down, safe_fs_y_end, self.num_boundary)
        x_seepage = self.vertices[1][0] + y_seepage * self.slope_right

        x_bottom = np.random.uniform(self.x_min, self.vertices[1][0], self.num_boundary)
        y_bottom = np.zeros_like(x_bottom)

        def to_tensor(x, y):
            pts = np.stack([x, y], axis=1)
            t = torch.tensor(pts, dtype=target_dtype, device=self.device)
            return t

        return {
            "domain": to_tensor(x_dom, y_dom),
            "up_bound": to_tensor(x_up, y_up),
            "down_bound": to_tensor(x_down, y_down),
            "seepage_bound": to_tensor(x_seepage, y_seepage),
            "bottom_bound": to_tensor(x_bottom, y_bottom),
            "fs_bound": to_tensor(self.fs_x, self.fs_y),
            "fs_normals": torch.tensor(self.get_fs_normals(), dtype=target_dtype, device=self.device)
        }
