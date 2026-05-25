import torch

def get_gradient(y, x):
    return torch.autograd.grad(
        outputs=y, inputs=x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]

class SeepageLoss:
    def __init__(self, model, k=1.0):
        self.model = model
        self.k = float(k)

    def __call__(self, batch_data, h_up, h_down):
        S = self.model.scale_global
        H_scale = self.model.scale_h

        def get_norm_pts(key):
            return (batch_data[key] / S).clone().detach().requires_grad_(True)

        # 1. PDE: k * (u_xx + u_yy)^2
        xy_dom_norm = get_norm_pts('domain')
        u_dom = self.model.net(xy_dom_norm)
        du = get_gradient(u_dom, xy_dom_norm)
        u_xx = get_gradient(du[:, 0:1], xy_dom_norm)[:, 0:1]
        u_yy = get_gradient(du[:, 1:2], xy_dom_norm)[:, 1:2]
        loss_pde = self.k * torch.mean((u_xx + u_yy)**2)

        # 2. Dirichlet BCs
        loss_up = torch.mean((self.model.net(get_norm_pts('up_bound')) - h_up/H_scale)**2)
        loss_down = torch.mean((self.model.net(get_norm_pts('down_bound')) - h_down/H_scale)**2)

        xy_seep_norm = get_norm_pts('seepage_bound')
        y_phys_seep = batch_data['seepage_bound'][:, 1:2]
        loss_seep = torch.mean((self.model.net(xy_seep_norm) - y_phys_seep/H_scale)**2)

        # 3. Neumann BCs
        xy_bot_norm = get_norm_pts('bottom_bound')
        du_bot = get_gradient(self.model.net(xy_bot_norm), xy_bot_norm)
        loss_bot = torch.mean(du_bot[:, 1:2]**2)

        xy_fs_norm = get_norm_pts('fs_bound')
        du_fs = get_gradient(self.model.net(xy_fs_norm), xy_fs_norm)
        n_fs = batch_data['fs_normals']
        du_dn_fs = du_fs[:, 0:1] * n_fs[:, 0:1] + du_fs[:, 1:2] * n_fs[:, 1:2]
        loss_fs = torch.mean(du_dn_fs**2)

        return loss_pde + 10.0 * (loss_up + loss_down + loss_seep) + 2.0 * (loss_bot + loss_fs)

    def evaluate_unweighted_loss(self, batch_data, h_up, h_down):
        S = self.model.scale_global
        H_scale = self.model.scale_h

        def get_norm_pts(key):
            return (batch_data[key] / S).clone().detach().requires_grad_(True)

        xy_dom_norm = get_norm_pts('domain')
        u_dom = self.model.net(xy_dom_norm)
        du = get_gradient(u_dom, xy_dom_norm)
        u_xx = get_gradient(du[:, 0:1], xy_dom_norm)[:, 0:1]
        u_yy = get_gradient(du[:, 1:2], xy_dom_norm)[:, 1:2]
        loss_pde = torch.mean((u_xx + u_yy)**2)

        loss_up = torch.mean((self.model.net(get_norm_pts('up_bound')) - h_up/H_scale)**2)
        loss_down = torch.mean((self.model.net(get_norm_pts('down_bound')) - h_down/H_scale)**2)

        xy_seep_norm = get_norm_pts('seepage_bound')
        y_phys_seep = batch_data['seepage_bound'][:, 1:2]
        loss_seep = torch.mean((self.model.net(xy_seep_norm) - y_phys_seep/H_scale)**2)

        xy_bot_norm = get_norm_pts('bottom_bound')
        loss_bot = torch.mean(get_gradient(self.model.net(xy_bot_norm), xy_bot_norm)[:, 1:2]**2)

        xy_fs_norm = get_norm_pts('fs_bound')
        du_fs = get_gradient(self.model.net(xy_fs_norm), xy_fs_norm)
        n_fs = batch_data['fs_normals']
        loss_fs = torch.mean((du_fs[:, 0:1] * n_fs[:, 0:1] + du_fs[:, 1:2] * n_fs[:, 1:2])**2)

        return (loss_pde + loss_up + loss_down + loss_seep + loss_bot + loss_fs).item()

    def compute_velocity_field(self, xy_tensor):
        """Compute Darcy velocity field q = -k * grad(H).

        Args:
            xy_tensor: (N, 2) tensor of physical coordinates, requires_grad will be set.

        Returns:
            qx, qy: (N, 1) tensors of velocity components.
        """
        S = self.model.scale_global
        H_scale = self.model.scale_h
        xy_norm = (xy_tensor / S).clone().detach().requires_grad_(True)
        u = self.model.net(xy_norm)
        du = get_gradient(u, xy_norm)
        # du/dx_norm and du/dy_norm in normalized space; convert to physical:
        # H = u * H_scale, x_norm = x/S => dH/dx = (dH/du)*(du/dx_norm)*(dx_norm/dx) = H_scale * du_dx_norm * (1/S)
        dH_dx = du[:, 0:1] * H_scale / S
        dH_dy = du[:, 1:2] * H_scale / S
        qx = -self.k * dH_dx
        qy = -self.k * dH_dy
        return qx, qy
