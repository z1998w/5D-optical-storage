import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
from tqdm import tqdm
import math
import os
import pandas as pd


# ==============================================================================
# Part 1: Global Setup and Data Generation
# ==============================================================================
print("="*60)
print("Part 1: Defining Global Ground Truth, Measurement Parameters, and Full Physics Model")
print("="*60)

M, N = 10, 10
np.random.seed(42)
DELTA_TRUE = torch.from_numpy(np.pi * np.random.rand(M, N)).float()
PSI_TRUE = torch.from_numpy((np.pi / 2) * np.random.rand(M, N)).float()
#真解（DELTA_TRUE 和 PSI_TRUE）的生成确实使用了固定的随机种子
# 训练过程中涉及到的数据增强（即图像生成）**使用的随机数（如高斯噪声、随机角度、扰动等）没有设置固定种子
GLOBAL_LABEL = torch.stack([DELTA_TRUE, PSI_TRUE], dim=0)#转化为torch
print(f"Global unique Delta_true and Psi_true created with resolution {M}x{N}.")

num_train_pairs = 20
alphas = np.linspace(0, np.pi, num_train_pairs)
betas = np.random.rand(num_train_pairs) * np.pi
train_alpha_beta_pairs = list(zip(alphas, betas))

#五组测试数据，不包含在训练集里，因为训练集的beta是随机生成的
test_alpha_beta_pairs = [
    (np.deg2rad(10), np.deg2rad(10)), (np.deg2rad(80), np.deg2rad(170)),
    (np.deg2rad(100), np.deg2rad(100)), (np.deg2rad(130), np.deg2rad(50)),
    (np.deg2rad(20), np.deg2rad(75))
]
K_LAYER = 10
print(f"Training data pairs: {len(train_alpha_beta_pairs)}. Test data pairs: {len(test_alpha_beta_pairs)}.")

class FullPhysicsGenerator:
    def __init__(self, m=10, n=10):
        self.m, self.n = m, n; self.N_layers, self.h, self.w, self.d, self.theta = 40, 1.0, 1.0, 0.02, np.deg2rad(20)
        self.I_min, self.sigma_add, self.sigma_scr, self.tau = 0.001, 0.05, 0.05, 0.98
        self.sigma_Delta, self.sigma_Psi = 0.05, 0.05; self.E0 = np.array([[1], [0]], dtype=np.complex128)
    def _R(self, theta): c, s = np.cos(theta), np.sin(theta); return np.array([[c, -s], [s, c]], dtype=np.complex128)
    def _Jr(self, phi): return np.array([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]], dtype=np.complex128)
    def _Jr_ret(self, Delta, psi): return self._R(-psi) @ self._Jr(Delta) @ self._R(psi)
    def _sum_rand_Jones(self, Nn):
        if Nn == 0: return np.zeros((2, 2), dtype=np.complex128)
        eps_vals = 0.01 * np.random.randn(Nn); theta_vals = np.pi * np.random.rand(Nn); c, s = np.cos(theta_vals), np.sin(theta_vals)
        J_rands = np.zeros((Nn, 2, 2), dtype=np.complex128); J_rands[:, 0, 0] = np.exp(-1j * eps_vals / 2); J_rands[:, 1, 1] = np.exp(1j * eps_vals / 2)
        R_negs = np.array([[c, s], [-s, c]]).transpose(2, 0, 1); R_poss = np.array([[c, -s], [s, c]]).transpose(2, 0, 1)
        return np.sum(R_negs @ J_rands @ R_poss, axis=0)
    def generate_one_noisy_image(self, alpha, beta, k, delta_true_np, psi_true_np):
        I_noisy = np.zeros_like(delta_true_np)
        eps_scr_img = np.random.randn() * self.sigma_scr; I_add_img = np.random.randn() * self.sigma_add
        for i in range(self.m):
            for j in range(self.n):
                J1 = self._R(-np.pi/4)@self._Jr(np.pi/2)@self._R(np.pi/4); J2 = self._Jr(alpha); J3 = self._R(-np.pi/4)@self._Jr(beta)@self._R(np.pi/4)
                delta_val = delta_true_np[i,j] + np.random.randn() * self.sigma_Delta; psi_val = psi_true_np[i,j] + np.random.randn() * self.sigma_Psi
                J4 = np.eye(2, dtype=np.complex128)
                for nlayer in range(self.N_layers, 0, -1):
                    Rn = abs(nlayer - k) * self.h * np.tan(self.theta); r = Rn / self.w
                    if nlayer == k: Jn = self._Jr_ret(delta_val, psi_val)
                    else:
                        r_floor = int(np.floor(r))
                        if r_floor > 0: Nn = int(1 + 4 * r_floor + 4 * np.sum(np.floor(np.sqrt(r**2 - np.arange(1, r_floor + 1)**2))))
                        else: Nn = 1
                        Jn = np.eye(2, dtype=np.complex128) + (self.d / Nn) * self._sum_rand_Jones(Nn)
                    J4 = J4 @ Jn
                J5 = self._R(-np.pi/4); Jps = J5 @ J4 @ J3 @ J2 @ J1; Eout = Jps @ self.E0
                I_raw = self.tau * np.linalg.norm(Eout)**2; I_noisy[i, j] = I_raw * (1 + eps_scr_img) + self.I_min + I_add_img
        return torch.from_numpy(I_noisy).float()

#产生无噪音数据
def generate_one_ideal_image(alpha, beta, delta_true, psi_true):
    tau, I_max, I_min = 0.98, 1.0, 0.001
    term1 = torch.sin(torch.tensor(alpha)) * torch.cos(torch.tensor(beta)) * torch.cos(delta_true)
    term2 = -torch.sin(torch.tensor(alpha)) * torch.sin(torch.tensor(beta)) * torch.cos(2 * psi_true) * torch.sin(delta_true)
    term3 = torch.cos(torch.tensor(alpha)) * torch.sin(2 * psi_true) * torch.sin(delta_true)
    return (0.5 * tau * I_max * (1 + term1 + term2 + term3) + I_min).float()

physics_generator = FullPhysicsGenerator(M, N)
delta_np, psi_np = DELTA_TRUE.numpy(), PSI_TRUE.numpy()

# ======================= [MODIFICATION START] =======================

# 定义训练数据中无噪音和有噪音数据的比例
ideal_ratio = 0.6
num_ideal_train = int(num_train_pairs * ideal_ratio)
num_noisy_train = num_train_pairs - num_ideal_train

print(f"\nGenerating training data: {num_ideal_train} ideal (70%) and {num_noisy_train} noisy (30%).")

# 为不同类型的数据分配 (alpha, beta) 对。
# 注意：这里只是按顺序分割了原始的 train_alpha_beta_pairs 列表。
# DataLoader 中的 shuffle=True 会确保在训练时数据是随机混合的。
ideal_pairs = train_alpha_beta_pairs[:num_ideal_train]
noisy_pairs = train_alpha_beta_pairs[num_ideal_train:]

# 生成70%的无噪音（理想）数据
ideal_train_images = [generate_one_ideal_image(a, b, DELTA_TRUE, PSI_TRUE)
                      for a, b in tqdm(ideal_pairs, desc="Generating ideal train data")]

# 生成30%的有噪音数据
noisy_train_images = [physics_generator.generate_one_noisy_image(a, b, K_LAYER, delta_np, psi_np)
                      for a, b in tqdm(noisy_pairs, desc="Generating noisy train data")]

# 合并成最终的训练图像列表。
# 图像的顺序与 train_alpha_beta_pairs 列表的顺序保持一致。
train_images = ideal_train_images + noisy_train_images
print("Mixed training data generated.")
# ======================= [MODIFICATION END] =========================


#生成测试数据
print("\nGenerating test data...")
test_I_noisy_list = [physics_generator.generate_one_noisy_image(a, b, K_LAYER, delta_np, psi_np) for a, b in tqdm(test_alpha_beta_pairs, desc="Generating noisy test data")]
test_I_ideal_list = [generate_one_ideal_image(a, b, DELTA_TRUE, PSI_TRUE) for a, b in tqdm(test_alpha_beta_pairs, desc="Generating ideal test data")]
print("Test data generated.")


# ==============================================================================
# Part 2: Custom Dataset and Model
# ==============================================================================
class SingleObjectDataset(Dataset):
    def __init__(self, images, alpha_beta_list, global_label):
        self.images, self.params, self.label = images, [torch.tensor(p, dtype=torch.float32) for p in alpha_beta_list], global_label
    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx], self.params[idx], self.label

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, param_channels=2):
        #UNet 第一个卷积层接收 3 个通道（1 图像 + 2 参数）
        super(ConditionalUNet, self).__init__()
        combined_channels = in_channels + param_channels
        def _conv_block(in_c, out_c): return nn.Sequential(nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True), nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True))
        # 将 $(\alpha, \beta)$ 展开成两个通道，与图像数据合并，构成输入张量的 3 个通道：[I, α, β]
        self.pool = nn.MaxPool2d(2)
        self.enc1 = _conv_block(combined_channels, 64)# 输入通道:3 (1+2), 输出:64
        self.enc2 = _conv_block(64, 128)# 输入:64, 输出:128
        self.bottleneck = _conv_block(128, 256)# 输入:128, 输出:256
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)# 上采样2倍
        self.dec2 = _conv_block(256, 128)# 输入256(128上采样+128跳跃), 输出:128
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)# 上采样2倍
        self.dec1 = _conv_block(128, 64)# 输入128(64上采样+64跳跃), 输出:64
        self.out_conv = nn.Conv2d(64, out_channels, 1)# 1x1卷积, 输出2通道
        self.final_activation = nn.Sigmoid()#输出范围 (0, 1)
    def forward(self, x, params):
        if x.dim() == 3: x = x.unsqueeze(1)
        params_tiled = params.view(params.shape[0], -1, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        combined_input = torch.cat([x, params_tiled], dim=1)
        e1 = self.enc1(combined_input); e2 = self.enc2(self.pool(e1)); b = self.bottleneck(self.pool(e2))
        up2 = self.upconv2(b)
        up2_resized = F.interpolate(up2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([up2_resized, e2], dim=1))
        up1 = self.upconv1(d2); d1 = self.dec1(torch.cat([up1, e1], dim=1))
        out = self.final_activation(self.out_conv(d1))
        pi = torch.tensor(math.pi, device=out.device, dtype=out.dtype)
        scaling = torch.tensor([pi, pi / 2.0], device=out.device, dtype=out.dtype).view(1, 2, 1, 1)
        # 使用了 Sigmoid + 缩放来限制输出范围在 $[0, \pi]$ 和 $[0, \pi/2]$
        return out * scaling


# ==============================================================================
# Part 3: Training and Evaluation
# ==============================================================================
train_dataset = SingleObjectDataset(train_images, train_alpha_beta_pairs, GLOBAL_LABEL)
test_noisy_dataset = SingleObjectDataset(test_I_noisy_list, test_alpha_beta_pairs, GLOBAL_LABEL)
test_ideal_dataset = SingleObjectDataset(test_I_ideal_list, test_alpha_beta_pairs, GLOBAL_LABEL)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_noisy_loader = DataLoader(test_noisy_dataset, batch_size=5, shuffle=False)
test_ideal_loader = DataLoader(test_ideal_dataset, batch_size=5, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalUNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=150, verbose=True)
print(f"✅ Model, datasets, and optimizer are ready. Training will run on {device}.")

print("\n" + "="*60 + "\nPart 3: Training the expert network...\n" + "="*60)
epochs = 4000
start_time = time.time()
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for i_data, params, label in train_loader:
        i_data, params, label = i_data.to(device), params.to(device), label.to(device)
        optimizer.zero_grad(); prediction = model(i_data, params)
        loss = criterion(prediction, label); loss.backward(); optimizer.step()
        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / len(train_loader)
    scheduler.step(avg_epoch_loss)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Avg Train Loss: {avg_epoch_loss:.8f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
print(f"✅ Training finished in {time.time() - start_time:.2f} seconds.")

def evaluate_and_save_all_samples(loader, title_prefix, device):
    print("\n" + "="*60 + f"\nEVALUATING ALL TEST SAMPLES: {title_prefix}\n" + "="*60)
    model.eval()
    with torch.no_grad():
        all_preds = []
        for i_data, params, _ in loader:
            i_data, params = i_data.to(device), params.to(device)
            prediction = model(i_data, params)  # shape: [B, 2, M, N]
            all_preds.append(prediction)
    
    all_preds = torch.cat(all_preds, dim=0)  # shape: [num_samples, 2, M, N]
    delta_true, psi_true = GLOBAL_LABEL[0].to(device), GLOBAL_LABEL[1].to(device)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # 用于保存每个样本误差指标的汇总表
    summary_rows = []

    for idx in range(all_preds.shape[0]):
        prediction = all_preds[idx]
        delta_pred, psi_pred = prediction[0], prediction[1]

        # Error metrics
        delta_error = torch.abs(delta_pred - delta_true)
        psi_error = torch.abs(psi_pred - psi_true)

        fro_error_delta = torch.norm(delta_pred - delta_true, p='fro').item()
        fro_error_psi = torch.norm(psi_pred - psi_true, p='fro').item()

        max_delta_error = torch.max(delta_error).item()
        max_psi_error = torch.max(psi_error).item()

        summary_rows.append({
            "Sample": idx,
            "Frobenius_Delta": fro_error_delta,
            "Frobenius_Psi": fro_error_psi,
            "Max_Abs_Delta": max_delta_error,
            "Max_Abs_Psi": max_psi_error
        })

        # Save per-sample .csv
        np.savetxt(os.path.join(output_dir, f"delta_pred_{title_prefix}_sample{idx}.csv"), delta_pred.cpu().numpy(), delimiter=',')
        np.savetxt(os.path.join(output_dir, f"psi_pred_{title_prefix}_sample{idx}.csv"), psi_pred.cpu().numpy(), delimiter=',')
        np.savetxt(os.path.join(output_dir, f"delta_error_map_{title_prefix}_sample{idx}.csv"), delta_error.cpu().numpy(), delimiter=',')
        np.savetxt(os.path.join(output_dir, f"psi_error_map_{title_prefix}_sample{idx}.csv"), psi_error.cpu().numpy(), delimiter=',')

        # import matplotlib.pyplot as plt
        
        # # 可视化输出
        # fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # titles = [
        #     r"Ground Truth $\Delta$", r"Predicted $\widehat{\Delta}$", r"Abs Error $|\Delta - \widehat{\Delta}|$",
        #     r"Ground Truth $\psi$", r"Predicted $\widehat{\psi}$", r"Abs Error $|\psi - \widehat{\psi}|$"
        # ]
        # images = [
        #     delta_true.cpu().numpy(), delta_pred.cpu().numpy(), delta_error.cpu().numpy(),
        #     psi_true.cpu().numpy(), psi_pred.cpu().numpy(), psi_error.cpu().numpy()
        # ]
        
        # for ax, img, title in zip(axes.flatten(), images, titles):
        #     im = ax.imshow(img, cmap='viridis', vmin=0, vmax=np.pi if 'Delta' in title else np.pi/2)
        #     ax.set_title(title, fontsize=10)
        #     ax.axis('off')
        #     fig.colorbar(im, ax=ax, shrink=0.7)
        
        # plt.tight_layout()
        # save_path = os.path.join(output_dir, f"vis_{title_prefix}_sample{idx}.pdf")
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"✅ Visualization saved: {save_path}")
        import matplotlib.pyplot as plt
        
        plt.rcParams.update({'font.size': 16})  # 全局字体设置
        
        # 画图设置
        def plot_single_map(img, vmin, vmax, label, save_path):
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax)
            
           
            
            # 添加色条
            cbar = plt.colorbar(im, ax=ax, fraction=0.08, pad=0.02, aspect=10)
            cbar.ax.tick_params(labelsize=22)  # 色条刻度大小
            ax.axis('off')
            
            # 不显示标题
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {save_path}")
        
        # 定义图像项
        vis_items = [
            ("delta_gt", delta_true.cpu().numpy(), (0, np.pi)),
            ("delta_pred", delta_pred.cpu().numpy(), (0, np.pi)),
            ("delta_err", delta_error.cpu().numpy(), (0, np.pi/2)),
            ("psi_gt", psi_true.cpu().numpy(), (0, np.pi/2)),
            ("psi_pred", psi_pred.cpu().numpy(), (0, np.pi/2)),
            ("psi_err", psi_error.cpu().numpy(), (0, np.pi/4)),
        ]
        
        # 批量保存
        for tag, img, (vmin, vmax) in vis_items:
            save_path = os.path.join(output_dir, f"{title_prefix}_sample{idx}_{tag}.pdf")
            plot_single_map(img, vmin, vmax, label=tag, save_path=save_path)


    # 保存汇总误差为 .csv
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, f"error_summary_{title_prefix}.csv"), index=False)
    print(f"\n✅ All predictions + error maps + summary saved in '{output_dir}/' directory.")

evaluate_and_save_all_samples(test_noisy_loader, "Noisy_Test_Set_N_40", device)
evaluate_and_save_all_samples(test_ideal_loader, "Ideal_Noiseless_Test_Set_N_40", device)


#保存真解
output_dir = "results"
np.savetxt(os.path.join(output_dir, "delta_ground_truth.csv"), DELTA_TRUE.numpy(), delimiter=',')
np.savetxt(os.path.join(output_dir, "psi_ground_truth.csv"), PSI_TRUE.numpy(), delimiter=',')
print(f"\n✅ Ground truth saved to CSV in '{output_dir}/' directory.")

