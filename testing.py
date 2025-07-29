import torch
import torch.nn.functional as F
import numpy as np

print("=" * 60)
print("F.binary_cross_entropy vs F.cross_entropy 詳細對比")
print("=" * 60)

# ================== 1. F.binary_cross_entropy ==================
print("\n1. F.binary_cross_entropy - 二分類/多標籤問題")
print("-" * 50)

# 場景：軌跡關聯問題 - 判斷每對軌跡是否屬於同一個目標
# 輸入：經過 sigmoid 的概率值 [0, 1]
# 目標：0 或 1 (不關聯/關聯)

# 例子：3個軌跡的相似度矩陣
B, N = 2, 3  # 2個batch，每個batch有3條軌跡

# 預測的相似度矩陣（已經過 sigmoid，值在 [0,1] 之間）
pred_similarity = torch.tensor(
    [
        # Batch 1: 軌跡0和1相似，軌跡2獨立
        [
            [1.0, 0.8, 0.2],  # 軌跡0 與 [0,1,2] 的相似度
            [0.8, 1.0, 0.1],  # 軌跡1 與 [0,1,2] 的相似度
            [0.2, 0.1, 1.0],
        ],  # 軌跡2 與 [0,1,2] 的相似度
        # Batch 2: 軌跡1和2相似，軌跡0獨立
        [[1.0, 0.3, 0.2], [0.3, 1.0, 0.9], [0.2, 0.9, 1.0]],
    ],
    dtype=torch.float32,
)

# 真實標籤（0=不關聯，1=關聯）
gt_similarity = torch.tensor(
    [
        # Batch 1: 只有軌跡0和1關聯
        [[1, 1, 0], [1, 1, 0], [0, 0, 1]],
        # Batch 2: 只有軌跡1和2關聯
        [[1, 0, 0], [0, 1, 1], [0, 1, 1]],
    ],
    dtype=torch.float32,
)

# 計算 BCE loss
bce_loss = F.binary_cross_entropy(pred_similarity, gt_similarity)
print(f"預測相似度矩陣:\n{pred_similarity}")
print(f"真實相似度矩陣:\n{gt_similarity}")
print(f"BCE Loss: {bce_loss:.4f}")

# 手動計算驗證
manual_bce = -torch.mean(
    gt_similarity * torch.log(pred_similarity + 1e-8)
    + (1 - gt_similarity) * torch.log(1 - pred_similarity + 1e-8)
)
print(f"手動計算的 BCE: {manual_bce:.4f}")

# ================== 2. F.cross_entropy ==================
print("\n\n2. F.cross_entropy - 多分類問題")
print("-" * 50)

# 場景：軌跡ID分類問題 - 判斷每條軌跡屬於哪個ID類別
# 輸入：未經過 softmax 的 logits
# 目標：類別索引

# 例子：4條軌跡，5個可能的ID類別（包括"未知"類別）
batch_size, num_tracks, num_classes = 2, 4, 5

# 預測的 logits（未經過 softmax）
pred_logits = torch.tensor(
    [
        # Batch 1: 4條軌跡的分類 logits
        [
            [2.1, -1.0, 0.5, -0.8, 1.2],  # 軌跡0: 最可能是類別0
            [-0.5, 3.2, -1.1, 0.3, -0.9],  # 軌跡1: 最可能是類別1
            [0.1, -0.7, 2.8, -1.5, 0.4],  # 軌跡2: 最可能是類別2
            [-1.2, 0.3, -0.1, -0.4, 2.5],
        ],  # 軌跡3: 最可能是類別4(未知)
        # Batch 2: 4條軌跡的分類 logits
        [
            [1.8, -0.5, 0.2, -1.1, 0.9],
            [-0.8, 2.1, -0.9, 1.5, -1.2],
            [0.4, -1.3, 1.9, -0.6, 0.8],
            [-0.9, 1.1, -0.2, 2.7, -1.5],
        ],
    ],
    dtype=torch.float32,
)

# 真實類別標籤（類別索引）
gt_labels = torch.tensor(
    [
        [0, 1, 2, 4],  # Batch 1: 軌跡分別屬於類別 0,1,2,4
        [0, 3, 2, 3],  # Batch 2: 軌跡分別屬於類別 0,3,2,3
    ],
    dtype=torch.long,
)

# 計算 CrossEntropy loss
ce_loss = F.cross_entropy(pred_logits.view(-1, num_classes), gt_labels.view(-1))
print(f"預測 logits 形狀: {pred_logits.shape}")
print(f"真實標籤: {gt_labels}")
print(f"CrossEntropy Loss: {ce_loss:.4f}")

# 將 logits 轉換為概率來理解
pred_probs = F.softmax(pred_logits, dim=-1)
print(f"轉換為概率後的預測:\n{pred_probs[0]}")  # 只顯示第一個batch

# 手動計算驗證（簡化版）
log_probs = F.log_softmax(pred_logits.view(-1, num_classes), dim=-1)
manual_ce = F.nll_loss(log_probs, gt_labels.view(-1))
print(f"手動計算的 CE: {manual_ce:.4f}")

# ================== 3. 關鍵區別總結 ==================
print("\n\n3. 關鍵區別總結")
print("-" * 50)

print("F.binary_cross_entropy:")
print("  • 用途: 二分類 或 多標籤分類")
print("  • 輸入: 已經過 sigmoid 的概率值 [0,1]")
print("  • 目標: 0 或 1 (可以是 float)")
print("  • 場景: 軌跡關聯、目標檢測、多標籤分類")
print("  • 公式: -[y*log(p) + (1-y)*log(1-p)]")

print("\nF.cross_entropy:")
print("  • 用途: 多分類 (互斥類別)")
print("  • 輸入: 未經過 softmax 的 logits")
print("  • 目標: 類別索引 (必須是 long tensor)")
print("  • 場景: 圖像分類、軌跡ID分類、NLP分類")
print("  • 公式: -log(softmax(logits)[target_class])")

# ================== 4. 實際應用場景對比 ==================
print("\n\n4. 在軌跡追蹤中的應用場景")
print("-" * 50)

print("使用 BCE 的場景:")
print("  ✓ 軌跡關聯矩陣 [B, N, N] - 判斷軌跡對是否關聯")
print("  ✓ 目標檢測 - 判斷邊界框是否包含目標")
print("  ✓ 多標籤分類 - 一個軌跡可能有多個屬性")

print("\n使用 CrossEntropy 的場景:")
print("  ✓ 軌跡ID分類 [B, N] -> [B, N, num_ids] - 每條軌跡分配唯一ID")
print("  ✓ 行為分類 - 軌跡屬於 [走路/跑步/停止] 之一")
print("  ✓ 類別檢測 - 軌跡屬於 [人/車/動物] 之一")

# ================== 5. 常見錯誤示例 ==================
print("\n\n5. 常見使用錯誤")
print("-" * 50)

# 錯誤1: 對 logits 使用 BCE
try:
    logits = torch.randn(2, 3, 5)  # 未經過 sigmoid 的 logits
    targets = torch.randint(0, 2, (2, 3, 5)).float()
    wrong_loss = F.binary_cross_entropy(logits, targets)  # 錯誤！
except Exception as e:
    print(f"錯誤1 - 對 logits 直接使用 BCE: {type(e).__name__}")

# 正確做法：先 sigmoid
logits = torch.randn(2, 3, 5)
targets = torch.randint(0, 2, (2, 3, 5)).float()
probs = torch.sigmoid(logits)
correct_loss = F.binary_cross_entropy(probs, targets)
print(f"正確做法 - 先 sigmoid 再 BCE: {correct_loss:.4f}")

# 錯誤2: 目標類別用 float 給 CrossEntropy
try:
    logits = torch.randn(4, 5)
    targets = torch.tensor([0.0, 1.0, 2.0, 3.0])  # 錯誤：float 類型
    wrong_loss = F.cross_entropy(logits, targets)
except Exception as e:
    print(f"錯誤2 - CrossEntropy 目標用 float: {type(e).__name__}")

# 正確做法：用 long
logits = torch.randn(4, 5)
targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # 正確：long 類型
correct_loss = F.cross_entropy(logits, targets)
print(f"正確做法 - CrossEntropy 目標用 long: {correct_loss:.4f}")

print("\n" + "=" * 60)
print("總結：選擇損失函數的決策樹")
print("=" * 60)
print("問題類型 -> 輸出維度 -> 損失函數")
print("二分類 -> [B, 1] -> BCE")
print("多標籤 -> [B, K] -> BCE")
print("多分類 -> [B, C] -> CrossEntropy")
print("軌跡關聯 -> [B, N, N] -> BCE")
print("軌跡分類 -> [B, N, C] -> CrossEntropy")
