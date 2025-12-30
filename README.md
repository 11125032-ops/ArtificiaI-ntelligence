# 人工智慧期末作業：YOLOv5 口罩偵測（Google Colab）

## 組員
- 11125013 郭慧庭
- 11125032 林欣儀
- 11125036 夏振凱

---

## 一、前言與程式目的
本次期末作業的目標是使用 **YOLOv5 深度學習模型**，在 **Google Colab（GPU 環境）** 中完成「口罩配戴偵測（Mask Detection）」任務。  
依照教學流程下載公開資料集、訓練模型，並將訓練完成的權重實際應用於圖片與影片中，驗證模型對「有戴口罩 / 未戴口罩」之辨識能力。

- 題目教學來源：  
  https://www.topcfd.cn/18020/

---

## 二、資料集來源與說明
- 資料集來源：Roboflow 公開資料集  
  https://public.roboflow.com/
- 資料集名稱：Mask Wearing Dataset
- 類別（Classes）：
  - `mask`
  - `no-mask`
- 資料格式：YOLO v5 PyTorch 格式（含 `train / valid / test`）

---

## 三、實作環境
- 平台：Google Colab
- 硬體加速：GPU（Tesla T4，約 16GB）
- 深度學習框架：PyTorch
- 模型：YOLOv5
- torch / CUDA（依 Colab 實際分配為主；本次實作輸出為 torch 2.9.0、CUDA 12.6）

---

## 四、實作流程（依教學與作業文件步驟）

### 第一步：下載資料集（Roboflow）
1. 進入 https://public.roboflow.com/
2. 選擇 **Mask Wearing Dataset**
3. 點擊 **raw**
4. 下載格式選擇 **YOLO v5 PyTorch**（壓縮檔）

---

### 第二步：解壓與上傳至 Google Drive
1. 將下載的壓縮檔解壓
2. 將資料夾更名為 `Mask`
3. 上傳到 Google Drive 的 `MyDrive` 目錄底下  
   （路徑會是：`/content/drive/MyDrive/Mask`）

---

### 第三步：設定 Colab 使用 GPU
在 Colab：  
「執行階段」→「變更執行階段類型」→ 硬體加速器選 **GPU**

---

### 第四步：掛載 Google Drive 並確認資料集路徑
```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import os
os.chdir('/content/drive/MyDrive/Mask')
!ls
```

確認資料夾內包含 `train`、`valid`、`test` 與 `data.yaml`。

---

### 第五步：下載 YOLOv5 原始碼並確認 GPU
```python
%cd /content/drive/MyDrive
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
```

```python
import torch
from IPython.display import clear_output

clear_output()
print('Setup complete. Using torch %s %s' % (
    torch.__version__,
    torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'
))
```

若成功使用 GPU，輸出可看到 Tesla T4 等資訊。

---

### 第六步：安裝相依套件
```python
%pip install -qr requirements.txt
```

---

### 第七步：修正資料集設定檔（data.yaml）與修改類別數（nc）
#### (1) 修正 `Mask/data.yaml` 路徑
```python
import yaml

yaml_path = '/content/drive/MyDrive/Mask/data.yaml'

with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

data['path'] = '/content/drive/MyDrive/Mask'
data['train'] = 'train/images'
data['val'] = 'valid/images'
data['test'] = 'test/images'

with open(yaml_path, 'w') as f:
    yaml.dump(data, f)

print("✅ data.yaml 路徑已修正完畢！")
```

#### (2) 修改模型類別數 `nc`
開啟檔案：
- `yolov5/models/yolov5s.yaml`

將：
```yaml
nc: 80
```
改為：
```yaml
nc: 2
```
（因資料集只有 `mask` 與 `no-mask` 兩類）

---

### 第八步：模型訓練（300 epochs）
```python
%cd /content/drive/MyDrive/yolov5
!python train.py --img 640 --batch 16 --epochs 300 --data /content/drive/MyDrive/Mask/data.yaml --weights yolov5s.pt --cache
```

訓練完成後，權重檔通常會在（注意 exp 可能是 exp、exp2...）：
- `yolov5/runs/train/exp*/weights/best.pt`

---

### 第九步：圖片偵測測試（mask.jpg）
1. 準備一張口罩相關圖片，命名為 `mask.jpg`
2. 放進 `yolov5` 資料夾內（`/content/drive/MyDrive/yolov5/`）

```python
!python detect.py --weights runs/train/exp*/weights/best.pt --img 640 --conf 0.5 --source mask.jpg
```

範例輸出可看到抓到「7 個 mask、2 個 no-mask」。  
（若人物重疊，可能出現框被遮住的情況）

輸出圖片位置（detect 的 exp 也可能遞增）：
- `yolov5/runs/detect/exp*/mask.jpg`

---

### 第十步：準備影片（mask.mp4）
因原教學影音可能無法使用，本次改用自行下載之口罩相關影片：  
1. 準備影片並命名為 `mask.mp4`
2. 放進 `yolov5` 資料夾內

---

### 第十一步：影片偵測測試（mask.mp4）
```python
!python detect.py --weights runs/train/exp*/weights/best.pt --img 640 --conf 0.5 --source mask.mp4
```

輸出影片位置：
- `yolov5/runs/detect/exp*/mask.mp4`

本次影片連結（Google Drive）：
- https://drive.google.com/file/d/1-NRvPbJlOjaWktjQ8H-lDuW9dI2Ru02g/view?usp=drive_link

---

## 五、實驗結果
- 模型可辨識「有戴口罩（mask）」與「未戴口罩（no-mask）」
- 在多人或遮擋情況下仍可進行基本偵測，但偶有誤判或漏判
- 若需提升準確度，可增加資料量或調整訓練參數（如 epochs、batch size 等）

---

## 六、成果展示
- 圖片偵測成果（路徑依實際 exp 編號為主）：
  - `yolov5/runs/detect/exp*/mask.jpg`
- 影片偵測成果（路徑依實際 exp 編號為主）：
  - `yolov5/runs/detect/exp*/mask.mp4`
- 影片 Drive 連結：
  - https://drive.google.com/file/d/1-NRvPbJlOjaWktjQ8H-lDuW9dI2Ru02g/view?usp=drive_link

---

## 七、結論
本次實作成功完成 YOLOv5 口罩偵測模型之訓練與應用。  
透過 Google Colab GPU 環境，可有效降低本地端硬體限制，並驗證深度學習模型在影像辨識任務上的可行性。
