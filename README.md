# 人工智慧期末作業：口罩配戴偵測（YOLOv5 + Roboflow）

## 組員
- 11125013 郭慧庭  
- 11125032 林欣儀  
- 11125036 夏振凱  

---

## 一、前言與作業目的

本次期末作業的目標是使用 **深度學習模型 YOLOv5**，在 **Google Colab** 環境中進行「口罩配戴偵測（Mask / No Mask）」。

本組依照網路教學流程，結合 **Roboflow 提供的 Mask Wearing Dataset**，完成以下任務：

- 建立訓練資料集  
- 在 Google Colab 上訓練 YOLOv5 模型  
- 使用訓練完成的模型進行圖片與影片偵測  

---

## 二、教學與資料來源

- **參考教學網址：**  
  https://www.topcfd.cn/18020/

- **資料集來源（Roboflow）：**  
  Mask Wearing Dataset（含 Mask / No Mask 兩類）

---

## 三、實作環境說明

- 平台：Google Colab  
- Python 版本：Colab 預設  
- GPU：NVIDIA Tesla（Colab 提供）  
- 主要模型：YOLOv5s  

---

## 四、實作流程

> 本章節僅說明流程概念，實際指令與操作請見後續步驟。

整體流程如下：

1. 將 Roboflow 資料集下載並上傳至 Google Drive  
2. 掛載 Google Drive  
3. Clone YOLOv5 官方程式碼  
4. 修改資料集路徑與模型設定  
5. 啟用 GPU 並進行模型訓練  
6. 使用訓練完成的模型進行圖片與影片偵測  

---

## 五、Google Colab 實作步驟

### 1.掛載 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2.進入資料集資料夾

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3.Clone YOLOv5 原始碼

```bash
!git clone https://github.com/ultralytics/yolov5
```

```python
%cd yolov5
```

### 4.安裝必要套件

```bash
!pip install -r requirements.txt
```

### 5.修改資料集設定（data.yaml）

確認 data.yaml 內容如下（路徑依實際資料夾調整）：
```yami
train: /content/drive/MyDrive/Mask/train/images
val: /content/drive/MyDrive/Mask/valid/images

nc: 2
names: ['mask', 'no-mask']
```

### 6.修改模型類別數（yolov5s.yaml）

將模型中的類別數從 80 改為 2：
```yami
nc: 2
```

### 7.啟用 GPU 並開始訓練

將模型中的類別數從 80 改為 2：
```bash
!python train.py --img 640 --batch 16 --epochs 50 \
--data data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt
```

---

## 六、模型偵測成果

### 1.圖片偵測

```bash
!python detect.py --weights runs/train/exp/weights/best.pt \
--img 640 --conf 0.25 --source data/images
```

### 2.影片偵測

```bash
!python detect.py --weights runs/train/exp/weights/best.pt \
--img 640 --conf 0.25 --source test.mp4
```

---

## 七、成果展示

- 圖片偵測成果：  
- 影片偵測成果：

---

## 八、常見錯誤與解決方式

| 問題             | 原因             | 解決方式                  |
| -------------- | -------------- | --------------------- |
| `--cache` 參數錯誤 | Word 自動轉成長破折號  | 改為 `--cache` 或移除      |
| 訓練失敗           | data.yaml 路徑錯誤 | 檢查 Drive 路徑是否正確       |
| 偵測不到物件         | confidence 太高  | 將 `--conf` 調低（如 0.25） |

---

## 九、結論

本次作業成功使用 YOLOv5 與 Roboflow 資料集，在 Google Colab 上完成口罩配戴偵測模型的訓練與應用。
透過本次實作，我們學習到：
- 深度學習資料集的整理方式
- YOLOv5 模型訓練流程
- 模型應用於圖片與影片的實際偵測
未來可延伸應用於即時監控、公共安全與防疫相關場景。
