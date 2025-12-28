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

## 四、實作流程（摘要）

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

### 1️⃣ 掛載 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
