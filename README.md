# BikeSharingPrediction 项目说明

## 项目简介

本项目旨在基于历史共享单车租赁数据，预测未来的租赁数量。使用的模型包括：

LSTM 模型

Transformer 模型

自定义模型：TFT

自定义模型：Autoformer

支持两种预测任务：

**短期预测 (O=96)**：基于过去 96 小时的数据预测未来 96 小时的租赁数量。

**长期预测 (O=240)**：基于过去 96 小时的数据预测未来 240 小时的租赁数量。

## 数据说明

项目使用的训练数据和测试数据存放于`data/`文件夹内：

`train_data.csv`

`test_data.csv`

## 运行说明

### 环境依赖

运行本项目需要以下环境依赖：

Python 3.8+

PyTorch 1.13+

其他依赖请参考`requirements.txt`（如未提供，请根据运行环境安装所需包）。

### 快速开始

通过以下命令训练不同模型，并完成预测任务：

#### 1. LSTM 模型

**短期预测 (O=96)**



```
python main.py --model LSTM --output\_window 96
```

**长期预测 (O=240)**



```
python main.py --model LSTM --output\_window 240
```

#### 2. Transformer 模型

**短期预测 (O=96)**



```
python main.py --model Transformer --output\_window 96
```

**长期预测 (O=240)**



```
python main.py --model Transformer --output\_window 240
```

#### 3. 自定义模型：TFT

**短期预测 (O=96)**



```
python main.py --model TFT --output\_window 96
```

**长期预测 (O=240)**



```
python main.py --model TFT --output\_window 240
```

#### 4. 自定义模型：Autoformer

**短期预测 (O=96)**



```
python main.py --model Autoformer --output\_window 96
```

**长期预测 (O=240)**



```
python main.py --model Autoformer --output\_window 240
```

### 日志记录

每次运行会自动生成日志文件，保存于项目根目录。日志文件命名格式为：



```
training\_\<model\_name>\_output\<output\_window>.log
```

日志文件中包含每次运行的训练损失和测试结果（MSE、MAE）。

### 绘图

预测结果与真实值对比图可通过以下命令生成：



```
python plot\_results.py --predictions\_file \<predictions\_file.npy> --ground\_truth\_file \<ground\_truth\_file.npy> --title "Model\_Name O=\<output\_window>" --output\_file \<output\_image.png>
```

示例：



```
python plot\_results.py --predictions\_file lstm\_final\_predictions\_output96.npy --ground\_truth\_file lstm\_final\_ground\_truths\_output96.npy --title "LSTM Short-Term Prediction" --output\_file lstm\_short\_term.png
```

## 文件结构

项目文件组织如下：



```
BikeSharingPrediction/

├── data/

│   ├── train\_data.csv

│   ├── test\_data.csv

├── models/

│   ├── lstm\_model.py

│   ├── transformer\_model.py

│   ├── tft\_model.py

│   ├── autoformer\_model.py

├── utils/

│   ├── data\_loader.py

├── main.py

├── train.py

├── plot\_results.py

├── README.md
```

## 注意事项

确保训练时 GPU 可用，若环境无 GPU，将自动切换至 CPU。

模型复杂度较高（如 Autoformer）时，建议适当增加 epochs 或调整 batch\_size。
