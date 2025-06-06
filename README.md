# GOOJODOQ达人匹配系统 v2.0

> 基于真实越南GOOJODOQ达人数据的智能匹配系统，专为3C产品优化

## 系统简介

- **真实数据**：使用真实越南达人和产品数据
- **智能匹配**：多维度加权产品匹配算法
- **3C产品优化**：特别针对数码、电子产品进行优化

## 快速部署

1. 确保你的Node.js和npm已安装
2. 安装Vercel CLI: `npm i -g vercel`
3. 克隆本仓库: `git clone https://github.com/Ar1haraNaN7mI/GOOJODOQ_KOL_Selection.git`
4. 进入项目目录: `cd GOOJODOQ_KOL_Selection`
5. 部署到Vercel: `vercel`

## 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python app.py

# 访问系统
# 浏览器打开: http://localhost:8000
```

## 文件结构

- `app.py`: Flask主应用
- `real_data_matcher.py`: 达人匹配核心算法
- `Match_ProductCreator-main/`: 数据文件目录
- `static/`: 静态资源文件
- `index.html`: 前端界面

## 🎯 系统特点

### 核心优势
- **真实数据**：152,404条越南GOOJODOQ达人 + 8,619条产品数据
- **GMV导向**：完全替代ROI指标，使用7%基准计算模型
- **智能匹配**：多字段加权产品匹配（功能特性优先）
- **激进定价**：差异化价格体系，每千粉47-200元范围
- **预算控制**：智能预算约束，最大化投资回报

### 技术亮点
- **多因素评分**：GMV(40%) + 佣金率(30%) + 互动率(20%) + 客单价(10%)
- **动态权重**：产品匹配50% + 性能指标50%
- **实时计算**：平均响应时间0.3-0.6秒
- **双上限控制**：每千粉≤200元，总价≤粉丝数15%

## 📊 数据概览

| 指标 | 数值 |
|------|------|
| 达人总数 | 152,404条 |
| 产品数据 | 8,619条 |
| 有效GMV达人 | 4,813个 |
| 高质量达人 | GMV>10万：4,117个 |
| GMV范围 | ₫0 - ₫31.2亿 |
| 平均GMV | ₫139,706 |

## 🚀 快速启动

### 环境要求
- Python 3.8+
- 内存 2GB+ (推荐4GB+)
- Windows 10/11

### 30秒启动
```bash
# 1. 进入项目目录
cd C:\Users\你的用户名\Desktop\ChatBot\infound-inner

# 2. 安装依赖（首次运行）
pip install -r requirements.txt

# 3. 启动系统
python app.py

# 4. 访问系统
# 浏览器打开: http://localhost:8000
```

### 启动验证
看到以下信息表示启动成功：
```
✅ 成功加载 152,404 条真实达人记录
✅ 成功加载 8,619 条产品记录
✅ 系统初始化成功
🌟 GOOJODOQ达人匹配系统已启动
```

## 🔧 算法架构

### 1. 产品匹配算法
```
多字段加权匹配：
├── 功能特性：权重1.0 (最重要)
├── 商品介绍：权重0.9 (较高)
├── 商品类别：权重0.8 (中等)
└── 商品标题：权重0.7 (较低)

TF-IDF向量化 → 余弦相似度 → 加权平均
```

### 2. 价格计算算法
```
激进差异化定价：
├── ≤10k粉丝：0.025元/粉丝
├── 10k-50k：250 + (粉丝-10k) × 0.055
├── 50k-100k：2450 + (粉丝-50k) × 0.075
├── 100k-300k：6200 + (粉丝-100k) × 0.095
├── 300k-1M：25200 + (粉丝-300k) × 0.06
└── >1M：67200 + (粉丝-1M) × 0.03

多因素调整 × 双上限控制
```

### 3. 3C产品评分算法
```
3C产品专用评分：
├── 技术适应性评分: 25分
├── 价格区间适配度: 20分
├── 销售能力评分: 20分
├── 专业度评分: 15分
├── 粉丝规模适配度: 10分
└── 复杂产品适应性: 10分
```

## 🌐 API接口

### 推荐API
```http
POST /api/recommend
Content-Type: application/json

{
    "product_description": "GOOJODOQ metal adjustable tablet stand",
    "product_price": 299000,
    "commission_rate": 0.07,
    "top_k": 10,
    "min_followers": 20000
}
```

### 响应示例
```json
{
    "success": true,
    "recommendations": [
        {
            "rank": 1,
            "creator_username": "techreview_vn",
            "performance": {
                "gmv": 2500000,
                "estimated_gmv_price_commission": 1200000,
                "commission": 84000
            },
            "cost": {
                "estimated_cost": 45800,
                "cost_per_thousand_followers": 152.7
            },
            "product_matching": {
                "match_score": 0.89,
                "combined_score": 1.67,
                "product_3c_score": 83.5
            }
        }
    ],
    "statistics": {
        "total_cost": 458000,
        "budget_utilization": "91.6%",
        "cost_calculation_method": "激进差异化定价，双上限控制"
    }
}
```

### 其他API
- `GET /api/health` - 系统健康检查
- `GET /api/stats` - 系统统计信息
- `POST /api/estimate-gmv` - GMV预估API

## 📈 性能指标

### 查询性能
- **数据加载**：45-60秒（一次性）
- **查询响应**：0.3-0.6秒
- **内存使用**：~800MB
- **并发支持**：10-50用户

### 匹配精度
- **产品匹配**：基于8,619条产品数据
- **匹配准确率**：基于真实GMV/佣金验证
- **价格差异化**：47-200元每千粉丝范围

## 🎨 系统特色

### 1. 3C产品特性优化
- **技术适应性**：针对3C产品优化CTR评估
- **价格区间适配**：自动匹配适合的价格区间
- **专业度评分**：特别关注技术理解能力

### 2. 产品类型识别
- **手机数码**：注重CTR和粉丝质量
- **电脑配件**：强调GMV表现
- **音频设备**：重视客单价表现
- **智能设备**：平衡多指标综合评分

### 3. 风险控制
- **双上限保护**：防止价格过高
- **历史验证**：基于真实佣金数据校验
- **分级标记**：低/中/高风险等级

## 📁 文件结构

```
infound-inner/
├── app.py                     # Flask主应用
├── real_data_matcher.py       # 数据匹配引擎
├── index.html                 # 前端界面
├── requirements.txt           # Python依赖
├── 算法介绍.txt               # 详细算法说明
├── 一键启动教程.txt           # 启动教程
├── Match_ProductCreator-main/
│   ├── Creator_List_Viet.xlsx     # 达人数据
│   └── Product_Creator_OCRAll_VietLink0603.xlsx # 产品数据
└── static/
    └── styles.css             # 样式文件
```

## 🔗 相关文档

- [算法介绍.txt](./算法介绍.txt) - 详细的算法说明和技术规格
- [一键启动教程.txt](./一键启动教程.txt) - 完整的安装和启动指南
- [系统更新总结.md](./系统更新总结.md) - 版本更新历史

## 🏷️ 版本信息

- **当前版本**：v2.0 - GMV导向激进差异化版本
- **数据源**：越南GOOJODOQ市场
- **更新日期**：2025-06-04
- **技术栈**：Python + Flask + scikit-learn + pandas

---

**🚀 立即体验GOOJODOQ达人匹配系统，发现最适合您3C产品的优质达人！** 