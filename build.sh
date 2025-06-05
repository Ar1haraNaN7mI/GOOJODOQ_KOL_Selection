#!/bin/bash

# 优化部署脚本
echo "开始优化部署过程..."

# 删除不必要的文件
rm -rf .git
rm -rf __pycache__
rm -rf .pytest_cache
rm -rf .venv
rm -rf .vscode
rm -rf .idea

# 保留必要文件
echo "保留必要的文件..."
files_to_keep=(
  "app.py"
  "real_data_matcher.py"
  "index.html"
  "requirements.txt"
  "vercel.json"
  "static/styles.css"
  "Match_ProductCreator-main/Creator_List_Viet.xlsx"
  "Match_ProductCreator-main/Product_Creator_OCRAll_VietLink0603.xlsx"
)

# 检查requirements.txt文件
echo "确保requirements.txt只包含必要的依赖..."
cat > requirements.txt << EOF
flask==3.1.1
flask-cors==6.0.0
pandas==2.2.3
numpy==2.2.6
openpyxl==3.1.5
scikit-learn==1.6.1
joblib==1.3.0
tqdm==4.67.1
EOF

echo "部署优化完成!" 