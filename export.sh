#!/bin/bash

echo "开始准备GitHub导出文件..."

# 创建临时目录
EXPORT_DIR="goojodoq_export"
mkdir -p $EXPORT_DIR

# 拷贝必要文件
cp app.py $EXPORT_DIR/
cp real_data_matcher.py $EXPORT_DIR/
cp index.html $EXPORT_DIR/
cp requirements.txt $EXPORT_DIR/
cp vercel.json $EXPORT_DIR/
cp build.sh $EXPORT_DIR/
cp README.md $EXPORT_DIR/

# 创建Excel数据目录
mkdir -p $EXPORT_DIR/Match_ProductCreator-main
cp Match_ProductCreator-main/Creator_List_Viet.xlsx $EXPORT_DIR/Match_ProductCreator-main/
cp Match_ProductCreator-main/Product_Creator_OCRAll_VietLink0603.xlsx $EXPORT_DIR/Match_ProductCreator-main/

# 创建静态资源目录
mkdir -p $EXPORT_DIR/static
cp static/styles.css $EXPORT_DIR/static/

# 创建最小化的.gitignore
cat > $EXPORT_DIR/.gitignore << EOF
# Python相关
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.env
.venv
env/
venv/
ENV/
.pytest_cache/
.coverage
htmlcov/

# 其他
.DS_Store
.idea/
.vscode/
EOF

echo "导出文件准备完成: $EXPORT_DIR"
echo "可以将此目录上传到GitHub" 