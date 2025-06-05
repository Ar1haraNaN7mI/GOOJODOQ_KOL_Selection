#!/usr/bin/env python3
"""
TikTok达人匹配系统环境检查脚本
检查所有依赖包是否正确安装
"""

import sys
import importlib
from packaging import version

def check_package(package_name, min_version=None):
    """检查包是否安装并满足版本要求"""
    try:
        module = importlib.import_module(package_name)
        installed_version = getattr(module, '__version__', 'unknown')
        
        status = "✅"
        version_info = f"v{installed_version}"
        
        if min_version and installed_version != 'unknown':
            try:
                if version.parse(installed_version) < version.parse(min_version):
                    status = "⚠️"
                    version_info += f" (需要 >={min_version})"
            except:
                pass
                
        print(f"{status} {package_name:<15} {version_info}")
        return True
        
    except ImportError:
        print(f"❌ {package_name:<15} 未安装")
        return False

def main():
    print("=" * 50)
    print("TikTok达人匹配系统 - 环境依赖检查")
    print("=" * 50)
    
    # 关键依赖包
    packages = [
        ("flask", "3.0.0"),
        ("flask_cors", "6.0.0"),
        ("pandas", "2.1.0"),
        ("numpy", "1.24.0"),
        ("openpyxl", "3.1.0"),
        ("sklearn", "1.3.0"),
        ("requests", "2.31.0"),
        ("tqdm", "4.65.0"),
        ("scipy", "1.11.0"),
        ("matplotlib", "3.8.0"),
        ("seaborn", "0.13.0"),
    ]
    
    success_count = 0
    total_count = len(packages)
    
    print("\n核心依赖包:")
    for package_name, min_ver in packages:
        if check_package(package_name, min_ver):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"检查结果: {success_count}/{total_count} 个包正常")
    
    if success_count == total_count:
        print("🎉 所有依赖包都已正确安装！")
        print("✅ 系统可以正常运行")
        
        # 测试系统核心功能
        print("\n测试系统核心功能...")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import pandas as pd
            import numpy as np
            
            # 简单测试
            vectorizer = TfidfVectorizer()
            test_texts = ["测试文本1", "测试文本2"]
            vectors = vectorizer.fit_transform(test_texts)
            similarity = cosine_similarity(vectors)
            
            print("✅ TF-IDF向量化功能正常")
            print("✅ 余弦相似度计算功能正常")
            print("✅ 所有核心功能测试通过")
            
        except Exception as e:
            print(f"❌ 核心功能测试失败: {e}")
    else:
        print("⚠️  部分依赖包缺失，请运行以下命令安装:")
        print("pip install -r requirements.txt --upgrade")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 