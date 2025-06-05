#!/usr/bin/env python3
"""
GOOJODOQ达人匹配系统 - 纯真实数据版本
直接使用真实Excel数据进行筛选和排序，不生成任何假数据
专门针对3C产品优化的推荐算法
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDataMatcher:
    """基于真实数据的达人匹配器 - 纯筛选模式 + 产品匹配，专门针对3C产品优化"""
    
    def __init__(self):
        self.creators_df = None
        self.products_df = None
        self.is_initialized = False
        
        # 数据文件路径 - 更新为正确的文件路径
        self.creator_file = "Match_ProductCreator-main/Creator_List_Viet.xlsx"
        self.product_file = "Match_ProductCreator-main/Product_Creator_OCRAll_VietLink0603.xlsx"
        
        # 文本向量化器
        self.product_vectorizer = None
        self.product_vectors = None
        
        # 3C产品关键词库
        self.tech_3c_keywords = {
            '手机数码': ['手机', '智能手机', 'iPhone', 'Android', '华为', '小米', '三星', 'OPPO', 'vivo', '苹果'],
            '电脑配件': ['电脑', '笔记本', '台式机', '主板', '显卡', '内存', 'CPU', '硬盘', 'SSD', '键盘', '鼠标'],
            '音频设备': ['耳机', '音响', '蓝牙耳机', '降噪耳机', '音箱', '麦克风', '耳塞', '头戴式'],
            '智能设备': ['智能手表', '平板', 'iPad', '智能家居', '路由器', '充电器', '数据线', '移动电源'],
            '摄影设备': ['相机', '摄像头', '镜头', '三脚架', '闪光灯', '稳定器', '无人机'],
            '游戏设备': ['游戏机', 'PS5', 'Xbox', 'Switch', '游戏手柄', '游戏键盘', '游戏鼠标', '显示器']
        }
        
    def estimate_gmv_from_price_commission(self, product_price, commission_rate, creator_data):
        """
        根据商品价格和抽佣比例预估GMV
        结合达人历史表现和3C产品特性进行预估
        """
        try:
            # 获取达人基础数据
            followers = creator_data.get('Affiliate_followers', 0)
            historical_gmv = creator_data.get('Affiliate_GMV', 0)
            historical_orders = creator_data.get('Affiliate_orders', 0)
            ctr = creator_data.get('CTR', 0)
            avg_order_value = creator_data.get('Avg_order_value', 0)
            
            # 基础转化率预估（3C产品特化）
            if followers <= 10000:
                base_conversion_rate = 0.008  # 小达人，3C产品转化率较低
            elif followers <= 50000:
                base_conversion_rate = 0.012  # 中小达人
            elif followers <= 200000:
                base_conversion_rate = 0.015  # 中等达人
            else:
                base_conversion_rate = 0.018  # 大达人，3C产品信任度更高
            
            # 根据价格区间调整转化率（3C产品价格敏感度）
            if product_price < 100:
                price_multiplier = 1.5  # 低价3C产品更容易转化
            elif product_price < 500:
                price_multiplier = 1.2  # 中低价产品
            elif product_price < 1500:
                price_multiplier = 1.0  # 中等价位
            elif product_price < 5000:
                price_multiplier = 0.8  # 高价产品转化率下降
            else:
                price_multiplier = 0.6  # 极高价产品，需要更强信任度
            
            # 根据佣金率调整（激励效应）
            if commission_rate > 0.15:
                commission_multiplier = 1.3  # 高佣金激励
            elif commission_rate > 0.10:
                commission_multiplier = 1.15
            elif commission_rate > 0.05:
                commission_multiplier = 1.0
            else:
                commission_multiplier = 0.9  # 低佣金可能影响推广积极性
            
            # 历史表现调整
            if historical_gmv > 0 and historical_orders > 0:
                historical_conversion = historical_orders / (followers * 0.1)  # 假设10%的粉丝看到推广
                if base_conversion_rate > 0:  # 防止除零错误
                    performance_multiplier = min(2.0, max(0.5, historical_conversion / base_conversion_rate))
                else:
                    performance_multiplier = 1.0
            else:
                performance_multiplier = 1.0
            
            # CTR调整（3C产品重视参数和评测）
            if ctr > 8:
                ctr_multiplier = 1.2  # 高互动率说明内容质量好
            elif ctr > 4:
                ctr_multiplier = 1.1
            else:
                ctr_multiplier = 1.0
            
            # 综合转化率
            final_conversion_rate = (base_conversion_rate * 
                                   price_multiplier * 
                                   commission_multiplier * 
                                   performance_multiplier * 
                                   ctr_multiplier)
            
            # 预估订单数
            estimated_orders = followers * final_conversion_rate
            
            # 预估GMV
            estimated_gmv = estimated_orders * product_price
            
            # 合理性检查
            if historical_gmv > 0:
                # 不能超过历史表现的3倍
                max_reasonable_gmv = historical_gmv * 3
                estimated_gmv = min(estimated_gmv, max_reasonable_gmv)
            
            # 最低GMV保护
            min_gmv = max(product_price * 5, followers * 0.1)  # 至少5单或每粉丝0.1元
            estimated_gmv = max(estimated_gmv, min_gmv)
            
            logger.debug(f"GMV预估: 价格₫{product_price}, 佣金{commission_rate*100:.1f}%, "
                        f"转化率{final_conversion_rate*100:.3f}%, 预估GMV₫{estimated_gmv:.0f}")
            
            return round(estimated_gmv, 0)
            
        except Exception as e:
            logger.error(f"GMV预估计算错误: {e}")
            # 简单降级算法
            return max(product_price * 10, followers * 0.2)
    
    def classify_3c_product(self, product_description):
        """
        对3C产品进行分类，返回产品类别和关键特征
        """
        product_description = product_description.lower()
        
        matched_categories = []
        for category, keywords in self.tech_3c_keywords.items():
            for keyword in keywords:
                if keyword.lower() in product_description:
                    matched_categories.append(category)
                    break
        
        # 去重并返回主要类别
        if matched_categories:
            return matched_categories[0], matched_categories
        else:
            return '通用数码', ['通用数码']
    
    def calculate_3c_product_score(self, creator_data, product_category, product_price):
        """
        针对3C产品计算达人适配度评分 - 差异化版本
        重点考虑3C产品的特有属性：技术专业度、价格敏感度、产品复杂度适应性
        """
        score = 0.0
        
        # 1. 技术适应性评分 (25分) - 基于CTR和互动质量
        ctr = creator_data.get('CTR', 0)
        followers = creator_data.get('Affiliate_followers', 0)
        
        # 3C产品需要较高的技术理解能力，CTR是重要指标
        if ctr > 15:  # 高技术互动能力
            score += 25
        elif ctr > 8:
            score += 20
        elif ctr > 4:
            score += 15
        elif ctr > 2:
            score += 10
        else:
            score += 5  # 低互动率对3C产品不利
        
        # 2. 价格区间适配度 (20分) - 3C产品价格敏感度很高
        avg_order_value = creator_data.get('Avg_order_value', 0)
        if avg_order_value > 0:
            price_ratio = product_price / avg_order_value
            
            if 0.7 <= price_ratio <= 1.5:  # 价格高度匹配
                score += 20
            elif 0.5 <= price_ratio <= 2.0:  # 价格较匹配
                score += 15
            elif 0.3 <= price_ratio <= 3.0:  # 价格一般匹配
                score += 10
            else:  # 价格差异较大
                score += 5
        else:
            score += 8  # 没有历史价格数据，给中等分
        
        # 3. 销售能力评分 (20分) - 基于历史订单转化能力
        orders = creator_data.get('Affiliate_orders', 0)
        if followers > 0:
            order_conversion_rate = orders / followers  # 订单转化率
            
            if order_conversion_rate > 0.02:  # 2%以上转化率，优秀
                score += 20
            elif order_conversion_rate > 0.01:  # 1-2%转化率，良好
                score += 15
            elif order_conversion_rate > 0.005:  # 0.5-1%转化率，一般
                score += 10
            elif order_conversion_rate > 0.002:  # 0.2-0.5%转化率，偏低
                score += 5
            else:
                score += 2  # 转化率过低
        else:
            score += 8  # 无粉丝数据，给中等分
        
        # 4. 3C产品专业度评分 (15分) - 基于GMV规模和产品类别匹配
        gmv = creator_data.get('Affiliate_GMV', 0)
        
        # 3C产品需要一定的专业背景，GMV体现专业能力
        if gmv > 2000000:  # 高GMV说明有强专业能力
            score += 15
        elif gmv > 1000000:
            score += 12
        elif gmv > 500000:
            score += 10
        elif gmv > 100000:
            score += 8
        else:
            score += 5  # 低GMV对3C产品推广不利
        
        # 5. 粉丝规模适配度 (10分) - 3C产品的最佳粉丝范围
        if 30000 <= followers <= 150000:  # 3C产品黄金粉丝范围
            score += 10
        elif 15000 <= followers <= 300000:  # 较好范围
            score += 8
        elif 5000 <= followers <= 500000:  # 可接受范围
            score += 6
        else:
            score += 3  # 过高或过低都不适合3C产品
        
        # 6. 复杂产品适应性 (10分) - 基于平均订单价值的稳定性
        if avg_order_value > 200000:  # 习惯高价值产品
            score += 10
        elif avg_order_value > 100000:
            score += 8
        elif avg_order_value > 50000:
            score += 6
        else:
            score += 4
        
        # 根据产品类别进行微调
        category_bonus = 0
        if product_category == '手机数码':
            if ctr > 10:  # 手机类需要高互动
                category_bonus = 5
            if followers > 50000:  # 手机类适合中大型达人
                category_bonus += 3
        elif product_category == '电脑配件':
            if gmv > 1000000:  # 电脑配件需要专业性
                category_bonus = 8
        elif product_category == '音频设备':
            if avg_order_value > 100000:  # 音频设备价值敏感
                category_bonus = 6
        
        score += category_bonus
        
        return min(100, max(0, score))  # 确保分数在0-100范围内
        
    def load_data(self):
        """加载真实的GOOJODOQ达人数据和产品数据"""
        try:
            # 加载达人数据
            if not os.path.exists(self.creator_file):
                logger.error(f"达人数据文件不存在: {self.creator_file}")
                return False
                
            logger.info(f"加载真实GOOJODOQ达人数据: {self.creator_file}")
            self.creators_df = pd.read_excel(self.creator_file)
            logger.info(f"成功加载 {len(self.creators_df):,} 条真实达人记录")
            
            # 加载产品数据
            if not os.path.exists(self.product_file):
                logger.warning(f"产品数据文件不存在: {self.product_file}")
                self.products_df = pd.DataFrame()  # 空DataFrame
            else:
                logger.info(f"加载产品数据: {self.product_file}")
                self.products_df = pd.read_excel(self.product_file)
                logger.info(f"成功加载 {len(self.products_df):,} 条产品记录")
            
            # 数据预处理
            self._preprocess_data()
            
            # 初始化产品文本向量化
            if not self.products_df.empty:
                self._initialize_product_vectors()
            
            self.is_initialized = True
            logger.info("✅ 真实数据加载完成")
            return True
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return False
    
    def _preprocess_data(self):
        """数据预处理，针对新的数据结构"""
        # 显示达人数据基本信息
        logger.info(f"达人数据列: {list(self.creators_df.columns)}")
        logger.info(f"达人数据形状: {self.creators_df.shape}")
        
        # 显示产品数据基本信息
        if not self.products_df.empty:
            logger.info(f"产品数据列: {list(self.products_df.columns)}")
            logger.info(f"产品数据形状: {self.products_df.shape}")
        
        # 处理达人数据的缺失值和非数值字符
        numeric_columns = ['Affiliate_GMV', 'Affiliate_followers', 'Est_commission', 
                          'Avg_order_value', 'CTR', 'Affiliate_orders']
        
        for col in numeric_columns:
            if col in self.creators_df.columns:
                # 将非数值字符串替换为NaN
                self.creators_df[col] = pd.to_numeric(self.creators_df[col], errors='coerce')
                
                # 使用中位数填充缺失值
                median_val = self.creators_df[col].median()
                if pd.isna(median_val):
                    median_val = 0  # 如果中位数也是NaN，使用0
                self.creators_df[col] = self.creators_df[col].fillna(median_val)
                logger.info(f"达人数据列 {col}: 转换为数值类型，使用中位数 {median_val:.2f} 填充缺失值")
        
        # 处理产品数据的缺失值和非数值字符
        if not self.products_df.empty:
            product_numeric_columns = ['粉丝数', 'Affiliate_GMV', 'Affiliate_followers', 'Est_commission', 
                              'Avg_order_value', 'CTR', 'Affiliate_orders', '售价', '佣金比例']
            
            for col in product_numeric_columns:
                if col in self.products_df.columns:
                    # 将非数值字符串替换为NaN
                    self.products_df[col] = pd.to_numeric(self.products_df[col], errors='coerce')
                    
                    # 使用中位数填充缺失值
                    median_val = self.products_df[col].median()
                    if pd.isna(median_val):
                        median_val = 0  # 如果中位数也是NaN，使用0
                    self.products_df[col] = self.products_df[col].fillna(median_val)
                    logger.info(f"产品数据列 {col}: 转换为数值类型，使用中位数 {median_val:.2f} 填充缺失值")
            
            # 处理联系方式字段
            contact_columns = ['达人邮箱', '达人其他联系方式']
            for col in contact_columns:
                if col in self.products_df.columns:
                    self.products_df[col] = self.products_df[col].fillna('无')
        
        # 数据统计
        gmv_min, gmv_max, gmv_mean = self.creators_df['Affiliate_GMV'].min(), self.creators_df['Affiliate_GMV'].max(), self.creators_df['Affiliate_GMV'].mean()
        followers_min, followers_max, followers_mean = self.creators_df['Affiliate_followers'].min(), self.creators_df['Affiliate_followers'].max(), self.creators_df['Affiliate_followers'].mean()
        
        logger.info(f"达人GMV统计: 最小值={gmv_min:.0f}, 最大值={gmv_max:.0f}, 平均值={gmv_mean:.0f}")
        logger.info(f"达人粉丝数统计: 最小值={followers_min:.0f}, 最大值={followers_max:.0f}, 平均值={followers_mean:.0f}")
        
        # 统计有效数据
        gmv_positive = (self.creators_df['Affiliate_GMV'] > 0).sum()
        logger.info(f"有效达人数据: GMV>0的达人={gmv_positive}个")
        
        if not self.products_df.empty:
            has_email = (self.products_df.get('达人邮箱', pd.Series(['无'] * len(self.products_df))) != '无').sum()
            has_contact = (self.products_df.get('达人其他联系方式', pd.Series(['无'] * len(self.products_df))) != '无').sum()
            logger.info(f"产品数据联系方式: 有邮箱={has_email}个, 有其他联系方式={has_contact}个")
    
    def _initialize_product_vectors(self):
        """初始化产品文本向量化 - 针对3C产品优化"""
        try:
            # 检查产品相关字段
            text_fields = []
            field_weights = {}
            
            # 按优先级添加字段：功能特性 >= 商品介绍 >= 商品类别 >= 商品标题
            if '功能特性' in self.products_df.columns:
                text_fields.append('功能特性')
                field_weights['功能特性'] = 1.0  # 最高权重，3C产品重视功能
            if '商品介绍' in self.products_df.columns:
                text_fields.append('商品介绍')
                field_weights['商品介绍'] = 0.9  # 次高权重
            if '商品类别' in self.products_df.columns:
                text_fields.append('商品类别')
                field_weights['商品类别'] = 0.8  # 类别重要
            if '商品标题' in self.products_df.columns:
                text_fields.append('商品标题')
                field_weights['商品标题'] = 0.7
            
            if not text_fields:
                logger.warning("未找到产品文本字段，跳过产品向量化")
                return
            
            # 为每个字段单独创建向量化器和向量
            self.product_vectorizer = {}
            self.product_vectors = {}
            self.field_weights = field_weights
            
            for field in text_fields:
                # 获取该字段的文本
                field_texts = []
                for _, row in self.products_df.iterrows():
                    if pd.notna(row.get(field, '')):
                        field_texts.append(str(row[field]))
                    else:
                        field_texts.append("")
                
                if not any(field_texts):  # 如果字段都是空的
                    continue
                
                # 创建该字段的向量化器（针对3C产品优化）
                vectorizer = TfidfVectorizer(
                    max_features=1500,  # 增加特征数，3C产品术语丰富
                    ngram_range=(1, 3),  # 包含3-gram，捕捉技术术语
                    min_df=1,
                    max_df=0.95,
                    token_pattern=r'(?u)\b\w+\b',  # 支持中文字符
                    lowercase=True
                )
                
                # 训练并转换
                vectors = vectorizer.fit_transform(field_texts)
                
                self.product_vectorizer[field] = vectorizer
                self.product_vectors[field] = vectors
                
                logger.info(f"字段 '{field}' 向量化完成: {vectors.shape}, 权重: {field_weights[field]}")
            
            logger.info(f"✅ 多字段产品向量化完成，字段权重: {field_weights}")
            
        except Exception as e:
            logger.error(f"产品向量化失败: {e}")
            self.product_vectorizer = None
            self.product_vectors = None
    
    def calculate_product_match_score(self, query_text):
        """计算加权产品匹配度分数"""
        if not self.product_vectorizer or not self.product_vectors:
            return {}
        
        try:
            # 获取匹配的产品和对应的达人
            product_matches = {}
            
            # 为每个字段计算相似度并加权
            field_similarities = {}
            
            for field, vectorizer in self.product_vectorizer.items():
                # 向量化查询文本
                query_vector = vectorizer.transform([query_text])
                
                # 计算余弦相似度
                similarities = cosine_similarity(query_vector, self.product_vectors[field]).flatten()
                field_similarities[field] = similarities
            
            # 计算加权平均相似度
            for idx in range(len(self.products_df)):
                weighted_score = 0
                total_weight = 0
                
                for field, similarities in field_similarities.items():
                    if similarities[idx] > 0.01:  # 降低相似度阈值
                        weight = self.field_weights[field]
                        weighted_score += similarities[idx] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    final_score = weighted_score / total_weight
                    product_row = self.products_df.iloc[idx]
                    creator_username = product_row.get('Creator_username', '')
                    
                    if creator_username and creator_username.strip():
                        # 如果达人已存在，取最高相似度
                        if creator_username in product_matches:
                            product_matches[creator_username] = max(product_matches[creator_username], final_score)
                        else:
                            product_matches[creator_username] = final_score
            
            logger.info(f"产品匹配: 找到 {len(product_matches)} 个相关达人 (加权平均算法)")
            return product_matches
            
        except Exception as e:
            logger.error(f"产品匹配计算失败: {e}")
            return {}
    
    def filter_creators_with_price_commission(self, product_price, commission_rate, min_followers=None, 
                                            max_followers=None, min_estimated_gmv=None, max_estimated_gmv=None,
                                            top_k=10, product_query=None, budget=None):
        """
        新的筛选方法：基于商品价格和抽佣比例筛选达人
        专门针对3C产品优化
        """
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                return {"success": False, "error": "系统未初始化"}
            
            # 复制数据进行筛选
            filtered_df = self.creators_df.copy()
            original_count = len(filtered_df)
            
            # 分类3C产品
            product_category, categories = self.classify_3c_product(product_query or "")
            logger.info(f"产品分类: {product_category}, 所有类别: {categories}")
            
            # 为每个达人计算预估GMV和适配度评分
            estimated_gmvs = []
            product_scores = []
            
            for idx, row in filtered_df.iterrows():
                # 转换为字典格式
                creator_data = row.to_dict()
                
                # 计算预估GMV
                estimated_gmv = self.estimate_gmv_from_price_commission(
                    product_price, commission_rate, creator_data
                )
                estimated_gmvs.append(estimated_gmv)
                
                # 计算3C产品适配度评分
                score = self.calculate_3c_product_score(creator_data, product_category, product_price)
                product_scores.append(score)
            
            # 添加计算结果到DataFrame
            filtered_df['estimated_gmv_price_commission'] = estimated_gmvs
            filtered_df['product_3c_score'] = product_scores
            
            # 应用筛选条件
            if min_followers is not None:
                filtered_df = filtered_df[filtered_df['Affiliate_followers'] >= min_followers]
                logger.info(f"粉丝数 >= {min_followers} 筛选后: {len(filtered_df)} 个达人")
            
            if max_followers is not None:
                filtered_df = filtered_df[filtered_df['Affiliate_followers'] <= max_followers]
                logger.info(f"粉丝数 <= {max_followers} 筛选后: {len(filtered_df)} 个达人")
            
            if min_estimated_gmv is not None:
                filtered_df = filtered_df[filtered_df['estimated_gmv_price_commission'] >= min_estimated_gmv]
                logger.info(f"预估GMV >= {min_estimated_gmv} 筛选后: {len(filtered_df)} 个达人")
            
            if max_estimated_gmv is not None:
                filtered_df = filtered_df[filtered_df['estimated_gmv_price_commission'] <= max_estimated_gmv]
                logger.info(f"预估GMV <= {max_estimated_gmv} 筛选后: {len(filtered_df)} 个达人")
            
            # 计算产品匹配度
            product_matches = {}
            if product_query:
                product_matches = self.calculate_product_match_score(product_query)
            
            # 3C产品专用排序算法
            def calculate_comprehensive_score(row):
                base_score = 0
                
                # 历史GMV权重 (30%)
                historical_gmv = row.get('Affiliate_GMV', 0)
                if historical_gmv > 0:
                    gmv_score = min(100, np.log10(historical_gmv + 1) * 20)  # 对数缩放
                    base_score += gmv_score * 0.30
                
                # 预估GMV权重 (25%)
                estimated_gmv = row.get('estimated_gmv_price_commission', 0)
                if estimated_gmv > 0:
                    estimated_score = min(100, np.log10(estimated_gmv + 1) * 15)
                    base_score += estimated_score * 0.25
                
                # 3C产品适配度权重 (20%)
                product_score = row.get('product_3c_score', 0)
                base_score += product_score * 0.20
                
                # 产品匹配度权重 (15%)
                username = row.get('Creator_username', '')
                if username in product_matches:
                    match_score = product_matches[username] * 100
                    base_score += match_score * 0.15
                
                # 粉丝数权重 (10%) - 适中最佳
                followers = row.get('Affiliate_followers', 0)
                if 20000 <= followers <= 200000:  # 3C产品的最佳粉丝范围
                    followers_score = 100
                elif followers > 200000:
                    followers_score = max(50, 100 - (followers - 200000) / 10000)  # 超过后递减
                else:
                    followers_score = followers / 20000 * 100  # 低于线性增长
                base_score += followers_score * 0.10
                
                return base_score
            
            # 计算综合评分
            filtered_df['comprehensive_score'] = filtered_df.apply(calculate_comprehensive_score, axis=1)
            
            # 排序：综合评分优先
            filtered_df = filtered_df.sort_values('comprehensive_score', ascending=False)
            
            # 预算筛选（如果提供）
            if budget and budget > 0:
                # 计算每个达人的预估费用
                estimated_costs = []
                for _, row in filtered_df.iterrows():
                    creator_data = {
                        'followers': row.get('Affiliate_followers', 0),
                        'gmv': row.get('Affiliate_GMV', 0),
                        'commission': row.get('Est_commission', 0),
                        'avg_order_value': row.get('Avg_order_value', 0),
                        'ctr': row.get('CTR', 0),
                        'estimated_gmv_7pct': row.get('estimated_gmv_price_commission', 0)
                    }
                    # 使用现有的费用计算算法（需要从app.py导入或重新实现）
                    estimated_cost = self._calculate_creator_cost(creator_data)
                    estimated_costs.append(estimated_cost)
                
                filtered_df['estimated_cost'] = estimated_costs
                
                # 预算约束筛选
                cumulative_cost = 0
                selected_indices = []
                for idx, row in filtered_df.iterrows():
                    cost = row['estimated_cost']
                    if cumulative_cost + cost <= budget:
                        cumulative_cost += cost
                        selected_indices.append(idx)
                    else:
                        break
                
                if selected_indices:
                    filtered_df = filtered_df.loc[selected_indices]
                    logger.info(f"预算约束筛选后: {len(filtered_df)} 个达人, 总费用: ¥{cumulative_cost:,.0f}")
                else:
                    # 如果预算太少，至少返回最便宜的达人
                    filtered_df = filtered_df.head(1)
                    logger.warning(f"预算不足，仅返回最高评分达人")
            
            # 限制返回数量
            if len(filtered_df) > top_k:
                filtered_df = filtered_df.head(top_k)
            
            # 构建返回结果
            results = []
            for idx, (_, row) in enumerate(filtered_df.iterrows()):
                # 从产品数据中获取联系方式信息和昵称
                creator_username = row.get('Creator_username', '')
                email = '无'
                other_contact = '无'
                creator_nickname = '无'  # 默认昵称
                
                if creator_username and not self.products_df.empty:
                    # 查找该达人在产品数据中的联系方式和昵称
                    product_matches_for_creator = self.products_df[self.products_df['Creator_username'] == creator_username]
                    if not product_matches_for_creator.empty:
                        first_match = product_matches_for_creator.iloc[0]
                        email = first_match.get('达人邮箱', '无')
                        other_contact = first_match.get('达人其他联系方式', '无')
                        creator_nickname = first_match.get('达人昵称', '无')  # 获取达人昵称
                        
                        # 确保不是NaN值
                        if pd.isna(email) or email == '':
                            email = '无'
                        if pd.isna(other_contact) or other_contact == '':
                            other_contact = '无'
                        if pd.isna(creator_nickname) or creator_nickname == '':
                            creator_nickname = creator_username  # 如果昵称为空，使用用户名
                
                creator_data = {
                    'rank': idx + 1,
                    'creator_username': creator_username,
                    'creator_nickname': creator_nickname,
                    'creator_email': email,
                    'creator_other_contact': other_contact,
                    'audience': {
                        'followers': int(row.get('Affiliate_followers', 0))
                    },
                    'performance': {
                        'gmv': float(row.get('Affiliate_GMV', 0)),
                        'estimated_gmv_price_commission': float(row.get('estimated_gmv_price_commission', 0)),
                        'orders': int(row.get('Affiliate_orders', 0)),
                        'avg_order_value': float(row.get('Avg_order_value', 0)),
                        'ctr': float(row.get('CTR', 0))
                    },
                    'cost': {
                        'estimated_cost': float(row.get('estimated_cost', 0)) if 'estimated_cost' in row else 0,
                        'cost_per_thousand_followers': 0  # 稍后计算
                    },
                    'product_matching': {
                        'has_product_data': len(product_matches) > 0,
                        'match_score': product_matches.get(row.get('Creator_username', ''), 0),
                        'combined_score': float(row.get('comprehensive_score', 0)),
                        'product_3c_score': float(row.get('product_3c_score', 0))
                    },
                    'metadata': {
                        'ranking_method': f"3C产品专用算法 ({product_category})",
                        'data_source': 'GOOJODOQ真实数据',
                        'product_category': product_category
                    },
                    'extra_data': row.to_dict()  # 包含所有原始数据
                }
                
                # 计算每千粉丝成本
                if creator_data['cost']['estimated_cost'] > 0 and creator_data['audience']['followers'] > 0:
                    creator_data['cost']['cost_per_thousand_followers'] = round(
                        (creator_data['cost']['estimated_cost'] / creator_data['audience']['followers']) * 1000, 2
                    )
                
                results.append(creator_data)
            
            processing_time = time.time() - start_time
            
            # 统计信息
            statistics = {
                'total_count': original_count,
                'filtered_count': len(filtered_df),
                'returned_count': len(results),
                'product_category': product_category,
                'has_product_matching': len(product_matches) > 0,
                'ranking_method': f"3C产品专用算法 ({product_category})",
                'processing_time': processing_time
            }
            
            if results:
                statistics.update({
                    'total_cost_formatted': f"₫{sum(r['cost']['estimated_cost'] for r in results):,.0f}",
                    'average_gmv': f"₫{sum(r['performance']['gmv'] for r in results) / len(results):,.0f}",
                    'average_estimated_gmv': f"₫{sum(r['performance']['estimated_gmv_price_commission'] for r in results) / len(results):,.0f}"
                })
            
            return {
                'success': True,
                'recommendations': results,
                'statistics': statistics,
                'processing_time': processing_time,
                'query': {
                    'product_price': product_price,
                    'commission_rate': commission_rate,
                    'product_query': product_query,
                    'top_k': top_k,
                    'budget': budget
                }
            }
            
        except Exception as e:
            logger.error(f"筛选处理失败: {e}")
            return {"success": False, "error": f"筛选失败: {str(e)}"}
    
    def _calculate_creator_cost(self, creator_data):
        """简化版费用计算（从app.py移植）"""
        try:
            followers = creator_data.get('followers', 0)
            gmv = creator_data.get('gmv', 0)
            
            if followers <= 0:
                return 50
            
            # 基础费用计算
            if followers <= 10000:
                base_cost = followers * 0.025
            elif followers <= 50000:
                base_cost = 250 + (followers - 10000) * 0.055
            elif followers <= 100000:
                base_cost = 2450 + (followers - 50000) * 0.075
            else:
                base_cost = 6200 + (followers - 100000) * 0.095
            
            # GMV调整
            if gmv > 500000:
                gmv_multiplier = 1.2
            elif gmv > 100000:
                gmv_multiplier = 1.1
            else:
                gmv_multiplier = 1.0
            
            estimated_cost = base_cost * gmv_multiplier
            
            # 上限控制
            max_cost_by_cpm = (followers / 1000) * 200
            max_cost_by_followers = followers * 0.15
            
            estimated_cost = min(estimated_cost, max_cost_by_cpm, max_cost_by_followers)
            estimated_cost = max(estimated_cost, 50)
            
            return round(estimated_cost, 0)
            
        except Exception as e:
            logger.error(f"费用计算错误: {e}")
            return max(50, followers * 0.02) if followers > 0 else 50

    def filter_creators(self, min_gmv=None, max_gmv=None, min_followers=None, max_followers=None,
                       min_commission=None, max_commission=None, min_ctr=None, max_ctr=None,
                       top_k=10, sort_by='GMV', product_query=None):
        """筛选达人 - 保持向后兼容性的旧方法"""
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                return {"success": False, "error": "系统未初始化"}
            
            # 复制数据进行筛选
            filtered_df = self.creators_df.copy()
            original_count = len(filtered_df)
            
            # 计算产品匹配度
            product_matches = {}
            if product_query:
                product_matches = self.calculate_product_match_score(product_query)
            
            # 应用筛选条件 - 改为GMV导向
            if min_gmv is not None:
                filtered_df = filtered_df[filtered_df['Affiliate_GMV'] >= min_gmv]
                logger.info(f"GMV >= {min_gmv} 筛选后: {len(filtered_df)} 个达人")
            
            if max_gmv is not None:
                filtered_df = filtered_df[filtered_df['Affiliate_GMV'] <= max_gmv]
                logger.info(f"GMV <= {max_gmv} 筛选后: {len(filtered_df)} 个达人")
            
            if min_followers is not None:
                filtered_df = filtered_df[filtered_df['Affiliate_followers'] >= min_followers]
                logger.info(f"粉丝数 >= {min_followers} 筛选后: {len(filtered_df)} 个达人")
            
            if max_followers is not None:
                filtered_df = filtered_df[filtered_df['Affiliate_followers'] <= max_followers]
                logger.info(f"粉丝数 <= {max_followers} 筛选后: {len(filtered_df)} 个达人")
            
            if min_commission is not None:
                filtered_df = filtered_df[filtered_df['Est_commission'] >= min_commission]
                logger.info(f"佣金 >= {min_commission} 筛选后: {len(filtered_df)} 个达人")
            
            if max_commission is not None:
                filtered_df = filtered_df[filtered_df['Est_commission'] <= max_commission]
                logger.info(f"佣金 <= {max_commission} 筛选后: {len(filtered_df)} 个达人")
            
            if min_ctr is not None and 'CTR' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['CTR'] >= min_ctr]
                logger.info(f"CTR >= {min_ctr} 筛选后: {len(filtered_df)} 个达人")
            
            if max_ctr is not None and 'CTR' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['CTR'] <= max_ctr]
                logger.info(f"CTR <= {max_ctr} 筛选后: {len(filtered_df)} 个达人")
            
            # 添加产品匹配度分数并排序
            if product_matches:
                filtered_df['product_match_score'] = filtered_df['Creator_username'].map(product_matches).fillna(0)
                
                # 分离有匹配和无匹配的达人
                matched_creators = filtered_df[filtered_df['product_match_score'] > 0].copy()
                unmatched_creators = filtered_df[filtered_df['product_match_score'] == 0].copy()
                
                if len(matched_creators) > 0:
                    # 计算动态权重 - 根据匹配质量调整权重分配
                    avg_match_score = matched_creators['product_match_score'].mean()
                    max_match_score = matched_creators['product_match_score'].max()
                    min_match_score = matched_creators['product_match_score'].min()
                    
                    # 动态权重：匹配质量高时减少产品权重，让多因素发挥更大作用
                    if avg_match_score > 0.7:  # 高质量匹配
                        product_weight = 0.30  # 产品匹配30%
                        gmv_weight = 0.40      # GMV 40%
                        commission_weight = 0.20  # 佣金 20%
                        engagement_weight = 0.10  # 互动指标 10%
                    elif avg_match_score > 0.4:  # 中等质量匹配
                        product_weight = 0.40  # 产品匹配40%
                        gmv_weight = 0.35      # GMV 35%
                        commission_weight = 0.10  # 佣金 10%
                        engagement_weight = 0.15  # 互动指标 15%
                    else:  # 低质量匹配
                        product_weight = 0.50  # 产品匹配50%
                        gmv_weight = 0.25      # GMV 25%
                        commission_weight = 0.15  # 佣金 15%
                        engagement_weight = 0.10  # 互动指标 10%
                    
                    # 标准化各项指标
                    max_gmv = matched_creators['Affiliate_GMV'].max() if matched_creators['Affiliate_GMV'].max() > 0 else 1
                    max_commission = matched_creators['Est_commission'].max() if matched_creators['Est_commission'].max() > 0 else 1
                    max_ctr = matched_creators['CTR'].max() if 'CTR' in matched_creators.columns and matched_creators['CTR'].max() > 0 else 1
                    max_followers = matched_creators['Affiliate_followers'].max() if matched_creators['Affiliate_followers'].max() > 0 else 1
                    
                    # 有产品匹配的达人分数范围：1.0 - 2.0
                    matched_creators['combined_score'] = 1.0 + (
                        (matched_creators['product_match_score'] / max_match_score) * product_weight +
                        (matched_creators['Affiliate_GMV'] / max_gmv) * gmv_weight +
                        (matched_creators['Est_commission'] / max_commission) * commission_weight +
                        ((matched_creators.get('CTR', 0) / max_ctr) * 0.6 + 
                         (matched_creators['Affiliate_followers'] / max_followers) * 0.4) * engagement_weight
                    )
                    
                    logger.info(f"动态权重匹配: 产品{product_weight*100:.0f}% + GMV{gmv_weight*100:.0f}% + 佣金{commission_weight*100:.0f}% + 互动{engagement_weight*100:.0f}%")
                    
                    # 对无产品匹配的达人：多因素权重，分数范围 0.0 - 1.0
                    if len(unmatched_creators) > 0:
                        unmatched_creators['combined_score'] = (
                            (unmatched_creators['Affiliate_GMV'] / max_gmv) * 0.35 +          # GMV 35%
                            (unmatched_creators['Est_commission'] / max_commission) * 0.30 +   # 佣金 30%
                            (unmatched_creators.get('CTR', 0) / max_ctr) * 0.20 +             # CTR 20%
                            (unmatched_creators['Affiliate_followers'] / max_followers) * 0.15 # 粉丝数 15%
                        ) * 0.8  # 总体降权到0.8以下
                    
                    # 先排序有匹配的达人，再排序无匹配的达人
                    matched_creators = matched_creators.sort_values('combined_score', ascending=False)
                    if len(unmatched_creators) > 0:
                        unmatched_creators = unmatched_creators.sort_values('combined_score', ascending=False)
                        # 合并结果，有产品匹配的排在前面
                        filtered_df = pd.concat([matched_creators, unmatched_creators], ignore_index=True)
                    else:
                        filtered_df = matched_creators
                    
                    logger.info(f"使用动态权重产品匹配排序 (匹配质量: {avg_match_score:.2f})")
                else:
                    # 没有任何产品匹配，使用多元化GMV排序
                    max_gmv = filtered_df['Affiliate_GMV'].max() if filtered_df['Affiliate_GMV'].max() > 0 else 1
                    max_commission = filtered_df['Est_commission'].max() if filtered_df['Est_commission'].max() > 0 else 1
                    max_ctr = filtered_df['CTR'].max() if 'CTR' in filtered_df.columns and filtered_df['CTR'].max() > 0 else 1
                    max_followers = filtered_df['Affiliate_followers'].max() if filtered_df['Affiliate_followers'].max() > 0 else 1
                    
                    filtered_df['combined_score'] = (
                        (filtered_df['Affiliate_GMV'] / max_gmv) * 0.40 +          # GMV 40%
                        (filtered_df['Est_commission'] / max_commission) * 0.30 +   # 佣金 30%
                        (filtered_df.get('CTR', 0) / max_ctr) * 0.20 +             # CTR 20%
                        (filtered_df['Affiliate_followers'] / max_followers) * 0.10 # 粉丝数 10%
                    )
                    filtered_df = filtered_df.sort_values('combined_score', ascending=False)
                    logger.info(f"无产品匹配，使用多因素GMV优先排序")
            else:
                filtered_df['product_match_score'] = 0
                # 使用多元化GMV排序
                max_gmv = filtered_df['Affiliate_GMV'].max() if filtered_df['Affiliate_GMV'].max() > 0 else 1
                max_commission = filtered_df['Est_commission'].max() if filtered_df['Est_commission'].max() > 0 else 1
                max_ctr = filtered_df['CTR'].max() if 'CTR' in filtered_df.columns and filtered_df['CTR'].max() > 0 else 1
                max_followers = filtered_df['Affiliate_followers'].max() if filtered_df['Affiliate_followers'].max() > 0 else 1
                
                filtered_df['combined_score'] = (
                    (filtered_df['Affiliate_GMV'] / max_gmv) * 0.40 +          # GMV 40%
                    (filtered_df['Est_commission'] / max_commission) * 0.30 +   # 佣金 30%
                    (filtered_df.get('CTR', 0) / max_ctr) * 0.20 +             # CTR 20%
                    (filtered_df['Affiliate_followers'] / max_followers) * 0.10 # 粉丝数 10%
                )
                filtered_df = filtered_df.sort_values('combined_score', ascending=False)
            
            # 取前K个结果
            top_results = filtered_df.head(top_k)
            
            # 构建结果
            results = []
            for rank, (_, creator) in enumerate(top_results.iterrows(), 1):
                # 计算预估GMV基于7%佣金率
                estimated_gmv_7pct = creator['Est_commission'] / 0.07 if creator['Est_commission'] > 0 else 0
                
                result = {
                    "rank": rank,
                    "creator_username": creator['Creator_username'],
                    "gmv": float(creator['Affiliate_GMV']),
                    "estimated_gmv_7pct": float(estimated_gmv_7pct),  # 新增：基于7%计算的预估GMV
                    "followers": int(creator['Affiliate_followers']),
                    "commission": float(creator['Est_commission']),
                    "ctr": float(creator.get('CTR', 0)),
                    "avg_order_value": float(creator.get('Avg_order_value', 0)),
                    "product_match_score": float(creator.get('product_match_score', 0)),
                    "combined_score": float(creator.get('combined_score', 0)),
                    "raw_data": creator.to_dict()
                }
                results.append(result)
            
            # 统计信息
            statistics = {
                "total_count": len(results),
                "filtered_count": len(filtered_df),
                "original_count": original_count,
                "average_gmv": float(top_results['Affiliate_GMV'].mean()) if len(top_results) > 0 else 0,
                "total_commission": float(top_results['Est_commission'].sum()) if len(top_results) > 0 else 0,
                "has_product_matching": len(product_matches) > 0
            }
            
            processing_time = time.time() - start_time
            logger.info(f"筛选完成: 从{original_count}个达人中筛选出{len(results)}个结果，用时{processing_time:.2f}秒")
            
            return {
                "success": True,
                "results": results,
                "statistics": statistics,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"筛选过程出错: {e}")
            return {"success": False, "error": str(e)}
    
    def get_system_stats(self):
        """获取系统统计信息"""
        if not self.is_initialized:
            return {"success": False, "error": "系统未初始化"}
        
        stats = {
            "creator_count": len(self.creators_df),
            "product_count": len(self.products_df) if self.products_df is not None and not self.products_df.empty else 0,
            "total_creators": len(self.creators_df),
            "total_products": len(self.products_df) if self.products_df is not None and not self.products_df.empty else 0,
            "gmv_range": {
                "min": float(self.creators_df['Affiliate_GMV'].min()),
                "max": float(self.creators_df['Affiliate_GMV'].max()),
                "mean": float(self.creators_df['Affiliate_GMV'].mean())
            },
            "followers_range": {
                "min": int(self.creators_df['Affiliate_followers'].min()),
                "max": int(self.creators_df['Affiliate_followers'].max()),
                "mean": int(self.creators_df['Affiliate_followers'].mean())
            },
            "has_product_data": self.product_vectorizer is not None and not self.products_df.empty,
            "cache_exists": self.product_vectorizer is not None and not self.products_df.empty,
            "data_sources": {
                "creator_file": self.creator_file,
                "product_file": self.product_file,
                "creator_loaded": True,
                "product_loaded": not self.products_df.empty if self.products_df is not None else False
            }
        }
        
        return {"success": True, "stats": stats}
    
    def get_creator_by_username(self, username):
        """根据用户名获取达人详情"""
        if not self.is_initialized:
            return {"success": False, "error": "系统未初始化"}
        
        creator = self.creators_df[self.creators_df['Creator_username'] == username]
        if creator.empty:
            return {"success": False, "error": "达人不存在"}
        
        creator_data = creator.iloc[0].to_dict()
        return {"success": True, "creator": creator_data} 