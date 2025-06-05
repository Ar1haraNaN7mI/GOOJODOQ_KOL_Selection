#!/usr/bin/env python3
"""
GOOJODOQ达人匹配系统 - 纯筛选版本
将产品描述转换为筛选条件，直接使用真实数据筛选
专门针对3C产品优化
"""

import os
import logging
import re
import math
from flask import Flask, request, jsonify, send_file
from real_data_matcher import RealDataMatcher

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)

# 全局匹配器实例
matcher = RealDataMatcher()

def calculate_estimated_cost(creator_data):
    """
    计算达人推广预估费用
    基于多因素综合算法 - 超保守定价策略，严格双上限控制
    """
    try:
        # 获取基础数据
        followers = creator_data.get('followers', 0)
        gmv = creator_data.get('gmv', 0)
        commission = creator_data.get('commission', 0)
        avg_order_value = creator_data.get('avg_order_value', 0)
        ctr = creator_data.get('ctr', 0)
        estimated_gmv_7pct = creator_data.get('estimated_gmv_7pct', 0)
        
        if followers <= 0:
            return 50  # 最低费用
        
        # 基础费用计算（增大价格差异，策略适中）
        if followers <= 10000:
            base_cost = followers * 0.025  # 小达人：每粉丝0.025元
        elif followers <= 50000:
            base_cost = 250 + (followers - 10000) * 0.055  # 中小达人（递增率提高）
        elif followers <= 100000:
            base_cost = 2450 + (followers - 50000) * 0.075  # 中等达人（显著提高）
        elif followers <= 300000:
            base_cost = 6200 + (followers - 100000) * 0.095  # 大达人（大幅提高）
        elif followers <= 1000000:
            base_cost = 25200 + (followers - 300000) * 0.06  # 超大达人
        else:
            base_cost = 67200 + (followers - 1000000) * 0.03  # 顶级达人
        
        # 多因素调整系数（增加差异化）
        
        # 1. GMV表现系数（增强差异）
        if gmv > 5000000:
            gmv_multiplier = 1.25  # 极高GMV显著提升
        elif gmv > 2000000:
            gmv_multiplier = 1.18
        elif gmv > 1000000:
            gmv_multiplier = 1.12
        elif gmv > 500000:
            gmv_multiplier = 1.06
        elif gmv > 100000:
            gmv_multiplier = 1.0
        else:
            gmv_multiplier = 0.95  # 低GMV略降
        
        # 2. 佣金效率系数（增强差异）
        commission_rate = commission / gmv if gmv > 0 else 0
        if commission_rate > 0.25:  # 极高佣金率
            commission_multiplier = 1.18
        elif commission_rate > 0.15:
            commission_multiplier = 1.12
        elif commission_rate > 0.10:
            commission_multiplier = 1.06
        elif commission_rate > 0.05:
            commission_multiplier = 1.0
        else:
            commission_multiplier = 0.94
        
        # 3. 互动表现系数（增强差异）
        if ctr > 15:
            engagement_multiplier = 1.15
        elif ctr > 10:
            engagement_multiplier = 1.10
        elif ctr > 5:
            engagement_multiplier = 1.05
        elif ctr > 2:
            engagement_multiplier = 1.0
        else:
            engagement_multiplier = 0.95
        
        # 4. 客单价系数（增强差异）
        if avg_order_value > 1000000:  # 极高客单价
            aov_multiplier = 1.12
        elif avg_order_value > 500000:
            aov_multiplier = 1.06
        else:
            aov_multiplier = 1.0
        
        # 加权综合计算（权重平衡）
        comprehensive_multiplier = (
            gmv_multiplier * 0.40 +           # GMV权重40%
            commission_multiplier * 0.30 +    # 佣金率权重30%
            engagement_multiplier * 0.20 +    # 互动率权重20%
            aov_multiplier * 0.10             # 客单价权重10%
        )
        
        # 综合计算预估费用
        estimated_cost = base_cost * comprehensive_multiplier
        
        # 严格双上限控制
        
        # 上限1：每千粉价格不超过200元（提高上限）
        cost_per_thousand = (estimated_cost / followers) * 1000
        max_cost_by_cpm = (followers / 1000) * 200  # 每千粉丝200元上限
        if estimated_cost > max_cost_by_cpm:
            estimated_cost = max_cost_by_cpm
            logger.debug(f"费用被每千粉上限限制: {cost_per_thousand:.2f} -> 200.00")
        
        # 上限2：总价格不超过粉丝数的15%（大幅提高上限，允许激进定价）
        max_cost_by_followers = followers * 0.15  # 粉丝数15%上限
        if estimated_cost > max_cost_by_followers:
            estimated_cost = max_cost_by_followers
            logger.debug(f"费用被粉丝数比例上限限制: {estimated_cost:.0f} -> {max_cost_by_followers:.0f}")
        
        # 最小费用保护（激进策略）
        min_cost = max(50, followers * 0.003)  # 最少每粉丝0.003元，最低50元
        estimated_cost = max(estimated_cost, min_cost)
        
        # 历史数据校验（更加保守）
        if commission > 500 and gmv > 5000:
            # 基于历史表现的费用校验（极其保守）
            performance_based_cost = commission * 1.5  # 假设费用是佣金的1.5倍
            # 取较小值（更保守）
            estimated_cost = min(estimated_cost, performance_based_cost)
        
        # 最终验证双上限
        final_cpm = (estimated_cost / followers) * 1000 if followers > 0 else 0
        final_percentage = (estimated_cost / followers) * 100 if followers > 0 else 0
        
        logger.debug(f"最终费用: ¥{estimated_cost:.0f}, 每千粉: ¥{final_cpm:.2f}, 占粉丝数: {final_percentage:.3f}%")
        
        return round(estimated_cost, 0)
        
    except Exception as e:
        logger.error(f"费用计算错误: {e}")
        # 降级到激进算法（与新体系保持一致）
        followers = creator_data.get('followers', 0)
        if followers <= 5000:
            return max(50, followers * 0.02)
        elif followers <= 15000:
            return max(150, followers * 0.08) 
        elif followers <= 40000:
            return max(950, followers * 0.15)
        else:
            return max(4700, followers * 0.25)

def calculate_cost_per_thousand_followers(estimated_cost, followers):
    """计算每千粉丝成本（CPM）"""
    if followers <= 0:
        return 0
    return round((estimated_cost / followers) * 1000, 2)

def initialize_system():
    """初始化系统"""
    logger.info("🚀 初始化GOOJODOQ达人匹配系统...")
    
    if matcher.load_data():
        logger.info("✅ 系统初始化成功 - 使用GOOJODOQ真实数据")
        return True
    else:
        logger.error("❌ 系统初始化失败")
        return False

def parse_product_to_filters(product_description, budget=None, min_gmv=None, min_followers=None):
    """将产品描述转换为筛选条件 - 改为GMV导向，并加强预算约束"""
    filters = {}
    
    # 基础筛选条件
    if min_gmv is not None:
        filters['min_gmv'] = min_gmv
    
    if min_followers is not None:
        filters['min_followers'] = min_followers
    
    # 根据预算设置合理的粉丝数和GMV上限（更加合理的预算分配）
    if budget and budget > 0:
        # 根据预算估算合理的粉丝数上限（保守估算每千粉丝40元）
        estimated_max_followers = (budget / 40) * 1000
        
        # 根据预算设置GMV范围，避免推荐过于昂贵的达人
        if budget < 1000:
            filters['max_followers'] = min(estimated_max_followers, 50000)
            filters['max_gmv'] = 100000
            logger.info(f"低预算模式: 最大粉丝{filters['max_followers']:,.0f}, 最大GMV{filters['max_gmv']:,.0f}")
        elif budget < 3000:
            filters['max_followers'] = min(estimated_max_followers, 150000)
            filters['max_gmv'] = 300000
            logger.info(f"小预算模式: 最大粉丝{filters['max_followers']:,.0f}, 最大GMV{filters['max_gmv']:,.0f}")
        elif budget < 8000:
            filters['max_followers'] = min(estimated_max_followers, 400000)
            filters['max_gmv'] = 800000
            logger.info(f"中等预算模式: 最大粉丝{filters['max_followers']:,.0f}, 最大GMV{filters['max_gmv']:,.0f}")
        elif budget < 20000:
            filters['max_followers'] = min(estimated_max_followers, 1000000)
            filters['max_gmv'] = 2000000
            logger.info(f"较高预算模式: 最大粉丝{filters['max_followers']:,.0f}, 最大GMV{filters['max_gmv']:,.0f}")
        else:
            # 高预算，相对宽松限制
            filters['max_followers'] = min(estimated_max_followers, 3000000)
            filters['max_gmv'] = 5000000
            logger.info(f"高预算模式: 最大粉丝{filters['max_followers']:,.0f}, 最大GMV{filters['max_gmv']:,.0f}")
    
    # 根据产品类型设置筛选条件
    product_lower = product_description.lower()
    
    # 高端产品 - 需要高GMV和一定粉丝基础
    if any(keyword in product_lower for keyword in ['高端', '奢侈', '豪华', '智能', '科技', '专业']):
        filters['min_gmv'] = max(filters.get('min_gmv', 0), 100000)
        filters['min_followers'] = max(filters.get('min_followers', 0), 30000)
        logger.info("检测到高端产品，设置高GMV和粉丝要求")
    
    # 大众产品 - 注重性价比
    elif any(keyword in product_lower for keyword in ['便宜', '实惠', '性价比', '日用', '家用']):
        filters['min_gmv'] = max(filters.get('min_gmv', 0), 30000)
        if not filters.get('max_gmv'):  # 只有在预算没有设置时才设置
            filters['max_gmv'] = 400000
        logger.info("检测到大众产品，设置适中GMV要求")
    
    # 美妆护肤 - 需要高GMV和互动
    elif any(keyword in product_lower for keyword in ['化妆', '护肤', '美妆', '口红', '面膜']):
        filters['min_gmv'] = max(filters.get('min_gmv', 0), 80000)
        filters['min_followers'] = max(filters.get('min_followers', 0), 20000)
        logger.info("检测到美妆产品，设置GMV和互动要求")
    
    # 服装配饰 - 需要一定粉丝基础
    elif any(keyword in product_lower for keyword in ['服装', '衣服', '鞋子', '包包', '配饰']):
        filters['min_followers'] = max(filters.get('min_followers', 0), 15000)
        filters['min_gmv'] = max(filters.get('min_gmv', 0), 60000)
        logger.info("检测到服装产品，设置粉丝和GMV要求")
    
    # 食品饮料 - 注重转化率
    elif any(keyword in product_lower for keyword in ['食品', '零食', '饮料', '茶', '咖啡']):
        filters['min_gmv'] = max(filters.get('min_gmv', 0), 80000)
        logger.info("检测到食品产品，设置GMV转化要求")
    
    # 母婴用品 - 需要信任度
    elif any(keyword in product_lower for keyword in ['母婴', '儿童', '玩具', '奶粉', '尿布']):
        filters['min_gmv'] = max(filters.get('min_gmv', 0), 100000)
        filters['min_followers'] = max(filters.get('min_followers', 0), 25000)
        logger.info("检测到母婴产品，设置信任度要求")
    
    # 默认筛选条件（更宽松）
    if not filters.get('min_gmv'):
        filters['min_gmv'] = 10000  # 最低GMV要求降低
    
    return filters

# ===== 路由定义 =====

@app.route('/')
def index():
    """首页"""
    return send_file('index.html')

@app.route('/api/health')
def health_check():
    """健康检查"""
    try:
        return jsonify({
            "success": True,
            "status": "healthy",
            "initialized": matcher.is_initialized,
            "message": "GOOJODOQ达人匹配系统运行正常"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/stats')
def system_stats():
    """获取系统统计信息"""
    try:
        stats_result = matcher.get_system_stats()
        if stats_result.get('success'):
            # 确保返回正确的字段名给前端
            stats_data = stats_result['stats']
            return jsonify({
                "success": True,
                "stats": {
                    "creator_count": stats_data.get('creator_count', 0),
                    "product_count": stats_data.get('product_count', 0),
                    "total_creators": stats_data.get('total_creators', 0),
                    "total_products": stats_data.get('total_products', 0),
                    "gmv_range": stats_data.get('gmv_range', {}),
                    "followers_range": stats_data.get('followers_range', {}),
                    "has_product_data": stats_data.get('has_product_data', False),
                    "cache_exists": stats_data.get('cache_exists', False),
                    "data_sources": stats_data.get('data_sources', {})
                }
            })
        else:
            return jsonify({"success": False, "error": stats_result.get('error', '获取统计信息失败')}), 500
    except Exception as e:
        logger.error(f"获取系统统计信息失败: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """新的推荐API - 支持商品价格+抽佣比例模式"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求数据为空"}), 400
        
        # 检查新模式参数
        product_price = data.get('product_price')
        commission_rate = data.get('commission_rate')
        
        if product_price is not None and commission_rate is not None:
            # 新模式：商品价格 + 抽佣比例
            logger.info(f"使用新模式: 商品价格¥{product_price}, 抽佣比例{commission_rate*100:.1f}%")
            
            # 参数验证
            if product_price <= 0:
                return jsonify({"success": False, "error": "商品价格必须大于0"}), 400
            if not (0 < commission_rate <= 1):
                return jsonify({"success": False, "error": "抽佣比例必须在0-1之间"}), 400
            
            # 获取其他参数
            product_description = data.get('product_description', '').strip()
            budget = data.get('budget')
            top_k = int(data.get('top_k', 10))
            min_followers = data.get('min_followers')
            max_followers = data.get('max_followers')
            min_estimated_gmv = data.get('min_estimated_gmv')
            max_estimated_gmv = data.get('max_estimated_gmv')
            
            # 调用新的筛选算法
            result = matcher.filter_creators_with_price_commission(
                product_price=product_price,
                commission_rate=commission_rate,
                min_followers=min_followers,
                max_followers=max_followers,
                min_estimated_gmv=min_estimated_gmv,
                max_estimated_gmv=max_estimated_gmv,
                top_k=top_k,
                product_query=product_description,
                budget=budget
            )
            
            if result['success']:
                logger.info(f"新模式推荐成功: 返回 {len(result['recommendations'])} 个达人")
            else:
                logger.error(f"新模式推荐失败: {result.get('error', '未知错误')}")
            
            return jsonify(result)
        
        else:
            # 旧模式：兼容性处理
            logger.info("使用兼容模式: 传统GMV筛选")
            
            product_description = data.get('product_description', '').strip()
            budget = data.get('budget')
            top_k = int(data.get('top_k', 10))
            min_gmv = data.get('min_gmv')
            min_followers = data.get('min_followers')
            
            if not product_description:
                return jsonify({"success": False, "error": "请输入产品描述"}), 400
            
            # 调用传统筛选算法
            result = matcher.filter_creators(
                min_gmv=min_gmv,
                min_followers=min_followers,
                top_k=top_k,
                product_query=product_description
            )
            
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"推荐处理失败: {e}")
        return jsonify({
            "success": False,
            "error": f"推荐失败: {str(e)}"
        }), 500

@app.route('/api/creators/<username>')
def get_creator_details(username):
    """获取达人详细信息"""
    try:
        creator = matcher.get_creator_by_username(username)
        if creator:
            return jsonify({"success": True, "creator": creator})
        else:
            return jsonify({"success": False, "error": "达人不存在"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/estimate-gmv', methods=['POST'])
def estimate_gmv():
    """GMV预估API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求数据为空"}), 400
        
        product_price = data.get('product_price')
        commission_rate = data.get('commission_rate')
        creator_username = data.get('creator_username')
        
        if not all([product_price, commission_rate, creator_username]):
            return jsonify({"success": False, "error": "缺少必要参数"}), 400
        
        # 获取达人数据
        creator = matcher.get_creator_by_username(creator_username)
        if not creator:
            return jsonify({"success": False, "error": "达人不存在"}), 404
        
        # 计算预估GMV
        estimated_gmv = matcher.estimate_gmv_from_price_commission(
            product_price, commission_rate, creator
        )
        
        # 分类3C产品
        product_category, categories = matcher.classify_3c_product(data.get('product_description', ''))
        
        # 计算适配度评分
        product_score = matcher.calculate_3c_product_score(creator, product_category, product_price)
        
        return jsonify({
            "success": True,
            "estimated_gmv": estimated_gmv,
            "product_category": product_category,
            "product_score": product_score,
            "creator_username": creator_username
        })
        
    except Exception as e:
        logger.error(f"GMV预估失败: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({"success": False, "error": "接口不存在"}), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    logger.error(f"服务器内部错误: {error}")
    return jsonify({"success": False, "error": "服务器内部错误"}), 500

# 启动应用
if __name__ == "__main__":
    if initialize_system():
        logger.info("🎯 启动GOOJODOQ达人匹配系统服务器...")
        # 生产环境配置
        app.run(
            host='0.0.0.0',
            port=8000,
            debug=False,
            threaded=True
        )
    else:
        logger.error("❌ 系统初始化失败，服务器启动中止")
        exit(1) 