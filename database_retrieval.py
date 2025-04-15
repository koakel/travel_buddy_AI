import pandas as pd
import numpy as np
import os
import re
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
import faiss
import datetime


class DataRetrievalModule:
    """数据检索模块：处理景点和商家信息检索"""

    def __init__(self, data_dir: str = "data", model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"):
        """
        初始化数据检索模块

        Args:
            data_dir: 数据文件存放目录
            model_name: 使用的向量模型名称
            use_local_model: 是否使用本地模型
        """
        self.data_dir = data_dir
        self.spots_df = None  # 景点数据
        self.merchants_df = None  # 商家数据
        
        # 确保数据目录存在
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"已创建数据目录: {data_dir}")

        # 加载数据
        self._load_data()

        # FAISS向量索引
        self.spots_index = None
        self.merchants_index = None
        self.embedding_dim = 384  # 默认嵌入维度，将在生成嵌入时更新

        # 初始化向量模型
        try:
            # 检查当前目录下是否存在model文件夹
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
            if os.path.exists(local_model_path):
                print(f"使用本地模型: {local_model_path}")
                self.model = SentenceTransformer(local_model_path)
            else:
                print(f"本地模型目录不存在: {local_model_path}，将从互联网下载模型")
                self.model = SentenceTransformer(model_name)
                # 保存模型到本地（可选）
                os.makedirs(local_model_path, exist_ok=True)
                self.model.save(local_model_path)
                print(f"已将模型保存到本地: {local_model_path}")
                
            self.spots_embeddings = None
            self.merchants_embeddings = None
            self._generate_embeddings()
        except Exception as e:
            print(f"向量模型加载失败: {e}")
            self.model = None

    def _load_data(self):
        """加载数据文件（支持Excel和CSV格式）"""
        # 尝试加载Excel文件
        spots_xlsx_path = os.path.join(self.data_dir, "大理景点整理.xlsx")
        merchants_csv_path = os.path.join(self.data_dir, "意向合作商家.csv")

        # 初始化DataFrame
        self.spots_df = pd.DataFrame()
        self.merchants_df = pd.DataFrame()

        # 处理景点数据
        if os.path.exists(spots_xlsx_path):
            try:
                self.spots_df = pd.read_excel(spots_xlsx_path)
                print(f"成功加载{len(self.spots_df)}个景点数据 (Excel格式)")
            except Exception as e:
                print(f"Excel景点数据加载失败: {e}")
        else:
            print(f"警告: 无法加载景点数据文件")

        # 处理商家数据
        if os.path.exists(merchants_csv_path):
            try:
                self.merchants_df = pd.read_csv(merchants_csv_path, encoding='utf-8')
                print(f"成功加载{len(self.merchants_df)}个商家数据 (CSV格式, 编码: utf-8)")
            except Exception as e:
                print(f"商家数据加载失败: {e}")
        else:
            print(f"警告: 无法加载商家数据文件")

        # 如果两个数据文件都不存在或无法加载，创建示例数据
        if self.spots_df.empty and self.merchants_df.empty:
            print("无法加载任何数据")

    def _generate_embeddings(self):
        """为景点和商家数据生成向量嵌入并创建FAISS索引"""
        if self.model is None:
            return

        # 为景点生成向量
        if not self.spots_df.empty:
            # 将景点名称和简介拼接作为特征文本
            spot_texts = []
            for _, row in self.spots_df.iterrows():
                text = f"{row['景点名称']} {row['简介']} {row['特色看点']}"
                spot_texts.append(text)

            # 生成向量
            self.spots_embeddings = self.model.encode(spot_texts)
            
            # 确保向量是float32类型并进行L2归一化
            self.spots_embeddings = self.spots_embeddings.astype(np.float32)
            faiss.normalize_L2(self.spots_embeddings)  # 归一化向量
            
            print(f"已生成{len(spot_texts)}个景点向量")
            
            # 更新嵌入维度
            self.embedding_dim = self.spots_embeddings.shape[1]
            
            # 创建FAISS索引 - 使用IndexFlatIP进行内积相似度（类似于余弦相似度）
            self.spots_index = self._create_faiss_index(self.spots_embeddings)
            print(f"已创建景点FAISS索引")

        # 为商家生成向量
        if not self.merchants_df.empty:
            # 将商家名称和位置拼接作为特征文本
            merchant_texts = []
            for _, row in self.merchants_df.iterrows():
                text = f"{row['商家']} {row['位置']} {row['房间型号']}"
                merchant_texts.append(text)

            # 生成向量
            self.merchants_embeddings = self.model.encode(merchant_texts)
            
            # 确保向量是float32类型并进行L2归一化
            self.merchants_embeddings = self.merchants_embeddings.astype(np.float32)
            faiss.normalize_L2(self.merchants_embeddings)  # 归一化向量
            
            print(f"已生成{len(merchant_texts)}个商家向量")
            
            # 更新嵌入维度
            self.embedding_dim = self.merchants_embeddings.shape[1]
            
            # 创建FAISS索引
            self.merchants_index = self._create_faiss_index(self.merchants_embeddings)
            print(f"已创建商家FAISS索引")
            
        # 尝试保存索引到文件
        self._save_indexes()

    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        创建FAISS索引
        
        Args:
            embeddings: 嵌入向量数组
            
        Returns:
            FAISS索引对象
        """
        # 获取维度
        dim = embeddings.shape[1]
        
        # 创建索引 - 使用IndexFlatIP适合小数据集，对于大数据集可以使用量化器
        index = faiss.IndexFlatIP(dim)  # 内积相似度，适用于L2归一化的向量
        
        # 添加向量到索引
        index.add(embeddings)
        
        return index
    
    def _save_indexes(self):
        """保存FAISS索引到文件"""
        try:
            os.makedirs(os.path.join(self.data_dir, 'indexes'), exist_ok=True)
            
            # 保存嵌入和索引数据
            if self.spots_embeddings is not None and self.spots_index is not None:
                faiss.write_index(self.spots_index, os.path.join(self.data_dir, 'indexes', 'spots_index.faiss'))
                with open(os.path.join(self.data_dir, 'indexes', 'spots_embeddings.pkl'), 'wb') as f:
                    pickle.dump(self.spots_embeddings, f)
                
            if self.merchants_embeddings is not None and self.merchants_index is not None:
                faiss.write_index(self.merchants_index, os.path.join(self.data_dir, 'indexes', 'merchants_index.faiss'))
                with open(os.path.join(self.data_dir, 'indexes', 'merchants_embeddings.pkl'), 'wb') as f:
                    pickle.dump(self.merchants_embeddings, f)
                    
            print("已保存FAISS索引到文件")
            
        except Exception as e:
            print(f"保存索引失败: {e}")
    
    def _load_indexes(self):
        """从文件加载FAISS索引"""
        try:
            # 检查索引文件
            spots_index_path = os.path.join(self.data_dir, 'indexes', 'spots_index.faiss')
            merchants_index_path = os.path.join(self.data_dir, 'indexes', 'merchants_index.faiss')
            
            # 加载景点索引
            if os.path.exists(spots_index_path):
                self.spots_index = faiss.read_index(spots_index_path)
                embeddings_path = os.path.join(self.data_dir, 'indexes', 'spots_embeddings.pkl')
                if os.path.exists(embeddings_path):
                    with open(embeddings_path, 'rb') as f:
                        self.spots_embeddings = pickle.load(f)
                print("已加载景点FAISS索引")
                
            # 加载商家索引
            if os.path.exists(merchants_index_path):
                self.merchants_index = faiss.read_index(merchants_index_path)
                embeddings_path = os.path.join(self.data_dir, 'indexes', 'merchants_embeddings.pkl')
                if os.path.exists(embeddings_path):
                    with open(embeddings_path, 'rb') as f:
                        self.merchants_embeddings = pickle.load(f)
                print("已加载商家FAISS索引")
                
            return True
            
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False

    def keyword_search(self, query: str, search_type: str = "all", limit: int = 3) -> Dict[str, List[Dict]]:
        """
        基于关键词的搜索

        Args:
            query: 搜索关键词
            search_type: 搜索类型('spots', 'merchants', 'all')
            limit: 返回结果数量限制

        Returns:
            搜索结果字典
        """
        results = {"spots": [], "merchants": []}

        # 规范化查询关键词 - 更细粒度地分词
        keywords = re.split(r'\s+|,|，|、|。|！|？|；|:|：', query.lower())
        # 过滤太短的关键词，但保留数字
        keywords = [k for k in keywords if len(k) > 1 or k.isdigit()]  
        
        if not keywords:
            # 如果没有有效关键词，使用整个查询作为一个关键词
            keywords = [query.lower()]

        # 搜索景点
        if search_type in ["spots", "all"] and not self.spots_df.empty:
            for idx, row in self.spots_df.iterrows():
                score = 0
                matched_keywords = set()  # 记录已匹配的关键词，避免重复计分
                
                # 计算匹配分数
                for col in ['景点名称', '地址', '简介', '特色看点']:
                    if pd.notna(row[col]):
                        text = str(row[col]).lower()
                        for keyword in keywords:
                            # 如果关键词已匹配过，跳过
                            if keyword in matched_keywords:
                                continue
                                
                            # 完全匹配给更高分
                            if keyword == text or f" {keyword} " in f" {text} ":
                                score += 10 if col == '景点名称' else 5
                                matched_keywords.add(keyword)
                            # 部分匹配
                            elif keyword in text:
                                score += 3 if col == '景点名称' else 1
                                matched_keywords.add(keyword)

                if score > 0:
                    results["spots"].append({
                        "data": row.to_dict(),
                        "score": score,
                        "index": idx
                    })

            # 按匹配分数排序并限制结果数量
            results["spots"] = sorted(results["spots"], key=lambda x: x["score"], reverse=True)[:limit]

        # 搜索商家
        if search_type in ["merchants", "all"] and not self.merchants_df.empty:
            for idx, row in self.merchants_df.iterrows():
                score = 0
                matched_keywords = set()  # 记录已匹配的关键词，避免重复计分
                
                # 计算匹配分数
                for col in ['商家', '位置', '房间型号', '价格']:
                    if pd.notna(row[col]):
                        text = str(row[col]).lower()
                        for keyword in keywords:
                            # 如果关键词已匹配过，跳过
                            if keyword in matched_keywords:
                                continue
                                
                            # 完全匹配给更高分
                            if keyword == text or f" {keyword} " in f" {text} ":
                                score += 10 if col == '商家' else 5
                                matched_keywords.add(keyword)
                            # 部分匹配
                            elif keyword in text:
                                score += 3 if col == '商家' else 1
                                matched_keywords.add(keyword)

                if score > 0:
                    results["merchants"].append({
                        "data": row.to_dict(),
                        "score": score,
                        "index": idx
                    })

            # 按匹配分数排序并限制结果数量
            results["merchants"] = sorted(results["merchants"], key=lambda x: x["score"], reverse=True)[:limit]

        return results

    def vector_search(self, query: str, search_type: str = "all", limit: int = 3) -> Dict[str, List[Dict]]:
        """
        基于向量的语义搜索，使用FAISS

        Args:
            query: 搜索查询
            search_type: 搜索类型('spots', 'merchants', 'all')
            limit: 返回结果数量限制

        Returns:
            搜索结果字典
        """
        if self.model is None:
            return {"spots": [], "merchants": []}

        results = {"spots": [], "merchants": []}

        # 生成查询向量
        query_embedding = self.model.encode([query])
        # 确保向量是float32类型并进行L2归一化
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # 搜索景点
        if search_type in ["spots", "all"] and self.spots_index is not None and not self.spots_df.empty:
            # 使用FAISS执行相似度搜索
            # distances是相似度分数，indices是向量的索引位置
            distances, indices = self.spots_index.search(query_embedding, limit)
            
            # 转换结果
            for i, idx in enumerate(indices[0]):
                if idx < len(self.spots_df) and idx >= 0:  # 确保索引有效
                    results["spots"].append({
                        "data": self.spots_df.iloc[idx].to_dict(),
                        "score": float(distances[0][i]),
                        "index": int(idx)
                    })

        # 搜索商家
        if search_type in ["merchants", "all"] and self.merchants_index is not None and not self.merchants_df.empty:
            # 使用FAISS执行相似度搜索
            distances, indices = self.merchants_index.search(query_embedding, limit)
            
            # 转换结果
            for i, idx in enumerate(indices[0]):
                if idx < len(self.merchants_df) and idx >= 0:  # 确保索引有效
                    results["merchants"].append({
                        "data": self.merchants_df.iloc[idx].to_dict(),
                        "score": float(distances[0][i]),
                        "index": int(idx)
                    })

        return results

    def search(self, query: str, search_type: str = "all", limit: int = 3) -> Dict[str, Any]:
        """
        综合搜索接口：结合关键词搜索和向量搜索

        Args:
            query: 搜索查询
            search_type: 搜索类型('spots', 'merchants', 'all')
            limit: 返回结果数量限制

        Returns:
            搜索结果
        """
        # 执行关键词搜索
        keyword_results = self.keyword_search(query, search_type, limit)

        # 执行向量搜索
        vector_results = self.vector_search(query, search_type, limit)

        # 合并结果（去重）
        combined_results = {"spots": [], "merchants": []}

        # 合并景点结果
        seen_spots = set()
        for result_list in [keyword_results["spots"], vector_results["spots"]]:
            for item in result_list:
                spot_name = item["data"]["景点名称"]
                if spot_name not in seen_spots:
                    seen_spots.add(spot_name)
                    combined_results["spots"].append(item)

        # 合并商家结果
        seen_merchants = set()
        for result_list in [keyword_results["merchants"], vector_results["merchants"]]:
            for item in result_list:
                merchant_name = item["data"]["商家"]
                if merchant_name not in seen_merchants:
                    seen_merchants.add(merchant_name)
                    combined_results["merchants"].append(item)

        # 限制结果数量
        combined_results["spots"] = combined_results["spots"][:limit]
        combined_results["merchants"] = combined_results["merchants"][:limit]

        return combined_results

    def format_llm_card(self, query: str, content: Any, card_type: str = "llm", llm_content: str = None) -> str:
        """
        统一的卡片格式化函数 - 处理所有类型的数据
        
        Args:
            query: 用户查询
            content: 内容数据 (可以是LLM文本、景点数据字典或商家数据字典)
            card_type: 卡片类型 ("llm", "spot", "merchant")
            llm_content: LLM生成的回复内容（用于在景点或商家卡片中展示）
            
        Returns:
            格式化的卡片信息
        """
        # 获取当前日期作为建议参考时间点
        current_date = datetime.datetime.now().strftime("%Y年%m月")
        
        # 根据卡片类型处理不同的内容
        if card_type == "spot":
            # 处理景点数据
            spot_data = content
            # 提取主要信息并确保值为字符串
            name = str(spot_data.get("景点名称", "未知景点"))
            address = str(spot_data.get("地址", "地址未提供"))
            fee = str(spot_data.get("费用", "费用信息未提供"))
            open_time = str(spot_data.get("开放时间", "开放时间未提供"))
            duration = str(spot_data.get("用时参考", "参考用时未提供"))
            intro = str(spot_data.get("简介", "暂无简介"))
            highlights = str(spot_data.get("特色看点", "暂无特色看点"))
            contact = str(spot_data.get("联系方式", "联系方式未提供"))
            
            # 处理nan值
            if name == "nan": name = "未知景点"
            if address == "nan": address = "地址未提供"
            if fee == "nan": fee = "费用信息未提供"
            if open_time == "nan": open_time = "开放时间未提供"
            if duration == "nan": duration = "参考用时未提供"
            if intro == "nan": intro = "暂无简介"
            if highlights == "nan": highlights = "暂无特色看点"
            if contact == "nan": contact = "联系方式未提供"

            # 简化内容 - 截断过长文本
            if len(intro) > 100:
                intro = intro[:97] + "..."
            if len(highlights) > 80:
                highlights = highlights[:77] + "..."

            # 构建卡片信息 - 改进可视化布局
            card_content = f"""【景点信息】{name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 地址：{address}
🎫 门票：{fee}
⏰ 开放时间：{open_time}
⌛ 参考游览时间：{duration}
📞 联系电话：{contact}

📝 景点简介：
{intro}

✨ 特色看点：
{highlights}"""

            # 添加LLM的观点和建议（如果有）
            if llm_content:
                card_content += f"""

🤖 AI旅伴点评：
{llm_content}"""

        elif card_type == "merchant":
            # 处理商家数据
            merchant_data = content
            # 提取主要信息并确保值为字符串
            name = str(merchant_data.get("商家", "未知商家"))
            location = str(merchant_data.get("位置", "位置未提供"))
            room_type = str(merchant_data.get("房间型号", "房型未提供"))
            price = str(merchant_data.get("价格", "价格未提供"))
            ground_rooms = str(merchant_data.get("一楼房间数量", "未知"))
            contact = str(merchant_data.get("联系方式", "联系方式未提供"))
            
            # 处理nan值
            if name == "nan": name = "未知商家"
            if location == "nan": location = "位置未提供"
            if room_type == "nan": room_type = "房型未提供"
            if price == "nan": price = "价格未提供"
            if ground_rooms == "nan": ground_rooms = "未知"
            if contact == "nan": contact = "联系方式未提供"

            # 构建卡片信息 - 改进可视化布局
            card_content = f"""【住宿信息】{name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 地址：{location}
🏠 房型：{room_type}
💰 价格：{price}
🧓 一楼房间数量：{ground_rooms} (适合行动不便人士)
📞 联系电话：{contact}"""

            # 添加LLM的观点和建议（如果有）
            if llm_content:
                card_content += f"""

🤖 AI旅伴点评：
{llm_content}"""

        else:
            # 处理LLM生成内容
            llm_content = content
            
            # 根据查询内容智能选择卡片标题
            title = "【AI旅伴点评】"
            
            # 基于查询类型选择更具体的标题
            if any(keyword in query for keyword in ["景点", "游玩", "参观", "景区", "玩", "去", "游览"]):
                title = "【景点推荐】"
            elif any(keyword in query for keyword in ["住宿", "酒店", "民宿", "客栈", "住", "房间"]):
                title = "【住宿建议】"
            elif any(keyword in query for keyword in ["美食", "吃", "餐厅", "小吃", "菜"]):
                title = "【美食指南】"
            elif any(keyword in query for keyword in ["交通", "怎么去", "路线", "到达", "车"]):
                title = "【交通指南】"
            elif any(keyword in query for keyword in ["建议", "攻略", "行程", "路线", "规划"]):
                title = "【行程建议】"
                
            # 构建卡片信息 - 使用更精美的格式
            card_content = f"""{title}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{llm_content}"""
        
        # 检查是否需要添加适老提示（适用于所有卡片类型）
        accessibility_note = ""
        if any(keyword in query.lower() for keyword in ["老人", "年龄大", "父母", "爷爷", "奶奶", "老年", "长辈"]):
            # 添加适老化提示
            accessibility_note = """
👴👵 适老贴士：
• 请选择缓坡通道和电梯设施
• 建议错峰出行，避开人流高峰
• 随身携带常用药品和老人证件
• 安排充足休息时间，避免过度疲劳
"""
        
        # 检查是否涉及行动不便
        if any(keyword in query.lower() for keyword in ["轮椅", "行动不便", "腿脚", "不方便", "残疾", "无障碍"]):
            # 添加无障碍提示
            accessibility_note = """
♿ 无障碍贴士：
• 出发前请电话确认景点/酒店无障碍设施情况
• 选择配有轮椅通道和无障碍卫生间的场所
• 建议提前预约无障碍服务或设施
• 考虑选择一楼房间或有电梯的住宿
"""
        
        # 添加页脚信息
        footer = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🕒 {current_date}更新
📝 以上信息由大理旅伴AI提供，仅供参考
📞 需了解更多详情，请联系当地旅游服务
"""
        
        # 组合最终卡片
        return card_content + accessibility_note + footer
    
    # 提供向后兼容的函数
    def format_spot_card(self, spot_data: Dict[str, Any]) -> str:
        """向后兼容的景点卡片格式化函数"""
        return self.format_llm_card("景点", spot_data, card_type="spot")
        
    def format_merchant_card(self, merchant_data: Dict[str, Any]) -> str:
        """向后兼容的商家卡片格式化函数"""
        return self.format_llm_card("住宿", merchant_data, card_type="merchant")

    def find_nearby_merchants(self, spot_address: str) -> List[Dict[str, Any]]:
        """
        根据景点地址查找附近的商家

        Args:
            spot_address: 景点地址

        Returns:
            附近商家列表
        """
        if self.merchants_df.empty:
            return []
            
        # 处理spot_address为nan的情况
        if pd.isna(spot_address) or str(spot_address).lower() == "nan":
            spot_address = ""

        nearby_merchants = []
        # 提取地址中的区域信息
        address_parts = re.split(r'[市区县乡镇村]', str(spot_address))

        for _, merchant in self.merchants_df.iterrows():
            # 获取商家位置并确保是字符串
            location = str(merchant.get("位置", ""))
            if location.lower() == "nan":
                location = ""
                
            match = False

            # 检查地址是否包含相同的区域关键词
            for part in address_parts:
                if part and len(part) > 1 and part in location:
                    match = True
                    break

            if match:
                nearby_merchants.append(merchant.to_dict())

        return nearby_merchants[:3]  # 限制返回结果数量

    def format_as_card(self, query: str, llm_content: str) -> str:
        """
        将LLM生成的回复转换为卡片格式 (兼容旧接口)
        
        Args:
            query: 用户查询
            llm_content: LLM生成的内容
            
        Returns:
            格式化的卡片或纯文本
        """
        # 尝试解析JSON（适配新格式）
        try:
            if isinstance(llm_content, str) and (llm_content.startswith('{') or llm_content.strip().startswith('{')):
                llm_json = json.loads(llm_content)
                needs_card = llm_json.get("needs_card", "no").lower() == "yes"
                content = llm_json.get("content", "")
                
                if not needs_card:
                    # 直接返回内容，不使用卡片格式
                    return content
                else:
                    # 使用卡片格式
                    return self.format_llm_card(query, content, card_type="llm")
        except (json.JSONDecodeError, AttributeError):
            pass  # 不是JSON格式，继续使用旧的处理方式
        
        # 默认使用通用卡片格式
        return self.format_llm_card(query, llm_content, card_type="llm")

    def process_query(self, query: str, user_profile: Optional[Dict] = None, llm_content: Optional[str] = None) -> Dict[str, Any]:
        """
        处理用户查询，返回结构化信息

        Args:
            query: 用户查询
            user_profile: 用户画像信息
            llm_content: LLM生成的内容（如果有）

        Returns:
            处理结果
        """
        # 分析查询意图
        if any(keyword in query for keyword in ["住宿", "酒店", "民宿", "客栈", "住", "房间"]):
            search_type = "merchants"
        elif any(keyword in query for keyword in ["景点", "游玩", "参观", "景区", "玩", "去"]):
            search_type = "spots"
        else:
            search_type = "all"

        # 执行搜索
        results = self.search(query, search_type=search_type, limit=2)
        
        # 提取景点和商家数据（如果有）
        spot_data = None
        merchant_data = None
        
        if results["spots"]:
            spot_data = results["spots"][0]["data"]
            
        if results["merchants"]:
            merchant_data = results["merchants"][0]["data"]

        # 默认回复类型为卡片格式的LLM内容
        default_content = "非常抱歉，我无法找到与您查询相关的具体信息。请尝试询问大理的热门景点、古城周边住宿等更具体的问题，我会尽力为您提供帮助。"
        
        # 使用传入的LLM内容（如果有）
        llm_text = llm_content if llm_content else default_content
        
        # 处理LLM的JSON响应
        try:
            # 只有当LLM内容是字符串且可能是JSON时才尝试解析
            if llm_text and isinstance(llm_text, str) and ('{' in llm_text and '}' in llm_text):
                # 清理可能包含的代码块标记
                json_str = llm_text.strip()
                if "```json" in json_str:
                    # 提取JSON部分
                    start = json_str.find("```json") + 7
                    end = json_str.rfind("```")
                    if end > start:
                        json_str = json_str[start:end].strip()
                
                # 解析JSON
                llm_json = json.loads(json_str)
                needs_card = llm_json.get("needs_card", "no").lower() == "yes"
                content = llm_json.get("content", "")
                
                # 首先检查是否需要卡片
                if needs_card:
                    # 如果需要卡片，并且有相关数据，则优先使用数据信息并整合LLM内容
                    if search_type == "spots" and spot_data:
                        response = {
                            "type": "spot", 
                            "content": self.format_llm_card(query, spot_data, card_type="spot", llm_content=content)
                        }
                    elif search_type == "merchants" and merchant_data:
                        response = {
                            "type": "merchant", 
                            "content": self.format_llm_card(query, merchant_data, card_type="merchant", llm_content=content)
                        }
                    else:
                        # 没有特定数据，使用LLM卡片
                        response = {
                            "type": "llm_card", 
                            "content": self.format_llm_card(query, content, card_type="llm")
                        }
                else:
                    # 如果不需要卡片，直接返回LLM内容
                    response = {
                        "type": "text", 
                        "content": content
                    }
            else:
                # 当输入不是JSON格式时，尝试寻找旧格式的标记
                if isinstance(llm_text, str) and llm_text.startswith("[CARD]"):
                    # 旧格式卡片标记
                    content = llm_text.replace("[CARD]", "").strip()
                    response = {
                        "type": "llm_card", 
                        "content": self.format_llm_card(query, content, card_type="llm")
                    }
                elif isinstance(llm_text, str) and llm_text.startswith("[TEXT]"):
                    # 旧格式文本标记
                    content = llm_text.replace("[TEXT]", "").strip()
                    response = {
                        "type": "text", 
                        "content": content
                    }
                else:
                    # 默认使用LLM回复内容
                    response = {
                        "type": "text", 
                        "content": llm_text
                    }
        except (json.JSONDecodeError, AttributeError) as e:
            # 仅在调试模式下输出详细错误
            print(f"JSON解析提示 (不影响使用): {str(e)}")
            
            # 尝试使用内容的基本格式
            if isinstance(llm_text, str) and ("[CARD]" in llm_text or "needs_card" in llm_text.lower()):
                # 看起来像是有卡片标记，使用卡片格式
                if "[CARD]" in llm_text:
                    content = llm_text.replace("[CARD]", "").strip()
                else:
                    content = llm_text
                    
                response = {
                    "type": "llm_card", 
                    "content": self.format_llm_card(query, content, card_type="llm")
                }
            else:
                # 默认使用文本格式
                response = {
                    "type": "text", 
                    "content": llm_text if isinstance(llm_text, str) else default_content
                }

        return response

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        检索与查询相关的信息，并格式化为文本以传递给LLM
        
        Args:
            query: 用户查询
            top_k: 返回的结果数量
            
        Returns:
            检索到的信息文本
        """
        # 确定搜索类型
        if any(keyword in query for keyword in ["住宿", "酒店", "民宿", "客栈", "住", "房间"]):
            search_type = "merchants"
        elif any(keyword in query for keyword in ["景点", "游玩", "参观", "景区", "玩", "去", "游览"]):
            search_type = "spots"
        else:
            search_type = "all"
            
        # 执行搜索
        results = self.search(query, search_type=search_type, limit=top_k)
        
        # 格式化结果为文本
        retrieved_info = []
        
        # 处理景点信息
        if results["spots"]:
            retrieved_info.append("相关景点信息：")
            for i, item in enumerate(results["spots"], 1):
                spot = item["data"]
                name = str(spot.get("景点名称", "未知景点"))
                if name == "nan": name = "未知景点"
                
                address = str(spot.get("地址", "地址未提供"))
                if address == "nan": address = "地址未提供"
                
                fee = str(spot.get("费用", "费用信息未提供"))
                if fee == "nan": fee = "费用信息未提供"
                
                intro = str(spot.get("简介", "暂无简介"))
                if intro == "nan": intro = "暂无简介"
                if len(intro) > 100:
                    intro = intro[:97] + "..."
                
                retrieved_info.append(f"{i}. {name}：位于{address}，门票{fee}。{intro}")
                
        # 处理商家信息
        if results["merchants"]:
            retrieved_info.append("\n相关住宿信息：")
            for i, item in enumerate(results["merchants"], 1):
                merchant = item["data"]
                name = str(merchant.get("商家", "未知商家"))
                if name == "nan": name = "未知商家"
                
                location = str(merchant.get("位置", "位置未提供"))
                if location == "nan": location = "位置未提供"
                
                room_type = str(merchant.get("房间型号", "房型未提供"))
                if room_type == "nan": room_type = "房型未提供"
                
                price = str(merchant.get("价格", "价格未提供"))
                if price == "nan": price = "价格未提供"
                
                retrieved_info.append(f"{i}. {name}：位于{location}，提供{room_type}，价格{price}")
                
        # 如果没有找到任何结果
        if not retrieved_info:
            return "我没有找到与您查询相关的具体景点或住宿信息，但我会尽力回答您的问题。"
            
        return "\n".join(retrieved_info)


# 示例用法
if __name__ == "__main__":
    retrieval = DataRetrievalModule()

    # 测试查询景点
    query1 = "有什么适合老人的景点推荐"
    result1 = retrieval.process_query(query1, {"age": "65", "mobility_status": "行走缓慢"})
    print(f"查询: {query1}")
    print(result1["content"])
    print("-" * 50)

    # 测试查询住宿
    query2 = "大理古城附近有什么便宜的住宿"
    result2 = retrieval.process_query(query2)
    print(f"查询: {query2}")
    print(result2["content"])
