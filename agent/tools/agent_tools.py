import os
from utils.logger_handler import logger
from langchain_core.tools import tool
import requests
import streamlit as st
from requests.exceptions import RequestException
from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from dotenv import load_dotenv
load_dotenv()
rag = RagSummarizeService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}

@tool
def rag_summarize(query: str) -> str:
    """专业的知识库检索工具。
    当用户询问关于扫地机器人的：
    1. 错误代码（如 E88, E21 等）
    2. 故障现象（如不回充、噪音大）
    3. 产品说明或使用技巧时，
    必须调用此工具获取官方解决方案。"""

    # 1. 核心修改：从 rag 实例的属性中获取状态，而不是 st.session_state
    # 这里的 rag 是你在文件顶部初始化的 RagSummarizeService 实例
    # 它的 active_bm25 属性已经在 ReactAgent.execute_stream 中被同步过了
    use_bm25 = getattr(rag, "active_bm25", True)

    # 2. 增强型日志：让你在控制台一眼看到当前到底是哪种模式在跑
    mode_desc = "【语义+关键词】混合模式" if use_bm25 else "【仅语义】向量模式"
    logger.info(f"🤖 [AgentTool] Agent 发起检索请求 -> 关键词: '{query}' | 运行模式: {mode_desc}")

    # 3. 调用业务逻辑
    # 这里的 rag_summarize 是你 Service 类里的方法，里面有我们加的 ✅/❌ 打印
    res = rag.rag_summarize(query, use_bm25=use_bm25)

    logger.info(f"🤖 [AgentTool] 检索结果已就绪，返回给 Agent 进行总结。")
    return res



@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的实时天气情况。
    输入参数必须是城市中文名称，例如 '深圳'、'北京'。
    """
    # 1. 从环境变量获取 API Key
    api_key = os.getenv("SENIVERSE_API_KEY")
    if not api_key:
        logger.error("[get_weather] 缺少 SENIVERSE_API_KEY 环境变量")
        return "系统配置缺失，暂时无法获取天气信息。"

    # 2. 拼接第三方 API 的 URL
    # 这是心知天气 V3 版本的请求格式
    url = f"https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={city}&language=zh-Hans&unit=c"

    try:
        # 3. 发送 HTTP GET 请求，务必设置 timeout 超时时间！
        response = requests.get(url, timeout=5.0)

        # 检查 HTTP 状态码，如果不是 200 会抛出异常
        response.raise_for_status()

        # 4. 解析第三方返回的 JSON 数据
        data = response.json()

        # 提取我们需要的天气字段
        weather_info = data["results"][0]["now"]
        text = weather_info["text"]  # 天气现象，如：晴、多云
        temperature = weather_info["temperature"]  # 温度

        # 5. 格式化为人类和 LLM 都能看懂的自然语言字符串返回
        # 解析出天气数据后
        result_str = f"{city}当前天气：{text}，气温：{temperature}摄氏度。"

        # 加这一行！直接让中间件/控制台打印出来
        logger.info(f"🌤️ [get_weather] 成功获取到外部数据: {result_str}")

        return result_str

    except RequestException as e:
        # 捕获网络异常（如超时、DNS 解析失败等）
        logger.error(f"[get_weather] 调用天气API失败，网络异常: {e}")
        return f"当前网络拥堵，暂时无法获取 {city} 的天气信息。"
    except KeyError as e:
        # 捕获 JSON 解析异常（比如第三方 API 偷偷改了返回格式）
        logger.error(f"[get_weather] 解析天气API响应失败，缺少字段: {e}")
        return f"获取 {city} 的天气数据格式异常。"
    except Exception as e:
        # 兜底捕获其他未知异常
        logger.error(f"[get_weather] 发生未知错误: {e}")
        return f"查询 {city} 天气时发生系统错误。"


@tool
def get_user_location() -> str:
    """获取用户所在城市的名称，以纯字符串形式返回"""
    try:
        # 这里以免费的 ip-api 为例（不需要鉴权，直接获取请求机器的公网IP归属地）
        # 注意：如果你的 Agent 部署在服务器上，这样查出来的可能是服务器的地址！
        response = requests.get("http://ip-api.com/json/?lang=zh-CN", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return data.get("city", "未知城市")
    except Exception as e:
        print(f"获取位置失败: {e}")

    return "未知城市"  # 兜底策略


@tool
def get_user_id() -> str:
    """获取用户的ID，以纯字符串形式返回"""
    return random.choice(user_ids)


@tool
def get_current_month() -> str:
    """获取当前月份，以纯字符串形式返回"""
    return random.choice(month_arr)


def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool
def fetch_external_data(user_id: str, month: str) -> str:
    """从外部系统中获取指定用户在指定月份的使用记录，以JSON字符串形式返回，如果未检索到返回空字符串"""
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""
# if __name__ =='__main__':
#     # 注意两点：
#     # 1. 使用 .invoke() 方法
#     # 2. 参数必须以字典键值对的形式传入
#     result = fetch_external_data.invoke({"user_id": "1001", "month": "2025-01"})
#     # 正常的打印
#     print(result)

@tool
def fill_context_for_report():
    """无入参，调用后触发中间件自动为报告生成的场景动态注入上下文信息。返回执行成功的状态字符串。"""
    return "fill_context_for_report已调用"
