"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
import warnings
import logging

# 1. 屏蔽 Python 标准警告
warnings.filterwarnings("ignore", message="Accessing __path__ from")
warnings.filterwarnings("ignore", category=UserWarning)

# 2. 顺便把 transformers 的日志级别调高，只看 Error
import transformers
transformers.utils.logging.set_verbosity_error()
import os
# 设置 HuggingFace 的国内加速镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder  # [新增] 导入精排核心库
import jieba
import torch
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


# 定义分词函数
def chinese_tokenizer(text):
    # 使用 jieba.lcut 得到一个词语列表
    return list(jieba.cut(text))


class RagSummarizeService(object):
    def __init__(self):
        # 先检查 CUDA 是否真的可用（防御性编程）
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 精排模型运行设备锁定为: {device.upper()}")
        self.vector_store = VectorStoreService()

        # 最终喂给大模型的文档数量 (决赛名额)
        self.k = chroma_conf.get("k", 3)

        # 精排之前的海选召回量 (扩大海选池，保证不错过 E88)
        self.retrieval_k = 10

        # 1. 初始化向量检索器
        self.vector_retriever = self.vector_store.get_retriever()
        # (可选) 如果你的 retriever 支持动态修改 k 值，强制设为 retrieval_k
        if hasattr(self.vector_retriever, 'search_kwargs'):
            self.vector_retriever.search_kwargs['k'] = self.retrieval_k

        # 2. 加载 BGE 精排模型 (首次运行会下载，之后走本地缓存)
        print("⏳ 正在加载 BGE-Reranker 精排模型，请稍候...")
        self.reranker = CrossEncoder(
            'BAAI/bge-reranker-base',
            max_length=512,
            device=device,  # 显式指定使用 CUDA
            model_kwargs={"local_files_only": True} # 同时保留离线模式，防止之前的网络报错
        )
        print("✅ BGE-Reranker 加载完成！")

        # 3. 初始化 BM25 检索器
        all_docs = self.vector_store.get_all_documents()
        print(f"DEBUG: 当前 BM25 索引库中的文档总数: {len(all_docs)}")
        if all_docs:
            self.bm25_retriever = BM25Retriever.from_documents(
                documents=all_docs,
                preprocess_func=chinese_tokenizer
            )
            # BM25 也召回 retrieval_k 条
            self.bm25_retriever.k = self.retrieval_k
        else:
            self.bm25_retriever = None

        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()



    def _init_chain(self):
        return self.prompt_template | print_prompt | self.model | StrOutputParser()

    def retriever_docs(self, query: str, use_bm25: bool = True) -> list[Document]:
        print(f"\n🚀 开始纯 AI 检索链路 -> Query: '{query}'")

        # ==========================================
        # 1. 海选阶段：多路召回与去重
        # ==========================================
        vector_docs = self.vector_retriever.invoke(query)
        print(f"\n📡 [第一阶段：向量检索] 召回数量: {len(vector_docs)}")
        for i, doc in enumerate(vector_docs[:3]):  # 只打印前3条预览
            print(f"   └─ Rank {i + 1} | 内容预览: {doc.page_content[:50].replace(chr(10), '')}...")
        # 利用字典推导式实现优雅去重
        all_candidates = list({doc.page_content: doc for doc in vector_docs}.values())

        if use_bm25 and self.bm25_retriever:
            bm25_docs = self.bm25_retriever.invoke(query)
            print(f"\n🔍 [第二阶段：BM25检索] 召回数量: {len(bm25_docs)}")
            for i, doc in enumerate(bm25_docs[:3]):
                print(f"   └─ Rank {i + 1} | 内容预览: {doc.page_content[:50].replace(chr(10), '')}...")
            seen = {doc.page_content for doc in all_candidates}
            # 仅将未见过的 BM25 文档加入候选池
            all_candidates.extend([d for d in bm25_docs if d.page_content not in seen])

        if not all_candidates:
            print(" ⚠️ 警告：海选阶段未找到任何文档！")
            return []

        # ==========================================
        # 2. 决赛阶段：Cross-Encoder 纯语义精排
        # ==========================================
        sentence_pairs = [[query, doc.page_content] for doc in all_candidates]
        scores = self.reranker.predict(sentence_pairs)

        # 仅记录 AI 给出的真实原始分数
        for i, score in enumerate(scores):
            all_candidates[i].metadata["rerank_score"] = float(score)

        # ==========================================
        # 3. 排序与截断输出
        # ==========================================
        all_candidates.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
        final_docs = all_candidates[:self.k]

        # 干净清爽的日志输出
        print(f" 🏆 [纯 AI 精排完成] 最终 Top {self.k}：")
        for i, doc in enumerate(final_docs):
            score = doc.metadata.get('rerank_score', 0)
            print(f"   └─ Top {i + 1} | 分数: {score:7.4f} | 预览: {doc.page_content[:50].replace(chr(10), '')}...")
        print("-" * 60)

        return final_docs

    def rag_summarize(self, query: str, use_bm25: bool = True) -> str:
        # 获取精排后的文档
        context_docs = self.retriever_docs(query, use_bm25=use_bm25)

        context = ""
        for counter, doc in enumerate(context_docs, 1):
            # 将精排分数也拼接到 prompt 里，虽然大模型不一定看，但日志里看得很爽
            score_info = f"[精排得分: {doc.metadata.get('rerank_score', 0):.4f}] "
            context += f"【参考资料{counter}】 {score_info}: {doc.page_content}\n"

        print("\n" + "=" * 20 + " 发送给 AI 的上下文 " + "=" * 20)
        print(context)
        print("=" * 60 + "\n")

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()

    # 测试查询，你可以故意弄脏这个词看看精排的威力
    # test_query = "E88 故障 修复方法"
    test_query = "传感器水渍 故障排查 报错清理方法 吹风机吹干 防撞条"
    print(f"\n🔍 正在开始全量深度排查: {test_query}")

    # 直接调用完整的检索链即可
    docs = rag.retriever_docs(test_query, use_bm25=True)

    if docs:
        print("\n✨ 测试完成，以上为进入大模型的最终参考资料。")
    else:
        print("\n! 检索库中没有任何资料。")