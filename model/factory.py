from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf
from dotenv import load_dotenv

load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"])


# 云端通义千问 Embedding 工厂
class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


# 【新增】本地 BGE 模型工厂
class LocalEmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        print("⏳ 正在由 LocalFactory 实例化本地 BGE-Large 模型，请稍候...")
        model_name = "BAAI/bge-large-zh-v1.5"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}  # 开启归一化
        )


chat_model = ChatModelFactory().generator()

embed_model = EmbeddingsFactory().generator()

local_embed_model = LocalEmbeddingsFactory().generator()
