from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf

# from model.factory import embed_model
from model.factory import local_embed_model as embed_model

from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import time
import os
from dotenv import load_dotenv
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
load_dotenv()


class VectorStoreService:
    def __init__(self):
        abs_persist_path = get_abs_path(chroma_conf["persist_directory"])

        logger.info(f"🚩 向量数据库物理路径锁定为: {abs_persist_path}")
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=abs_persist_path,  # 这里必须传入绝对路径
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    # 在 VectorStoreService 类末尾添加
    def get_all_documents(self) -> list[Document]:
        """获取 Chroma 中所有文档块，用于 BM25 初始化"""
        try:
            results = self.vector_store.get(limit=10000)  # 获取全量数据
            docs = [
                Document(
                    page_content=results['documents'][i],
                    metadata=results['metadatas'][i] if results['metadatas'] else {}
                )
                for i in range(len(results['documents']))
            ]
            return docs
        except Exception as e:
            logger.error(f"获取全量文档失败: {str(e)}")
            return []

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False  # md5 没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True  # md5 处理过

                return False  # md5 没处理过

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)
                # --- 临时调试代码：看看 E88 到底在哪 ---
                # for doc in split_document:
                #     if "E88" in doc.page_content:
                #         logger.info(f"✅ 成功切分出包含 E88 的块: {doc.page_content}")
                # ------------------------------------
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                # self.vector_store.add_documents(split_document)
                # 修改后的代码：每次最多发送 10 个 chunk 给大模型
                # batch_size = 10
                # 本地显卡推理极快，无需 sleep 限流，直接设置较大的 batch_size
                batch_size = 100
                total_batches = (len(split_document) + batch_size - 1) // batch_size
                for i in range(0, len(split_document), batch_size):
                    batch = split_document[i: i + batch_size]
                    self.vector_store.add_documents(batch)

                    print(f"📦 正在写入第 {i // batch_size + 1}/{total_batches} 批次...")  # 监控进度
                    # time.sleep(1)  # 增加延迟，稳如老狗

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
