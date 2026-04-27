import chromadb
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path

# 手动连接底层库
path = get_abs_path(chroma_conf["persist_directory"])
client = chromadb.PersistentClient(path=path)
collection = client.get_collection(name=chroma_conf["collection_name"])

print(f"--- 物理层检查 ---")
print(f"数据库路径: {path}")
print(f"当前集合中的原始条数: {collection.count()}")

# 如果这里不是 0，也不是 132，而是一个几十的小数字，说明你终于写进去了！