import os


def delete_file():
    """
    删除指定目录下的所有文件
    :return:
    """
    directory = "../data/raw/"  # 指定要删除文件的目录

    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在。")
        return

    for root, _, files in os.walk(directory):
        for file in files:
            # 构建文件的完整路径
            if file.startswith("."):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"已删除文件: {file_path}")
                except Exception as e:
                    print(f"删除文件 {file_path} 时出错: {e}")



delete_file()