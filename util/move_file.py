import os
import shutil


def move_all_files(src_dir,dest_dir,overwrite=True):
    """
    移动源目录下的所有文件到目标目录
    :param src_dir:
    :param dest_dir:
    :return:
    """

    os.makedirs(dest_dir,exist_ok=True)

    for root, _ , files in os.walk(src_dir):
        for file in files:
            if not file.startswith("._"):
                src_path = os.path.join(root, file)
                print(f"正在处理文件: {src_path}")
                # 处理同名文件冲突
                dst_path = os.path.join(dest_dir, file)
                if os.path.exists(dst_path) and not overwrite:
                    base, ext = os.path.splitext(file)
                    count = 1
                    while True:
                        new_file = f"{base}({count}){ext}"
                        new_path = os.path.join(dest_dir, new_file)
                        if not os.path.exists(new_path):
                            dst_path = new_path
                            break
                        count += 1

                # 移动文件
                shutil.move(src_path, dst_path)
                print(f"文件 {src_path} 已移动到 {dst_path}")
            else:
                print(f"文件：{file},跳过")

if __name__ == '__main__':
    src_directory = "/Users/liupeize/Desktop/牙齿/儿童牙齿数据"  # 源目录
    dest_directory = "../data/raw/child/"  # 目标目录

    move_all_files(src_directory, dest_directory, overwrite=True)
    print("所有文件已成功移动。")