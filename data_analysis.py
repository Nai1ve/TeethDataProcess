from data_analysis import data_analysis

# 1.原始数据统计，牙片数、废片数、多生牙、缺牙、困难、中等、简单，牙片的分辨率
hospital = data_analysis.raw_photo_count()


# 2.分割后的统计 FDI分类、牙齿数、牙齿的尺寸、和牙片的比例、灰度