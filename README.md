遗留问题：

1、point_cloud_project_to_image .py需要修改成使用单应性矩阵的方法
2、

使用流程：
1、使用point_selector.py选取像素点
2、所有的图片选取好后，使用extract_incenter.py进行中心点处理，合并到一个csv中
3、把像素点和对应的雷达点云写入到calibration.py中，进行标定
4、把标定结果得到的单应性矩阵，写入到point_cloud_project_to_image .py，进行投影。