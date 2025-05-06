使用流程：

1、pixel_point_selector.py：选取图片中角反射器的三个角点后会自动计算出内心点，每个图片对应一个csv文件，里面有三个角点像素坐标和一个内心点坐标，当所有的图片都选取好后，再使用extract_incenter.py进行中心点处理，合并到一个csv中。(代码里面有图片的路径，选取一次后需要修改路径中的图片名称进行下一次的选取)

2、extract_incenter.py：提取上一步生成的CSV文件中的内心点，把每张图片的内心点写入到一个csv文件中，方便后续使用。

3、point_cloud_representative_calculator.py：计算点云的代表点，把每帧的雷达点云代表点写入到一个csv文件中，得到代表点后需要根据记录的角反射器与雷达的距离进行对比，如果距离相差太大，我们需要从点云原始文件中手动选取角反射器的代表点填写到CSV中。（注意输入文件的CSV格式，在空旷环境中只有角反射器这一个强反射物体的时候我们可以根据RCS/SNR进行判断，角反射器点云的RCS/SNR会显著大于普通点云）

4、calibration.py：计算单应性矩阵，把点云代表点和像素代表点进行输入，最终得到单应性矩阵。

5、point_cloud_project_to_image.py：根据单应性矩阵把点云投影到图片上，得到点云在图片上的投影。



注意！！！：
1、每个py文件中的输入文件路径和输出文件路径需要修改成自己的路径。
2、这里提供的dataset是一个示例数据集(仅供参考)，里面包含了点云、图片,自采数据集请按照示例的格式进行调整文件名称和格式内容。
3、如果我们选取点云代表点的时候，点云代表点和像素代表点的距离相差太大，我们需要从点云原始文件中手动选取角反射器的代表点填写到CSV中。（根据采集数据时测量的距离寻找对应的点云XY，并且使用RCS/SNR进行辅助判断，金属角反射器的RCS/SNR会显著大于普通点云）

数据采集要求：
1、尽量在空旷的环境下没有金属强反射物，建议在操场这种场景进行采集
2、放置角反射器在相机和雷达的FOV范围内，具体距离根据实际情况进行调整，每次摆放的距离请进行记录，方便后续挑选出角反射器的对应点云。
3、录制尽量多的数据组，一组数据中包含：点云（3-5帧）、图片（一帧）、角反射器到Radar的距离，建议录制25组左右。


开发环境：
python3.9.10

依赖库安装：
pip install opencv-python pandas
pip install opencv-python-headless opencv-contrib-python
pip install requests
pip install bs4
pip install open3d
pip install pyntcloud numpy scipy scikit-image
pip install matplotlib 
pip install scikit-learn
