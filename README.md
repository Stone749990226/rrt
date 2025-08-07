- 后端启动的文件：main.py 
- RRT算法代码：rrt.py 
- 人工势场法代码：apf.py 
- 与一些其他算法的比对的代码：astar.py、test.py 
- 路径重规划算法：replan.py，用到了utils.py中的代码

  核心文件：main.py(具体每个参数的含义已经写在help里了，代码有注释，Astar部分由于是很成熟的算法没有注释，可以问ai)

docker里运行：
```shell
docker build -t path_planning:1.0 .
docker run -it -p 8123:8123 \
  --name path_planning_container \
  -v /data/ImageData:/data/ImageData \
  -v $(pwd)/logs:/logs \
  path_planning
```
如果想后台运行加上-d参数。
如果需要切换端口和图片路径，改成下面的命令
```shell
PORT=8882 && docker run -p ${PORT}:8123 \
  --name path_planning_container \
  -v /mnt/disk1/caddy/ImageData:/data/ImageData \
  -v $(pwd)/logs:/logs \
  path_planning:1.0 
```

停止运行方法：
```shell
docker stop path_planning_container && docker rm path_planning_container
```

查看运行错误：
```shell
docker logs path_planning_container
```

保存镜像：
```shell
docker save -o path_planning.tar path_planning:1.0
```

加载打包好的镜像：
```shell
sudo docker load -i path_planning.tar
```

拷贝到另一台主机：
```shell
scp path_planning.tar ices@10.249.44.78:/home/ices/rrt
```
