直接运行：

```
uvicorn app2:app --log-level info --port 8123 --host 127.0.0.1 > uvicorn.log 2>&1
```


docker里运行：
```shell
docker build -t path_planning:1.0 .
PORT=8882 && docker run -p ${PORT}:${PORT} \
  --name path_planning_container \
  -v /data/ImageData:/data/ImageData \
  -v $(pwd)/logs:/logs \
  path_planning
```
如果想后台运行加上-d参数。
如果需要修改图片路径，改成下面的命令
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