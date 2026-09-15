docker run -itd --shm-size=100g --privileged=true --name <容器名> \
--net=host \
-v /mnt:/mnt \
-v /home:/home \
-v /data:/data \
-v /var/queue_schedule:/var/queue_schedule \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/sbin:/usr/local/sbin \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
--device=/dev/davinci0:/dev/davinci0 \
--device=/dev/davinci1:/dev/davinci1 \
--device=/dev/davinci2:/dev/davinci2 \
--device=/dev/davinci3:/dev/davinci3 \
--device=/dev/davinci4:/dev/davinci4 \
--device=/dev/davinci5:/dev/davinci5 \
--device=/dev/davinci6:/dev/davinci6 \
--device=/dev/davinci7:/dev/davinci7 \
--device=/dev/davinci8:/dev/davinci8 \
--device=/device/davinci_manager:/device/davinci_manager \
--device=/dev/hisi_hdc:/dev/hisi_hdc \
<镜像名>

docker exec -it <容器名> bash

cd /home/y00951466