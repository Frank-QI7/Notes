54.206.43.206     学校的ssh：   [a1700172@uss.cs.adelaide.edu.au](/var/folders/bk/wzdlzcx90tn997tgv8dk78tm0000gn/T/abnerworks.Typora/49374DE2-BF4E-4315-A0CF-ED22545C661C/mailto:a1700172@uss.cs.adelaide.edu.au) 

#### 路由器安装[wireguard](https://www.youtube.com/watch?v=gbnOoHfx3IQ)

学校的HPC ssh:       a1700172@phoenix.adelaide.edu.au

iproxy 8100 8100

Mac os用命令下载文件： curl -o +[文件名字] + [url ]

#### [conda的命令](https://www.jianshu.com/p/d2e15200ee9b)

#### [Jupyter notebook的conda环境设置](https://zhuanlan.zhihu.com/p/29564719)

GCP: [*35.237.219.199* ](https://35.237.219.199/) (在实例里的网络设置里设置防火墙)

```
For GCP:
首先遇到的是要用自己的电脑创建一对key：
ssh-keygen -t rsa -f ~/.ssh/my-ssh-key -C yunqingqi  （会给我们设置一个private和public的文件，在pub结尾的就是public的key）
chmod 400 ~/.ssh/my-ssh-key
然后在点击instance，点击edit，给ssh设置key
cat ~/.ssh/my-ssh-key.pub
把内容复制到设置的框里去
我们在连接我们的谷歌云的时候，要用private的key，也就是不带.pub结尾的那个
```

**另外一个设置谷歌云服务器密码的方法就是**：

1. 在谷歌云服务器页面点击创建ssh，然后 sudo -i 进入管理员模式
2. 输入passwd root，进行重新的密码设置
3. 输入 vi /etc/ssh/sshd_config, 对ssh进行修改，使其支持ssh登陆

![image-20181121112530181](/Users/yunqingqi/Desktop/note/image-20181121112530181-2761730.png)

4. 进入 config文件后，按键 i, 进入insert 模式

5. ![image-20181121112728309](/Users/yunqingqi/Desktop/note/image-20181121112728309-2761848.png)

6. 然后找到图片中 PermitRootLogin 这个语句，把 no 改为 yes

7. 然后再空白的地方输入下面这两行，允许输入密码登录

8. ![image-20181121112923550](/Users/yunqingqi/Desktop/note/image-20181121112923550-2761963.png)

9. 然后保存退出，关闭ssh页面，在谷歌云服务器页面上 重置 该事例

   #### [wireguard vpn设置](https://www.mfzy.cf/archives/58)

[手把手教你用Google云平台搭建自己的深度学习工作站](https://hikapok.github.io/2018/01/19/intro-to-google-cloud/)

[动手学深度学习](https://zh.gluon.ai/chapter_prerequisite/install.html#%E8%8E%B7%E5%8F%96%E4%BB%A3%E7%A0%81%E5%B9%B6%E5%AE%89%E8%A3%85%E8%BF%90%E8%A1%8C%E7%8E%AF%E5%A2%83)

```
nvidia-smi 来看 GPU 的运行情况
设置环境变量：
sudo vim ~/.bashrc
现在是下载conda： wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Mini( 用tap 自动补全)
然后是：source ~/.bashrc
然后是：下载包含本书全部代码的压缩包，解压后进入文件夹。运行以下命令。
mkdir gluon_tutorials_zh-1.0 && cd gluon_tutorials_zh-1.0
curl https://zh.gluon.ai/gluon_tutorials_zh-1.0.tar.gz -o tutorials.tar.gz
tar -xzvf tutorials.tar.gz && rm tutorials.tar.gz
# 使用清华 conda 镜像。
conda config --prepend channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
然后是：conda env create -f environment.yml
然后是：source activate gluon      (安装成功后，以后就直接跳到这一步来记过gluon，然后打开jupyter)
最后是：打开 Juputer 笔记本。
jupyter notebook
这时在浏览器打开 http://localhost:8888 （通常会自动打开）就可以查看和运行本书中每一节的代码了。
本书中若干章节的代码会自动下载数据集和预训练模型，并默认使用美国站点下载。我们可以在运行 Jupyter 前指定 MXNet 使用国内站点下载书中的数据和模型。
set MXNET_GLUON_REPO=https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/ jupyter notebook

在那个书的目录下运行： pip uninstall mxnet
然后 pip install --pre mxnet-cu80
然后运行 python
import mxnet as mx
看是否安装成功，如果出现错误，就用pip3 install --pre mxnet-cu80，然后运行的时候运行python3

然后：a = mx.nd.array([1,2,3], mx.gpu())
输出 a 会返回这样的：
[ 1.  2.  3.]
<NDArray 3 @gpu(0)>

然后把vim environment.yml里的mxnet改成mxnet-cu80
再更新一下：conda env update -f environment.yml
重新开启：source activate gluon
再运行：jupyter notebook
然后是把本地的8000端口连到 云服务器到8888端口，就可以在本地看到书的内容了
ssh -L8000:localhost:8888 -i "~/.ssh/my-ssh-key" yunqingqi@35.229.83.36 
如果忘记了 token 的话，就先进入那个目录，然后激活gluon，运行：jupyter notebook list
就会返回token，把token后面的值输入进去就可以了
```

**[Jupyter Notebook介绍、安装及使用教程](https://zhuanlan.zhihu.com/p/33105153)** **最简单的使用方法就是jupyter notebook --port 9999**

dBFTTrBA5hfFM
用scp传文件(在Mac下首先 cd ~ 不然yunqing.pem 获取不到)：
先打包：
tar zcvf xxxx.tar.gz 目标文件      ##打包压缩文件
tar zxvf xxxx.tar.gz              ##解包解压缩xxxx.tar.gz===>xxxx

如果是zip 类型的文件，就直接用 unzip 解压就ok了

再：
scp -i "yunqing.pem" -r /Users/yunqingqi/Express_frame/blog ubuntu@ec2-54-206-43-206.ap-southeast-2.compute.amazonaws.com:~/shuoshuo

还有从服务器往本地传输文件 scp -r a1700172@uss.cs.adelaide.edu.au:~/Desktop/GuiOptimiser ~/Desktop/

vnc连接：dBFTTrBA5hfFM: 现在只能用下面的方法连接了，再mac的terminal下输入：
ssh -L 5901:localhost:5901 -i "yunqing.pem" ubuntu@ec2-54-206-43-206.ap-southeast-2.compute.amazonaws.com
然后打开vnc，输入localhost:5901 再输入密码就好了

学校的服务器是：ssh a1700172@uss.cs.adelaide.edu.au
PTE 账号：FRANK_QI 密码:xiyangyanG7!

微信开发ID：wx13bd7fe96434f4ce
Mac下 open . 即可打开当前文件夹 我的mac本地时代配置是redis-server /usr/local/etc/redis.conf
阿里云上的vnc账号：mahbubtuto 密码haibo 阿里云上的是 redis-server /etc/redis/redis.conf
另外停止和启动 Redis 服务的命令如下：
sudo /etc/init.d/redis-server stop
sudo /etc/init.d/redis-server start

mongodb安装过程：
1. sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 0C49F3730359A14518585931BC711F9BA15703C6

2. echo "deb [ arch=amd64,arm64 ] http://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.4 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.4.list

3. sudo apt-get update

4. sudo apt-get install -y mongodb-org
    然后
    mongod —-auth --port 27017 --dbpath /data/db
    后台开启mongod！！！！
    sudo mongod --fork --logpath ~/log/mongodb.log --dbpath=/data/db
    再打开一个terminal运行mongo就好了
    然后use admin（一定要在admin下创建新的用户，然后再db.auth！！！！！！！！！！）
    再执行 db.createUser({user: 'frank', pwd: 'dBFTTrBA5hfFM1', roles: [{role: 'root', db: 'admin'}]})
    一定要 db.auth('frank', 'dBFTTrBA5hfFM1')，才可以真正成功

再修改文件：sudo vi /etc/mongod.conf，把下面的改成这样
net:
  port: 27017
  bindIp: 0.0.0.0
还有添加验证信息：要把security取消注释
security:
 authorization: enabled
配置完后重启！！！！

以后开机就运行命令：sudo mongod 

ps ax | grep mongod 用来查询当前mongodb 进程并用：kill -9 进程号 杀掉
mongodb 数据导出 ：  
mongoexport --host 127.0.0.1 --port 27017 -u admin -p dBFTTrBA5hfFM --db Coles -c product --type=csv -o coles.csv --authenticationDatabase admin -f "name,image,price"
-f后面跟的是csv的field

Robo3T上查询命令： 
db.getCollection('product').find({"name":{$regex:'voltaren',$options:'$i'}})

Vnc开启是 vncserver :1
修改分配的端口范围：vim /etc/sysctl.conf 然后添加：
net.ipv4.ip_local_port_range = 32768 59000
然后 sysctl -p /etc/sysctl.conf 重新加载

gerapy使用：
Gerapy init 创建一个gerapy文件夹，里面有一个project文件夹
cd gerapy后 gerapy migrate目录下生成一个 SQLite 数据库，同时建立数据库表
gerapy runserver 后，就可以打开localhost:8000来管理了


#docker运行 splash
docker run -p 8050:8050 scrapinghub/splash

剩下的去Pycharm 里的 internship里看setting还有spider怎么设置





安装桌面:
sudo apt-get install gnome-shell ubuntu-gnome-desktop 
安装好后 reboot
配置 sudo nano /etc/gdm3/custom.conf
  AutomaticLoginEnable = true
  AutomaticLogin = root

scrapy里cookie的设置参考myproject里的tencent,现在cookie都是key，value对，要自己改
scrapy 里设置log，log的文件要在setting里设置
要在每一个for循环里 设定 item = 。。。 在每一个for里面也要写   yield item
scrapy.Request里的meta是这样赋值的 meta={"haha":"123","xixi":"456"})

下面写在init里面～～～   引包：import logging
logger = logging.getLogger(__name__)    （有可能要用self.）
使用： logger.warning(item)  (有可能要用到self)
self.logger.debug('正在爬取：')
self.logger.debug(request.url)

正则表达式删除空行 ^[\s]*\n

Pm2:
npm install pm2 -g        pm2 start app.js      
查看所有的运行状态： pm2 list 或者 pm2 ls
监控所有的运行的进程：pm2 monit
查看所有运行的程序的日志：pm2 logs
查看单个进程的的日志: pm2 logs  app_name/id
暂停进程： pm2 stop  app_name/id
删除进程：pm2 delete app_name/id
重启进程：pm2 restart app_name/id
显示一个进程的详细信息：pm2 describe app_name/id  或者   pm2 show app_name/id
如果你不仅仅想监控被pm2管理的进程，还需要监控进程所运行的机器的信息，你可以使用下面这个API： pm2 web
pm2会启动一个叫做pm2-http-interface的进程提供web服务。你打开浏览器输入http：//127.0.0.1:9615，会把部署的服务器的信息和程序的信息都显示出来
pm2 stop all  #停止PM2列表中所有的进程

另外一个坑就是，在处理unicode的中文无法换行的时候，是因为中文保存在string里的时候，string的末尾会有个／r，导致强行换行，需要把这个去掉

scrapy 步骤：

```
scrapy startproject myproject
```

```
cd myproject
scrapy genspider mydomain mydomain.com
```

```python
scrapy shell http://doc.scrapy.org/en/latest/_static/selectors-sample1.html
```

**Cookie 替换取值**

```pytho n
(.*)?:(.*)
'$1':'$2',

(.+?)=(.+?);
'$1':'$2',  会有空格，要把空格去掉
最有一个因为没有;结尾，所以要手动去换

替换python2.7里面的print
print (.*)
print ($1)    如果print后面有注释，则需要改一下
```

**这样删除空白行**

```
^[\s]*\n
```

pyquery 是可以选择第几个的

```pyth
body= selector("div.a-wrap.corner  td.a-content p").eq(0)
就取了第一个！！！！！，然后body.text() 就可以输出内容了
还有一个可以输出html的就是 .html()
```

python里用yagmail发送邮件

```py
import yagmail

#链接邮箱服务器

yag=yagmail.SMTP( user="user@126.com", password="1234", host='smtp.126.com')

# 邮箱正文

contents=['This is the body, and here is just text http://somedomain/image.png','You can find an audio file attached.','/local/path/song.mp3']

# 发送邮件

yag.send('taaa@126.com','subject', contents)
```

Splash 等待元素直到出现

```py
function main(splash)
  splash:set_user_agent(splash.args.ua)
  assert(splash:go(splash.args.url))

  -- requires Splash 2.3  
  while not splash:select('.my-element') do
    splash:wait(0.1)
  end
  return {html=splash:html()}
end
```

```
splash:runjs("document.getElementsByClassName('pagingNext')[0].scrollIntoView(true)")
```

woolworths比较特殊，页面无法用splash获取，那么我们就利用这个特性：当pagenumber超过存在的数的时候，就会导致



python保存文件

```python
import urllib.request
from urllib import error
import json
import urllib
import os

class Laotuo93Pipeline(object):
    def open_spider(self, spider):
        self.file = open('content.json', 'w')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        songName = item['name']
        songDownloadURL = item['url']

        try:
            data = urllib.request.urlopen(songDownloadURL).read()
        except error.URLError as e:
            print(e.reason,"######链接不存在，继续下载下一首########")
        file_path = "audio/" + songName + ".mp3"

        with open(file_path, 'wb') as f:
            f.write(data)

        line = json.dumps(dict(item),ensure_ascii=False,indent=2)
        self.file.write(line)
        return item
```

#### Appium macOS 下的 Appium 安装与配置 Appium Installation & Setup With macOS

#### 最主要的就是去官网下一个appium-desktop for Mac

appium本质上就是一个Nodejs应用，我们可以使用npm对其进行安装，安装完毕后就可以使用命令行启动

```
npm install -g appium
```

可以使用appium-doctor来确认安装环境是否完成

```
npm install -g appium-doctor
```

```
appium-doctor
```

```
brew install carthage
```

```
brew install libimobiledevice --HEAD
brew install ideviceinstaller
下面这个可以不安装
# npm install -g ios-deploy
npm install app-inspector -g
sudo pip3 install Appium-Python-Client
```

Mac安装使用tesseract-ocr

```
brew install tesseract

然后
cd /usr/local/Cellar/tesseract/3.04.01_2/share/tessdata (版本可能不同，可以先进入/usr/local/Cellar/tesseract/)

最后 pip3 install pytesseract

在python中使用就是：
from PIL import Image
import pytesseract
words = pytesseract.image_to_string(Image.open('...xxx/test.png'), lang='chi_sim')
print(words)
```

#### 在用 [vpn设置](https://github.com/hwdsl2/setup-ipsec-vpn/blob/master/README-zh.md) 它的shell 设置vpn的时候，要把aws的500 和4500 端口打开！！

