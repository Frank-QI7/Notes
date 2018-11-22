# 安装

# 环境要求

使用工具前确保如下几点:(确保已经安装了xcode)

- 安装 MonkeyDev

```
git clone https://github.com/AloneMonkey/MonkeyDev.git
cd MonkeyDev/bin
sudo ./md-install
```

- 安装最新的[theos](https://github.com/theos/theos/wiki/Installation)

```
sudo git clone --recursive https://github.com/theos/theos.git /opt/theos
```

- 安装ldid(如安装theos过程安装了ldid，跳过)

```
brew install ldid
```

- 配置免密码登录越狱设备(如果没有越狱设备，跳过)

```
ssh-keygen -t rsa -P ''
ssh-copy-id -i /Users/username/.ssh/id_rsa root@ip
```

或者安装`sshpass`自己设置密码:

```
brew install https://raw.githubusercontent.com/kadwanev/bigboybrew/master/Library/Formula/sshpass.rb
```

然后打开xcode，new project 就会出现 MonkeyDev。

但是想要为所欲为，还需要一个被砸过壳的ipa或者app，我们的任务就是对想逆向的app先砸壳



