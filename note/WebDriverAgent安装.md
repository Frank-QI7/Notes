#### Requirement

```
还要安装 nodejs 网上查一下，具体的忘了
brew install node
brew install npm
brew install libimobiledevice --HEAD
brew install ideviceinstaller  
brew install carthage
cd /usr/local/lib
npm install -g ios-deploy  
npm install -g appium-doctor  #用于检测ios环境是否安装完全
appium-doctor --ios   #用于检测ios环境是否安装完全
```

```
用来查询手机上应用的ipa
ideviceinstaller -l
```

```
git clone https://github.com/facebook/WebDriverAgent
sudo chmod -R a+rwx WebDriverAgent
cd WebDriverAgent
运行 sh ./Scripts/bootstrap.sh      
再使用 xCode build 或者product-Test一次。
build WebDriverAgent文件下的 WebDriverAgent.xcodeproj 文件
最重要的就是要修改 WebDriverAgentLib 下面的这个，
```

![image-20180925172023856](/Users/yunqingqi/Library/Application Support/typora-user-images/image-20180925172023856.png)

然后是修改 WebDriverAgentRunner 里的![image-20180925172123549](/Users/yunqingqi/Library/Application Support/typora-user-images/image-20180925172123549.png)

![image-20180925172137691](/Users/yunqingqi/Library/Application Support/typora-user-images/image-20180925172137691.png)

把product bundle identifier 修改一下，也就是加个 .fangzhou 

然后点击![image-20180925172248191](/Users/yunqingqi/Library/Application Support/typora-user-images/image-20180925172248191.png)

设置下面这个：

![image-20180925172258733](/Users/yunqingqi/Library/Application Support/typora-user-images/image-20180925172258733.png)

最后是就 点击product里的 test ，如果terminal里面出现了ip地址，就显示成功了

最后还在terminal里输入 

```
iproxy 8100 8100
```









