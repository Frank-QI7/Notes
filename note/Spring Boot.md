spring boot  的父级依赖 spring-boot-starter-parent配置之后，那么当前的项目就是spring boot项目；

spring-boot-starter-parent 是一个特殊的 starter依赖，它用来提供相关的 **Maven** 默认依赖，使用它之后，常用的jar包依赖就可以省去version配置；(在pom文件里我们可以看到，dependencies里的文件都没有版本号，就是因为这个父级依赖的原因)

spring boot提供了哪些默认jar包的依赖，可以在pom文件下的parent里看到

![image-20181122121450175](/Users/yunqingqi/Desktop/note/image-20181122121450175-2851090.png)

然后在version 那个tag 的地方 按command + 左键，点击版本就可以看到依赖的具体内容了，我们可以看到

![image-20181122122329954](/Users/yunqingqi/Desktop/note/image-20181122122329954-2851609.png)

**它还有parent依赖 **，按command + 左键，在properities里就可以看到了真正的**依赖包**和**版本**

![image-20181122122516735](/Users/yunqingqi/Desktop/note/image-20181122122516735-2851716.png)

再往下看，我们可以看到 一个 dependencyManagement 的标签 

![image-20181122123023945](/Users/yunqingqi/Desktop/note/image-20181122123023945-2852023.png)

这个标签下的 dependencies 标签里的内容 **当我们在用到的时候，我们是需要声明的！！！**，再来看一下pom文件里的 dependencies，

![image-20181122123143759](/Users/yunqingqi/Desktop/note/image-20181122123143759-2852103.png)

我们用到了 

```
<groupId>org.springframework.boot</groupId>
<artifactId>spring-boot-starter-web</artifactId>
```

这两个，再来看我们上面 dependencyManagement 下的 dependencies，里面是有我们上面的那个依赖的，只是说我们不用再声明它的版本号～～～～

![image-20181122123336403](/Users/yunqingqi/Desktop/note/image-20181122123336403-2852216.png)

如果不想使用某个默认的版本依赖，可以通过pom.xml文件里的properities属性覆盖各个依赖项，比如覆盖spring版本：

```java
<properities>
	<spring.version>5.0.0RELEASE</spring.version>
</properities>
```

在我们的main，java文件夹下的Application文件里，

![image-20181122123749205](/Users/yunqingqi/Desktop/note/image-20181122123749205-2852469.png)

我们需要写上

![image-20181122123809511](/Users/yunqingqi/Desktop/note/image-20181122123809511-2852489.png)

@SpringBootAllication 这个注解，作用就是 **开启sprint的自动配置**，**有一点需要注意的就是,** 这个配置文件只能扫描与其同一级别，或其子级的文件，比如我们的controller文件是和Application文件是同一级别的，我们的controller文件下有有一个HelloController文件，那么Application文件可以配置HelloController文件~~~~

@Controller 和 @ResponseBody 都是Spring mvc



  现在来看一下Spring boot的**核心配置文件**，

第一个就是 .properties 文件：键值对的properities属性文件配置方式

![image-20181122124746255](/Users/yunqingqi/Desktop/note/image-20181122124746255-2853066.png)



