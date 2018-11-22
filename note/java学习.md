![image-20180917154248653](/Users/yunqingqi/Desktop/note/image-20180917154248653.png)

![image-20180917154326108](/Users/yunqingqi/Desktop/note/image-20180917154326108.png)

![image-20180917154359054](/Users/yunqingqi/Desktop/note/image-20180917154359054.png)

#### 习惯添加try 和 catch，system.err 会输出红色的错误提示

![image-20180917152128482](/Users/yunqingqi/Desktop/note/image-20180917152128482.png)

#### [Eclipse小技巧](https://www.zhihu.com/question/29013594)

```
Ctrl+1 快速修复(最经典的快捷键,就不用多说了)
Ctrl+D: 删除当前行 
Ctrl+Alt+↓ 复制当前行到下一行(复制增加)
Ctrl+Alt+↑ 复制当前行到上一行(复制增加)
Ctrl+W 关闭当前Editer
Alt+Shift+M 自动引包
```

#### 在java中求两个矩形的交集和并集

```java
import java.awt.Rectangle;
 
public class Main {
	public static void main(String[] args) {
		//形参分别是   左上角X坐标 ，左上角Y坐标 ，宽，高
		Rectangle re1 = new Rectangle(0, 0, 100, 200);
		Rectangle re2 = new Rectangle(50, 50, 100, 200);
		System.out.println("矩形A:"+"x坐标="+re1.x+" y坐标="+re1.y+" 宽:"+re1.width+" 高="+re1.height);
		System.out.println("矩形B:"+"x坐标="+re2.x+" y坐标="+re2.y+" 宽:"+re2.width+" 高="+re2.height);
		System.out.println();
		
		//得到交集
		Rectangle intersection = re1.intersection(re2);// 交集
		
		//得到并集
		Rectangle union = re1.union(re2);// 并集
		
		// 交集坐标
		int intrX = intersection.x;// 水平坐标
		int intrY = intersection.y;// 垂直坐标
		//交集宽高
		int intrwidth = intersection.width;// 宽
		int intrheight = intersection.height;// 高
		System.out.println("交集是:"+"x坐标="+intrX+" y坐标="+intrY+" 宽:"+intrwidth+" 高="+intrheight);
		
		// 并集坐标
		int unionX = union.x;// 水平坐标
		int unionY = union.y;// 垂直坐标
		//并集宽高
		int unionwidth = union.width;// 宽
		int unionheight = union.height;// 高
		System.out.println("并集是:"+"x坐标="+unionX+" y坐标="+unionY+" 宽:"+unionwidth+" 高="+unionheight);
	}
}
```

#### intersects 是直接判断是否有交集！！！！



### 具体代码去eclipse 下面的Java学习 还有Snack 看看

java.awt:  (比swing要快，因为是c++编写)
​	Abstract Window ToolKit(抽象窗口工具包),需要
调用本地系统方法实现功能。属重量级控件。

javax.swing:
​	在AWT的基础上,建立的一套图形界面系统,其中
提供了更多的组件,而且完全由Java实现。增强了移植性,属轻量级控件。



Swing 的使用主要就是以下步骤：

**1. JFrame** – java的GUI程序的基本思路是以JFrame为基础，它是屏幕上window的对象，能够最大化、最小化、关闭。

**2. JPanel** – Java图形用户界面(GUI)工具包swing中的面板容器类，**可以放到 JFrame 里**

**3. JLabel** – JLabel 对象可以显示文本、图像或同时显示二者。**可以放到 JPane 里**

实例：

```java
JFrame frame = new JFrame("Login Example");   //创建了一个名为Login Example 的显示框，我们可以设置位置，大小等等。
// 设置界面可见
frame.setVisible(true);

JPanel panel = new JPanel(); 

frame.add(panel);   //这样我们就把 JPanel 放到 JFrame 里了

JLabel userLabel = new JLabel("User:"); //这样类似于html里的 p 或者 h 就是用来显示一些文字
/* 这个方法定义了组件的位置。
* setBounds(x, y, width, height)
* x 和 y 指定左上角的新位置，由 width 和 height 指定新的大小。
*/
userLabel.setBounds(10,20,80,25);
panel.add(userLabel); // //这样我们就把 JLabel 放到 JPanel 里了

//剩下的元素，比如button，textfield之类的 就类似于上面了，最后的实例结果就是 下图
```

![image-20181001124713946](/Users/yunqingqi/Desktop/note/image-20181001124713946.png)

#### 先说一下如何对 JFrame 添加组件

```java
对JFrame添加组件有两种方式：
1)用getContentPane()方法获得JFrame的内容面板，再对其加入组件：frame.getContentPane().add(childComponent)
2)建立一个Jpanel或JDesktopPane之类的中间容器，把组件添加到容器中，用setContentPane()方法把该容器置为JFrame的内容面板：
JPanel  contentPane=new  JPanel();
……//把其它组件添加到Jpanel中;
frame.setContentPane(contentPane);     // 但是上面我们用的是frame.add(panel); 
//把contentPane对象设置成为frame的内容面板
```



#### 这里有一个设置button各种属性的

```java
automanualbutton = new JButton("manual"); //添加一个button
ImageIcon manual = new ImageIcon("Icon/manual.png");  //图片来源
Image manualimage = manual.getImage().getScaledInstance(50, 50, Image.SCALE_SMOOTH);
automanualbutton.setIcon(new ImageIcon(manualimage)); //给button设置icon
automanualbutton.setBounds(60, 300, 180, 81); //设置butoon的位置
automanualbutton.setForeground(Color.GREEN); // 设置前景颜色, 就是指button上的字颜色
automanualbutton.addKeyListener(new keyboardListener()); //添加key监听事件
automanualbutton.setToolTipText("Robot will start up"); //还可以给button添加一个tooltip，就是说，你鼠标放在上面的时候提示出来的内容

```

#### public class和class的区别

如果一个类声明的时候使用了public class进行了声明，则类名称必须与文件名称完全一致，如果类的声明使用了class的话，则类名称可以与文件名称不一致。 **范例：定义一个类(文件名称为：Hello.java)**

```java
public class HelloDemo{    
    public static void main(String args[]){   
        System.out.println("HelloWorld!!!");  
    }
}
```

此类使用public class声明，类名称是HelloDemo，但是文件名称Hello.java，所以，此时编译时会出现如下问题：

```text
Hello.java:类 HelloDemo 是公共的，应在名为HelloDemo.java文件中声明
```

**范例：有如下代码(文件名称为:Hello.java)**

```java
class HelloDemo{

    public static void main(String args[]){

       
       System.out.println("HelloWorld!!!");

    }

}
```

文件名称为Hello.java，文件名称与类名称不一致，但是因为使用了class声明所以，此时编译不会产生任何错误，但是生成之后的*.class文件的名称是和class声明的类名称完全一致的:HelloDemo.class 执行的时候不能再执行java Hello，而是应该执行java HelloDemo

**在一个*.java的文件中，只能有一个public class的声明，但是允许有多个class的声明**

```java
public class Hello{
   public static void main(String args[]){  
       System.out.println("HelloWorld!!!");
    }
}
class A{}

class B{}
```

在以上的文件中，定义了三个类，那么此时程序编译之后会形成三个*.class文件。

#### static 声明的含义

public void 修饰是非静态方法，该类方法属于**对象**，在对象初始化（new Object()）后才能被调用；
public static void 修饰是静态方法，属于**类**，使用类名.方法名直接调用

**下面来看例子**

```java
public class Demo {
    static int i = 10;     // 静态变量
    int j;                 // 实例变量
    Demo() {
        this.j = 20;       // 对于静态变量，类里的函数可以直接用this来调用，实例变量不行
    }
    public static void main(String[] args) {
        System.out.println("类变量 i=" + Demo.i); // 静态变量属于类，使用类名.方法名直接调用！！！
        Demo obj = new Demo(); // 实例变量属于 ！！对象！！，在对象初始化（new Object()）后才能被调用！！！！！！！！！！！！
        System.out.println("实例变量 j=" + obj.j);
    }
}
```

**下面还有一个例子，很重要,  是关于 static 的内存分配**

静态变量属于  **类**，不属于任何独立的  **对象**，**所以无需创建类的实例就可以访问静态变量。** 之所以会产生这样的结果，**是因为编译器只为整个类创建了一个静态变量的副本，也就是只分配一个内存空间，虽然有多个 实例，但这些 实例 共享该内存。** 实例变量则不同，**每创建一个对象，都会分配一次内存空间，**不同变量的内存相互独立，互不影响，改变  a 对象的实例变量不会影响 b 对象 ,   下面看例子：

```java
public class Demo{
    static int i;
    int j;
    public static void main(String[] args) {
        Demo obj1 = new Demo();
        obj1.i = 10;
        obj1.j = 20;
        Demo obj2 = new Demo();
        System.out.println("obj1.i=" + obj1.i + ", obj1.j=" + obj1.j);
        System.out.println("obj2.i=" + obj2.i + ", obj2.j=" + obj2.j);
    }
}
运行结果：
obj1.i=10, obj1.j=20
obj2.i=10, obj2.j=0
```

**注意：外面可以看到，静态变量虽然也可以通过对象来访问，但是不被提倡，编译器也会产生警告。**

<font color=#B22222 >所以！！！！！ 以后在写代码的时候，不要通过new一个类创建一个对象，然后用对象来获得类里的static变量！！！！！！！！！！！！！！！！ </font>

#### Java——Super关键字

**一. 调用父类的属性**

```java
class Person{
    String name = "李四";
    int age;
}
class Student extends Person{
    String name = "张三";
    void print() {
        //调用父类的属性
        System.out.println("父类的属性：" + super.name);
        //调用本类的属性
        System.out.println("子类的属性：" + this.name);
    }
}
public class test{
    public static void main(String[] args) {
        new Student().print();
    }
}
结果是：
父类的属性：李四
子类的属性：张三
```

**二. 调用父类的普通方法**

```java
class Person{
    public void print() {
        System.out.println("父类的方法");
    }
}
class Student extends Person{
    public void print(){
        //调用父类的普通方法
        super.print();
        System.out.println("子类的方法");
    }
}
public class test{
    public static void main(String[] args) {
        new Student().print();
    }
}
结果是：
父类的方法
子类的方法
```

**三. 调用父类的构造方法**

```java
class Person {
    private String name;
    private int age;
    public Person(String name,int age){
        this.setName(name);
        this.setAge(age);
    }
    public void setName(String name) {
        this.name = name;
    }
    public String getName() {
        return name;
    }
    public void setAge(int age) {
        this.age = age;
    }
    public int getAge() {
        return age;
    }
    public String getInfo(){
        return "姓名:" + this.name + "年龄:" + this.age;
    }
}
class Student extends Person{
    String school;
    public Student(String name,int age,String school) {
        super(name,age);      // super调用构造方法也必须放在子类构造方法的首行
        this.setSchool(school);
    }
    public void setSchool(String school) {
        this.school = school;
    }
    public String getSchool() {
        return school;
    }
    public String getInfo() {
        return super.getInfo() + "学校:" + this.getSchool();
    }
}
public class Superkey{
    public static void main(String[] args) {
        Student student = new Student("张三",30,"清华大学");
        System.out.println(student.getInfo());
    }
}
结果是：
姓名:张三年龄:30学校:清华大学
```

<font color=#B22222 > **super调用构造方法也必须放在子类构造方法的首行。**</font>

