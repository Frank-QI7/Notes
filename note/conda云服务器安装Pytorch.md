## 环境

OS：CentOS 6.3

CUDA：8.0

CUDNN：7.0

## 步骤

**说明**：CUDA和CUDNN需要提前安装好。



1. 安装mini conda 2,版本4.5.4

```bash
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
sh Miniconda2-latest-Linux-x86_64.sh 
```

\2. 创建虚拟环境

```text
conda create -n pytorch python=2
```

\3. 安装pytorch

copy [PyTorch](https://link.zhihu.com/?target=https%3A//pytorch.org/) 官网的命令即可

```text
conda install pytorch torchvision -c pytorch 
```

\4. 验证 torch是否安装成功

```text
source activate pytorch
python
>>> import torch
>>> torch.cuda.is_available()
```

输出True表示安装成功，且GPU在pytorch中可用。

\5. 安装jupyter notebook

```text
conda install jupyter
# 生成配置
jupyter notebook --generate-config
```

6.修改 jupyter配置

```text
vim ~/.jupyter/jupyter_notebook_config.py 
```

修改默认绑定IP

```text
c.NotebookApp.ip = '0.0.0.0'
```

修改笔记位置，注意修改为自己需要存放的绝对路径，也可以不改，默认为启动jupyter时当前路径。

```text
c.NotebookApp.notebook_dir = u'存放笔记目录绝对路径'
```

修改打开浏览器选项，服务器运行时，通常我们不需要启动jupyter时打开浏览器，所以关闭此选项。

```text
c.NotebookApp.open_browser = False
```

\7. 修改密码

```text
jupyter notebook password
```

\8. 启动jupyter

```text
jupyter notebook
```

\9. 浏览器打开jupyter notebook

启动成功后会输出访问地址，默认端口为8888。如果是在自己的个人电脑上安装的torch和jupyter，则 [http://localhost:8888](https://link.zhihu.com/?target=http%3A//localhost%3A8888/) 即可打开jupyter。



\10. 新建python 2 notebook，执行以下代码

```text
import torch
torch.cuda.is_available()
```

如果输出为True表示成功。

## 遇到的问题

1.第4步`import torch`，出现错误:

```text
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/xxx/miniconda2/envs/pytorch/lib/python2.7/site-packages/torch/__init__.py", line 78, in <module>
    from torch._C import *
ImportError: /lib64/libc.so.6: version `GLIBC_2.14' not found (required by /home/xxx/miniconda2/envs/pytorch/lib/python2.7/site-packages/torch/_C.so)
```

分析：GLIBC版本太低，检查GLIBC版本

```text
strings /lib64/libc.so.6 |grep GLIBC
```

输出

```text
GLIBC_2.2.5
GLIBC_2.2.6
GLIBC_2.3
GLIBC_2.3.2
GLIBC_2.3.3
GLIBC_2.3.4
GLIBC_2.4
GLIBC_2.5
GLIBC_2.6
GLIBC_2.7
GLIBC_2.8
GLIBC_2.9
GLIBC_2.10
GLIBC_2.11
GLIBC_2.12
GLIBC_PRIVATE
```

发现GLIBC最高版本为2.12，而pytorch程序需要2.14，手动编译 2.14 至 /opt/glibc-2.14:

```text
su - root
wget https://ftp.gnu.org/gnu/glibc/glibc-2.14.tar.gz
tar zxf glibc-2.14.tar.gz
cd glibc-2.14
mkdir build
cd build
../configure --prefix=/opt/glibc-2.14
make -j4
make install
```

编译完成后，添加glibc-2.14到 LD_LIBRARYPATH。

```text
export LD_LIBRARY_PATH=/opt/glibc-2.14/lib:$LD_LIBRARY_PATH
```

再次执行第4步，不再报错。



2.启动jupyter后，jupyter notebook中执行 import torch 出现kernel died

分析：直接在python 中 `import torch` 是正常的，在ipython中执行`import torch`出现 Segmentation fault ，怀疑与GLIBC有关。尝试通过新版GLIBC启动ipython:

```text
source activate pytorch
/opt/glibc-2.14/lib/ld-2.14.so  --library-path $LD_LIBRARY_PATH:/lib64 /home/xxx/miniconda2/envs/pytorch/bin/python /home/xxx/miniconda2/envs/pytorch/bin/ipython
```

执行第4步，不再报错。

## 解决办法:

a.列出jupyter内核：

```text
jupyter kernelspec list
```

输出

```text
Available kernels:
  python2    /home/xxx/miniconda2/envs/pytorch/share/jupyter/kernels/python2
```

b. 编写通过glibc启动程序脚本

```text
touch /home/xxx/bin/glibc_2.14_run.sh
chmod +x /home/xxx/bin/glibc_2.14_run.sh
```

脚本内容

```text
#!/bin/sh

/opt/glibc-2.14/lib/ld-2.14.so  --library-path $LD_LIBRARY_PATH:/lib64 $*
```

c.修改内核配置中的启动参数

```text
vim  /home/xxx/miniconda2/envs/pytorch/share/jupyter/kernels/python2/kernel.json 
```

修改前

```text
{
 "display_name": "Python 2",
 "language": "python",
 "argv": [
  "/home/xxx/miniconda2/envs/pytorch/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ]
}
```

argv 中第一行增加 **/home/xxx/bin/glibc_2.14_run.sh** ，修改后

```text
{
 "display_name": "Python 2",
 "language": "python",
 "argv": [
  "/home/xxx/bin/glibc_2.14_run.sh",
  "/home/xxx/miniconda2/envs/pytorch/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ]
}
```

d.重启 jupyter notebook，重试第10步正常。

本方法参考：[10859 在 glibc < 2.17 的系统上安装 TensorFlow](https://zhuanlan.zhihu.com/p/33059558)

## 尝试过几种无效的办法：

a.因为先装的miniconda，后编译的GLIBC，尝试重装miniconda并重新创建虚拟环境，无效。

b.修改`LD_PRELOAD=/opt/glibc-2.14/lib/libc.so.6`，无效，并且执行其他系统命令出现Segmentation fault ，再执行 `unset LD_PRELOAD` 恢复。