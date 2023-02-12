# This file records all the pytorch methods that I have used for this project.
## 1. Datatype Related Fucntion in Pytorch:
Datatypes in Torch: \
tensor.IntTensor
tensor.FloatTensor
### 1.1 - torch.randn(m,n):
Create a matrix with size of (m,n), based on Normal Distribution
### 1.2 - isinstance(obj, type):
Return a boolean value dependeing whether obj belongs to the type parameter.
<br>
### 1.3 - torch.Inttensor(a):
Assign an int value to varaible a in torch.
### 1.4 - torch.tensor(a):
生成一个标量, dimension is 0.
### 1.5 - torch.tensor([a]):
生成一个张量, dimension is 1.
### 1.6 - torch.set_default_tensor_type(torch.DoubleTensor)
set double_tensor as the default data type
### 1.7 - torch.empty(m,n)
create an uninitialized tensor with the size of m,n
### 1.8 - torch.rand(m,n):
均匀0到1之间的数字，比较均匀的sampling
### 1.9 - torch.rand_like(a):
所有带_like的，括号里传入的都是tensor, return a tensor that has the same size as the input tensor.
### 1.10 - torch.normal(mean=torch.full([10],0.), std=torch.arrange(1,0,-1))
生成 dimension=1 的十个数，他们的mean是0，std由0.1逐渐过渡到0最后过渡到-0.1。\
注意：torch.full([10], 0.)里的点必须要有，否则因为版本问题而报错。
### 1.11 - a[:, :, :, :]:
Tensor Indexing, the same as python indexing.
### 1.12 - torch.index_select(input, dim, index, *, output=None)
Select the tensor by specific dimension/index.\
Attention:\
**input** - the tensor input\
**dim** - the dimension in which we index\
**index** - the 1-D tensor containing the indices to index\
**Example**\
a = torch.tensor(4,3,28,28)\
indices = torch.tensor([0,2])\
torch.index_select(a, 0, indices)\
or a.index_select(0, indices)\
By code above, we are able to select the first dimension with index 0,1
### 1.13 - torch.masked_select(input, index):
similar to torch.index_select
### 1.14 - torch.take(input, index):
flatten the torch and then take the selected index tensor
### 1.15 - torch.view & toych.reshape
全连接层非常适合， 纬度变换，缺点：容易造成数据污染