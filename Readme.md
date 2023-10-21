此项目的目的是督促自己持续学习
直到完成属于自己的“百科全书”

目前正在更新“使用Numpy实现Neural Network”部分...

更新日志：
1. 2023/09/25 增加激活函数的Sigmoid的forward和grad部分，实现Softmax的fordward部分，增加linear layer的forward部分，并做了基础测试
2. 2023/09/26 更新单个输入样本分别在 1) 单个linear layer； 2) 单个linear layer加Sigmoid函数情况下的backward部分，并做了基础测试
3. 2023/09/28 增加单位函数的forward和grad部分，更新多个样本在Fully Connected Network中的backward部分，并做了基础测试
4. 2023/09/30 增加了Fully Connected Network的Forward计算过程，保存为draw.io和pdf文件
5. 2023/10/08 更新了Fully Connected Network的Backward计算过程，更新文件为fully_connected_neural_network.drawio
6. 2023/10/11 更新了Gradient Descent基本运算的实现，更新了Mean Squared Error损失函数的实现，添加了Gradient Descent基本功能的测试函数；新增losses.py，optimiaer.py文件，更新了test.py文件
7. 2023/10/14 新增了ParameterBasic类用于表示可学习参数；新增了ModuleBase类作为Module组件和Neural Network的基类，并重载了class内属性赋值方法，实现了Module可学习参数列表动态记录的过程；实现了ModuleBase的backward方法；增加了相关测试函数。新增parameter.py、model.py，更新了layer.py、test.py
8. 2023/10/21 优化了Sigmoid激活函数出现大数溢出的问题；ModuleBase类中增加了对于Module Sequence参数注册的情况；新增了Convolutional Layer 2D的基本前向传播运算，并做了基础测试。修改了activations.py、modle.py、layers.py
