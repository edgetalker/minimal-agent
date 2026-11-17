## Mock 版本实现
### 整体控制流
agent.run(task) -> step() -> query() -> execute_actions() -> exit 显示退出
+ 动态 Prompt 渲染逻辑 - jinja2 条件化渲染
+ Mock Agent && Mock Environment

## 真实 API 接入
+ Responses API - 无状态设计 - 如何维护每次调用的上下文
+ 在顶层目录通过 Protocol 定义子类行为 -> Model、Environment、Agent
+ model: 连接client - 发起query，得到结果 - 解析tool calling
+ environment: subprocess实现
+ 通过异常来控制中止条件的触发：sp 中显示指定结束命令，env中捕获命令并给出退出信号

## 统一消息结构
+ 一条完整的模型响应是一个复杂的结构化对象，最需要关注的是 output 字段
+ 将错误信息转换为一条告诉模型如何改正的消息

## Agent 执行轨迹保存
+ serialize() -> save() 

## 完整的计费实现
+ cost