## Mock 版本实现
### 整体控制流
agent.run(task) -> step() -> query() -> execute_actions() -> exit 显示退出
+ 动态 Prompt 渲染逻辑 - jinja2 条件化渲染
+ Mock Agent && Mock Environment

## 真实 API 接入
+ Responses API
+ 在顶层目录通过 Protocol 定义子类行为 -> Model、Environment、Agent
+ model: 连接client - 发起query，得到结果 - 解析tool calling
+ environment: subprocess实现
+ 通过异常来控制中止条件的触发：sp 中显示指定结束命令，env中捕获命令并给出退出信号