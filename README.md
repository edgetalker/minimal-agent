## Mock 版本实现
### 整体控制流
agent.run(task) -> step() -> query() -> execute_actions() -> exit 显示退出
+ 动态 Prompt 渲染逻辑 - jinja2 条件化渲染
+ Mock Agent && Mock Environment