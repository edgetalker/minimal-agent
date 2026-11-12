from agent import DefaultAgent
from mock import MockModel, MockEnvironment

def main():
    agent = DefaultAgent(
        model = MockModel(),
        env = MockEnvironment(),
        system_template = "You are a coding agent.",
        instance_template = "Please solve: {{ task }}",
        step_limit = 10,
    )

    result = agent.run(task="Print hello")
    print("Result:", result)
    print("Model calls:", agent.n_calls)
    print("Messages:")

    for message in agent.messages:
        print(message)


if __name__ == "__main__":
    main()