from mellea.backends import model_ids
from examples.mellea.agents import DoclingWritingAgent


def main():
    model_id = model_ids.OPENAI_GPT_OSS_20B

    # tools_config = MCPConfig()
    # tools = setup_mcp_tools(config=tools_config)
    tools = []

    agent = DoclingWritingAgent(model_id=model_id, tools=tools)
    document = agent.run("Write me a document on polymers in food-packaging.")

    # Save the document
    os.makedirs("./scratch", exist_ok=True)
    fname = datetime.now().strftime("%Y%m%d_%H%M%S")

    document.save_as_markdown(filename=f"./scratch/{fname}.md", text_width=72)
    document.save_as_html(filename=f"./scratch/{fname}.html")
    document.save_as_json(filename=f"./scratch/{fname}.json")

    logger.info(f"report written to `./scratch/{fname}.html`")


if __name__ == "__main__":
    main()
