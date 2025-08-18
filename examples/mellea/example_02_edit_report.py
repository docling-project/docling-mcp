import os

from pathlib import Path

from docling_core.types.doc.document import (
    DoclingDocument,
)

from mellea.backends import model_ids
from examples.mellea.agents import DoclingEditingAgent


def main():
    model_id = model_ids.OPENAI_GPT_OSS_20B

    # tools_config = MCPConfig()
    # tools = setup_mcp_tools(config=tools_config)
    tools = []

    document = DoclingDocument.load_from_json(Path("./scratch/20250815_125216.json"))

    agent = DoclingEditingAgent(model_id=model_id, tools=tools)

    document_ = agent.run(
        "Put the polymer abbreviations in a seperate column in the first table.",
        document=document,
    )

    document_ = agent.run(
        "Expand the Introduction to three paragraphs.", document=document
    )

    document_ = agent.run("Make the title longer!", document=document)

    document_ = agent.run(
        "Ensure that the section-headers have the correct level!", document=document
    )

    # Save the document
    """
    os.makedirs("./scratch", exist_ok=True)
    fname = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    document.save_as_markdown(filename=f"./scratch/{fname}.md", text_width=72)
    document.save_as_html(filename=f"./scratch/{fname}.html")
    document.save_as_json(filename=f"./scratch/{fname}.json")
    
    logger.info(f"report written to `./scratch/{fname}.html`")
    """


if __name__ == "__main__":
    main()
