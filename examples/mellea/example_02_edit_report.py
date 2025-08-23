import os


from pathlib import Path

from docling_core.types.doc.document import (
    DoclingDocument,
)

from mellea.backends import model_ids
from examples.mellea.agents import DoclingEditingAgent, logger


def new_path(ipath: Path, ending: str) -> Path:
    return Path(str(ipath).replace(".json", ending))


def run_task(
    ipath: Path,
    opath: Path,
    task: str,
    model_id=model_ids.OPENAI_GPT_OSS_20B,
    tools: list = [],
):
    document = DoclingDocument.load_from_json(ipath)

    agent = DoclingEditingAgent(model_id=model_id, tools=tools)

    document_ = agent.run(
        task=task,
        document=document,
    )
    document.save_as_html(filename=opath)

    logger.info(f"report written to `{opath}`")


def main():
    model_id = model_ids.OPENAI_GPT_OSS_20B

    # tools_config = MCPConfig()
    # tools = setup_mcp_tools(config=tools_config)
    tools = []

    os.makedirs("./scratch", exist_ok=True)
    ipath = Path("./scratch/20250815_125216.json")

    for _ in [
        # ("Put the polymer abbreviations in a seperate column in the first table.", new_path(ipath, "_updated_table.html")),
        # ("Make the title longer!", new_path(ipath, "_updated_title.html")),
        # ("Ensure that the section-headers have the correct level!", new_path(ipath, "_updated_headings.html")),
        (
            "Expand the Introduction to three paragraphs.",
            new_path(ipath, "_updated_introduction.html"),
        ),
    ]:
        run_task(ipath=ipath, opath=_[1], task=_[0], model_id=model_id)


if __name__ == "__main__":
    main()
