"""parallax_audit — comparative black-box auditing of proprietary alignment in LLMs.

Implements the framework of *Auditing Proprietary Alignment in Large Language
Models: A Comparative Framework Without a Ground-Truth Standard*.

Pipeline stages, each a CLI subcommand (``python -m parallax_audit.cli --help``):

    ingest   legacy corpus  -> canonical tables      (parallax_audit.data)
    query    questions      -> model responses       (parallax_audit.query)
    embed    responses      -> embedding matrices    (parallax_audit.score.embed)
    judge    responses      -> ordinal scores        (parallax_audit.score.judge)
    test     scores         -> hypothesis tests      (parallax_audit.stats)
    report   tests          -> LaTeX tables, figures (parallax_audit.report)

Submodules are imported lazily: ``parallax_audit.score.embed`` pulls in torch, and
``parallax_audit.query.openrouter`` pulls in the OpenAI SDK, so importing the package
root must stay cheap and dependency-free.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
