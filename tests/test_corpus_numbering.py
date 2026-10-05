"""Unit tests for line-numbered ask corpus assembly."""

from __future__ import annotations

from coworker import cli


def test_corpus_prefixes_every_line_with_its_number(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("".join(f"line {n}\n" for n in range(1, 12)))

    corpus = cli._build_corpus([str(doc)])

    assert corpus.startswith(f"<file path='{doc}' lines='numbered'>\n")
    assert " 1| line 1\n" in corpus
    assert "11| line 11\n</file>" in corpus


def test_number_lines_keeps_blank_lines_addressable():
    assert cli._number_lines("a\n\nc") == "1| a\n2| \n3| c"
