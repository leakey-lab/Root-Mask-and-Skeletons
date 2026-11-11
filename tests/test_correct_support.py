"""The prompt answers of the Correct page fixtures (``tests/correct_support.answer_prompts``):
each box shown is recorded and answered from a queue, and a box nobody queued an answer for
fails the test."""

import os

import pytest

from tests.correct_support import SAVE, draw_line, paint, same_path

pytestmark = pytest.mark.usefixtures("correct_guards")


def test_answers_answer_the_prompt_of_a_save_then_open(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    draw_line(editor)
    answers.will(SAVE)

    editor.load_image(project.b, project.folder)

    assert len(answers.seen) == 1 and "a.png" in answers.seen[0][1]
    assert os.path.isfile(project.correction())
    assert same_path(editor.current_image_path, project.b) and not editor.has_unsaved_changes()


def test_an_unanswered_prompt_fails_the_test(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (60, 100))
    with pytest.raises(pytest.fail.Exception, match="unexpected prompt"):
        editor.maybe_save_changes()
