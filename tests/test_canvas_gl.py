"""OpenGL viewport policy for the image canvases (app/gui/canvas_gl.py).

GL viewports are used only when the env opt-out allows them AND a one-time probe
found a usable hardware OpenGL context (UI-16, PE-05). GL viewports ask for
``gl_samples()`` MSAA samples (``ROOT_VIEWER_GL_SAMPLES``, default 4, 0 = off).
"""

import pytest

from app.gui import canvas_gl
from app.gui.canvas_view import CanvasViewMixin
from tests.qt_teardown import flush_deleted


@pytest.fixture(autouse=True)
def _not_macos(monkeypatch):
    """Default platform for these tests; the macOS tests patch it again."""
    monkeypatch.setattr(canvas_gl, "_platform", lambda: "linux")


def test_env_flag_unset_requests_opengl(monkeypatch):
    monkeypatch.delenv(canvas_gl.ENV_VAR, raising=False)
    assert canvas_gl.opengl_viewport_requested() is True


def test_macos_defaults_to_the_raster_viewport(monkeypatch):
    monkeypatch.delenv(canvas_gl.ENV_VAR, raising=False)
    monkeypatch.setattr(canvas_gl, "_platform", lambda: "darwin")
    assert canvas_gl.opengl_viewport_requested() is False


def test_macos_can_opt_in_to_opengl(monkeypatch):
    monkeypatch.setenv(canvas_gl.ENV_VAR, "1")
    monkeypatch.setattr(canvas_gl, "_platform", lambda: "darwin")
    assert canvas_gl.opengl_viewport_requested() is True


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_other_platforms_still_default_on(monkeypatch, platform):
    monkeypatch.delenv(canvas_gl.ENV_VAR, raising=False)
    monkeypatch.setattr(canvas_gl, "_platform", lambda: platform)
    assert canvas_gl.opengl_viewport_requested() is True


def test_macos_default_is_a_chosen_path_not_a_slow_one(monkeypatch):
    def probe():
        raise AssertionError("the macOS default must not probe")

    monkeypatch.setattr(canvas_gl, "opengl_support", probe)
    monkeypatch.delenv(canvas_gl.ENV_VAR, raising=False)
    monkeypatch.setattr(canvas_gl, "_platform", lambda: "darwin")
    assert canvas_gl.canvas_line() == (f"Software drawing · {canvas_gl.DARWIN_REASON}", False)
    assert canvas_gl.slow_path_notice("cocoa") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        (" Yes ", True),
        ("ON", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("", False),
    ],
)
def test_env_flag_is_read_at_call_time(monkeypatch, value, expected):
    monkeypatch.setenv(canvas_gl.ENV_VAR, value)
    assert canvas_gl.opengl_viewport_requested() is expected


@pytest.mark.parametrize(
    ("renderer", "software"),
    [
        ("Gallium 0.4 on llvmpipe (LLVM 3.6, 128 bits)", True),
        ("llvmpipe (LLVM 15.0.7, 256 bits)", True),
        ("GDI Generic", True),
        ("Microsoft Basic Render Driver", True),
        ("Google SwiftShader", True),
        ("NVIDIA GeForce RTX 5070 Ti/PCIe/SSE2", False),
        ("AMD Radeon(TM) Graphics", False),
        ("Intel(R) UHD Graphics 620", False),
        ("", False),
    ],
)
def test_software_rasterizers_are_recognised(renderer, software):
    assert canvas_gl.is_software_renderer(renderer) is software


def test_env_opt_out_never_runs_the_probe(monkeypatch):
    calls = []

    def probe():
        calls.append(1)
        return canvas_gl.GLSupport(True, "fake")

    monkeypatch.setattr(canvas_gl, "opengl_support", probe)
    monkeypatch.setenv(canvas_gl.ENV_VAR, "0")
    assert canvas_gl.opengl_viewports_allowed() is False
    assert calls == []


@pytest.mark.parametrize("supported", [True, False])
def test_allowed_follows_the_probe_when_requested(monkeypatch, supported):
    monkeypatch.setattr(canvas_gl, "opengl_support", lambda: canvas_gl.GLSupport(supported, "fake"))
    monkeypatch.setenv(canvas_gl.ENV_VAR, "1")
    assert canvas_gl.opengl_viewports_allowed() is supported


def test_probe_finds_no_opengl_on_the_offscreen_platform(app):
    if app.platformName() != "offscreen":
        pytest.skip("the offscreen platform is the one known to have no OpenGL")
    support = canvas_gl.probe_opengl_support()
    assert support.supported is False
    assert support.reason


def _fake_support(monkeypatch, support):
    monkeypatch.setattr(canvas_gl, "opengl_support", lambda: support)


def test_canvas_line_names_the_renderer(monkeypatch):
    monkeypatch.delenv(canvas_gl.ENV_VAR, raising=False)
    _fake_support(
        monkeypatch,
        canvas_gl.GLSupport(True, "hardware OpenGL", "NVIDIA GeForce RTX 5070 Ti"),
    )
    assert canvas_gl.canvas_line() == ("OpenGL · NVIDIA GeForce RTX 5070 Ti", False)


def test_canvas_line_says_the_user_turned_opengl_off(monkeypatch):
    def probe():
        raise AssertionError("an opted-out session must not probe")

    monkeypatch.setattr(canvas_gl, "opengl_support", probe)
    monkeypatch.setenv(canvas_gl.ENV_VAR, "0")
    assert canvas_gl.canvas_line() == (
        "Software drawing · turned off by ROOT_VIEWER_ENABLE_OPENGL_VIEWPORT",
        False,
    )


def test_canvas_line_flags_an_unchosen_slow_path(monkeypatch):
    monkeypatch.delenv(canvas_gl.ENV_VAR, raising=False)
    _fake_support(
        monkeypatch, canvas_gl.GLSupport(False, "software renderer 'llvmpipe'", "llvmpipe")
    )
    assert canvas_gl.canvas_line() == (
        "Software drawing (slower) · software renderer 'llvmpipe'",
        True,
    )


def test_slow_path_notice_only_off_offscreen(monkeypatch):
    monkeypatch.delenv(canvas_gl.ENV_VAR, raising=False)
    _fake_support(
        monkeypatch, canvas_gl.GLSupport(False, "software renderer 'llvmpipe'", "llvmpipe")
    )
    assert canvas_gl.slow_path_notice("windows") == (
        "Canvases draw without OpenGL (slower): software renderer 'llvmpipe'. See About."
    )
    assert canvas_gl.slow_path_notice("offscreen") is None
    assert canvas_gl.slow_path_notice("minimal") is None
    _fake_support(monkeypatch, canvas_gl.GLSupport(True, "hardware OpenGL", "GPU"))
    assert canvas_gl.slow_path_notice("windows") is None
    monkeypatch.setenv(canvas_gl.ENV_VAR, "0")
    assert canvas_gl.slow_path_notice("windows") is None


def test_the_canvas_notice_is_one_warn_toast(monkeypatch):
    from types import SimpleNamespace

    from app.gui.window import body

    calls = []
    window = SimpleNamespace(notify=lambda text, kind: calls.append((text, kind)))
    monkeypatch.setattr(canvas_gl, "slow_path_notice", lambda platform: "slow canvases")
    assert body.notify_canvas_path(window) is True
    assert calls == [("slow canvases", "warn")]
    calls.clear()
    monkeypatch.setattr(canvas_gl, "slow_path_notice", lambda platform: None)
    assert body.notify_canvas_path(window) is False
    assert calls == []


def test_no_application_means_no_probe_and_no_cache(monkeypatch):
    from PyQt6.QtGui import QGuiApplication

    def probe():
        raise AssertionError("must not probe without an application")

    monkeypatch.setattr(canvas_gl, "probe_opengl_support", probe)
    monkeypatch.setattr(QGuiApplication, "instance", staticmethod(lambda: None))
    monkeypatch.delenv(canvas_gl.ENV_VAR, raising=False)
    assert canvas_gl.opengl_support().checked is False
    assert canvas_gl.canvas_line() == ("OpenGL not checked yet", False)
    assert canvas_gl.slow_path_notice("windows") is None


# ----------------------------------------------------------------------------- MSAA
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 4),
        ("0", 0),
        ("8", 8),
        (" 2 ", 2),
        ("abc", 4),
        ("", 4),
        ("4.5", 4),
        ("64", 16),
        ("-3", 0),
    ],
)
def test_gl_samples_reads_the_env_var_at_call_time(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(canvas_gl.SAMPLES_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(canvas_gl.SAMPLES_ENV_VAR, value)
    assert canvas_gl.gl_samples() == expected


@pytest.mark.parametrize("value", [None, "0", "8"])
def test_gl_viewport_requests_the_msaa_samples(app, monkeypatch, value):
    if value is None:
        monkeypatch.delenv(canvas_gl.SAMPLES_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(canvas_gl.SAMPLES_ENV_VAR, value)
    viewport = CanvasViewMixin._new_gl_viewport()  # constructed, never shown
    try:
        assert viewport.format().samples() == canvas_gl.gl_samples()
    finally:
        viewport.deleteLater()
        flush_deleted()


# ------------------------------------------------------------- the probe's verdict
class _Format:
    def __init__(self, major: int, minor: int) -> None:
        self._version = (major, minor)

    def majorVersion(self) -> int:  # noqa: N802 - Qt name
        return self._version[0]

    def minorVersion(self) -> int:  # noqa: N802 - Qt name
        return self._version[1]


def _stub_context(monkeypatch, *, created: bool, version=(4, 6), renderer: str = "") -> None:
    """``QOpenGLContext`` as a stand-in GPU: whether a context is made, its GL version and
    ``GL_RENDERER`` (the real renderer read needs a current context, which offscreen has not)."""
    import PyQt6.QtGui

    class Context:
        def create(self) -> bool:
            return created

        def format(self) -> _Format:
            return _Format(*version)

    monkeypatch.setattr(PyQt6.QtGui, "QOpenGLContext", Context)
    monkeypatch.setattr(canvas_gl, "_renderer_name", lambda context: renderer)


@pytest.mark.parametrize(
    ("created", "version", "renderer", "supported", "reason"),
    [
        (False, (4, 6), "", False, "no OpenGL context on this platform"),
        (True, (1, 1), "", False, "OpenGL 1.1 is below 2.0"),
        (
            True,
            (3, 3),
            "llvmpipe (LLVM 15.0.7, 256 bits)",
            False,
            "software renderer 'llvmpipe (LLVM 15.0.7, 256 bits)'",
        ),
        (True, (4, 6), "NVIDIA GeForce RTX 5070 Ti/PCIe/SSE2", True, "hardware OpenGL"),
    ],
    ids=["no-context", "gl-1", "software", "hardware"],
)
def test_the_probe_judges_the_context_it_made(
    monkeypatch, created, version, renderer, supported, reason
):
    _stub_context(monkeypatch, created=created, version=version, renderer=renderer)
    support = canvas_gl.probe_opengl_support()
    assert (support.supported, support.reason) == (supported, reason)
    assert support.renderer == (renderer if created and version >= (2, 0) else "")


@pytest.fixture
def fresh_probe_cache():
    """The per-process probe cache, empty before and after the test."""
    canvas_gl._probed_support.cache_clear()
    yield
    canvas_gl._probed_support.cache_clear()


@pytest.mark.parametrize(
    ("env", "supported", "level", "message"),
    [
        (None, False, "WARNING", "OpenGL canvas viewports off (slower drawing): fake reason"),
        ("0", False, "INFO", "OpenGL canvas viewports off: fake reason"),
        (None, True, "INFO", "OpenGL canvas viewports available (GPU)"),
    ],
    ids=["requested-unsupported", "opted-out", "supported"],
)
def test_the_probe_runs_once_per_process_and_warns_only_when_gl_was_wanted(
    app, monkeypatch, caplog, fresh_probe_cache, env, supported, level, message
):
    calls = []

    def probe():
        calls.append(1)
        return canvas_gl.GLSupport(supported, "fake reason", "GPU" if supported else "")

    monkeypatch.setattr(canvas_gl, "probe_opengl_support", probe)
    if env is None:
        monkeypatch.delenv(canvas_gl.ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(canvas_gl.ENV_VAR, env)
    with caplog.at_level("INFO", logger=canvas_gl.__name__):
        first, second = canvas_gl.opengl_support(), canvas_gl.opengl_support()

    assert first is second and first.supported is supported and calls == [1]
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [(level, message)]
