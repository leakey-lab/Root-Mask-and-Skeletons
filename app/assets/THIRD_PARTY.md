# Third-party files in `app/assets/`

Dash serves this folder at `/assets/` and auto-includes every `.css` / `.js` file in
alphabetical order, so the vendored stylesheets are prefixed `00_` / `01_` and the
SPROUTS theme (`style.css`) is applied last. The SPROUTS scripts `sprouts-autosize.js` and
`sprouts-uirevision.js` must sort after `plotly-basic.min.js`, which they extend. Nothing
here is fetched from the internet at runtime: the dashboards render identically on an
offline lab or field PC.

| File | Upstream | Version | License | SHA-256 (as vendored) |
|---|---|---|---|---|
| `00_bootswatch-darkly.min.css` | https://cdn.jsdelivr.net/npm/bootswatch@5.3.6/dist/darkly/bootstrap.min.css (Bootswatch "Darkly", built on Bootstrap) | Bootswatch 5.3.6 / Bootstrap 5.3.6 | MIT — `licenses/bootswatch-MIT.txt`, `licenses/bootstrap-MIT.txt` | `8dd768cfc9a9dfd14ece38c806683451d1b2301e712270ca23bb54c0ea54f417` |
| `fonts/IBMPlexSans-Regular-Latin1.woff2` | https://cdn.jsdelivr.net/npm/@ibm/plex-sans@1.1.0/fonts/split/woff2/ | @ibm/plex-sans 1.1.0 | SIL OFL 1.1 — `licenses/IBM-Plex-OFL-1.1.txt` | `b5ad7bd39f996144915f0ad9849a90183b27d8c28ad97ed98af5b1bebc51f6b1` |
| `fonts/IBMPlexSans-Medium-Latin1.woff2` | same | @ibm/plex-sans 1.1.0 | SIL OFL 1.1 | `b5610af04d0d4b5a14a621d96d974b993e945a065db1a8861918f69ef9321934` |
| `fonts/IBMPlexSans-SemiBold-Latin1.woff2` | same | @ibm/plex-sans 1.1.0 | SIL OFL 1.1 | `fff0ab3a88b0b4aa0b693e4f0201359a15183b08e3fa5696d1918d8f0ade8ad5` |
| `fonts/IBMPlexMono-Regular-Latin1.woff2` | https://cdn.jsdelivr.net/npm/@ibm/plex-mono@2.5.0/fonts/split/woff2/ | @ibm/plex-mono 2.5.0 | SIL OFL 1.1 | `e8993d946649b9d01abb1ed06d574b19d8ea3e66b5c3948602db335c44c18e56` |
| `fonts/IBMPlexMono-Medium-Latin1.woff2` | same | @ibm/plex-mono 2.5.0 | SIL OFL 1.1 | `41201b658a328b9d00368215c2f1102770f80b15952ab82631e4006255e6365d` |
| `plotly-basic.min.js` | https://cdn.jsdelivr.net/npm/plotly.js-basic-dist-min@4.1.1/plotly-basic.min.js (plotly.js "basic" partial bundle: scatter, bar and pie traces; byte-identical to the npm tarball, `sha512-1oNF8PeQx2hv/y7tqEVWKD3JGwn/MGEMQ8mL9HUwsaqG9FGy/OU5FHUHvzqbqxN4mnwZGI9abTiE775C4kkjMQ==`) | plotly.js 4.1.1 | MIT — `licenses/plotly.js-MIT.txt` (the licences of the libraries it bundles are kept in the file's comments) | `99784e3abfb07b142a15e521a439e6d9addceadb3f79553fd24c72774f0649ba` |

The SHA-256 values are those of the committed files. `.gitattributes` checks
`plotly-basic.min.js` out with LF line endings whatever `core.autocrlf` says, so the copy
Dash serves and a PyInstaller build packs has these bytes on Windows too;
`tests/test_viz_dash_plotly_bundle.py` compares the file with its row's SHA-256.

## Why a plotly.js bundle is vendored

dcc.Graph uses `window.Plotly` when the page already defines it. Without this file it
fetched the full 4.8 MB `plotly.min.js` (served by plotly.py) after the page had
rendered and spent ~0.3 s evaluating it on every dashboard open; the basic bundle is
1.2 MB and is loaded with the page. On a 46k-row synthetic trial the first chart of an
open came 208-291 ms sooner in every pairing measured (cold and warm opens, with and
without the prewarm; median of 5 each). It must be the plotly.js version plotly.py
builds figures for, and the dashboards may only use its trace types:
`tests/test_viz_dash_plotly_bundle.py` checks both.

## Local modifications

- `00_bootswatch-darkly.min.css`: the upstream file (SHA-256
  `075e3cf8031ccbe320d0399b393bbabd21d7bf1b0921507bd65d2f8fbf374b83`) starts with
  `@import url(https://fonts.googleapis.com/css2?family=Lato...)`. That single rule is
  replaced by a comment so no request leaves the machine. Lato is never used: the
  SPROUTS theme sets the Bootstrap font variables to IBM Plex (`style.css`). The rest
  of the file, including its copyright headers, is byte-identical to upstream.
- The fonts are unmodified. Only the weights the theme uses are vendored (Sans 400 /
  500 / 600, Mono 400 / 500), as IBM's `Latin1` subsets: ASCII, Latin-1 (`±`, `²`,
  `µ`, `·`) and common punctuation (`–`, `—`, `…`). Other characters fall back to
  the system font, as they did with the Google Fonts `latin` subset.

## Updating

Download the new upstream files, repeat the modification above, update this table
(versions and SHA-256) and run `pytest tests/test_viz_dash_offline_assets.py`, which
fails if any served stylesheet references a remote host.

After a plotly.py upgrade, download `plotly.js-basic-dist-min@<version>` for the
plotly.js version `plotly.offline.get_plotlyjs_version()` reports, replace
`plotly-basic.min.js` unmodified, update its row and run
`pytest tests/test_viz_dash_plotly_bundle.py tests/test_viz_dash_autosize_guard.py
tests/test_viz_dash_uirevision.py`. Read plotly.js's release notes for changed defaults: the
dashboards set plotly.js 3's back where plotly.js 4 changed them (`showSendToCloud` and
`doubleClickDelay` in `dash_shell.graph_config`); the bar value labels take plotly.js 4's
own contrast colour (owner decision #51, 2026-10-31). If a dashboard needs a trace type the basic bundle lacks,
delete the file instead: dcc.Graph then loads the full bundle.

## Qt UI fonts (`resources/fonts/`)

The desktop UI (the Qt stylesheet and `app/design/tokens.py`) names IBM Plex Sans and IBM
Plex Mono first. Qt cannot read the `.woff2` files above (they stay for the dashboards), so
the same releases ship as TTF files in `resources/fonts/`, which `app/gui/widgets/fonts.py`
registers at start-up, before the stylesheet loads. They are copied unmodified from the
`fonts/complete/ttf/` folder of IBM's GitHub release archives:

- IBM Plex Sans 1.1.0: https://github.com/IBM/plex/releases/download/%40ibm%2Fplex-sans%401.1.0/ibm-plex-sans.zip
  (SHA-256 `fb365d910566e6d199cc2c15579a7dd9a267128e18431a394ed81f1970c69200`; GitHub lists
  no digest for this 2024 asset, so it was checked against the npm package
  `@ibm/plex-sans@1.1.0`, whose registry checksum matched: its 171 files are byte-identical
  in the archive).
- IBM Plex Mono 2.5.0: https://github.com/IBM/plex/releases/download/%40ibm%2Fplex-mono%402.5.0/ibm-plex-mono.zip
  (SHA-256 `6d23f01257663d8cc49a0d64c22ced630b79e0e2a0ac08a0da86e9a38bbc481c`, the digest
  GitHub lists for the asset).

Only the weights the UI uses are shipped: Sans 400 / 500 / 600 / 700 and Mono 400 / 600
(the 800 of two titles draws in Bold). The fonts are complete (no subset). Licence: SIL OFL
1.1, `resources/fonts/IBM-Plex-OFL-1.1.txt` (the archives' `LICENSE.txt`, the same text as
`licenses/IBM-Plex-OFL-1.1.txt`).

| File | Version | License | SHA-256 (as vendored) |
|---|---|---|---|
| `resources/fonts/IBMPlexSans-Regular.ttf` | IBM Plex Sans 1.1.0 | SIL OFL 1.1 | `975dcda37d80f038dcd143c22e33ca2d97a0cc5a929aace1c749153b0fe1afa5` |
| `resources/fonts/IBMPlexSans-Medium.ttf` | IBM Plex Sans 1.1.0 | SIL OFL 1.1 | `331c8639d7598b2cde62a911a71db195e30cb655cd6bdf2e324a7e984955f907` |
| `resources/fonts/IBMPlexSans-SemiBold.ttf` | IBM Plex Sans 1.1.0 | SIL OFL 1.1 | `a20caf8286023a6a7a85e40b1d2a4ae9fc3e3b1f9eda8f4c542dd4986af67bb1` |
| `resources/fonts/IBMPlexSans-Bold.ttf` | IBM Plex Sans 1.1.0 | SIL OFL 1.1 | `9e6c74a889a700d707613d24548fe4ffa6bc59559a0689d2cf9e133bdcdafb2f` |
| `resources/fonts/IBMPlexMono-Regular.ttf` | IBM Plex Mono 2.5.0 | SIL OFL 1.1 | `7c6fbddca4b700be918f5f6183d9bd4464fa427fe435f0b480d77fe2bb8c5a43` |
| `resources/fonts/IBMPlexMono-SemiBold.ttf` | IBM Plex Mono 2.5.0 | SIL OFL 1.1 | `f04d7c488ddf7d1fa99f2574efc3406ea4cbe17bb1af3a1ab960f84d0c96a172` |

To update: download the new release archives, check their SHA-256 against the release,
copy the same TTFs in unmodified, update this table and run `pytest tests/test_ui_fonts.py
tests/test_window_width_budget.py tests/test_shell_chrome_topbar.py` (a new font changes
text widths).

Code outside this folder: see `THIRD_PARTY_NOTICES.md` at the repository root.
