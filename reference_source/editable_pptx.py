"""Dependency-free editable PowerPoint export for Matplotlib figures.

Each output is a one-slide PPTX whose slide size matches the saved Matplotlib
canvas.  By default, text is rendered as font-independent SVG glyph outlines
and handed to Windows PowerPoint for conversion with its ``SVGEdit`` command.
The resulting labels are editable vector geometry, not string-editable
PowerPoint text.  A PNG fallback remains available for the explicitly requested
non-native compatibility mode.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid
import warnings
from xml.etree import ElementTree
from xml.sax.saxutils import escape
import zipfile


EMU_PER_INCH = 914_400
_REQUIRED_PPTX_PARTS = {
    "[Content_Types].xml",
    "_rels/.rels",
    "docProps/app.xml",
    "docProps/core.xml",
    "ppt/presentation.xml",
    "ppt/_rels/presentation.xml.rels",
    "ppt/slideLayouts/slideLayout1.xml",
    "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
    "ppt/slideMasters/slideMaster1.xml",
    "ppt/slideMasters/_rels/slideMaster1.xml.rels",
    "ppt/slides/slide1.xml",
    "ppt/slides/_rels/slide1.xml.rels",
    "ppt/theme/theme1.xml",
}
_FALLBACK_MEDIA_PARTS = {
    "ppt/media/image1.png",
    "ppt/media/image1.svg",
}
_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"
_SVG_GROUP_TAG = f"{{{_SVG_NAMESPACE}}}g"
_XLINK_HREF_ATTR = f"{{{_XLINK_NAMESPACE}}}href"
_ARIAL_FONT_PATHS = {
    'regular': "__FONT_ROOT__/arial.ttf",
    'bold': "__FONT_ROOT__/arialbd.ttf",
}
_ARIAL_FONT_FILES = tuple(_ARIAL_FONT_PATHS.values())
_ARIAL_FONT_FILES_REGISTERED = False


# Shared output validation, transactional promotion, and review typography.


def validate_final_figure_canvas(fig, tolerance_px=2.0):
    """Check that key layout artists are within the designed canvas before saving."""
    import numpy as np

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    offenders = []

    def _check(artist, label):
        if artist is None or not artist.get_visible():
            return
        try:
            bbox = artist.get_window_extent(renderer)
        except Exception:
            return
        if bbox is None or not np.all(np.isfinite(bbox.extents)):
            return
        if (bbox.x0 < canvas.x0 - tolerance_px or bbox.y0 < canvas.y0 - tolerance_px
                or bbox.x1 > canvas.x1 + tolerance_px or bbox.y1 > canvas.y1 + tolerance_px):
            offenders.append(
                f"{label}: ({bbox.x0:.1f}, {bbox.y0:.1f}, {bbox.x1:.1f}, {bbox.y1:.1f})"
            )

    from matplotlib.text import Text
    from matplotlib.legend import Legend
    for artist in fig.findobj(match=lambda obj: isinstance(obj, (Text, Legend))):
        if isinstance(artist, Text) and not artist.get_text().strip():
            continue
        label = artist.__class__.__name__
        if isinstance(artist, Text):
            label += f"[{artist.get_text()[:60]!r}]"
        _check(artist, label)

    if offenders:
        details = "\n  - ".join(offenders[:20])
        raise RuntimeError(
            "Figure layout extends beyond the canvas. Fix the layout instead of cropping/resizing:\n  - "
            + details
        )
    return True


def _validate_staged_final_png(path):
    """Require a complete, readable PNG before pair promotion."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f'PNG staging output is missing or empty: {path}')
    from PIL import Image as _PILImage
    with _PILImage.open(path) as image:
        image_format = image.format
        width, height = image.size
        image.verify()
    if image_format != 'PNG' or width < 1 or height < 1:
        raise RuntimeError(
            f'PNG staging output validation failed: format={image_format}, size={width}x{height}'
        )
    return width, height


def _validate_staged_final_pptx(path):
    """Require a valid one-slide native-shape PPTX before pair promotion."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f'PPTX staging output is missing or empty: {path}')
    required = {
        '[Content_Types].xml', 'ppt/presentation.xml',
        'ppt/slides/slide1.xml',
        'ppt/theme/theme1.xml',
        'ppt/slideMasters/slideMaster1.xml',
        'ppt/slideLayouts/slideLayout1.xml',
    }
    with zipfile.ZipFile(path, 'r') as archive:
        names = set(archive.namelist())
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f'Required PPTX staging output parts are missing: {missing}')
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f'PPTX staging output ZIP CRC failed: {bad_member}')
        xml_roots = {}
        forbidden_font_members = []
        for member in sorted(
            name for name in names
            if name.casefold().endswith(('.xml', '.rels'))
        ):
            payload = archive.read(member)
            try:
                root = ElementTree.fromstring(payload)
            except ElementTree.ParseError as exc:
                raise RuntimeError(
                    f'PPTX staging output XML parsing failed: {member}'
                ) from exc
            xml_roots[member] = root
            values = []
            for element in root.iter():
                values.extend(str(value) for value in element.attrib.values())
                if element.text:
                    values.append(element.text)
                if element.tail:
                    values.append(element.tail)
            searchable = (
                payload.decode('utf-8', errors='ignore')
                + '\n' + '\n'.join(values)
            ).casefold()
            if any(
                token in searchable
                for token in ('helvetica', 'dejavu sans', 'dejavusans')
            ):
                forbidden_font_members.append(member)
        if forbidden_font_members:
            raise RuntimeError(
                'PPTX staging output XML contains forbidden Helvetica/DejaVu Sans fonts: '
                f'{forbidden_font_members}'
            )
        slide_root = xml_roots['ppt/slides/slide1.xml']
        theme_root = xml_roots['ppt/theme/theme1.xml']
    namespaces = {
        'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
        'asvg': 'http://schemas.microsoft.com/office/drawing/2016/SVG/main',
    }
    font_scheme = theme_root.find('.//a:fontScheme', namespaces)
    if font_scheme is None:
        raise RuntimeError('PPTX staging output theme1.xml lacks fontScheme.')
    invalid_theme_fonts = []
    for font_group_name in ('majorFont', 'minorFont'):
        font_group = font_scheme.find(f'a:{font_group_name}', namespaces)
        if font_group is None:
            invalid_theme_fonts.append(f'{font_group_name}=missing')
            continue
        for face_name in ('latin', 'ea', 'cs'):
            face = font_group.find(f'a:{face_name}', namespaces)
            typeface = None if face is None else face.attrib.get('typeface')
            if typeface != 'Arial':
                invalid_theme_fonts.append(
                    f'{font_group_name}/{face_name}={typeface!r}'
                )
    if invalid_theme_fonts:
        raise RuntimeError(
            'PPTX staging output theme1.xml violates the Arial contract: '
            + ', '.join(invalid_theme_fonts)
        )
    native_shape_count = sum(
        len(slide_root.findall(query, namespaces))
        for query in ('.//p:sp', './/p:grpSp', './/p:cxnSp', './/p:graphicFrame')
    )
    if slide_root.findall('.//asvg:svgBlip', namespaces):
        raise RuntimeError('PPTX staging output contains unconverted SVG pictures.')
    if slide_root.findall('.//a:t', namespaces):
        raise RuntimeError('PPTX staging output contains text nodes rather than glyph outlines.')
    if native_shape_count < 1:
        raise RuntimeError('PPTX staging output contains no editable native shapes.')
    return native_shape_count


def _new_final_pair_transaction_paths(png_path, pptx_path):
    """Allocate call-unique hidden staging and backup names beside the targets."""
    for _ in range(16):
        token = uuid.uuid4().hex
        prefix = f'.final-pair-{token}'
        paths = {
            'png_stage': png_path.with_name(f'{prefix}.stage.png'),
            'pptx_stage': pptx_path.with_name(f'{prefix}.stage.pptx'),
            'png_backup': png_path.with_name(f'{prefix}.backup.png'),
            'pptx_backup': pptx_path.with_name(f'{prefix}.backup.pptx'),
        }
        if not any(path.exists() for path in paths.values()):
            return paths
    raise RuntimeError('Could not allocate unique Figure pair staging/backup paths.')


def _promote_staged_final_pair(paths, png_path, pptx_path, original_exists):
    """Promote both staged files, restoring the exact prior pair on failure."""
    states = [
        {
            'kind': 'PNG', 'stage': paths['png_stage'], 'final': png_path,
            'backup': paths['png_backup'], 'had_original': original_exists[0],
            'backed_up': False, 'promoted': False,
        },
        {
            'kind': 'PPTX', 'stage': paths['pptx_stage'], 'final': pptx_path,
            'backup': paths['pptx_backup'], 'had_original': original_exists[1],
            'backed_up': False, 'promoted': False,
        },
    ]
    try:
        changed_during_staging = [
            str(state['final']) for state in states
            if state['final'].exists() != state['had_original']
        ]
        if changed_during_staging:
            raise RuntimeError(
                'Figure pair target state changed externally during staging: '
                + ', '.join(changed_during_staging)
            )
        for state in states:
            if state['had_original']:
                os.replace(state['final'], state['backup'])
                state['backed_up'] = True
        for state in states:
            if state['final'].exists():
                raise FileExistsError(
                    f"Target was recreated immediately before Figure pair promotion: {state['final']}"
                )
            os.replace(state['stage'], state['final'])
            state['promoted'] = True
    except BaseException as exc:
        rollback_errors = []
        for state in reversed(states):
            promotion_completed = state['promoted'] or (
                not state['stage'].exists() and state['final'].exists()
            )
            if promotion_completed and state['final'].exists():
                try:
                    state['final'].unlink()
                except BaseException as rollback_exc:
                    rollback_errors.append(
                        f"Failed to remove new {state['kind']} ({state['final']}): {rollback_exc}"
                    )
        for state in states:
            if state['backup'].exists():
                try:
                    if state['final'].exists():
                        raise FileExistsError(
                            f"Rollback target already exists: {state['final']}"
                        )
                    os.replace(state['backup'], state['final'])
                except BaseException as rollback_exc:
                    rollback_errors.append(
                        f"Failed to restore original {state['kind']} ({state['backup']}): {rollback_exc}"
                    )
        if rollback_errors:
            raise RuntimeError(
                'Figure pair promotion failed and rollback was incomplete. '
                'Unique backups were preserved:\n  - ' + '\n  - '.join(rollback_errors)
            ) from exc
        raise
    for state in states:
        if state['backup'].exists():
            try:
                state['backup'].unlink()
            except OSError as cleanup_exc:
                warnings.warn(
                    f"Failed to clean up unique backup after Figure pair commit: {state['backup']}: "
                    f"{cleanup_exc}", RuntimeWarning,
                )


def _configure_font(regular_path: Path, bold_path: Path) -> str:
    import matplotlib as mpl
    from matplotlib import font_manager

    if not regular_path.is_file() or not bold_path.is_file():
        raise FileNotFoundError(
            "Regular/Bold font files are required for a new review Figure"
        )
    font_manager.fontManager.addfont(str(regular_path))
    font_manager.fontManager.addfont(str(bold_path))
    family = font_manager.FontProperties(fname=str(regular_path)).get_name()
    bold_family = font_manager.FontProperties(fname=str(bold_path)).get_name()
    if family != bold_family:
        raise RuntimeError(f"Font family mismatch: {family!r} versus {bold_family!r}")
    if family != "Arial":
        raise RuntimeError(f"Strict Arial was requested; resolved {family!r}")
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 9,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
        }
    )
    resolved = font_manager.findfont(
        font_manager.FontProperties(family=family), fallback_to_default=False
    )
    if Path(resolved).resolve() != regular_path.resolve():
        raise RuntimeError(f"Requested font did not resolve to {regular_path}: {resolved}")
    resolved_bold = font_manager.findfont(
        font_manager.FontProperties(family=family, weight="bold"),
        fallback_to_default=False,
    )
    if Path(resolved_bold).resolve() != bold_path.resolve():
        raise RuntimeError(
            f"Requested bold font did not resolve to {bold_path}: {resolved_bold}"
        )
    return family


def _normalized_font_path(path: str) -> str:
    """Return the canonical form used for exact approved-font comparisons."""
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _register_matplotlib_arial_fonts() -> tuple[str, ...]:
    """Register available Arial faces once, before Matplotlib creates text."""
    global _ARIAL_FONT_FILES_REGISTERED
    existing = tuple(path for path in _ARIAL_FONT_FILES if os.path.isfile(path))
    if not _ARIAL_FONT_FILES_REGISTERED:
        from matplotlib import font_manager
        for path in existing:
            font_manager.fontManager.addfont(path)
        _ARIAL_FONT_FILES_REGISTERED = True
    return existing


def _validate_matplotlib_arial_font_contract(fig=None) -> dict[str, str]:
    """Require real Arial Regular/Bold faces and Arial for visible figure text."""
    from matplotlib import font_manager, rcParams
    from matplotlib.text import Text

    _register_matplotlib_arial_fonts()
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']

    resolved = {}
    for face_name, weight in (('regular', 'normal'), ('bold', 'bold')):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=['Arial'], weight=weight),
                fallback_to_default=False,
            )
            face = font_manager.get_font(path)
        except Exception as exc:
            raise RuntimeError(f'Could not resolve Arial {face_name.title()} font.') from exc
        family_name = str(face.family_name)
        style_name = str(face.style_name)
        style_lower = style_name.casefold()
        valid_style = (
            style_lower in {'regular', 'normal', 'roman'}
            if face_name == 'regular'
            else 'bold' in style_lower and 'italic' not in style_lower and 'oblique' not in style_lower
        )
        actual_path = _normalized_font_path(path)
        expected_path = _normalized_font_path(_ARIAL_FONT_PATHS[face_name])
        if (
            family_name.casefold() != 'arial'
            or not valid_style
            or actual_path != expected_path
        ):
            raise RuntimeError(
                f'Arial {face_name.title()} font contract failed: '
                f'family={family_name!r}, style={style_name!r}, '
                f'path={actual_path}, expected={expected_path}'
            )
        resolved[face_name] = actual_path

    if fig is None:
        return resolved

    offenders = []
    for artist in fig.findobj(match=lambda obj: isinstance(obj, Text)):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        properties = artist.get_fontproperties()
        font_style = str(properties.get_style()).casefold()
        if font_style != 'normal':
            offenders.append(
                f'{artist.get_text()[:60]!r}: unsupported font style={font_style!r}; '
                'visible italic/oblique text is outside the Arial Regular/Bold contract'
            )
            continue
        try:
            path = font_manager.findfont(properties, fallback_to_default=False)
            face = font_manager.get_font(path)
            family_name = str(face.family_name)
            style_name = str(face.style_name)
        except Exception as exc:
            offenders.append(f'{artist.get_text()[:60]!r}: unresolved ({exc})')
            continue
        weight = properties.get_weight()
        is_bold = (
            ('bold' in weight.casefold() or weight.casefold() in {'demi', 'heavy', 'black'})
            if isinstance(weight, str)
            else float(weight) >= 600
        )
        expected_face = 'bold' if is_bold else 'regular'
        actual_path = _normalized_font_path(path)
        expected_path = resolved[expected_face]
        style_lower = style_name.casefold()
        valid_style = (
            'bold' in style_lower and 'italic' not in style_lower and 'oblique' not in style_lower
            if is_bold
            else style_lower in {'regular', 'normal', 'roman'}
        )
        if (
            family_name.casefold() != 'arial'
            or not valid_style
            or actual_path != expected_path
        ):
            offenders.append(
                f'{artist.get_text()[:60]!r}: family={family_name!r}, '
                f'style={style_name!r}, path={actual_path}, expected={expected_path}'
            )
    if offenders:
        raise RuntimeError(
            'Figure contains visible text that does not resolve to Arial:\n  - '
            + '\n  - '.join(offenders[:20])
        )
    return resolved


def _xml_attr(value: object) -> str:
    return escape(str(value), {'"': "&quot;", "'": "&apos;"})


def _length_to_inches(value: str | None) -> float | None:
    if not value:
        return None
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*(in|pt|px|cm|mm)?\s*",
        str(value),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    number = float(match.group(1))
    unit = (match.group(2) or "px").lower()
    return {
        "in": number,
        "pt": number / 72.0,
        "px": number / 96.0,
        "cm": number / 2.54,
        "mm": number / 25.4,
    }[unit]


def _svg_size_inches(svg_bytes: bytes, fig) -> tuple[float, float]:
    root = ElementTree.fromstring(svg_bytes)
    width = _length_to_inches(root.attrib.get("width"))
    height = _length_to_inches(root.attrib.get("height"))
    if width and height and width > 0 and height > 0:
        return width, height

    view_box = root.attrib.get("viewBox", "").replace(",", " ").split()
    if len(view_box) == 4:
        try:
            view_width = float(view_box[2])
            view_height = float(view_box[3])
        except ValueError:
            view_width = view_height = 0.0
        if view_width > 0 and view_height > 0:
            fig_width, fig_height = (float(v) for v in fig.get_size_inches())
            return fig_width, fig_width * view_height / view_width

    fig_width, fig_height = (float(v) for v in fig.get_size_inches())
    return fig_width, fig_height


def _dense_artist_size(artist) -> int:
    """Return an approximate point/cell count for a Matplotlib data artist."""
    try:
        offsets = artist.get_offsets()
    except (AttributeError, TypeError):
        offsets = None
    if offsets is not None:
        try:
            offset_count = len(offsets)
        except TypeError:
            offset_count = 0
        if offset_count > 1:
            return int(offset_count)

    try:
        data_array = artist.get_array()
    except (AttributeError, TypeError):
        data_array = None
    if data_array is not None:
        try:
            array_size = int(data_array.size)
        except (AttributeError, TypeError, ValueError):
            array_size = 0
        if array_size > 1:
            return array_size

    try:
        x_data = artist.get_xdata(orig=False)
    except (AttributeError, TypeError):
        x_data = None
    if x_data is not None:
        try:
            x_count = len(x_data)
        except TypeError:
            x_count = 0
        if x_count > 1:
            return int(x_count)

    try:
        paths = artist.get_paths()
    except (AttributeError, TypeError):
        paths = None
    if paths:
        path_vertex_count = 0
        for path in paths:
            try:
                path_vertex_count += len(path.vertices)
            except (AttributeError, TypeError):
                continue
        return int(path_vertex_count)
    return 0


def _find_dense_artists(fig, point_threshold: int) -> list:
    """Find data artists that would create an impractical number of shapes."""
    dense_artists = []
    for artist in fig.findobj():
        already_rasterized = bool(
            artist.get_rasterized()
            if hasattr(artist, "get_rasterized")
            else False
        )
        if not already_rasterized and _dense_artist_size(artist) < point_threshold:
            continue
        if not hasattr(artist, "get_visible") or not artist.get_visible():
            continue
        dense_artists.append(artist)
    return dense_artists


def _set_artists_visible(artists, visible: bool) -> list:
    changed = []
    for artist in artists:
        if not hasattr(artist, "get_visible") or not hasattr(artist, "set_visible"):
            continue
        original = artist.get_visible()
        if original == visible:
            continue
        artist.set_visible(visible)
        changed.append((artist, original))
    return changed


def _restore_artist_visibility(changed) -> None:
    for artist, original in reversed(changed):
        artist.set_visible(original)


def _render_dense_overlay_png(
    fig,
    dense_artists: list,
    save_kwargs: dict,
    *,
    bbox_inches=None,
) -> bytes:
    """Render dense data artists alone on a transparent full-canvas PNG."""
    dense_set = set(dense_artists)
    keep_axes = {artist.axes for artist in dense_artists if getattr(artist, "axes", None)}
    hide_artists = []
    for artist in fig.findobj():
        if artist is fig or artist in dense_set or artist in keep_axes:
            continue
        hide_artists.append(artist)

    changed = _set_artists_visible(hide_artists, False)
    png_buffer = BytesIO()
    png_kwargs = dict(save_kwargs)
    png_kwargs.pop("format", None)
    png_kwargs.pop("facecolor", None)
    png_kwargs.pop("edgecolor", None)
    png_kwargs["facecolor"] = "none"
    png_kwargs["edgecolor"] = "none"
    png_kwargs.setdefault("dpi", 300)
    if bbox_inches is not None:
        png_kwargs["bbox_inches"] = bbox_inches
        png_kwargs["pad_inches"] = 0
    try:
        fig.canvas.print_figure(png_buffer, format="png", **png_kwargs)
    finally:
        _restore_artist_visibility(changed)
    return png_buffer.getvalue()


def _render_dense_overlay_specs(
    fig,
    dense_artists: list,
    save_kwargs: dict,
    *,
    dense_layer_mode: str,
) -> list[dict[str, object]]:
    """Render dense layers as either one page overlay or axis-local pictures."""
    if not dense_artists:
        return []
    if dense_layer_mode == "full_canvas":
        width_inches, height_inches = (float(v) for v in fig.get_size_inches())
        return [
            {
                "png_bytes": _render_dense_overlay_png(
                    fig, dense_artists, save_kwargs
                ),
                "left_inches": 0.0,
                "top_inches": 0.0,
                "width_inches": width_inches,
                "height_inches": height_inches,
                "name": "Dense overlay full canvas",
            }
        ]
    if dense_layer_mode != "axis_local":
        raise ValueError(
            "dense_layer_mode must be 'full_canvas' or 'axis_local': "
            f"{dense_layer_mode!r}"
        )
    if save_kwargs.get("bbox_inches") is not None:
        raise ValueError(
            "axis_local dense overlays require the direct figure canvas, "
            "not bbox_inches cropping"
        )

    from matplotlib.transforms import Bbox

    width_inches, height_inches = (float(v) for v in fig.get_size_inches())
    axis_groups: list[tuple[object, list]] = []
    assigned_ids: set[int] = set()
    for axis in fig.axes:
        axis_artists = [
            artist
            for artist in dense_artists
            if getattr(artist, "axes", None) is axis
        ]
        if axis_artists:
            axis_groups.append((axis, axis_artists))
            assigned_ids.update(id(artist) for artist in axis_artists)
    unassigned = [
        artist for artist in dense_artists if id(artist) not in assigned_ids
    ]
    if unassigned:
        raise RuntimeError(
            "axis_local dense overlay found artists without a registered axes: "
            + ", ".join(type(artist).__name__ for artist in unassigned[:10])
        )

    overlays: list[dict[str, object]] = []
    for index, (axis, axis_artists) in enumerate(axis_groups, start=1):
        left, bottom, width, height = (
            float(value) for value in axis.get_position().bounds
        )
        if (
            width <= 0
            or height <= 0
            or left < 0
            or bottom < 0
            or left + width > 1
            or bottom + height > 1
        ):
            raise RuntimeError(
                "axis_local dense overlay axes bounds are outside the canvas: "
                f"{(left, bottom, width, height)}"
            )
        crop_bbox = Bbox.from_bounds(
            left * width_inches,
            bottom * height_inches,
            width * width_inches,
            height * height_inches,
        )
        axis_title = str(axis.get_title() or axis.get_label() or "").strip()
        suffix = f" - {axis_title}" if axis_title else ""
        overlays.append(
            {
                "png_bytes": _render_dense_overlay_png(
                    fig,
                    axis_artists,
                    save_kwargs,
                    bbox_inches=crop_bbox,
                ),
                "left_inches": left * width_inches,
                "top_inches": (1.0 - bottom - height) * height_inches,
                "width_inches": width * width_inches,
                "height_inches": height * height_inches,
                "name": f"Dense overlay axis {index:02d}{suffix}",
            }
        )
    return overlays


def _render_svg(
    fig,
    save_kwargs: dict,
    *,
    rasterize_dense_layers: bool,
    dense_layer_point_threshold: int,
    dense_layer_mode: str,
    vectorize_text_as_paths: bool,
) -> tuple[bytes, list[dict[str, object]]]:
    import matplotlib

    svg_buffer = BytesIO()
    svg_kwargs = dict(save_kwargs)
    svg_kwargs.pop("format", None)
    dense_artists = (
        _find_dense_artists(fig, dense_layer_point_threshold)
        if rasterize_dense_layers
        else []
    )
    background_patches = (
        [fig.patch] + [axis.patch for axis in fig.axes]
        if dense_artists
        else []
    )
    changed_artists = _set_artists_visible(
        dense_artists + background_patches, False
    )
    if dense_artists:
        svg_kwargs.pop("facecolor", None)
        svg_kwargs.pop("edgecolor", None)
        svg_kwargs["facecolor"] = "none"
        svg_kwargs["edgecolor"] = "none"
    # Glyph outlines are font-independent vector paths, but PowerPoint cannot
    # edit them as text strings after native-shape conversion.
    svg_fonttype = "path" if vectorize_text_as_paths else "none"
    try:
        with matplotlib.rc_context({"svg.fonttype": svg_fonttype}):
            fig.canvas.print_figure(svg_buffer, format="svg", **svg_kwargs)
    finally:
        _restore_artist_visibility(changed_artists)
    svg_bytes = svg_buffer.getvalue()
    if vectorize_text_as_paths:
        _validate_svg_has_no_text_nodes(svg_bytes)
        svg_bytes = _expand_matplotlib_text_glyph_uses(svg_bytes)
        _validate_svg_has_no_text_nodes(svg_bytes)
        _validate_svg_has_no_text_glyph_uses(svg_bytes)
    overlay_specs = _render_dense_overlay_specs(
        fig,
        dense_artists,
        save_kwargs,
        dense_layer_mode=dense_layer_mode,
    )
    return svg_bytes, overlay_specs


def _validate_svg_has_no_text_nodes(svg_bytes: bytes) -> int:
    """Require glyph-outline SVG content with no string-editable text nodes."""
    root = ElementTree.fromstring(svg_bytes)
    text_node_count = sum(
        1
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "text"
    )
    if text_node_count:
        raise RuntimeError(
            "SVG glyph-outline validation failed: "
            f"text_node={text_node_count}. "
            "Outlined glyphs must be vector paths, not string-editable text."
        )
    return text_node_count


def _svg_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _matplotlib_text_glyph_uses(root) -> list:
    """Return only ``use`` nodes nested under Matplotlib ``text_N`` groups."""
    glyph_uses = []

    def visit(element, inside_text_group: bool = False) -> None:
        local_name = _svg_local_name(element.tag)
        inside_text_group = inside_text_group or (
            local_name == "g"
            and element.attrib.get("id", "").startswith("text_")
        )
        if inside_text_group and local_name == "use":
            glyph_uses.append(element)
        for child in element:
            visit(child, inside_text_group)

    visit(root)
    return glyph_uses


def _validate_svg_has_no_text_glyph_uses(svg_bytes: bytes) -> int:
    """Require every Matplotlib text glyph reference to be an inline path."""
    root = ElementTree.fromstring(svg_bytes)
    glyph_use_count = len(_matplotlib_text_glyph_uses(root))
    if glyph_use_count:
        raise RuntimeError(
            "SVG glyph-use validation failed: "
            f"glyph_use={glyph_use_count}. "
            "Only inline glyph paths may be passed to PowerPoint."
        )
    return glyph_use_count


def _svg_use_href(use_element) -> str:
    href = use_element.attrib.get("href")
    xlink_href = use_element.attrib.get(_XLINK_HREF_ATTR)
    if href and xlink_href and href != xlink_href:
        raise RuntimeError(
            "SVG glyph use has inconsistent href and xlink:href values: "
            f"href={href!r}, xlink:href={xlink_href!r}"
        )
    reference = href or xlink_href
    if not reference or not reference.startswith("#") or len(reference) == 1:
        raise RuntimeError(
            "SVG glyph use must reference a local fragment path: "
            f"reference={reference!r}"
        )
    return reference[1:]


def _svg_numeric_coordinate(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        return "0"
    stripped = value.strip()
    if re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
        stripped,
    ) is None:
        raise RuntimeError(
            f"Cannot safely expand SVG glyph use coordinate {name}: "
            f"{value!r}"
        )
    return stripped


def _expand_matplotlib_text_glyph_uses(svg_bytes: bytes) -> bytes:
    """Inline Matplotlib text glyph paths without touching marker/scatter uses.

    Each replacement preserves SVG transform order as
    ``parent -> use transform -> translate(x, y) -> referenced path transform``.
    Unsupported targets and missing references fail strictly before any PPTX
    staging or target replacement.
    """
    root = ElementTree.fromstring(svg_bytes)
    glyph_uses = _matplotlib_text_glyph_uses(root)
    if not glyph_uses:
        return svg_bytes

    parents = {child: parent for parent in root.iter() for child in parent}
    elements_by_id = {}
    duplicate_ids = set()
    for element in root.iter():
        element_id = element.attrib.get("id")
        if not element_id:
            continue
        if element_id in elements_by_id:
            duplicate_ids.add(element_id)
        else:
            elements_by_id[element_id] = element
    if duplicate_ids:
        raise RuntimeError(
            "Duplicate IDs in SVG prevent expansion of glyph references: "
            f"{sorted(duplicate_ids)}"
        )

    for use_element in glyph_uses:
        reference_id = _svg_use_href(use_element)
        referenced = elements_by_id.get(reference_id)
        if referenced is None:
            raise RuntimeError(
                "Referenced SVG glyph use path is missing: "
                f"reference=#{reference_id}"
            )
        referenced_type = _svg_local_name(referenced.tag)
        if referenced_type != "path":
            raise RuntimeError(
                "SVG glyph use references an unsupported element: "
                f"reference=#{reference_id}, type={referenced_type}"
            )
        ancestor = parents.get(referenced)
        inside_defs = False
        while ancestor is not None:
            if _svg_local_name(ancestor.tag) == "defs":
                inside_defs = True
                break
            ancestor = parents.get(ancestor)
        if not inside_defs:
            raise RuntimeError(
                "Referenced SVG glyph use path is not inside defs: "
                f"reference=#{reference_id}"
            )
        if len(referenced):
            raise RuntimeError(
                "SVG glyph path contains unsupported child elements: "
                f"reference=#{reference_id}"
            )
        if len(use_element) or (use_element.text and use_element.text.strip()):
            raise RuntimeError(
                "SVG glyph use contains unsupported child/text content: "
                f"reference=#{reference_id}"
            )

        parent = parents.get(use_element)
        if parent is None:
            raise RuntimeError(
                f"Cannot find the parent of SVG glyph use: #{reference_id}"
            )
        child_index = list(parent).index(use_element)
        path_data = referenced.attrib.get("d", "")
        if not path_data.strip():
            # Matplotlib emits empty paths for spaces and some zero-width
            # glyphs. Later glyphs use absolute x positions, so removing this
            # non-rendering use preserves text advance and visible placement.
            use_tail = use_element.tail
            parent.remove(use_element)
            if use_tail:
                if child_index:
                    previous = parent[child_index - 1]
                    previous.tail = (previous.tail or "") + use_tail
                else:
                    parent.text = (parent.text or "") + use_tail
            continue

        replacement_group = ElementTree.Element(_SVG_GROUP_TAG)
        for attribute, value in use_element.attrib.items():
            if attribute not in {
                "href",
                _XLINK_HREF_ATTR,
                "x",
                "y",
                "transform",
            }:
                replacement_group.set(attribute, value)
        transforms = []
        use_transform = use_element.attrib.get("transform", "").strip()
        if use_transform:
            transforms.append(use_transform)
        if "x" in use_element.attrib or "y" in use_element.attrib:
            x_value = _svg_numeric_coordinate(use_element.attrib.get("x"), "x")
            y_value = _svg_numeric_coordinate(use_element.attrib.get("y"), "y")
            transforms.append(f"translate({x_value} {y_value})")
        if transforms:
            replacement_group.set("transform", " ".join(transforms))

        inline_path = deepcopy(referenced)
        inline_path.attrib.pop("id", None)
        inline_path.tail = None
        replacement_group.append(inline_path)
        replacement_group.tail = use_element.tail
        parent.remove(use_element)
        parent.insert(child_index, replacement_group)

    ElementTree.register_namespace("", _SVG_NAMESPACE)
    ElementTree.register_namespace("xlink", _XLINK_NAMESPACE)
    expanded_svg = ElementTree.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
    )
    _validate_svg_has_no_text_glyph_uses(expanded_svg)
    return expanded_svg


def _render_png(fig, save_kwargs: dict) -> bytes:
    png_buffer = BytesIO()
    png_kwargs = dict(save_kwargs)
    png_kwargs.pop("format", None)
    png_kwargs.setdefault("dpi", 300)
    fig.canvas.print_figure(png_buffer, format="png", **png_kwargs)
    return png_buffer.getvalue()


def _content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Default Extension="svg" ContentType="image/svg+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def _package_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def _presentation_xml(slide_width: int, slide_height: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>
  <p:sldSz cx="{slide_width}" cy="{slide_height}" type="custom"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>
"""


def _presentation_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>
"""


def _slide_master_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>
"""


def _slide_master_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
"""


def _slide_layout_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>
"""


def _slide_layout_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
"""


def _theme_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Roider Figure Theme">
  <a:themeElements>
    <a:clrScheme name="Roider"><a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="172033"/></a:dk2><a:lt2><a:srgbClr val="F4F6F8"/></a:lt2><a:accent1><a:srgbClr val="2B7FB8"/></a:accent1><a:accent2><a:srgbClr val="D84A43"/></a:accent2><a:accent3><a:srgbClr val="F28C24"/></a:accent3><a:accent4><a:srgbClr val="4D8D3A"/></a:accent4><a:accent5><a:srgbClr val="7A5BA6"/></a:accent5><a:accent6><a:srgbClr val="D8B70F"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Roider Arial"><a:majorFont><a:latin typeface="Arial"/><a:ea typeface="Arial"/><a:cs typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Arial"/><a:ea typeface="Arial"/><a:cs typeface="Arial"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Roider"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/><a:extraClrSchemeLst/>
</a:theme>
"""


def _slide_xml(slide_width: int, slide_height: int, title: str) -> str:
    safe_title = _xml_attr(title)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    <p:pic>
      <p:nvPicPr><p:cNvPr id="2" name="{safe_title}" descr="Outlined SVG figure; glyphs are vector paths and are not string-editable text."/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
      <p:blipFill>
        <a:blip r:embed="rId2">
          <a:extLst><a:ext uri="{{96DAC541-7B7A-43D3-8B79-37D633B846F1}}"><asvg:svgBlip xmlns:asvg="http://schemas.microsoft.com/office/drawing/2016/SVG/main" r:embed="rId3"/></a:ext></a:extLst>
        </a:blip>
        <a:stretch><a:fillRect/></a:stretch>
      </p:blipFill>
      <p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{slide_width}" cy="{slide_height}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:ln><a:noFill/></a:ln></p:spPr>
    </p:pic>
  </p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


def _slide_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image1.png"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image1.svg"/>
</Relationships>
"""


def _core_xml(title: str) -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    safe_title = escape(title)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{safe_title}</dc:title><dc:creator>Roider Final</dc:creator><cp:lastModifiedBy>Roider Final</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def _app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Roider editable SVG PPTX exporter</Application><PresentationFormat>Custom</PresentationFormat><Slides>1</Slides>
</Properties>
"""


def _validate_pptx_archive(
    path: Path,
    *,
    require_fallback_media: bool = False,
) -> None:
    """Validate the package before it is eligible to replace the target."""
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        required = set(_REQUIRED_PPTX_PARTS)
        if require_fallback_media:
            required.update(_FALLBACK_MEDIA_PARTS)
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"Required PPTX parts are missing: {missing}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"PPTX ZIP CRC validation failed: {bad_member}")
        ElementTree.fromstring(archive.read("[Content_Types].xml"))
        ElementTree.fromstring(archive.read("ppt/presentation.xml"))
        ElementTree.fromstring(archive.read("ppt/slides/slide1.xml"))
        slide_rels = ElementTree.fromstring(
            archive.read("ppt/slides/_rels/slide1.xml.rels")
        )
        if require_fallback_media:
            relationship_targets = {
                rel.attrib.get("Target", "") for rel in slide_rels
            }
            expected_targets = {"../media/image1.png", "../media/image1.svg"}
            if not expected_targets.issubset(relationship_targets):
                raise RuntimeError(
                    "Invalid SVG/PNG fallback relationship: "
                    f"targets={sorted(relationship_targets)}"
                )


def _normalize_native_pptx_theme_to_arial(path: Path) -> None:
    """Set only the six native-theme font faces to Arial, transactionally."""
    theme_member = "ppt/theme/theme1.xml"
    drawingml_ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
    token = uuid.uuid4().hex
    rewrite_candidate = path.with_name(f".{path.stem}.{token}.pptx")

    def _zipinfo_contract(info: zipfile.ZipInfo) -> tuple:
        return (
            info.filename,
            info.date_time,
            info.compress_type,
            info.comment,
            info.extra,
            info.create_system,
            info.create_version,
            info.extract_version,
            info.flag_bits,
            info.volume,
            info.internal_attr,
            info.external_attr,
        )

    try:
        with zipfile.ZipFile(path, "r") as source_archive:
            source_infos = source_archive.infolist()
            if sum(info.filename == theme_member for info in source_infos) != 1:
                raise RuntimeError(
                    "Expected exactly one native PPTX theme part: "
                    f"{theme_member}"
                )
            source_comment = source_archive.comment
            source_payloads = [
                (deepcopy(info), source_archive.read(info)) for info in source_infos
            ]

        theme_bytes = next(
            payload
            for info, payload in source_payloads
            if info.filename == theme_member
        )
        theme_root = ElementTree.fromstring(theme_bytes)
        font_schemes = theme_root.findall(f".//{{{drawingml_ns}}}fontScheme")
        if len(font_schemes) != 1:
            raise RuntimeError(
                "Expected exactly one native PPTX fontScheme: "
                f"count={len(font_schemes)}"
            )
        font_scheme = font_schemes[0]
        font_nodes = []
        for family_name in ("majorFont", "minorFont"):
            families = font_scheme.findall(f"{{{drawingml_ns}}}{family_name}")
            if len(families) != 1:
                raise RuntimeError(
                    f"Expected exactly one native PPTX {family_name}: "
                    f"count={len(families)}"
                )
            for face_name in ("latin", "ea", "cs"):
                faces = families[0].findall(f"{{{drawingml_ns}}}{face_name}")
                if len(faces) != 1:
                    raise RuntimeError(
                        f"Expected exactly one native PPTX {family_name}/{face_name}: "
                        f"count={len(faces)}"
                    )
                font_nodes.append(faces[0])
        for node in font_nodes:
            node.set("typeface", "Arial")
        ElementTree.register_namespace("a", drawingml_ns)
        normalized_theme = ElementTree.tostring(
            theme_root,
            encoding="utf-8",
            xml_declaration=True,
        )

        with zipfile.ZipFile(rewrite_candidate, "w") as target_archive:
            target_archive.comment = source_comment
            for info, payload in source_payloads:
                target_archive.writestr(
                    info,
                    normalized_theme if info.filename == theme_member else payload,
                )

        _validate_pptx_archive(rewrite_candidate)
        with zipfile.ZipFile(rewrite_candidate, "r") as target_archive:
            target_infos = target_archive.infolist()
            if [info.filename for info in target_infos] != [
                info.filename for info, _ in source_payloads
            ]:
                raise RuntimeError("Native PPTX ZIP member order changed.")
            if target_archive.comment != source_comment:
                raise RuntimeError("Native PPTX ZIP comment changed.")
            for target_info, (source_info, source_payload) in zip(
                target_infos,
                source_payloads,
            ):
                if _zipinfo_contract(target_info) != _zipinfo_contract(source_info):
                    raise RuntimeError(
                        "Native PPTX ZIP metadata changed: "
                        f"{source_info.filename}"
                    )
                target_payload = target_archive.read(target_info)
                if (
                    source_info.filename != theme_member
                    and target_payload != source_payload
                ):
                    raise RuntimeError(
                        "A native PPTX part other than the theme changed: "
                        f"{source_info.filename}"
                    )
            normalized_root = ElementTree.fromstring(
                target_archive.read(theme_member)
            )
        normalized_scheme = normalized_root.findall(
            f".//{{{drawingml_ns}}}fontScheme"
        )
        if len(normalized_scheme) != 1:
            raise RuntimeError("fontScheme validation failed after Arial normalization.")
        normalized_faces = []
        for family_name in ("majorFont", "minorFont"):
            family = normalized_scheme[0].find(f"{{{drawingml_ns}}}{family_name}")
            if family is None:
                raise RuntimeError(
                    f"{family_name} validation failed after Arial normalization."
                )
            normalized_faces.extend(
                family.find(f"{{{drawingml_ns}}}{face_name}")
                for face_name in ("latin", "ea", "cs")
            )
        if any(
            node is None or node.attrib.get("typeface") != "Arial"
            for node in normalized_faces
        ):
            raise RuntimeError("Native PPTX theme Arial validation failed.")
        os.replace(rewrite_candidate, path)
    finally:
        rewrite_candidate.unlink(missing_ok=True)


def _process_diagnostics(returncode: int | str, stdout: str, stderr: str) -> str:
    """Keep both PowerShell streams in conversion errors for later diagnosis."""
    return (
        f"exit_code={returncode}\n"
        f"STDOUT:\n{stdout.strip() or '<empty>'}\n"
        f"STDERR:\n{stderr.strip() or '<empty>'}"
    )


def _windows_path(path: Path) -> str:
    if os.name == "nt":
        return str(path.resolve())
    wslpath = shutil.which("wslpath")
    if wslpath is None:
        raise RuntimeError(
            "Native editable PPTX conversion requires Windows PowerPoint or wslpath."
        )
    result = subprocess.run(
        [wslpath, "-w", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _convert_svg_to_native_powerpoint_shapes(
    path: Path,
    svg_bytes: bytes,
    overlay_specs: list[dict[str, object]],
    width_inches: float,
    height_inches: float,
    timeout_seconds: int,
) -> int:
    """Create a native-shape candidate through installed Windows PowerPoint.

    ``path`` is a caller-owned staging path, never the final user-visible target.
    """
    powershell = shutil.which("powershell.exe")
    if powershell is None and os.name == "nt":
        powershell = shutil.which("powershell")
    if powershell is None:
        raise RuntimeError(
            "Windows PowerPoint/PowerShell for native editable PPTX conversion was not found."
        )

    token = uuid.uuid4().hex
    svg_temp_path = path.with_name(f".{path.stem}.{token}.svg")
    native_temp_path = path.with_name(f".{path.stem}.{token}.pptx")
    svg_temp_path.write_bytes(svg_bytes)
    overlay_temp_paths: list[Path] = []
    overlay_commands: list[str] = []
    for index, spec in enumerate(overlay_specs, start=1):
        overlay_temp_path = path.with_name(
            f".{path.stem}.{token}.overlay{index:02d}.png"
        )
        overlay_bytes = spec.get("png_bytes")
        if not isinstance(overlay_bytes, bytes):
            raise TypeError(f"Dense overlay {index} does not contain PNG bytes")
        overlay_temp_path.write_bytes(overlay_bytes)
        overlay_temp_paths.append(overlay_temp_path)
        overlay_windows_path = _windows_path(overlay_temp_path).replace("'", "''")
        overlay_name = str(spec.get("name") or f"Dense overlay {index:02d}").replace(
            "'", "''"
        )
        left_points = float(spec["left_inches"]) * 72.0
        top_points = float(spec["top_inches"]) * 72.0
        overlay_width_points = float(spec["width_inches"]) * 72.0
        overlay_height_points = float(spec["height_inches"]) * 72.0
        if overlay_width_points <= 0 or overlay_height_points <= 0:
            raise ValueError(f"Dense overlay {index} has invalid dimensions")
        overlay_commands.extend(
            [
                "$overlay = $slide.Shapes.AddPicture(",
                f"    '{overlay_windows_path}', 0, -1, "
                f"{left_points:.8f}, {top_points:.8f},",
                f"    {overlay_width_points:.8f}, {overlay_height_points:.8f}",
                ")",
                f"$overlay.Name = '{overlay_name}'",
                "$overlay.ZOrder(1)",
            ]
        )
    overlay_commands_text = "\n".join(overlay_commands)
    pptx_windows_path = _windows_path(native_temp_path).replace("'", "''")
    svg_windows_path = _windows_path(svg_temp_path).replace("'", "''")
    width_points = width_inches * 72.0
    height_points = height_inches * 72.0
    script = f"""
$ErrorActionPreference = 'Stop'
$pptxPath = '{pptx_windows_path}'
$svgPath = '{svg_windows_path}'
function Invoke-TransientPowerPointComRetry {{
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [Parameter(Mandatory = $true)][string]$Label,
        [int]$Attempts = 20,
        [int]$DelayMilliseconds = 250
    )
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {{
        try {{
            return & $Action
        }}
        catch {{
            $cursor = $_.Exception
            $isComFailure = $false
            while ($null -ne $cursor) {{
                if ($cursor -is [System.Runtime.InteropServices.COMException]) {{
                    $isComFailure = $true
                    break
                }}
                $cursor = $cursor.InnerException
            }}
            if (-not $isComFailure -or $attempt -eq $Attempts) {{
                throw
            }}
            Start-Sleep -Milliseconds $DelayMilliseconds
        }}
    }}
}}
$powerPoint = New-Object -ComObject PowerPoint.Application
$powerPoint.Visible = -1
Start-Sleep -Milliseconds 250
$presentation = $null
$window = $null
$slide = $null
$shape = $null
$overlay = $null
try {{
    $presentation = $powerPoint.Presentations.Add()
    Start-Sleep -Milliseconds 250
    Invoke-TransientPowerPointComRetry -Label 'PageSetup readiness' -Action {{
        $presentation.PageSetup.SlideWidth = {width_points:.8f}
        $presentation.PageSetup.SlideHeight = {height_points:.8f}
        $null
    }}
    $slide = $presentation.Slides.Add(1, 12)
    $shape = $slide.Shapes.AddPicture(
        $svgPath, 0, -1, 0.0, 0.0,
        {width_points:.8f}, {height_points:.8f}
    )
    $window = $presentation.Windows.Item(1)
    Invoke-TransientPowerPointComRetry -Label 'active SVG editing view' -Action {{
        $window.Activate()
        $window.ViewType = 9
        $window.View.GotoSlide(1)
        $shape.Select()
        $null
    }}
    $powerPoint.CommandBars.ExecuteMso("SVGEdit")
    Start-Sleep -Milliseconds 500
    if ($slide.Shapes.Count -eq 1 -and $slide.Shapes.Item(1).Type -eq 6) {{
        $converted = $slide.Shapes.Item(1).Ungroup()
    }}
    if ($slide.Shapes.Count -le 1) {{
        throw "PowerPoint did not convert the SVG into multiple editable shapes."
    }}
    $nativeCount = 0
    foreach ($item in $slide.Shapes) {{
        if ($item.Type -notin @(11, 13, 28)) {{
            $nativeCount += 1
        }}
    }}
    if ($nativeCount -lt 1) {{
        throw "Converted slide still contains only picture/graphic shapes."
    }}
    {overlay_commands_text}
    $presentation.SaveAs($pptxPath, 24)
    Write-Output "NATIVE_SHAPES=$nativeCount"
    Write-Output "DENSE_OVERLAYS={len(overlay_specs)}"
}}
finally {{
    if ($null -ne $presentation) {{
        try {{ $presentation.Close() }} catch {{}}
    }}
    try {{ $powerPoint.Quit() }} catch {{}}
    if ($null -ne $shape) {{
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($shape) | Out-Null
    }}
    if ($null -ne $overlay) {{
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($overlay) | Out-Null
    }}
    if ($null -ne $slide) {{
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($slide) | Out-Null
    }}
    if ($null -ne $window) {{
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($window) | Out-Null
    }}
    if ($null -ne $presentation) {{
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
    }}
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint) | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}}
"""
    script_temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            suffix=".ps1",
            prefix="roider_editable_pptx_",
            dir=tempfile.gettempdir(),
            delete=False,
        ) as handle:
            handle.write(script)
            script_temp_path = Path(handle.name)
        script_windows_path = _windows_path(script_temp_path)
        try:
            result = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    script_windows_path,
                ],
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_stdout = (
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else str(exc.stdout or "")
            )
            timeout_stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr or "")
            )
            raise RuntimeError(
                "PowerPoint native-shape conversion timed out:\n"
                + _process_diagnostics(
                    f"timeout>{timeout_seconds}s",
                    timeout_stdout,
                    timeout_stderr,
                )
            ) from exc
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(
                "PowerPoint native-shape conversion failed:\n"
                + _process_diagnostics(result.returncode, stdout, stderr)
            )
        match = re.search(r"NATIVE_SHAPES=(\d+)", stdout)
        if match is None or int(match.group(1)) < 1:
            raise RuntimeError(
                "Cannot inspect PowerPoint native-shape validation results:\n"
                + _process_diagnostics(result.returncode, stdout, stderr)
            )
        if not native_temp_path.exists():
            raise RuntimeError(
                f"PowerPoint did not save the native PPTX: {native_temp_path}\n"
                + _process_diagnostics(result.returncode, stdout, stderr)
            )
        os.replace(native_temp_path, path)
        return int(match.group(1))
    finally:
        if script_temp_path is not None:
            script_temp_path.unlink(missing_ok=True)
        svg_temp_path.unlink(missing_ok=True)
        for overlay_temp_path in overlay_temp_paths:
            overlay_temp_path.unlink(missing_ok=True)
        native_temp_path.unlink(missing_ok=True)


def _validate_native_shape_slide(
    path: Path,
    *,
    require_outlined_text: bool = True,
) -> dict[str, int]:
    """Require native shapes and, by default, glyph outlines instead of text."""
    namespaces = {
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "asvg": "http://schemas.microsoft.com/office/drawing/2016/SVG/main",
    }
    with zipfile.ZipFile(path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Native PPTX ZIP CRC validation failed: {bad_member}")
        slide_root = ElementTree.fromstring(archive.read("ppt/slides/slide1.xml"))
    counts = {
        "shape": len(slide_root.findall(".//p:sp", namespaces)),
        "group_shape": len(slide_root.findall(".//p:grpSp", namespaces)),
        "connector_shape": len(slide_root.findall(".//p:cxnSp", namespaces)),
        "graphic_frame": len(slide_root.findall(".//p:graphicFrame", namespaces)),
        "picture": len(slide_root.findall(".//p:pic", namespaces)),
        "svg_picture": len(slide_root.findall(".//asvg:svgBlip", namespaces)),
        "text_node": len(slide_root.findall(".//a:t", namespaces)),
    }
    counts["native_shape"] = (
        counts["shape"]
        + counts["group_shape"]
        + counts["connector_shape"]
        + counts["graphic_frame"]
    )
    summary = ", ".join(f"{key}={value}" for key, value in counts.items())
    if counts["svg_picture"] > 0:
        raise RuntimeError(
            "PPTX slide contains unconverted SVG pictures: " + summary
        )
    if require_outlined_text and counts["text_node"] > 0:
        raise RuntimeError(
            "PPTX slide still contains DrawingML text. "
            "Glyph-outline mode requires vector paths rather than "
            "string-editable text: " + summary
        )
    if counts["native_shape"] < 1:
        raise RuntimeError(
            "PPTX slide contains no editable native shapes: " + summary
        )
    return counts


def _validate_axis_local_dense_overlays(
    path: Path,
    *,
    expected_overlay_count: int,
) -> dict[str, int]:
    """Require named, bounded picture objects instead of a page-wide overlay."""
    namespaces = {
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    }
    with zipfile.ZipFile(path, "r") as archive:
        slide_root = ElementTree.fromstring(archive.read("ppt/slides/slide1.xml"))
        presentation_root = ElementTree.fromstring(
            archive.read("ppt/presentation.xml")
        )
    slide_size = presentation_root.find("p:sldSz", namespaces)
    if slide_size is None:
        raise RuntimeError("PPTX presentation is missing p:sldSz")
    slide_width = int(slide_size.attrib["cx"])
    slide_height = int(slide_size.attrib["cy"])
    overlay_pictures = []
    for picture in slide_root.findall(".//p:pic", namespaces):
        properties = picture.find("p:nvPicPr/p:cNvPr", namespaces)
        if properties is None:
            continue
        if properties.attrib.get("name", "").startswith("Dense overlay axis "):
            overlay_pictures.append(picture)
    if len(overlay_pictures) != expected_overlay_count:
        raise RuntimeError(
            "Axis-local dense overlay count mismatch: "
            f"expected={expected_overlay_count}, actual={len(overlay_pictures)}"
        )
    for picture in overlay_pictures:
        transform = picture.find("p:spPr/a:xfrm", namespaces)
        if transform is None:
            raise RuntimeError("Axis-local dense overlay is missing a:xfrm")
        offset = transform.find("a:off", namespaces)
        extent = transform.find("a:ext", namespaces)
        if offset is None or extent is None:
            raise RuntimeError("Axis-local dense overlay geometry is incomplete")
        left = int(offset.attrib["x"])
        top = int(offset.attrib["y"])
        width = int(extent.attrib["cx"])
        height = int(extent.attrib["cy"])
        if (
            left < 0
            or top < 0
            or width <= 0
            or height <= 0
            or left + width > slide_width + 2
            or top + height > slide_height + 2
        ):
            raise RuntimeError(
                "Axis-local dense overlay escapes the slide: "
                f"{(left, top, width, height)} vs "
                f"{(slide_width, slide_height)}"
            )
        if width >= slide_width or height >= slide_height:
            raise RuntimeError(
                "Axis-local mode produced a full-slide dense overlay picture"
            )
    return {
        "axis_local_overlay": len(overlay_pictures),
        "slide_width": slide_width,
        "slide_height": slide_height,
    }


def save_matplotlib_figure_as_editable_pptx(
    fig,
    pptx_path,
    *,
    png_fallback_path=None,
    title: str | None = None,
    validate: bool = True,
    convert_to_native_shapes: bool = True,
    conversion_timeout_seconds: int = 180,
    rasterize_dense_layers: bool = False,
    dense_layer_point_threshold: int = 100,
    dense_layer_mode: str = "full_canvas",
    fallback_to_vector_object: bool = False,
    return_native_shape_count: bool = False,
    vectorize_text_as_paths: bool = True,
    **save_kwargs,
):
    """Transactionally save one Matplotlib figure as a native-shape PPTX.

    ``save_kwargs`` should match the PNG save call so the PPTX and PNG use the
    same canvas or tight-bbox contract. Native conversion is strict by default:
    an SVG/PNG compatibility deck is staged separately, and the visible target
    is replaced only after the PowerPoint candidate passes archive and native
    shape validation. Set ``fallback_to_vector_object=True`` only to explicitly
    accept a non-native SVG picture when conversion fails. The default return
    remains the output ``Path``; ``return_native_shape_count=True`` returns
    ``(Path, count)`` without changing existing callers. By default,
    ``vectorize_text_as_paths=True`` freezes labels as font-independent glyph
    outlines; those vector paths are shape-editable but not string-editable
    PowerPoint text.
    """
    pptx_path = Path(pptx_path)
    if pptx_path.suffix.lower() != ".pptx":
        raise ValueError(f"PPTX output path must have a .pptx extension: {pptx_path}")
    _validate_matplotlib_arial_font_contract(fig)
    pptx_path.parent.mkdir(parents=True, exist_ok=True)

    if dense_layer_mode not in {"full_canvas", "axis_local"}:
        raise ValueError(
            "dense_layer_mode must be 'full_canvas' or 'axis_local': "
            f"{dense_layer_mode!r}"
        )
    svg_bytes, overlay_specs = _render_svg(
        fig,
        save_kwargs,
        rasterize_dense_layers=rasterize_dense_layers,
        dense_layer_point_threshold=dense_layer_point_threshold,
        dense_layer_mode=dense_layer_mode,
        vectorize_text_as_paths=vectorize_text_as_paths,
    )
    width_inches, height_inches = _svg_size_inches(svg_bytes, fig)
    slide_width = max(1, int(round(width_inches * EMU_PER_INCH)))
    slide_height = max(1, int(round(height_inches * EMU_PER_INCH)))

    png_bytes = None
    if png_fallback_path is not None:
        fallback_path = Path(png_fallback_path)
        if fallback_path.exists():
            png_bytes = fallback_path.read_bytes()
    if png_bytes is None:
        png_bytes = _render_png(fig, save_kwargs)

    deck_title = title or pptx_path.stem
    package_parts = {
        "[Content_Types].xml": _content_types_xml().encode("utf-8"),
        "_rels/.rels": _package_rels_xml().encode("utf-8"),
        "docProps/app.xml": _app_xml().encode("utf-8"),
        "docProps/core.xml": _core_xml(deck_title).encode("utf-8"),
        "ppt/presentation.xml": _presentation_xml(slide_width, slide_height).encode("utf-8"),
        "ppt/_rels/presentation.xml.rels": _presentation_rels_xml().encode("utf-8"),
        "ppt/slideLayouts/slideLayout1.xml": _slide_layout_xml().encode("utf-8"),
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": _slide_layout_rels_xml().encode("utf-8"),
        "ppt/slideMasters/slideMaster1.xml": _slide_master_xml().encode("utf-8"),
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": _slide_master_rels_xml().encode("utf-8"),
        "ppt/slides/slide1.xml": _slide_xml(slide_width, slide_height, deck_title).encode("utf-8"),
        "ppt/slides/_rels/slide1.xml.rels": _slide_rels_xml().encode("utf-8"),
        "ppt/theme/theme1.xml": _theme_xml().encode("utf-8"),
        "ppt/media/image1.png": png_bytes,
        "ppt/media/image1.svg": svg_bytes,
    }
    token = uuid.uuid4().hex
    fallback_candidate = pptx_path.with_name(
        f".{pptx_path.stem}.{token}.fallback.pptx"
    )
    native_candidate = pptx_path.with_name(
        f".{pptx_path.stem}.{token}.native.pptx"
    )
    native_shape_count = 0
    try:
        with zipfile.ZipFile(
            fallback_candidate, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for member_name, member_bytes in package_parts.items():
                archive.writestr(member_name, member_bytes)
        _validate_pptx_archive(
            fallback_candidate,
            require_fallback_media=True,
        )

        if not convert_to_native_shapes:
            os.replace(fallback_candidate, pptx_path)
            result = pptx_path
            return (
                (result, native_shape_count)
                if return_native_shape_count
                else result
            )

        try:
            reported_native_shape_count = _convert_svg_to_native_powerpoint_shapes(
                native_candidate,
                svg_bytes,
                overlay_specs,
                width_inches,
                height_inches,
                timeout_seconds=conversion_timeout_seconds,
            )
            _validate_pptx_archive(native_candidate)
            _normalize_native_pptx_theme_to_arial(native_candidate)
            _validate_pptx_archive(native_candidate)
            native_counts = _validate_native_shape_slide(
                native_candidate,
                require_outlined_text=vectorize_text_as_paths,
            )
            if dense_layer_mode == "axis_local" and overlay_specs:
                _validate_axis_local_dense_overlays(
                    native_candidate,
                    expected_overlay_count=len(overlay_specs),
                )
            native_shape_count = native_counts["native_shape"]
            if reported_native_shape_count < 1 or native_shape_count < 1:
                raise RuntimeError(
                    "Invalid PowerPoint native-shape count: "
                    f"reported={reported_native_shape_count}, xml={native_shape_count}"
                )
            os.replace(native_candidate, pptx_path)
        except Exception as exc:
            if not fallback_to_vector_object:
                raise
            warnings.warn(
                "PowerPoint native-shape conversion failed; saving an explicitly permitted "
                f"SVG vector object: {exc}",
                RuntimeWarning,
            )
            os.replace(fallback_candidate, pptx_path)
            native_shape_count = 0
        result = pptx_path
        return (
            (result, native_shape_count)
            if return_native_shape_count
            else result
        )
    finally:
        fallback_candidate.unlink(missing_ok=True)
        native_candidate.unlink(missing_ok=True)
