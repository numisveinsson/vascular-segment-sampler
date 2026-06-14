"""Render matching surface meshes (.vtp) side-by-side as .png for visual comparison.

This is the visualization counterpart to ``compute_metrics_meshes_comparison.py``: instead
of computing distance/Dice/volume metrics it renders the meshes with VTK (off-screen) and
composes a side-by-side comparison image per case using matplotlib.

Usage:
	# single prediction folder: GT vs pred, one comparison PNG per case
	python -m analysis.render_meshes_comparison --gt-dir /data/gt_vtps --pred-dir /data/pred_vtps \
			--out-dir /data/renders

	# multiple prediction folders: GT | pred_run1 | pred_run2 | ... per case
	python -m analysis.render_meshes_comparison --gt-dir /data/gt_vtps \
			--predictions-root /data/prediction_runs --out-dir /data/renders

	# render several camera views and color by a point/cell array (e.g. RegionId)
	python -m analysis.render_meshes_comparison --gt-dir /data/gt_vtps --pred-dir /data/pred_vtps \
			--out-dir /data/renders --views xy,xz,yz --color-by RegionId

	# clip meshes with centerline-based boxes before rendering
	python -m analysis.render_meshes_comparison --gt-dir /data/gt_vtps \
			--predictions-root /data/prediction_runs --out-dir /data/renders \
			--clip --centerline-dir /data/centerlines

Assumes meshes share the same filename in every directory (e.g. ``case001.vtp``), matching
the convention used by ``compute_metrics_meshes_comparison.py``.

Notes:
- Rendering is done off-screen so this works on headless machines.
- Requires the ``vtk`` Python package (``pip install vtk``). matplotlib is used to compose the
  grid of panels and add titles; if it is unavailable the script falls back to tiling the
  panels horizontally with numpy and writing a single PNG with VTK (no titles).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

try:
	import vtk
except Exception as e:  # pragma: no cover - import guard
	raise ImportError(
		"The 'vtk' Python package is required to run this script. "
		"Install it with `pip install vtk` or `conda install -c conda-forge vtk`."
	) from e

import numpy as np
from vtk.util import numpy_support

try:
	from tqdm import tqdm
except ImportError:
	def tqdm(iterable, desc=None, disable=False, **kwargs):
		return iterable

# Support both direct script execution and package/module execution.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _repo_root not in sys.path:
	sys.path.insert(0, _repo_root)

# Reuse the file/case matching helpers from the metrics comparison script so the two
# tools agree on which cases/folders are compared.
try:
	from analysis.compute_metrics_meshes_comparison import (
		clip_surface_mesh,
		find_cases_in_all_dirs,
		find_matching_vtps,
		load_vtp,
	)
except ImportError:
	from compute_metrics_meshes_comparison import (  # type: ignore
		clip_surface_mesh,
		find_cases_in_all_dirs,
		find_matching_vtps,
		load_vtp,
	)

# Optional centerline reader (for clipping). Falls back to the plain VTP reader.
try:
	from modules import vtk_functions as vf
except ImportError:
	vf = None


def load_centerline(path: str) -> vtk.vtkPolyData:
	"""Read a centerline .vtp, using modules.vtk_functions.read_geo when available."""
	if vf is not None:
		return vf.read_geo(path).GetOutput()
	return load_vtp(path)


# Camera view codes -> (position direction, view-up). The camera looks at the mesh
# center from `center + direction * distance`, oriented with `view_up` pointing up.
VIEW_DIRECTIONS: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
	'xy': ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),   # look down +z
	'xz': ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),  # look along -y
	'yz': ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),   # look along +x
}


def _named_color(name_or_rgb) -> Tuple[float, float, float]:
	"""Resolve a color given as an RGB tuple (0-1) or a VTK color name string."""
	if isinstance(name_or_rgb, (tuple, list)) and len(name_or_rgb) == 3:
		return (float(name_or_rgb[0]), float(name_or_rgb[1]), float(name_or_rgb[2]))
	colors = vtk.vtkNamedColors()
	c = colors.GetColor3d(str(name_or_rgb))
	return (c[0], c[1], c[2])


def _scalar_array(poly: vtk.vtkPolyData, array_name: str):
	"""Return (vtk_array, association) for `array_name` from point or cell data, else (None, None)."""
	pd = poly.GetPointData()
	arr = pd.GetArray(array_name)
	if arr is not None:
		return arr, 'point'
	cd = poly.GetCellData()
	arr = cd.GetArray(array_name)
	if arr is not None:
		return arr, 'cell'
	return None, None


def render_polydata_to_array(
	poly: vtk.vtkPolyData,
	view: str = 'xy',
	window_size: Tuple[int, int] = (700, 700),
	color=(0.85, 0.85, 0.9),
	background=(1.0, 1.0, 1.0),
	zoom: float = 1.2,
	color_by: Optional[str] = None,
	cmap: str = 'coolwarm',
	clim: Optional[Tuple[float, float]] = None,
	edges: bool = False,
	bounds: Optional[Sequence[float]] = None,
) -> np.ndarray:
	"""Render `poly` off-screen from a named camera `view` and return an (H, W, 3) uint8 array.

	If `color_by` names a point/cell scalar array present on the mesh, the surface is colored
	with a lookup table (`cmap` is a small built-in set: coolwarm, viridis, jet, grayscale);
	otherwise a solid `color` is used.

	If `bounds` (xmin, xmax, ymin, ymax, zmin, zmax) is given, the camera is framed using those
	shared bounds instead of the mesh's own bounds, so multiple meshes render at the same scale
	for side-by-side comparison.
	"""
	if view not in VIEW_DIRECTIONS:
		raise ValueError(f"Unknown view '{view}'. Valid: {', '.join(sorted(VIEW_DIRECTIONS))}")

	mapper = vtk.vtkPolyDataMapper()
	mapper.SetInputData(poly)

	scalars, assoc = (_scalar_array(poly, color_by) if color_by else (None, None))
	if scalars is not None:
		rng = scalars.GetRange() if clim is None else clim
		lut = _build_lookup_table(cmap, rng)
		mapper.SetScalarRange(rng[0], rng[1])
		mapper.SetLookupTable(lut)
		mapper.SetColorModeToMapScalars()
		mapper.ScalarVisibilityOn()
		if assoc == 'point':
			mapper.SetScalarModeToUsePointFieldData()
		else:
			mapper.SetScalarModeToUseCellFieldData()
		mapper.SelectColorArray(color_by)
	else:
		mapper.ScalarVisibilityOff()

	actor = vtk.vtkActor()
	actor.SetMapper(mapper)
	if scalars is None:
		actor.GetProperty().SetColor(*_named_color(color))
	if edges:
		actor.GetProperty().EdgeVisibilityOn()
		actor.GetProperty().SetLineWidth(0.5)

	renderer = vtk.vtkRenderer()
	renderer.AddActor(actor)
	renderer.SetBackground(*_named_color(background))

	render_window = vtk.vtkRenderWindow()
	render_window.SetOffScreenRendering(1)
	render_window.AddRenderer(renderer)
	render_window.SetSize(int(window_size[0]), int(window_size[1]))

	_set_camera_view(renderer, poly, view, zoom, bounds=bounds)
	render_window.Render()

	w2if = vtk.vtkWindowToImageFilter()
	w2if.SetInput(render_window)
	w2if.ReadFrontBufferOff()
	w2if.Update()
	img = _vtk_image_to_numpy(w2if.GetOutput())

	# Release GL resources for this window.
	render_window.Finalize()
	return img


def _union_bounds(polys: Sequence[vtk.vtkPolyData]) -> Optional[Tuple[float, float, float, float, float, float]]:
	"""Return the union (xmin, xmax, ymin, ymax, zmin, zmax) of all polydata bounds, or None."""
	xmin = ymin = zmin = float('inf')
	xmax = ymax = zmax = float('-inf')
	found = False
	for poly in polys:
		if poly is None or poly.GetNumberOfPoints() == 0:
			continue
		b = poly.GetBounds()
		if not all(np.isfinite(v) for v in b):
			continue
		xmin, xmax = min(xmin, b[0]), max(xmax, b[1])
		ymin, ymax = min(ymin, b[2]), max(ymax, b[3])
		zmin, zmax = min(zmin, b[4]), max(zmax, b[5])
		found = True
	if not found:
		return None
	return (xmin, xmax, ymin, ymax, zmin, zmax)


def _set_camera_view(
	renderer: vtk.vtkRenderer,
	poly: vtk.vtkPolyData,
	view: str,
	zoom: float,
	bounds: Optional[Sequence[float]] = None,
) -> None:
	direction, view_up = VIEW_DIRECTIONS[view]
	# Frame using shared `bounds` when provided so every panel renders at the same scale;
	# otherwise fall back to this mesh's own bounds.
	b = tuple(bounds) if bounds is not None else poly.GetBounds()
	cx = 0.5 * (b[0] + b[1])
	cy = 0.5 * (b[2] + b[3])
	cz = 0.5 * (b[4] + b[5])
	diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))
	if not np.isfinite(diag) or diag <= 0:
		diag = 1.0
	dist = diag * 2.0
	cam = renderer.GetActiveCamera()
	cam.SetFocalPoint(cx, cy, cz)
	cam.SetPosition(cx + direction[0] * dist, cy + direction[1] * dist, cz + direction[2] * dist)
	cam.SetViewUp(*view_up)
	# Reset to the shared bounds (not the renderer's actor bounds) so the framing is identical
	# across meshes; then apply the user zoom.
	renderer.ResetCamera(b[0], b[1], b[2], b[3], b[4], b[5])
	cam.Zoom(zoom)


def _build_lookup_table(cmap: str, rng: Tuple[float, float]) -> vtk.vtkLookupTable:
	lut = vtk.vtkLookupTable()
	lut.SetTableRange(rng[0], rng[1])
	cmap = (cmap or 'coolwarm').lower()
	if cmap in ('grayscale', 'gray', 'grey'):
		lut.SetHueRange(0.0, 0.0)
		lut.SetSaturationRange(0.0, 0.0)
		lut.SetValueRange(0.2, 1.0)
	elif cmap == 'jet':
		lut.SetHueRange(0.667, 0.0)
		lut.SetSaturationRange(1.0, 1.0)
		lut.SetValueRange(1.0, 1.0)
	elif cmap == 'viridis':
		# Approximate viridis via a few control colors.
		ctf = vtk.vtkColorTransferFunction()
		ctf.AddRGBPoint(rng[0], 0.267, 0.005, 0.329)
		ctf.AddRGBPoint(0.5 * (rng[0] + rng[1]), 0.128, 0.567, 0.551)
		ctf.AddRGBPoint(rng[1], 0.993, 0.906, 0.144)
		n = 256
		lut.SetNumberOfTableValues(n)
		for i in range(n):
			x = rng[0] + (rng[1] - rng[0]) * i / (n - 1)
			r, g, bl = ctf.GetColor(x)
			lut.SetTableValue(i, r, g, bl, 1.0)
		lut.Build()
		return lut
	else:  # coolwarm (default): blue -> white -> red
		ctf = vtk.vtkColorTransferFunction()
		ctf.AddRGBPoint(rng[0], 0.230, 0.299, 0.754)
		ctf.AddRGBPoint(0.5 * (rng[0] + rng[1]), 0.865, 0.865, 0.865)
		ctf.AddRGBPoint(rng[1], 0.706, 0.016, 0.150)
		n = 256
		lut.SetNumberOfTableValues(n)
		for i in range(n):
			x = rng[0] + (rng[1] - rng[0]) * i / (n - 1)
			r, g, bl = ctf.GetColor(x)
			lut.SetTableValue(i, r, g, bl, 1.0)
		lut.Build()
		return lut
	lut.Build()
	return lut


def _vtk_image_to_numpy(image: vtk.vtkImageData) -> np.ndarray:
	"""Convert a vtkImageData RGB(A) screenshot to an (H, W, 3) uint8 numpy array (top-down)."""
	dims = image.GetDimensions()  # (w, h, 1)
	scalars = image.GetPointData().GetScalars()
	arr = numpy_support.vtk_to_numpy(scalars)
	w, h = dims[0], dims[1]
	comps = arr.shape[1] if arr.ndim == 2 else 1
	arr = arr.reshape(h, w, comps)
	# VTK image origin is bottom-left; flip vertically for normal top-down display.
	arr = np.flipud(arr)
	if comps >= 3:
		arr = arr[:, :, :3]
	else:
		arr = np.repeat(arr[:, :, :1], 3, axis=2)
	return arr.astype(np.uint8)


def _crop_panels_uniform(
	panels: Sequence[np.ndarray],
	pad: int = 6,
	tol: int = 8,
) -> List[np.ndarray]:
	"""Crop the shared background margin off all panels using one common bounding box.

	The background color is sampled from the panels' corners. The crop box is the union of
	every panel's non-background content (so each panel keeps the same scale and alignment),
	then expanded by `pad` pixels. Returns the cropped panels (same order, same shared size).
	"""
	if not panels:
		return list(panels)
	# Only crop when all panels share the same shape (they do in normal operation).
	shape = panels[0].shape
	if any(p.shape != shape for p in panels):
		return list(panels)
	h, w = shape[:2]
	# Sample background from the four corners, averaged across panels.
	corners = np.array([
		[p[0, 0], p[0, -1], p[-1, 0], p[-1, -1]] for p in panels
	], dtype=np.float32).reshape(-1, shape[2] if len(shape) == 3 else 1)
	bg = np.median(corners, axis=0)

	any_content = np.zeros((h, w), dtype=bool)
	for p in panels:
		diff = np.abs(p.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
		any_content |= diff > tol
	if not any_content.any():
		return list(panels)

	rows = np.where(any_content.any(axis=1))[0]
	cols = np.where(any_content.any(axis=0))[0]
	r0 = max(0, int(rows[0]) - pad)
	r1 = min(h, int(rows[-1]) + 1 + pad)
	c0 = max(0, int(cols[0]) - pad)
	c1 = min(w, int(cols[-1]) + 1 + pad)
	return [p[r0:r1, c0:c1] for p in panels]


def _save_comparison_matplotlib(
	panels: Sequence[np.ndarray],
	titles: Sequence[str],
	out_path: str,
	suptitle: Optional[str] = None,
	dpi: int = 100,
) -> bool:
	"""Compose panels into a single row figure with titles. Returns True on success."""
	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
	except Exception:
		return False

	n = len(panels)
	if n == 0:
		return False
	h, w = panels[0].shape[:2]
	# Size the figure to the panels' aspect ratio so imshow (equal aspect) fills each axes
	# without leaving horizontal white bars. Small extra height covers titles/suptitle.
	panel_w_in = w / dpi
	panel_h_in = h / dpi
	title_pad_in = 0.35 + (0.4 if suptitle else 0.0)
	fig_w = max(3.0, panel_w_in * n)
	fig_h = max(3.0, panel_h_in + title_pad_in)
	fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), dpi=dpi, constrained_layout=True)
	if n == 1:
		axes = [axes]
	for ax, img, title in zip(axes, panels, titles):
		ax.imshow(img)
		ax.set_title(title, fontsize=11)
		ax.axis('off')
	if suptitle:
		fig.suptitle(suptitle, fontsize=13)
	# Collapse the gaps between panels and the figure margins.
	try:
		fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.0)
	except Exception:
		fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
	fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
	plt.close(fig)
	return True


def _save_comparison_numpy(panels: Sequence[np.ndarray], out_path: str) -> None:
	"""Fallback: tile panels horizontally (padding to equal height) and write a PNG via VTK."""
	if not panels:
		return
	max_h = max(p.shape[0] for p in panels)
	padded = []
	for p in panels:
		if p.shape[0] < max_h:
			pad = np.full((max_h - p.shape[0], p.shape[1], 3), 255, dtype=np.uint8)
			p = np.concatenate([p, pad], axis=0)
		padded.append(p)
	tiled = np.concatenate(padded, axis=1)
	_write_png(out_path, tiled)


def _write_png(out_path: str, rgb: np.ndarray) -> None:
	"""Write an (H, W, 3) uint8 array to a PNG using VTK (no matplotlib needed)."""
	h, w = rgb.shape[:2]
	flipped = np.flipud(rgb)  # VTK expects bottom-left origin
	flat = flipped.reshape(-1, 3).astype(np.uint8)
	vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
	vtk_arr.SetNumberOfComponents(3)
	image = vtk.vtkImageData()
	image.SetDimensions(w, h, 1)
	image.GetPointData().SetScalars(vtk_arr)
	writer = vtk.vtkPNGWriter()
	writer.SetFileName(out_path)
	writer.SetInputData(image)
	writer.Write()


def render_case_comparison(
	case: str,
	mesh_paths: Sequence[Tuple[str, str]],
	out_dir: str,
	views: Sequence[str],
	window_size: Tuple[int, int],
	color,
	background,
	zoom: float,
	color_by: Optional[str],
	cmap: str,
	clim: Optional[Tuple[float, float]],
	edges: bool,
	dpi: int,
	quiet: bool,
	clip: bool = False,
	centerline_path: Optional[str] = None,
	clip_temp_dir: Optional[str] = None,
) -> List[str]:
	"""Render one case across all (label, path) meshes and views; write one PNG per view.

	If `clip` is set and a `centerline_path` exists, each mesh is clipped with the
	centerline-based boxes (same logic as compute_metrics_meshes_comparison) before rendering.

	Returns the list of written PNG paths.
	"""
	written: List[str] = []
	# Load the centerline once (shared across all meshes/views) when clipping.
	centerline = None
	if clip:
		if centerline_path and os.path.exists(centerline_path):
			try:
				centerline = load_centerline(centerline_path)
			except Exception as e:
				if not quiet:
					print(f'  Warning: failed to load centerline {centerline_path}: {e}', file=sys.stderr)
		elif not quiet:
			print(f'  Warning: centerline not found for {case}: {centerline_path}. Rendering unclipped.', file=sys.stderr)

	# Load all meshes once; reuse across views.
	loaded: List[Tuple[str, vtk.vtkPolyData]] = []
	for label, path in mesh_paths:
		try:
			poly = load_vtp(path)
			if clip and centerline is not None:
				try:
					poly = clip_surface_mesh(poly, centerline, case_name=f'{case}_{label}', temp_dir=clip_temp_dir)
				except Exception as e:
					if not quiet:
						print(f'  Warning: clipping failed for {case} [{label}]: {e}. Using unclipped mesh.', file=sys.stderr)
			loaded.append((label, poly))
		except Exception as e:
			if not quiet:
				print(f'  Warning: failed to load {path}: {e}', file=sys.stderr)
	if not loaded:
		return written

	# Shared bounds across all meshes in this case so every panel renders at the same scale.
	shared_bounds = _union_bounds([poly for _, poly in loaded])

	for view in views:
		panels: List[np.ndarray] = []
		titles: List[str] = []
		for label, poly in loaded:
			try:
				img = render_polydata_to_array(
					poly, view=view, window_size=window_size, color=color,
					background=background, zoom=zoom, color_by=color_by, cmap=cmap,
					clim=clim, edges=edges, bounds=shared_bounds,
				)
			except Exception as e:
				if not quiet:
					print(f'  Warning: render failed for {case} [{label}] ({view}): {e}', file=sys.stderr)
				img = np.full((window_size[1], window_size[0], 3), 255, dtype=np.uint8)
			panels.append(img)
			titles.append(label)

		# Trim the shared empty background so panels sit close together (equal scale preserved).
		panels = _crop_panels_uniform(panels)

		out_path = os.path.join(out_dir, f'{case}_{view}.png')
		ok = _save_comparison_matplotlib(panels, titles, out_path, suptitle=f'{case} ({view})', dpi=dpi)
		if not ok:
			_save_comparison_numpy(panels, out_path)
		written.append(out_path)
		if not quiet:
			print(f'  Wrote {out_path}')
	return written


def parse_views_arg(s: str) -> List[str]:
	views = [v.strip().lower() for v in str(s).split(',') if v.strip()]
	invalid = [v for v in views if v not in VIEW_DIRECTIONS]
	if invalid:
		raise argparse.ArgumentTypeError(
			f"Invalid view(s): {invalid}. Valid: {', '.join(sorted(VIEW_DIRECTIONS))}"
		)
	return views or ['xy']


def parse_window_size_arg(s: str) -> Tuple[int, int]:
	parts = [p.strip() for p in str(s).replace('x', ',').split(',') if p.strip()]
	try:
		if len(parts) == 1:
			v = int(parts[0])
			return (v, v)
		if len(parts) == 2:
			return (int(parts[0]), int(parts[1]))
	except ValueError:
		pass
	raise argparse.ArgumentTypeError('Invalid window size. Use "700" or "800,600".')


def parse_clim_arg(s: Optional[str]) -> Optional[Tuple[float, float]]:
	if s is None:
		return None
	parts = [p.strip() for p in str(s).split(',') if p.strip()]
	try:
		if len(parts) == 2:
			return (float(parts[0]), float(parts[1]))
	except ValueError:
		pass
	raise argparse.ArgumentTypeError('Invalid clim. Use "min,max" (e.g. "0,9").')


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description='Render matching .vtp meshes side-by-side as .png for visual comparison.'
	)
	p.add_argument('--gt-dir', required=True, help='Ground-truth directory containing .vtp files')
	p.add_argument('--pred-dir', dest='pred_dir', default=None,
		help='Single prediction directory containing .vtp files')
	p.add_argument('--predictions-root', default=None,
		help='Folder containing multiple prediction subfolders. If set, each case is rendered as '
			'GT | <subfolder1> | <subfolder2> | ...')
	p.add_argument('--out-dir', default=None,
		help='Output directory for comparison PNGs. Default: <pred-dir>/renders or '
			'<predictions-root>/renders.')
	p.add_argument('--views', type=parse_views_arg, default=['xy'], metavar='V1,V2,...',
		help=f'Comma-separated camera views. Valid: {", ".join(sorted(VIEW_DIRECTIONS))}. Default: xy')
	p.add_argument('--ext', default='.vtp', help='File extension to look for (default: .vtp)')
	p.add_argument('--window-size', type=parse_window_size_arg, default=(1400, 1400),
		help='Per-panel render size, e.g. "700" or "800,600" (default: 1400,1400)')
	p.add_argument('--color', default='lightsteelblue',
		help='Solid surface color (VTK color name) when not using --color-by (default: lightsteelblue)')
	p.add_argument('--background', default='white', help='Background color (VTK color name, default: white)')
	p.add_argument('--zoom', type=float, default=1.3, help='Camera zoom factor (default: 1.3)')
	p.add_argument('--color-by', default=None,
		help='Name of a point/cell scalar array to color by (e.g. RegionId). Default: solid color.')
	p.add_argument('--cmap', default='coolwarm',
		help='Colormap when using --color-by: coolwarm, viridis, jet, grayscale (default: coolwarm)')
	p.add_argument('--clim', type=parse_clim_arg, default=None,
		help='Scalar color range "min,max" when using --color-by (default: array range)')
	p.add_argument('--edges', action='store_true', help='Draw mesh edges (wireframe overlay)')
	p.add_argument('--dpi', type=int, default=200, help='Output figure DPI (default: 200)')
	p.add_argument('--clip', action='store_true',
		help='Clip meshes using centerline-based boxes before rendering. Requires --centerline-dir.')
	p.add_argument('--centerline-dir', '--dir-cent', dest='centerline_dir', default=None,
		help='Directory containing centerline .vtp files (matched by case name). Required with --clip.')
	p.add_argument('--clip-temp-dir', default=None,
		help='Temporary directory for clipping box files (default: script directory)')
	p.add_argument('--quiet', action='store_true', help='Reduce logging')
	return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	gt_dir = args.gt_dir
	pred_dir = args.pred_dir
	predictions_root = args.predictions_root
	ext = args.ext if args.ext.startswith('.') else '.' + args.ext

	if not os.path.isdir(gt_dir):
		print(f'Error: gt-dir is not a directory: {gt_dir}', file=sys.stderr)
		return 1
	if predictions_root is None and pred_dir is None:
		print('Error: provide --pred-dir or --predictions-root', file=sys.stderr)
		return 1

	clip = args.clip
	centerline_dir = args.centerline_dir
	clip_temp_dir = args.clip_temp_dir
	if clip:
		if centerline_dir is None:
			print('Error: --centerline-dir (--dir-cent) is required when --clip is used', file=sys.stderr)
			return 1
		if not os.path.isdir(centerline_dir):
			print(f'Error: centerline directory does not exist: {centerline_dir}', file=sys.stderr)
			return 1

	output_base = args.out_dir if args.out_dir is not None else (
		os.path.join(predictions_root, 'renders') if predictions_root is not None
		else os.path.join(pred_dir, 'renders')
	)
	os.makedirs(output_base, exist_ok=True)

	color = args.color
	background = args.background

	# Multi-pred mode: GT | subfolder1 | subfolder2 | ...
	if predictions_root is not None:
		if not os.path.isdir(predictions_root):
			print(f'Error: predictions-root is not a directory: {predictions_root}', file=sys.stderr)
			return 1
		subdirs = sorted(
			d for d in os.listdir(predictions_root)
			if os.path.isdir(os.path.join(predictions_root, d)) and d != os.path.basename(output_base)
		)
		if not subdirs:
			print(f'Error: no subdirectories found in {predictions_root}', file=sys.stderr)
			return 1
		pred_dirs = [os.path.join(predictions_root, d) for d in subdirs]
		common_cases = find_cases_in_all_dirs(gt_dir, pred_dirs, ext=ext)
		if not common_cases:
			print('Error: no cases found in gt_dir and all prediction folders', file=sys.stderr)
			return 2
		cases = sorted(common_cases)
		if not args.quiet:
			print(f'Rendering {len(cases)} cases: GT + {len(subdirs)} prediction folders -> {output_base}')
		for case in tqdm(cases, desc='Cases', disable=args.quiet):
			mesh_paths = [('GT', os.path.join(gt_dir, case + ext))]
			for name, d in zip(subdirs, pred_dirs):
				mesh_paths.append((name, os.path.join(d, case + ext)))
			centerline_path = os.path.join(centerline_dir, case + ext) if clip and centerline_dir else None
			render_case_comparison(
				case, mesh_paths, output_base, args.views, args.window_size,
				color, background, args.zoom, args.color_by, args.cmap, args.clim,
				args.edges, args.dpi, args.quiet,
				clip=clip, centerline_path=centerline_path, clip_temp_dir=clip_temp_dir,
			)
		print(f'Done. Comparison images written to: {output_base}')
		return 0

	# Single-pred mode: GT | pred
	if not os.path.isdir(pred_dir):
		print(f'Error: pred-dir is not a directory: {pred_dir}', file=sys.stderr)
		return 1
	pairs = find_matching_vtps(gt_dir, pred_dir, ext=ext)
	if not pairs:
		print(f'No matching {ext} files found in {gt_dir} and {pred_dir}', file=sys.stderr)
		return 2
	if not args.quiet:
		print(f'Rendering {len(pairs)} cases: GT vs pred -> {output_base}')
	for case, gt_path, pred_path in tqdm(pairs, desc='Cases', disable=args.quiet):
		mesh_paths = [('GT', gt_path), ('pred', pred_path)]
		centerline_path = os.path.join(centerline_dir, case + ext) if clip and centerline_dir else None
		render_case_comparison(
			case, mesh_paths, output_base, args.views, args.window_size,
			color, background, args.zoom, args.color_by, args.cmap, args.clim,
			args.edges, args.dpi, args.quiet,
			clip=clip, centerline_path=centerline_path, clip_temp_dir=clip_temp_dir,
		)
	print(f'Done. Comparison images written to: {output_base}')
	return 0


if __name__ == '__main__':
	"""
	python analysis/render_meshes_comparison.py \
		--gt-dir "/Users/nsveinsson/Documents/datasets/vmr/vmr_all/surfaces" \
		--predictions-root "/Users/nsveinsson/Documents/data_papers/data_vesselsmoothnet/comparison_main" \
		--out-dir "/Users/nsveinsson/Documents/datasets/vmr/vmr_all/renders" \
		--views xy,xz
	"""
	raise SystemExit(main())
