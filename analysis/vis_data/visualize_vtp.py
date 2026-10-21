"""Render each surface mesh in a folder to XZ and YZ PNG images.

Usage:
    python analysis/vis_data/visualize_vtp.py -i /path/to/meshes -o /path/to/pngs
    python analysis/vis_data/visualize_vtp.py -i /path/to/meshes -o /path/to/pngs -color

`-color` paints the surface by the `RegionId` array (cell data, then point data).
If that array is missing it falls back to `Scalars_`. Without `-color` the mesh
is white. Rendering is off-screen.
"""

import argparse
import os
import sys

import numpy as np
import pyvista as pv

VIEWS = ('xz', 'yz')
MESH_EXTS = {'.vtp', '.stl'}
_SCALAR_NAMES = ('RegionId', 'Scalars_')


def list_meshes(input_dir):
    """Return sorted surface-mesh paths directly inside `input_dir`."""
    paths = []
    for name in sorted(os.listdir(input_dir)):
        if name.startswith('.'):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in MESH_EXTS:
            continue
        path = os.path.join(input_dir, name)
        if os.path.isfile(path):
            paths.append(path)
    return paths


def find_scalar_array(mesh):
    """Return (array_name, values) for RegionId, else Scalars_, else (None, None).

    Cell data is preferred over point data for each name.
    """
    for name in _SCALAR_NAMES:
        if name in mesh.cell_data:
            return name, np.asarray(mesh.cell_data[name])
        if name in mesh.point_data:
            return name, np.asarray(mesh.point_data[name])
    return None, None


def scalar_limits(values):
    """Data range for the colormap. A constant array is given a span of 1."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(finite.min())
    hi = float(finite.max())
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def render_mesh(mesh_file, output_folder, color_by_region=False):
    """Write `{stem}_xz.png` and `{stem}_yz.png`. Returns the written paths."""
    mesh = pv.read(mesh_file)
    if mesh.n_points == 0:
        raise ValueError('mesh has no points')
    scalar_name = None
    clim = None
    if color_by_region:
        scalar_name, values = find_scalar_array(mesh)
        if values is None:
            print(
                f'Warning: no RegionId or Scalars_ on {mesh_file}; rendering white.',
                file=sys.stderr,
            )
        else:
            clim = scalar_limits(values)

    stem = os.path.splitext(os.path.basename(mesh_file))[0]
    written = []
    for position in VIEWS:
        plotter = pv.Plotter(off_screen=True, window_size=(1024, 1024))
        try:
            if scalar_name is not None:
                plotter.add_mesh(
                    mesh,
                    scalars=scalar_name,
                    cmap='coolwarm',
                    clim=list(clim),
                    show_scalar_bar=False,
                )
            else:
                plotter.add_mesh(mesh, color='white')
            plotter.set_background('white')
            plotter.show_axes = False
            plotter.camera_position = position
            plotter.camera.zoom(0.9)
            out_path = os.path.join(output_folder, f'{stem}_{position}.png')
            plotter.screenshot(out_path)
            written.append(out_path)
        finally:
            plotter.close()
    return written


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Render surface meshes in a folder to XZ and YZ PNG images.'
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='Folder of surface meshes (.vtp or .stl)',
    )
    parser.add_argument(
        '-o', '--output', required=True,
        help='Folder to write PNG renders',
    )
    parser.add_argument(
        '-color', '--color', action='store_true',
        help='Color by RegionId, or Scalars_ if RegionId is absent',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_dir = args.input
    output_dir = args.output

    if not os.path.isdir(input_dir):
        print(f'Error: input is not a directory: {input_dir}', file=sys.stderr)
        return 1

    mesh_files = list_meshes(input_dir)
    if not mesh_files:
        print(f'Error: no .vtp or .stl files in {input_dir}', file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)
    n_ok = 0
    for mesh_file in mesh_files:
        try:
            written = render_mesh(mesh_file, output_dir, color_by_region=args.color)
        except Exception as exc:
            print(f'Warning: failed to render {mesh_file}: {exc}', file=sys.stderr)
            continue
        n_ok += 1
        for path in written:
            print(f'Wrote {path}')

    if n_ok == 0:
        print('Error: no meshes were rendered', file=sys.stderr)
        return 1
    print(f'Done. Rendered {n_ok} of {len(mesh_files)} meshes to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
