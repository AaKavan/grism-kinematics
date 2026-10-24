"""Execute the archived real-source notebook with fitting disabled."""
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def run(args):
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / 'notebooks/archive/real_source_kinematics.ipynb', as_version=4)
    settings = next(c for c in notebook.cells if c.cell_type == 'code' and c.source.startswith('SOURCE_ID'))
    settings.source += '\nRUN_FIT = False\n'
    for cell in notebook.cells:
        if args.skip_widgets and cell.cell_type == 'code' and 'review.inspector(' in cell.source:
            cell.source = 'print("Interactive controls omitted from this static export; use the source notebook for widgets.")'
        if cell.cell_type == 'code' and cell.source.startswith('RUN_MCMC'):
            cell.source = 'RUN_MCMC = False\n' + cell.source.split('\n', 1)[1]
            if args.saved_posterior:
                marker = 'from dingo.real_review import show_posterior'
                cell.source = cell.source.replace(marker,
                    f'SAVED_POSTERIOR = {str(args.saved_posterior.resolve())!r}\n\n' + marker)
    if args.source_id:
        settings.source += f'\nSOURCE_ID = {args.source_id}\n'
    if args.saved_run:
        settings.source += f'\nSAVED_RUN = {str(args.saved_run.resolve())!r}\n'
    nbformat.validate(notebook)
    NotebookClient(notebook, timeout=120, kernel_name='grism',
                   resources={'metadata': {'path': str(root)}}, allow_errors=False,
                   on_cell_start=lambda cell, cell_index: print(f'Cell {cell_index+1}/{len(notebook.cells)}', flush=True)).execute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    errors = [o for c in notebook.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o['output_type'] == 'error']
    images = sum('image/png' in o.get('data', {}) for c in notebook.cells if c.cell_type == 'code' for o in c.get('outputs', []))
    print(f'Executed {len(notebook.cells)} cells: {len(errors)} errors, {images} static PNG outputs. Saved {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-id', type=int)
    parser.add_argument('--saved-run', type=Path)
    parser.add_argument('--saved-posterior', type=Path)
    parser.add_argument('--skip-widgets', action='store_true',
                        help='Static export without live widget communication; keep all static fit/posterior plots')
    run(parser.parse_args())
