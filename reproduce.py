#!/usr/bin/env python3
"""Render declared panels from hash-pinned saved tables; never run analysis sources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def catalog():
    return json.loads((ROOT / 'manifests/figures.json').read_text())


def check_sources():
    records = json.loads((ROOT / 'manifests/sources.json').read_text())['files']
    paths = [r['path'] for r in records]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate source manifest entry')
    for record in records:
        path = ROOT / record['path']
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('Source escaped repository')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Source hash mismatch: ' + record['path'])
        ast.parse(path.read_text(), filename=str(path))
    print(f'{len(records)} unique reference sources: hash and syntax passed')


def read_table(record, source_root):
    import pandas as pd
    root = Path(source_root).resolve()
    path = root / record['input']
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Input escaped source-data directory')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('Input does not match the reviewed table: ' + record['input'])
    table = pd.read_csv(path)
    missing = set(record['columns']) - set(table.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    return table


def fonts(font_dir):
    from matplotlib import font_manager, rcParams
    root = Path(font_dir).resolve()
    for filename, style in [('arial.ttf', 'regular'), ('arialbd.ttf', 'bold')]:
        path = root / filename
        face = font_manager.get_font(str(path))
        if face.family_name != 'Arial' or ('bold' in face.style_name.lower()) != (style == 'bold'):
            raise ValueError('Expected Arial ' + style + ': ' + str(path))
        font_manager.fontManager.addfont(str(path))
    rcParams.update({'font.family':'Arial', 'font.size':10, 'svg.fonttype':'none',
                     'axes.spines.top':False, 'axes.spines.right':False})
    for filename, weight in [('arial.ttf','normal'), ('arialbd.ttf','bold')]:
        resolved = font_manager.findfont(font_manager.FontProperties(family='Arial',weight=weight), fallback_to_default=False)
        if Path(resolved).resolve() != root / filename:
            raise ValueError('Arial resolution did not select the supplied face')


def display(value):
    # Display names only. Saved column keys, scores, masks and statistics are unchanged.
    replacements = {'CD4_CTL':'CD4 CTL','CD8T':'CD8 T','Activation_Score':'Activation',
                    'Exhaustion_Score':'Exhaustion','HLA-DR+ B Cell':'HLA-II+ B cells',
                    'HLA-DR- B Cell':'HLA-II− B cells',
                    'DR+ B-dominant motif':'HLA-II+ B-dominant',
                    'DR- B-dominant motif':'HLA-II− B-dominant',
                    'Mixed B motif':'Mixed B','B-poor motif':'B-poor'}
    return replacements.get(str(value), str(value))


def heatmap(ax, matrix, title, cmap='viridis', symmetric=False):
    import numpy as np
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Missing or nonfinite heatmap cells; no imputation is performed')
    limit = float(np.abs(values).max()) if symmetric else None
    img = ax.imshow(values, aspect='auto', cmap=cmap,
                    vmin=-limit if symmetric else None, vmax=limit)
    ax.set_xticks(range(len(matrix.columns)), [display(x) for x in matrix.columns], rotation=40, ha='right')
    ax.set_yticks(range(len(matrix.index)), [display(x) for x in matrix.index])
    ax.set_title(title, fontweight='bold')
    return img


def draw(panel, table):
    import matplotlib.pyplot as plt
    import numpy as np
    if panel == '1b':
        # pivot is a reshape: duplicate tissue/lineage records fail instead of being averaged.
        order = list(dict.fromkeys(table['Run_Tissue_name']))
        matrix = table.pivot(index='Run_Tissue_name', columns='GT_Lineage_Broad', values='Tissue_lineage_percent').reindex(order).fillna(0)
        fig, ax = plt.subplots(figsize=(14,6)); bottom=np.zeros(len(matrix))
        for lineage in matrix:
            ax.bar(range(len(matrix)), matrix[lineage], bottom=bottom, label=lineage)
            bottom += matrix[lineage].to_numpy()
        ax.set_xticks(range(len(matrix)), matrix.index, rotation=75, ha='right')
        ax.set_ylabel('Cells within tissue (%)'); ax.legend(loc='upper left', bbox_to_anchor=(1.01,1))
        ax.set_ylim(0,100);fig.subplots_adjust(left=.06,right=.80,bottom=.35,top=.9)
    elif panel == '2a':
        markers=table.sort_values('Marker_order')['Marker'].drop_duplicates().tolist()
        models=table.sort_values('Model_index')['Model'].drop_duplicates().tolist()
        fig,ax=plt.subplots(figsize=(8,10))
        img=ax.scatter(table['Model'].map({x:i for i,x in enumerate(models)}),
                       table['Marker'].map({x:i for i,x in enumerate(markers)}),
                       s=table['Importance_norm']*140, c=table['NegLog10_P_display'],cmap='viridis',vmin=0,vmax=50)
        ax.set_xticks(range(len(models)),models,rotation=30,ha='right')
        ax.set_yticks(range(len(markers)),markers);ax.invert_yaxis()
        ax.set_xlabel('Model (bubble area: normalized importance)')
        fig.colorbar(img,ax=ax,label='−log10 one-sided Welch P (display cap 50)')
        fig.subplots_adjust(left=.18,right=.90,bottom=.12,top=.94)
    elif panel in ('3a','4a'):
        if panel=='3a':
            groups=[('Broad lineage',table)]; metrics=['Balanced Acc','Macro F1','Weighted F1']
        else:
            groups=list(table.groupby('Lineage',sort=False)); metrics=['Balanced_Accuracy','Macro_F1','Weighted_F1']
        fig, axes=plt.subplots(1,len(groups),figsize=(7*len(groups),7),squeeze=False)
        for ax,(name,group) in zip(axes[0],groups):
            matrix=group.set_index('Model')[metrics]
            img=heatmap(ax,matrix,name)
            img.set_clim(0,1);fig.colorbar(img,ax=ax,fraction=.04,pad=.02,label='Saved score')
        fig.subplots_adjust(left=.31 if len(groups)==1 else .07,right=.87 if len(groups)==1 else .98,bottom=.23,top=.9,wspace=.70)
    elif panel=='5c':
        scores=table['Score'].drop_duplicates(); subsets=table['T_Subset'].drop_duplicates()
        fig,axes=plt.subplots(len(scores),len(subsets),figsize=(18,7),squeeze=False)
        for i,score in enumerate(scores):
            for j,subset in enumerate(subsets):
                ax=axes[i,j]; group=table[(table.Score==score)&(table.T_Subset==subset)]
                for state,color in [('HLA-DR+ B Cell','#c44e52'),('HLA-DR- B Cell','#4c72b0')]:
                    part=group[group.Nearest_B_State==state].sort_values('Bin_midpoint_um')
                    ax.plot(part.Bin_midpoint_um,part.Mean_score,marker='o',color=color,label=display(state))
                ax.set_title(display(subset));ax.set_xlabel('Distance (µm)')
                if j==0:ax.set_ylabel(display(score)+' (saved pooled mean)')
        axes[0,-1].legend(loc='upper left',bbox_to_anchor=(1.03,1));fig.subplots_adjust(left=.06,right=.85,bottom=.10,top=.92,wspace=.50,hspace=.50)
    elif panel=='S1c':
        groups=list(table.groupby('Stage',sort=False));fig,axes=plt.subplots(1,len(groups),figsize=(13,6))
        for ax,(stage,group) in zip(np.atleast_1d(axes),groups):
            matrix=group.pivot(index='GT_lineage',columns='Predicted_lineage',values='GT_row_percent')
            img=heatmap(ax,matrix,stage,cmap='Blues');img.set_clim(0,100)
            ax.set_xlabel('Predicted lineage');ax.set_ylabel('GT lineage')
            fig.colorbar(img,ax=ax,label='Saved row percentage')
        fig.subplots_adjust(left=.13,right=.97,bottom=.27,top=.9,wspace=.55)
    elif panel=='S3b':
        fig,ax=plt.subplots(figsize=(7,5));t=table.sort_values('GRZB_mean')
        ax.plot(t.GRZB_mean,t.High_Component_Prob_median,marker='o',markersize=3)
        cutoffs=t.Selected_cutoff.drop_duplicates()
        if len(cutoffs)!=1:raise ValueError('Expected one saved cutoff')
        ax.axvline(cutoffs.iloc[0],color='black',linestyle='--',label='Saved cutoff')
        ax.set(xlabel='GZMB expression (saved standardized value)',ylabel='Median high-component posterior',ylim=(0,1.03))
        ax.legend();fig.subplots_adjust(left=.15,right=.96,bottom=.16,top=.9)
    elif panel=='S6e':
        scores=list(table.Score.drop_duplicates());fig,axes=plt.subplots(1,len(scores),figsize=(17,6))
        for ax,score in zip(np.atleast_1d(axes),scores):
            group=table[table.Score==score].copy()
            group['Comparison']=[display(a)+'\n− '+display(b) for a,b in zip(group.Group_A,group.Group_B)]
            matrix=group.pivot(index='T_Subset_Group',columns='Comparison',values='Mean_Delta_A_minus_B')
            img=heatmap(ax,matrix,display(score),cmap='coolwarm',symmetric=True)
            fig.colorbar(img,ax=ax,label='Saved pooled mean difference (A − B)')
        fig.subplots_adjust(left=.08,right=.98,bottom=.42,top=.9,wspace=.45)
    else:
        raise ValueError('No saved-table renderer for '+panel)
    fig.suptitle('Figure '+panel+' · saved-table replay',fontweight='bold')
    return fig


def validate_canvas(fig):
    from matplotlib.text import Text
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
    # Matplotlib creates extra ticks outside the view interval but does not draw
    # them. Exclude those undrawn labels, while checking every visible tick.
    undrawn = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    undrawn.update((id(tick.label1), id(tick.label2)))
    for artist in fig.findobj(match=Text):
        if id(artist) in undrawn or not artist.get_visible() or not artist.get_text().strip():continue
        box=artist.get_window_extent(renderer)
        if box.x0 < bounds.x0-1 or box.y0 < bounds.y0-1 or box.x1 > bounds.x1+1 or box.y1 > bounds.y1+1:
            raise ValueError('Text outside designed canvas: '+artist.get_text())


def render(args):
    records=catalog()['panels']; selected=args.panel
    if selected not in records or records[selected]['status']!='saved_table_replay':
        raise ValueError('Unsupported panel. Run list for coverage and required inputs.')
    record=records[selected];table=read_table(record,args.source_data)
    out=Path(args.output).expanduser().absolute()
    if any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('Output must not traverse symlinks')
    out = out.resolve()
    if out.exists():raise ValueError('Output must be a new directory; existing outputs are protected')
    for protected in [ROOT,Path(args.source_data).resolve()]:
        if out.is_relative_to(protected) or protected.is_relative_to(out):raise ValueError('Output overlaps code or source data')
    if any(part in ('Final','V9','Legacy','checkpoints','source_data') for part in out.parts):
        raise ValueError('Output must be outside protected analysis/source-data trees')
    import matplotlib
    matplotlib.use('Agg')
    fonts(args.font_dir);fig=draw(selected,table);validate_canvas(fig)
    out.mkdir(parents=True,exist_ok=False)
    for suffix in ['png','svg']:
        fig.savefig(out/f'Figure{selected}.{suffix}',dpi=160)
    (out/'replay.json').write_text(json.dumps({'panel':selected,'input':record['input'],
        'sha256':record['sha256'],'statistics_recomputed':False,'models_fitted':False,
        'manuscript_composite_reproduced':False,'formats':['png','svg']},indent=2)+'\n')
    import matplotlib.pyplot as plt
    plt.close(fig);print('Saved '+str(out))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list');sub.add_parser('check')
    p=sub.add_parser('render');p.add_argument('--panel',required=True);p.add_argument('--source-data',required=True)
    p.add_argument('--font-dir',required=True,help='Directory containing licensed arial.ttf and arialbd.ttf')
    p.add_argument('--output',required=True,help='New directory outside code, inputs and protected analysis trees')
    args=parser.parse_args()
    if args.command=='list':
        for key,value in catalog()['panels'].items():print(key+'\t'+value['status']+'\t'+value['description'])
    elif args.command=='check':check_sources()
    else:render(args)


if __name__=='__main__':
    try:main()
    except (ValueError,OSError) as error:raise SystemExit(str(error))
