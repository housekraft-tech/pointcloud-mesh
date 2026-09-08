"""Side-by-side native CAD renders, without modifying their geometry or camera."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[2]/'output_final/rectangular_rebuild_v1'


def main():
    fig,axes=plt.subplots(1,2,figsize=(18,8),facecolor='#f7f8fa')
    variants=[('mujammel_strict','STRICT RECTANGLES - NO FILL'),('mujammel','CONTINUITY-ASSISTED RECTANGLES')]
    for ax,(name,title) in zip(axes,variants):
        report=json.loads((BASE/name/'audit.json').read_text())['summary']
        support=report['area_weighted_scan_support']
        ax.imshow(plt.imread(BASE/name/'native_3d.png'));ax.axis('off')
        ax.set_title(title,fontsize=15,pad=16)
        ax.text(.5,-.02,f"{report['rectangles']} rectangles | {support['within_10mm_pct']:.1f}% within 10 mm | {support['within_50mm_pct']:.1f}% within 50 mm",transform=ax.transAxes,ha='center',fontsize=11)
    fig.suptitle('ENGRANCE  /  KOUSHIK - MUJAMMEL',fontsize=22,x=.04,ha='left')
    fig.text(.04,.02,'Left: no inferred continuity; more gaps.  Right: amber faces include continuity across small unobserved regions.\nBoth are partial surface models. Proximity to a return is not proof that it belongs to a wall. These are actual SketchUp renders.',fontsize=11)
    fig.subplots_adjust(left=.02,right=.98,top=.87,bottom=.12,wspace=.02)
    fig.savefig(BASE/'Engrance_two_flows.png',dpi=160);plt.close(fig)
    print(BASE/'Engrance_two_flows.png')
    fig,axes=plt.subplots(2,3,figsize=(18,10),facecolor='#f7f8fa')
    for row,(name,title) in enumerate(variants):
        for ax,view in zip(axes[row],['top','side','3d']):
            ax.imshow(plt.imread(BASE/name/f'native_{view}.png'));ax.axis('off')
            ax.set_title(f'{title}\n{view.upper()}',fontsize=12)
    fig.suptitle('Engrance - same native viewpoints, both flows',fontsize=20)
    fig.subplots_adjust(left=.01,right=.99,top=.9,bottom=.02,hspace=.18,wspace=.02)
    fig.savefig(BASE/'Engrance_two_flows_top_side_3d.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
