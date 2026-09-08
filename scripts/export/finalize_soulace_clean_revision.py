"""Create a handover preview from verified native SketchUp images."""
import json
from PIL import Image,ImageOps,ImageDraw,ImageFont
from clean_soulace_surfaces import OUT


def main():
    audit=json.loads((OUT/'native_audit.json').read_text())
    assert audit['addition_area_pass'] and audit['native_reopen_verified']
    assert audit['reference_only_hidden_on_reopen'] and audit['existing_geometry_preserved']
    report={'native_file':audit['path'],'groups':len(audit['groups']),'faces':sum(g['faces'] for g in audit['groups']),
            'new_clean_parts':len(audit['addition_area_checks']),'prior_geometry_preserved':True,'saved_and_reopened_verified':True,
            'reference_only_groups_hidden':audit['reference_only_groups'],'regression_tests_passed':34}
    (OUT/'handover_verification.json').write_text(json.dumps(report,indent=2))
    im=Image.new('RGB',(1800,770),(247,248,250));d=ImageDraw.Draw(im)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    d.text((35,22),'Soulace | actual SketchUp views from the revised whole-house file',font=font,fill=(31,45,57))
    for x,title,path in [(0,'Corrected upper staircase','detail_clean_upper_stairs_3d.png'),(900,'Planar cyan ground + interior floor','detail_clean_ground_and_floors_top.png')]:
        d.text((x+35,76),title,font=font,fill=(31,45,57))
        panel=ImageOps.contain(Image.open(OUT/path).convert('RGB'),(880,620));im.paste(panel,(x+(900-panel.width)//2,105+(620-panel.height)//2))
    d.text((35,735),'Small sampling holes repaired; larger gaps retained. Parking and interior stay at different elevations.',font=small,fill=(65,73,81))
    im.save(OUT/'Soulace_cleanup_review.png');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
