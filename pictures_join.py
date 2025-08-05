
from PIL import Image
import os
from pathlib import Path

base_dir = Path.cwd()
parent_path = base_dir.parent
pictures = ['TRENB', 'PSO']
datasets1 = ['Algerian','Banknote']
datasets2 = ['Climate','Diabetes']
datasets3 = ['Electrical','German']

bg = Image.new('RGB', (650, 900), 'white')

for i in range(3):
    for row, data in enumerate(datasets1 if i == 0 else datasets2 if i == 1 else datasets3 ):
        for col, pic_name in enumerate(pictures):
            img_path = os.path.join(parent_path, 'charts', f'{data}_{pic_name}_prior_c1_boxplot.jpg')
            img = Image.open(img_path).resize((300, 500))
            x = (col * 300) + 25          # 每張圖片寬 300，按欄擺放
            y = row * 450          # 每列高度 500，按行擺放
            bg.paste(img, (x, y))

    object_dir = os.path.join(parent_path, 'charts','TRENB_PSO_Compare')
    match i :
        case 0:
            bg.save(os.path.join(object_dir, f'{datasets1[0]}_{datasets1[1]}_prior_c1_boxplot.jpg'))
        case 1:
            bg.save(os.path.join(object_dir, f'{datasets2[0]}_{datasets2[1]}_prior_c1_boxplot.jpg'))
        case 2:
            bg.save(os.path.join(object_dir, f'{datasets3[0]}_{datasets3[1]}_prior_c1_boxplot.jpg'))