import pandas as pd
import affine
import os
import drago

print('Importing images.csv\n')
# Test mahmouds model
image_list = pd.read_csv('/osn/SpaceNet-MOD/testing/rgb-ps-dra/300/images.csv', index_col=0)
image_list['affine'] = image_list.affine.apply(lambda x: affine.Affine(*eval(x)[:6]))

print('Reading SSD results csv\n')

dtects = pd.read_csv('~/brian-ssd/ssd_keras/ssd_boats_multiclass_filtered/ssd_multiclass_filtered_results.csv', index_col=0)
# dtects = dtects[dtects.conf > 0.24284106791019441]
dtects['obj_class'] = dtects.class_id.astype(str)
dtects['box'] = [tuple(row) for row in dtects[['xmin', 'ymin', 'xmax', 'ymax']].values]

print('Performing pandas join\n')
joined = dtects.set_index('file_name').join(image_list.set_index('file_name'), how='inner')

print('Starting Drago logger\n')
dlog = drago.detection.DetectLogger()

print('Logging joined boxes\n')
for i, row in joined.iterrows():
    dlog.box(row.cat_id, row.obj_class, row.conf, row.affine, *row.box)

print('Saving geojson to disk\n')

dlog.save('~/brian-ssd/ssd_keras/ssd_boats_multiclass_filtered/ssd_boats_multiclass_filtered.geojson')
print('Done!')
