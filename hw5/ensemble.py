import sys
import keras as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from utils import *



dirlist = [
# './old/20/model.8.0.83465-0.84766.h5',
# './noupdate/0.83464/model.2.0.43227-0.85396.h5',
# './old/20/model.6.0.39986-0.84759.h5',
# './model.0.0.83417-0.85077.h5',
# './model.1.0.89832-0.84733.h5',
# './model.0.1.28535-0.84595.h5',
# './model.1.1.26059-0.84608.h5',
# -----------------
# './model.0.0.83945-0.37798.h5',
# './model.2.0.84310-0.37812.h5',
# './model.5.0.84050-0.38042.h5',
# './model.7.0.84340-0.38507.h5',
# './model.8.0.83795-0.37497.h5',
# './model.9.0.83810-0.38303.h5',
# './model.12.0.83910-0.37530.h5',
# './model.13.0.84115-0.38121.h5',
# './model.17.0.83880-0.38037.h5',
# './model.19.0.83825-0.38901.h5',
# -------------------------------
'./model.8.0.85198-0.46535.h5',
'./model.15.0.85192-0.47228.h5',
'./model.1.0.85213-0.47957.h5',
'./model.13.0.85185-0.47526.h5',
'./model.14.0.85118-0.46749.h5',
'./model.12.0.85117-0.46903.h5',
'./model.3.0.85068-0.46743.h5',
'./model.11.0.85047-0.49322.h5',
'./model.5.0.85058-0.47751.h5',
'./model.9.0.85045-0.44570.h5',
 
]
if len(sys.argv) > 2:
    dirlist = sys.argv[1:]

models = [K.models.load_model(d) for d in dirlist]
AllIn = models[0].input
embed = models[0].layers[0](AllIn)

dummyname = 0
new_out = []
for mo in models:
  # mo.input.name = mo.input.name + dummyname
  # dummyname += 'x'
  last = embed
  for i, layer in enumerate(mo.layers[1:]):
    layer.name = layer.name + str(dummyname)
    dummyname += 1
    print(layer)
    # if i == 0:
    #   continue
    last = layer(last)
    # layer.name = dummy_name
    # new_layer = layer(new_layer)
    # dummy_name += 'x'
  new_out.append(last)

new_out = K.layers.add(new_out)
nmodel = K.models.Model(inputs=AllIn, outputs=new_out)
nmodel.summary()
K.models.save_model(nmodel, './ensemble.h5')